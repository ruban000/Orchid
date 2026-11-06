// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <cstring>
#include <functional>
#include <chrono>
#include <cstdlib>
#include <string>
#include <string_view>
#include <unordered_set>
#include <fmt/ranges.h>
#include <array>
#include <vector>
#include <boost/container/small_vector.hpp>

#include "common/assert.h"
#include "common/cityhash.h"
#include "shader_recompiler/frontend/maxwell/control_flow.h"
#include "shader_recompiler/exception.h"
#include "shader_recompiler/object_pool.h"
#include "video_core/control/channel_state.h"
#include "video_core/dirty_flags.h"
#include "video_core/engines/kepler_compute.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/host1x/gpu_device_memory_manager.h"
#include "video_core/memory_manager.h"
#include "video_core/native_shader_hitch_trace.h"
#include "video_core/shader_cache.h"
#include "video_core/shader_environment.h"
#include "video_core/texture_cache/util.h"

namespace VideoCommon {

void ShaderCache::InvalidateRegion(VAddr addr, size_t size) {
    std::scoped_lock lock{invalidation_mutex};
    InvalidatePagesInRegion(addr, size);
    RemovePendingShaders();
}

void ShaderCache::InvalidateRegions(std::span<const std::pair<DAddr, size_t>> sequences) {
    if (sequences.empty()) {
        return;
    }
    std::scoped_lock lock{invalidation_mutex};
    for (const auto& [addr, size] : sequences) {
        InvalidatePagesInRegion(addr, size);
    }
    RemovePendingShaders();
}

void ShaderCache::OnCacheInvalidation(VAddr addr, size_t size) {
    std::scoped_lock lock{invalidation_mutex};
    InvalidatePagesInRegion(addr, size);
}

void ShaderCache::SyncGuestHost() {
    std::scoped_lock lock{invalidation_mutex};
    RemovePendingShaders();
}

ShaderCache::ShaderCache(Tegra::MaxwellDeviceMemoryManager& device_memory_)
    : device_memory{device_memory_} {}

// Task and mesh programs are translated with the mesh registers and the header of the pass-through
// geometry program (per-primitive outputs), so pipelines are keyed by them too.
u64 ShaderCache::MeshStateHash() const {
    using Regs = Tegra::Engines::Maxwell3D::Regs;
    const auto& regs{maxwell3d->regs};
    std::array<u32, 3 + 20> state{regs.reg_array[Regs::MeshShaderA], regs.reg_array[Regs::MeshShaderB],
                                  regs.reg_array[Regs::MeshInitShader]};
    if (regs.IsShaderConfigEnabled(Regs::ShaderType::Geometry)) {
        const auto& geometry{regs.pipelines[static_cast<size_t>(Regs::ShaderType::Geometry)]};
        const GPUVAddr address{(static_cast<u64>(geometry.reserved[0] & 0xff) << 32) |
                               geometry.reserved[1]};
        gpu_memory->ReadBlockUnsafe(address, state.data() + 3, 20 * sizeof(u32));
    }
    return Common::CityHash64(reinterpret_cast<const char*>(state.data()), sizeof(state)) | 1;
}

bool ShaderCache::RefreshStages(std::array<u64, 6>& unique_hashes) try {
    ReclaimInvalidatedShaders();
    auto& dirty{maxwell3d->dirty.flags};
    if (!dirty[VideoCommon::Dirty::Shaders]) {
        return last_shaders_valid;
    }
    NativeShaderHitchTrace::SlowScope refresh_hitch{
        "refresh-stages", 0, ~0U,
        Common::HitchDiagnostic::Enabled() && maxwell3d->GetNativeMacroEngine().native_turing};
    dirty[VideoCommon::Dirty::Shaders] = false;

    const GPUVAddr base_addr{maxwell3d->regs.program_region.Address()};
    const bool native = maxwell3d->GetNativeMacroEngine().native_turing;
    const bool mesh{native && maxwell3d->regs.IsMeshEnabled()};
    for (size_t index = 0; index < Tegra::Engines::Maxwell3D::Regs::MaxShaderProgram; ++index) {
        if (!maxwell3d->regs.IsShaderConfigEnabled(index)) {
            unique_hashes[index] = 0;
            continue;
        }
        const auto& shader_config{maxwell3d->regs.pipelines[index]};
        const auto program{static_cast<Tegra::Engines::Maxwell3D::Regs::ShaderType>(index)};
        if (program == Tegra::Engines::Maxwell3D::Regs::ShaderType::Pixel &&
            !maxwell3d->regs.rasterize_enable) {
            unique_hashes[index] = 0;
            continue;
        }
        if (mesh && program == Tegra::Engines::Maxwell3D::Regs::ShaderType::Geometry) {
            // A header without code: the mesh program's environment reads its output map.
            unique_hashes[index] = 0;
            continue;
        }
        const GPUVAddr shader_addr{native
            ? (static_cast<u64>(shader_config.reserved[0] & 0xff) << 32) | shader_config.reserved[1]
            : base_addr + shader_config.offset};
        const std::optional<VAddr> cpu_shader_addr{gpu_memory->GpuToCpuAddress(shader_addr)};
        if (!cpu_shader_addr) {
            LOG_ERROR(HW_GPU, "Invalid GPU address for shader {:#016x}", shader_addr);
            last_shaders_valid = false;
            return false;
        }
        const ShaderInfo* shader_info{TryGet(*cpu_shader_addr)};
        if (!shader_info && native && IsRejected(*cpu_shader_addr) &&
            !RecheckRejection(*cpu_shader_addr)) {
            // A rejected program fails the same way every time it is bound: skip its draws
            // without rethrowing (two C++ throws each time cost ~1-2% of the GPU thread).
            maxwell3d->GetNativeMacroEngine().native_execution_failed = true;
            last_shaders_valid = false;
            return false;
        }
        if (!shader_info) {
            const u32 start_address{native ? 0 : shader_config.offset};
            try {
                // Native programs are lowered while the environment is built (once: the
                // ShaderInfo keeps the result for the pipeline's environments).
                RethrowIfRejected(*cpu_shader_addr);
                GraphicsEnvironment env{*maxwell3d, *gpu_memory, program, native ? shader_addr : base_addr,
                                       start_address, {}, support_cooperative_matrix};
                NativeShaderHitchTrace::SlowScope info_hitch{
                    "shader-info", shader_addr, static_cast<u32>(env.ShaderStage()),
                    Common::HitchDiagnostic::Enabled() && native};
                info_hitch.SourceBytes(env.SourceSizeBytes());
                shader_info = MakeShaderInfo(env, *cpu_shader_addr);
                info_hitch.Finish("ok");
            } catch (const Shader::Exception&) {
                RememberRejection(*cpu_shader_addr, shader_addr);
                // Once per program: which stage slot it is bound to, its header and the slot's
                // registers (to tell how a pipeline with task/mesh programs is set up).
                static std::unordered_set<GPUVAddr> described;
                if (native && described.insert(shader_addr).second) {
                    std::array<u32, 5> header{};
                    gpu_memory->ReadBlockUnsafe(shader_addr, header.data(), sizeof(header));
                    std::array<u32, sizeof(shader_config) / sizeof(u32)> config{};
                    std::memcpy(config.data(), &shader_config, sizeof(config));
                    LOG_ERROR(HW_GPU, "Rejected program {:#x} is bound to stage {}: header {:08x}, slot {:08x}",
                              shader_addr, index, fmt::join(header, " "), fmt::join(config, " "));
                }
                throw;
            }
        }
        shader_infos[index] = shader_info;
        unique_hashes[index] = shader_info->unique_hash;
        if (mesh && (program == Tegra::Engines::Maxwell3D::Regs::ShaderType::VertexB ||
                     program == Tegra::Engines::Maxwell3D::Regs::ShaderType::Tessellation)) {
            unique_hashes[index] ^= MeshStateHash();
        }
    }
    last_shaders_valid = true;
    return true;
} catch (const Shader::Exception& error) {
    auto& native = maxwell3d->GetNativeMacroEngine();
    if (!native.native_turing) throw;
    native.native_execution_failed = true;
    last_shaders_valid = false;
    // Each rejected program is logged once (the draws that use it keep failing quietly).
    static std::unordered_set<std::string> logged;
    if (logged.insert(error.what()).second) {
        LOG_ERROR(HW_GPU, "Native graphics shader rejected: {}", error.what());
    }
    // Once: show what the vertex stage has bound, to find where an unbound bank's data lives.
    static std::vector<std::string> dumped_errors;
    if (const std::string what{error.what()};
        what.find("constant buffer") != std::string::npos && dumped_errors.size() < 12 &&
        std::count(dumped_errors.begin(), dumped_errors.end(), what) < 3) {
        dumped_errors.push_back(what);
        LOG_INFO(HW_GPU, "VS cbuf dump for: {}", what.substr(0, 40));
        for (u32 slot = 3; slot < 8; ++slot) {
            const auto& buffer = maxwell3d->state.shader_stages[0].const_buffers[slot];
            if (!buffer.enabled) continue;
            std::array<u32, 24> words{};
            gpu_memory->ReadBlockUnsafe(buffer.address, words.data(), sizeof(words));
            LOG_INFO(HW_GPU, "VS cbuf {} at {:#x} size {:#x}: {:08x}", slot, buffer.address, buffer.size,
                     fmt::join(words, " "));
        }
    }
    return false;
}

const ShaderInfo* ShaderCache::ComputeShader() {
    ReclaimInvalidatedShaders();
    const GPUVAddr program_base{kepler_compute->regs.code_loc.Address()};
    const auto& qmd{kepler_compute->launch_description};
    const GPUVAddr shader_addr{program_base + qmd.program_start};
    const std::optional<VAddr> cpu_shader_addr{gpu_memory->GpuToCpuAddress(shader_addr)};
    if (!cpu_shader_addr) {
        LOG_ERROR(HW_GPU, "Invalid GPU address for shader {:#016x}", shader_addr);
        return nullptr;
    }
    if (const ShaderInfo* const shader = TryGet(*cpu_shader_addr)) {
        return shader;
    }
    try {
        RecheckRejection(*cpu_shader_addr);
        RethrowIfRejected(*cpu_shader_addr);
        ComputeEnvironment env{*kepler_compute, *gpu_memory, program_base, qmd.program_start,
                               {}, support_cooperative_matrix};
        NativeShaderHitchTrace::SlowScope info_hitch{
            "shader-info", shader_addr, static_cast<u32>(env.ShaderStage()),
            Common::HitchDiagnostic::Enabled() && kepler_compute->IsNativeMode()};
        info_hitch.SourceBytes(env.SourceSizeBytes());
        const auto* result = MakeShaderInfo(env, *cpu_shader_addr);
        info_hitch.Finish("ok");
        return result;
    } catch (const Shader::Exception&) {
        RememberRejection(*cpu_shader_addr, shader_addr);
        throw;
    }
}

void ShaderCache::RethrowIfRejected(VAddr addr) {
    std::scoped_lock lock{rejected_mutex};
    if (const auto it = rejected_native.find(addr); it != rejected_native.end()) {
        std::rethrow_exception(it->second.error);
    }
}

bool ShaderCache::IsRejected(VAddr addr) {
    std::scoped_lock lock{rejected_mutex};
    return rejected_native.contains(addr);
}

// Programs can be rejected before the guest has finished writing them (Yoshi binds programs
// that still read as zeros), and their pages are not write tracked, so the rejection would
// stick for the rest of the run. Keep the bytes a rejection was made from and compare them
// again at most every 250 ms; a change drops the rejection. EDEN_NX2_REJECT_RECHECK=0 turns
// this off. Ported from zortos293/Eglantine 46a184c4e (opt-in there).
static bool RejectRecheckEnabled() {
    static const bool enabled = [] {
        const char* const value = std::getenv("EDEN_NX2_REJECT_RECHECK");
        return !value || std::string_view{value} != "0";
    }();
    return enabled;
}

static std::vector<u8> ReadRejectedWindow(Tegra::MemoryManager& gpu_memory, GPUVAddr addr) {
    size_t length = 0x2000;
    while (length > 0x80 && !gpu_memory.IsFullyMappedRange(addr, length)) {
        length /= 2;
    }
    std::vector<u8> bytes(length);
    gpu_memory.ReadBlockUnsafe(addr, bytes.data(), length);
    return bytes;
}

bool ShaderCache::RecheckRejection(VAddr addr) {
    if (!RejectRecheckEnabled()) {
        return false;
    }
    const auto now = std::chrono::steady_clock::now();
    GPUVAddr gpu_addr;
    std::vector<u8> before;
    std::chrono::steady_clock::time_point made;
    {
        std::scoped_lock lock{rejected_mutex};
        const auto it = rejected_native.find(addr);
        if (it == rejected_native.end() ||
            now - it->second.checked < std::chrono::milliseconds{250}) {
            return false;
        }
        it->second.checked = now;
        gpu_addr = it->second.gpu_addr;
        before = it->second.bytes;
        made = it->second.made;
    }
    const std::vector<u8> after = ReadRejectedWindow(*gpu_memory, gpu_addr);
    if (before == after) {
        return false;
    }
    {
        std::scoped_lock lock{rejected_mutex};
        rejected_native.erase(addr);
    }
    const size_t length = (std::min)(before.size(), after.size());
    const size_t offset = static_cast<size_t>(
        std::mismatch(before.begin(), before.begin() + static_cast<std::ptrdiff_t>(length),
                      after.begin())
            .first -
        before.begin());
    static std::atomic<u32> reports{};
    if (reports.fetch_add(1, std::memory_order_relaxed) < 256) {
        LOG_INFO(HW_GPU,
                 "Rejected program {:#x} changed {} ms after its rejection at +{:#x}, lowering again",
                 gpu_addr,
                 std::chrono::duration_cast<std::chrono::milliseconds>(now - made).count(), offset);
    }
    return true;
}

void ShaderCache::RememberRejection(VAddr addr, GPUVAddr gpu_addr) {
    Rejection rejection{};
    rejection.error = std::current_exception();
    rejection.gpu_addr = gpu_addr;
    if (RejectRecheckEnabled()) {
        rejection.bytes = ReadRejectedWindow(*gpu_memory, gpu_addr);
        rejection.made = rejection.checked = std::chrono::steady_clock::now();
    }
    std::scoped_lock lock{rejected_mutex};
    rejected_native.emplace(addr, std::move(rejection));
    rejected_lowest.store((std::min)(rejected_lowest.load(std::memory_order_relaxed), addr),
                          std::memory_order_relaxed);
    rejected_highest.store((std::max)(rejected_highest.load(std::memory_order_relaxed), addr),
                           std::memory_order_relaxed);
}

void ShaderCache::GetGraphicsEnvironments(GraphicsEnvironments& result,
                                          const std::array<u64, NUM_PROGRAMS>& unique_hashes) {
    size_t env_index{};
    const GPUVAddr base_addr{maxwell3d->regs.program_region.Address()};
    const bool native = maxwell3d->GetNativeMacroEngine().native_turing;
    for (size_t index = 0; index < NUM_PROGRAMS; ++index) {
        if (unique_hashes[index] == 0) {
            continue;
        }
        const auto program{static_cast<Tegra::Engines::Maxwell3D::Regs::ShaderType>(index)};
        auto& env{result.envs[index]};
        const auto& config = maxwell3d->regs.pipelines[index];
        const u32 start_address{native ? 0 : config.offset};
        const GPUVAddr address = native
            ? (static_cast<u64>(config.reserved[0] & 0xff) << 32) | config.reserved[1]
            : base_addr;
        env = GraphicsEnvironment{*maxwell3d, *gpu_memory, program, address, start_address,
                                  shader_infos[index]->lowered_native, support_cooperative_matrix};
        env.SetCachedSize(shader_infos[index]->size_bytes);
        result.env_ptrs[env_index++] = &env;
    }
}

ShaderInfo* ShaderCache::TryGet(VAddr addr) const {
    std::scoped_lock lock{lookup_mutex};

    const auto it = lookup_cache.find(addr);
    if (it == lookup_cache.end()) {
        return nullptr;
    }
    return it->second->data;
}

void ShaderCache::Register(std::unique_ptr<ShaderInfo> data, VAddr addr, size_t size) {
    std::scoped_lock lock{invalidation_mutex, lookup_mutex};

    const VAddr addr_end = addr + size;
    Entry* const entry = NewEntry(addr, addr_end, data.get());

    const u64 page_end = (addr_end + YUZU_PAGESIZE - 1) >> YUZU_PAGEBITS;
    for (u64 page = addr >> YUZU_PAGEBITS; page < page_end; ++page) {
        invalidation_cache[page].push_back(entry);
    }

    storage.push_back(std::move(data));

    device_memory.UpdatePagesCachedCount(addr, size, 1);
}

void ShaderCache::InvalidatePagesInRegion(VAddr addr, size_t size) {
    const VAddr addr_end = addr + size;
    // Rejected programs are few and in a few places: look at them only for writes that can
    // reach them (this runs for every inline upload and pre-draw invalidation).
    if (addr_end > rejected_lowest.load(std::memory_order_relaxed) &&
        addr <= rejected_highest.load(std::memory_order_relaxed)) {
        std::scoped_lock lock{rejected_mutex};
        boost::unordered::erase_if(rejected_native, [&](const auto& rejected) {
            return rejected.first >= addr && rejected.first < addr_end;
        });
    }
    const u64 page_end = (addr_end + YUZU_PAGESIZE - 1) >> YUZU_PAGEBITS;
    for (u64 page = addr >> YUZU_PAGEBITS; page < page_end; ++page) {
        auto it = invalidation_cache.find(page);
        if (it == invalidation_cache.end()) {
            continue;
        }
        InvalidatePageEntries(it->second, addr, addr_end);
    }
}

void ShaderCache::RemovePendingShaders() {
    if (marked_for_removal.empty()) {
        return;
    }
    // Remove duplicates
    std::ranges::sort(marked_for_removal);
    marked_for_removal.erase(std::unique(marked_for_removal.begin(), marked_for_removal.end()),
                             marked_for_removal.end());
    // Linear growth anyways - maybe consider static_vector instead?
    boost::container::small_vector<ShaderInfo*, 16> removed_shaders;
    std::scoped_lock lock{lookup_mutex};
    for (Entry* const entry : marked_for_removal) {
        removed_shaders.push_back(entry->data);
        auto const it = lookup_cache.find(entry->addr_start);
        ASSERT(it != lookup_cache.end());
        lookup_cache.erase(it);
    }
    marked_for_removal.clear();

    if (!removed_shaders.empty()) {
        const bool sorted = removed_shaders.size() > 16;
        if (sorted) {
            std::ranges::sort(removed_shaders, std::less<ShaderInfo*>{});
        }
        for (auto& shader : storage) {
            const bool removed = sorted
                ? std::binary_search(removed_shaders.begin(), removed_shaders.end(), shader.get(),
                                     std::less<ShaderInfo*>{})
                : std::ranges::find(removed_shaders, shader.get()) != removed_shaders.end();
            if (removed) {
                retired_shaders.push_back(std::move(shader));
            }
        }
        std::erase(storage, nullptr);
        invalidation_generation.fetch_add(1, std::memory_order_release);
    }
}

void ShaderCache::ReclaimInvalidatedShaders() {
    if (invalidation_generation.load(std::memory_order_acquire) ==
        observed_invalidation_generation) {
        return;
    }
    // Apart from the bound stages, no raw ShaderInfo pointer survives between pipeline lookups.
    // Environments sent to workers own their lowered code, so dropping those stages is sufficient.
    // CPU invalidation holds this same lock while retiring objects and publishing a generation.
    std::scoped_lock lock{lookup_mutex};
    shader_infos.fill(nullptr);
    last_shaders_valid = false;
    if (maxwell3d) {
        maxwell3d->dirty.flags[VideoCommon::Dirty::Shaders] = true;
    }
    retired_shaders.clear();
    observed_invalidation_generation = invalidation_generation.load(std::memory_order_relaxed);
}

void ShaderCache::InvalidatePageEntries(std::vector<Entry*>& entries, VAddr addr, VAddr addr_end) {
    size_t index = 0;
    while (index < entries.size()) {
        Entry* const entry = entries[index];
        if (!entry->Overlaps(addr, addr_end)) {
            ++index;
            continue;
        }

        UnmarkMemory(entry);
        RemoveEntryFromInvalidationCache(entry);
        marked_for_removal.push_back(entry);
    }
}

void ShaderCache::RemoveEntryFromInvalidationCache(const Entry* entry) {
    const u64 page_end = (entry->addr_end + YUZU_PAGESIZE - 1) >> YUZU_PAGEBITS;
    for (u64 page = entry->addr_start >> YUZU_PAGEBITS; page < page_end; ++page) {
        const auto entries_it = invalidation_cache.find(page);
        ASSERT(entries_it != invalidation_cache.end());
        std::vector<Entry*>& entries = entries_it->second;

        const auto entry_it = std::ranges::find(entries, entry);
        ASSERT(entry_it != entries.end());
        entries.erase(entry_it);
    }
}

void ShaderCache::UnmarkMemory(Entry* entry) {
    if (!entry->is_memory_marked) {
        return;
    }
    entry->is_memory_marked = false;

    const VAddr addr = entry->addr_start;
    const size_t size = entry->addr_end - addr;
    device_memory.UpdatePagesCachedCount(addr, size, -1);
}

ShaderCache::Entry* ShaderCache::NewEntry(VAddr addr, VAddr addr_end, ShaderInfo* data) {
    auto entry = std::make_unique<Entry>(Entry{addr, addr_end, data});
    Entry* const entry_pointer = entry.get();

    lookup_cache.emplace(addr, std::move(entry));
    return entry_pointer;
}

const ShaderInfo* ShaderCache::MakeShaderInfo(GenericEnvironment& env, VAddr cpu_addr) {
    auto info = std::make_unique<ShaderInfo>();
    if (const std::optional<u64> cached_hash{env.Analyze()}) {
        info->unique_hash = *cached_hash;
        info->size_bytes = env.CachedSizeBytes();
    } else {
        // Slow path, not really hit on commercial games
        // Build a control flow graph to get the real shader size
        Shader::ObjectPool<Shader::Maxwell::Flow::Block> flow_block;
        Shader::Maxwell::Flow::CFG cfg{env, flow_block, env.StartAddress()};
        info->unique_hash = env.CalculateHash();
        info->size_bytes = env.ReadSizeBytes();
    }
    info->lowered_native = env.LoweredNative();
    const size_t size_bytes{env.SourceSizeBytes() ? env.SourceSizeBytes() : info->size_bytes};
    const ShaderInfo* const result{info.get()};
    Register(std::move(info), cpu_addr, size_bytes);
    return result;
}

} // namespace VideoCommon
