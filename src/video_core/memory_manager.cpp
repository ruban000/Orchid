// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2018 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <cstdint>

#if (defined(__linux__) || defined(__APPLE__)) && __has_include(<execinfo.h>)
#include <dlfcn.h>
#include <execinfo.h>
#endif

#include "common/alignment.h"
#include "common/assert.h"
#include "common/logging.h"
#include "common/paged_scalar.h"
#include "common/unmapped_read_diagnostic.h"
#include "core/core.h"
#include "core/hle/kernel/k_page_table.h"
#include "core/hle/kernel/k_process.h"
#include "memory_manager.h"
#include "video_core/guest_memory.h"
#include "video_core/nx2.h"
#include "video_core/host1x/host1x.h"
#include "video_core/invalidation_accumulator.h"
#include "video_core/memory_manager.h"
#include "video_core/rasterizer_interface.h"
#include "video_core/renderer_base.h"
#include "video_core/texture_cache/util.h"

namespace Tegra {
namespace {
// Called only for an already-reported scalar fault with the explicit trace opt-in.
// Capture host callers here; source_location inside Read's lambda names the lambda itself.
void LogScalarReadFaultStack(u64 address, size_t bits, u32 record) {
#if (defined(__linux__) || defined(__APPLE__)) && __has_include(<execinfo.h>)
    void* frames[24]{};
    const int count = backtrace(frames, static_cast<int>(std::size(frames)));
    for (int index = 0; index < count; ++index) {
        Dl_info info{};
        const bool resolved = dladdr(frames[index], &info) != 0;
        const auto pc = reinterpret_cast<std::uintptr_t>(frames[index]);
        const auto base = reinterpret_cast<std::uintptr_t>(info.dli_fbase);
        LOG_ERROR(HW_GPU, "Scalar read fault stack bits={} fault={:#x} record={} frame={} "
                          "pc={:#x} module={} offset={:#x} symbol={}",
                  bits, address, record, index, pc, resolved && info.dli_fname ? info.dli_fname : "unknown",
                  resolved ? pc - base : pc,
                  resolved && info.dli_sname ? info.dli_sname : "unknown");
    }
#else
    LOG_ERROR(HW_GPU, "Scalar read fault stack unavailable on this platform: bits={} "
                      "fault={:#x} record={}", bits, address, record);
#endif
}
} // namespace

bool MemoryManager::IsShuttingDown() const {
    return system.IsShuttingDown();
}

using Tegra::Memory::GuestMemoryFlags;

std::atomic<size_t> MemoryManager::unique_identifier_generator{};

MemoryManager::MemoryManager(Core::System& system_, MaxwellDeviceMemoryManager& memory_, u64 address_space_bits_, GPUVAddr split_address_, u64 big_page_bits_, u64 page_bits_)
    : system{system_}, memory{memory_}, address_space_bits{address_space_bits_}
    , split_address{split_address_}, page_bits{page_bits_}, big_page_bits{big_page_bits_}
    , entries{}, big_entries{}
    , page_table{address_space_bits, address_space_bits + page_bits - 38, page_bits != big_page_bits ? page_bits : 0}
    , kind_map{PTEKind::INVALID}, unique_identifier{unique_identifier_generator.fetch_add(1, std::memory_order_acq_rel)}
    , accumulator{}
{
    address_space_size = 1ULL << address_space_bits;
    page_size = 1ULL << page_bits;
    page_mask = page_size - 1ULL;
    big_page_size = 1ULL << big_page_bits;
    big_page_mask = big_page_size - 1ULL;
    const u64 page_table_bits = address_space_bits - page_bits;
    const u64 big_page_table_bits = address_space_bits - big_page_bits;
    const u64 page_table_size = 1ULL << page_table_bits;
    const u64 big_page_table_size = 1ULL << big_page_table_bits;
    page_table_mask = page_table_size - 1;
    big_page_table_mask = big_page_table_size - 1;

    big_page_table_dev.ResizeAndClear(big_page_table_size);
    big_entries.resize(big_page_table_size / 32, 0);
    big_page_continuous.resize(big_page_table_size / continuous_bits, 0);
    entries.resize(page_table_size / 32, 0);
    // A new address space at the address of a destroyed one must not reuse its kinds.
    kind_generation.fetch_add(1, std::memory_order_release);
}

MemoryManager::MemoryManager(Core::System& system_, u64 address_space_bits_, GPUVAddr split_address_, u64 big_page_bits_, u64 page_bits_)
    : MemoryManager(system_, system_.Host1x().MemoryManager(), address_space_bits_, split_address_, big_page_bits_, page_bits_)
{}

void MemoryManager::SetNativeSyncpointShim(GPUVAddr address) {
    native_syncpoint_shim.store(address, std::memory_order_release);
}

MemoryManager::~MemoryManager() {
    kind_generation.fetch_add(1, std::memory_order_release);
}

MemoryManager::EntryType MemoryManager::GetEntry(size_t position, bool is_big_page) const {
    if (is_big_page) {
        position = position >> big_page_bits;
        const u64 entry_mask = big_entries[position / 32];
        const size_t sub_index = position % 32;
        return static_cast<EntryType>((entry_mask >> (2 * sub_index)) & 0x03ULL);
    } else {
        position = position >> page_bits;
        const u64 entry_mask = entries[position / 32];
        const size_t sub_index = position % 32;
        return static_cast<EntryType>((entry_mask >> (2 * sub_index)) & 0x03ULL);
    }
}

void MemoryManager::SetEntry(size_t position, MemoryManager::EntryType entry, bool is_big_page) {
    if (is_big_page) {
        position = position >> big_page_bits;
        const u64 entry_mask = big_entries[position / 32];
        const size_t sub_index = position % 32;
        big_entries[position / 32] =
            (~(3ULL << sub_index * 2) & entry_mask) | (static_cast<u64>(entry) << sub_index * 2);
    } else {
        position = position >> page_bits;
        const u64 entry_mask = entries[position / 32];
        const size_t sub_index = position % 32;
        entries[position / 32] =
            (~(3ULL << sub_index * 2) & entry_mask) | (static_cast<u64>(entry) << sub_index * 2);
    }
}

PTEKind MemoryManager::GetPageKind(GPUVAddr gpu_addr) const {
    std::unique_lock<std::mutex> lock(guard);
    return kind_map.GetValueAt(gpu_addr);
}

inline bool MemoryManager::IsBigPageContinuous(size_t big_page_index) const {
    const u64 entry_mask = big_page_continuous[big_page_index / continuous_bits];
    const size_t sub_index = big_page_index % continuous_bits;
    return ((entry_mask >> sub_index) & 0x1ULL) != 0;
}

inline void MemoryManager::SetBigPageContinuous(size_t big_page_index, bool value) {
    const u64 continuous_mask = big_page_continuous[big_page_index / continuous_bits];
    const size_t sub_index = big_page_index % continuous_bits;
    big_page_continuous[big_page_index / continuous_bits] =
        (~(1ULL << sub_index) & continuous_mask) | (value ? 1ULL << sub_index : 0);
}

GPUVAddr MemoryManager::PageTableOp(GPUVAddr gpu_addr, [[maybe_unused]] DAddr dev_addr, size_t size, PTEKind kind, MemoryManager::EntryType entry_type) {
    [[maybe_unused]] u64 remaining_size{size};
    if (entry_type == EntryType::Mapped) {
        page_table.ReserveRange(gpu_addr, size);
    }
    // The caches hear about each run of pages whose entry changed in one call, not page by page
    // (a 1.45 GB Switch 2 region made ~370,000 texture cache scans).
    GPUVAddr modified_start{};
    u64 modified_size{};
    const auto flush_modified = [&] {
        if (modified_size != 0) {
            rasterizer->ModifyGPUMemory(unique_identifier, modified_start, modified_size);
            modified_size = 0;
        }
    };
    for (u64 offset{}; offset < size; offset += page_size) {
        const GPUVAddr current_gpu_addr = gpu_addr + offset;
        [[maybe_unused]] const auto current_entry_type = GetEntry(current_gpu_addr, false);
        SetEntry(current_gpu_addr, entry_type, false);
        const auto index = PageEntryIndex(current_gpu_addr, false);
        const u32 sub_value = static_cast<u32>((dev_addr + offset) >> cpu_page_bits);
        const bool remapped = entry_type == EntryType::Mapped &&
                              current_entry_type == EntryType::Mapped &&
                              page_table[index] != sub_value;
        if (current_entry_type != entry_type || remapped) {
            if (modified_size == 0) {
                modified_start = current_gpu_addr;
            }
            modified_size += page_size;
        } else {
            flush_modified();
        }
        if (entry_type == EntryType::Mapped) {
            page_table[index] = sub_value;
        }
        remaining_size -= page_size;
    }
    flush_modified();
    {
        // As in BigPageTableOp: service threads map (nx2:gpu jobs) while the GPU thread reads the
        // kinds (GetPageKind, GetMemoryLayoutSize for every storage buffer binding).
        std::unique_lock<std::mutex> lock(guard);
        kind_map.Map(gpu_addr, gpu_addr + size, kind);
        kind_generation.fetch_add(1, std::memory_order_release);
    }
    return gpu_addr;
}

GPUVAddr MemoryManager::BigPageTableOp(GPUVAddr gpu_addr, [[maybe_unused]] DAddr dev_addr, size_t size, PTEKind kind, MemoryManager::EntryType entry_type) {
    [[maybe_unused]] u64 remaining_size{size};
    GPUVAddr modified_start{};
    u64 modified_size{};
    const auto flush_modified = [&] {
        if (modified_size != 0) {
            rasterizer->ModifyGPUMemory(unique_identifier, modified_start, modified_size);
            modified_size = 0;
        }
    };
    for (u64 offset{}; offset < size; offset += big_page_size) {
        const GPUVAddr current_gpu_addr = gpu_addr + offset;
        [[maybe_unused]] const auto current_entry_type = GetEntry(current_gpu_addr, true);
        SetEntry(current_gpu_addr, entry_type, true);
        const DAddr current_dev_addr = dev_addr + offset;
        const auto index = PageEntryIndex(current_gpu_addr, true);
        const u32 sub_value = static_cast<u32>(current_dev_addr >> cpu_page_bits);
        const bool remapped = entry_type == EntryType::Mapped &&
                              current_entry_type == EntryType::Mapped &&
                              big_page_table_dev[index] != sub_value;
        if (current_entry_type != entry_type || remapped) {
            if (modified_size == 0) {
                modified_start = current_gpu_addr;
            }
            modified_size += big_page_size;
        } else {
            flush_modified();
        }
        if (entry_type == EntryType::Mapped) {
            big_page_table_dev.Set(index, sub_value);
            const bool is_continuous = ([&] {
                uintptr_t base_ptr{
                    reinterpret_cast<uintptr_t>(memory.GetPointer<u8>(current_dev_addr))};
                if (base_ptr == 0) {
                    return false;
                }
                for (DAddr start_cpu = current_dev_addr + page_size;
                     start_cpu < current_dev_addr + big_page_size; start_cpu += page_size) {
                    base_ptr += page_size;
                    auto next_ptr = reinterpret_cast<uintptr_t>(memory.GetPointer<u8>(start_cpu));
                    if (next_ptr == 0 || base_ptr != next_ptr) {
                        return false;
                    }
                }
                return true;
            })();
            SetBigPageContinuous(index, is_continuous);
        }
        remaining_size -= big_page_size;
    }
    flush_modified();
    {
        std::unique_lock<std::mutex> lock(guard);
        kind_map.Map(gpu_addr, gpu_addr + size, kind);
        kind_generation.fetch_add(1, std::memory_order_release);
    }
    return gpu_addr;
}

void MemoryManager::BindRasterizer(VideoCore::RasterizerInterface* rasterizer_) {
    rasterizer = rasterizer_;
}

GPUVAddr MemoryManager::Map(GPUVAddr gpu_addr, DAddr dev_addr, std::size_t size, PTEKind kind, bool is_big_pages) {
    if (is_big_pages)
        return BigPageTableOp(gpu_addr, dev_addr, size, kind, EntryType::Mapped);
    return PageTableOp(gpu_addr, dev_addr, size, kind, EntryType::Mapped);
}

GPUVAddr MemoryManager::MapSparse(GPUVAddr gpu_addr, std::size_t size, bool is_big_pages) {
    if (is_big_pages)
        return BigPageTableOp(gpu_addr, 0, size, PTEKind::INVALID, EntryType::Reserved);
    return PageTableOp(gpu_addr, 0, size, PTEKind::INVALID, EntryType::Reserved);
}

void MemoryManager::Unmap(GPUVAddr gpu_addr, std::size_t size) {
    if (size == 0) {
        return;
    }
    GetSubmappedRangeImpl<false>(gpu_addr, size, page_stash);

    for (const auto& [map_addr, map_size] : page_stash) {
        rasterizer->UnmapMemory(map_addr, map_size);
    }
    page_stash.clear();

    BigPageTableOp(gpu_addr, 0, size, PTEKind::INVALID, EntryType::Free);
    PageTableOp(gpu_addr, 0, size, PTEKind::INVALID, EntryType::Free);
}

std::optional<DAddr> MemoryManager::GpuToCpuAddress(GPUVAddr gpu_addr) const {
    if (!IsWithinGPUAddressRange(gpu_addr)) [[unlikely]] {
        return std::nullopt;
    }
    DAddr dev_addr;
    if (GetEntry(gpu_addr, true) != EntryType::Mapped) [[unlikely]] {
        if (GetEntry(gpu_addr, false) != EntryType::Mapped)
            return std::nullopt;
        dev_addr = (DAddr(page_table[PageEntryIndex(gpu_addr, false)]) << cpu_page_bits) +
                   (gpu_addr & page_mask);
    } else {
        dev_addr = (DAddr(big_page_table_dev[PageEntryIndex(gpu_addr, true)]) << cpu_page_bits) +
                   (gpu_addr & big_page_mask);
    }
    if ((dev_addr >> MaxwellDeviceMemoryManager::AS_BITS) != 0) {
        LOG_ERROR(HW_GPU, "GPU address {:#x} maps outside the device space, to {:#x}", gpu_addr,
                  dev_addr);
        return std::nullopt;
    }
    return dev_addr;
}

std::optional<DAddr> MemoryManager::GpuToCpuAddress(GPUVAddr addr, std::size_t size) const {
    size_t page_index{addr >> page_bits};
    const size_t page_last{(addr + size + page_size - 1) >> page_bits};
    while (page_index < page_last) {
        const auto page_addr{GpuToCpuAddress(page_index << page_bits)};
        if (page_addr) {
            return page_addr;
        }
        ++page_index;
    }
    return std::nullopt;
}

template <typename T>
T MemoryManager::Read(GPUVAddr addr) const {
    if constexpr (sizeof(T) == sizeof(u32)) {
        const auto shim = native_syncpoint_shim.load(std::memory_order_acquire);
        if (shim && addr >= shim && addr - shim < NumShimSyncpoints * sizeof(u32) &&
            Common::IsAligned(addr - shim, sizeof(u32))) {
            return static_cast<T>(system.Host1x().GetSyncpointManager().GetGuestSyncpointValue(
                static_cast<u32>((addr - shim) / sizeof(u32))));
        }
    }

    // A host pointer is only valid within its 4 KiB device page, even inside a big GPU page.
    return Common::ReadPagedScalar<T>(addr, [&](GPUVAddr address) {
        const u8* const page_pointer = GetPointer(address);
        if (!page_pointer) {
            static std::atomic<u32> reported{};
            const u32 report = reported.fetch_add(1);
            if (report < 16) {
                LOG_ERROR(HW_GPU, "Read{} of unmapped GPU address {:#x}", sizeof(T) * 8,
                          address);
                if (Common::UnmappedReadDiagnostic::Enabled()) {
                    // Re-query only on this failure path. Mapping can change concurrently:
                    // these fields describe the diagnostic instant, not the original lookup.
                    const auto translated = GpuToCpuAddress(address);
                    const bool backed_now = translated && memory.GetPointer<u8>(*translated);
                    const char* const classification = !translated ? "gpu-translation"
                        : backed_now ? "mapping-changed-or-transient" : "missing-device-backing";
                    LOG_ERROR(HW_GPU, "Scalar read fault trace: memory_id={} requested={:#x}+{} "
                                      "fault={:#x} mapping_at_trace={} translated={} device={:#x} "
                                      "backed_now={} record={}/16",
                              GetID(), addr, sizeof(T), address, classification,
                              translated.has_value(), translated.value_or(0), backed_now, report + 1);
                    LogScalarReadFaultStack(address, sizeof(T) * 8, report + 1);
                    if (report == 15) {
                        LOG_ERROR(HW_GPU, "Scalar read fault trace cap reached for Read{}: "
                                          "additional_faults_not_traced=true", sizeof(T) * 8);
                    }
                }
            }
        }
        return page_pointer;
    });
}

template <typename T>
void MemoryManager::Write(GPUVAddr addr, T data) {
    Common::WritePagedScalar(addr, data, [&](GPUVAddr address) {
        u8* page_pointer = GetPointer(address);
        if (!page_pointer) {
            // A Switch 2 sparse block nobody wrote yet gets its memory now (the whole scalar,
            // a second block too if it straddles one).
            Tegra::Nx2BackSparseWrite(addr, sizeof(T), &data);
            page_pointer = GetPointer(address);
            ASSERT(page_pointer != nullptr);
        }
        return page_pointer;
    });
}

template u8 MemoryManager::Read<u8>(GPUVAddr addr) const;
template u16 MemoryManager::Read<u16>(GPUVAddr addr) const;
template u32 MemoryManager::Read<u32>(GPUVAddr addr) const;
template u64 MemoryManager::Read<u64>(GPUVAddr addr) const;
template void MemoryManager::Write<u8>(GPUVAddr addr, u8 data);
template void MemoryManager::Write<u16>(GPUVAddr addr, u16 data);
template void MemoryManager::Write<u32>(GPUVAddr addr, u32 data);
template void MemoryManager::Write<u64>(GPUVAddr addr, u64 data);

u8* MemoryManager::GetPointer(GPUVAddr gpu_addr) {
    const auto address{GpuToCpuAddress(gpu_addr)};
    if (!address) {
        return {};
    }

    return memory.GetPointer<u8>(*address);
}

const u8* MemoryManager::GetPointer(GPUVAddr gpu_addr) const {
    const auto address{GpuToCpuAddress(gpu_addr)};
    if (!address) {
        return {};
    }

    return memory.GetPointer<u8>(*address);
}

#if defined(_MSC_VER) && !defined(__clang__) // no need for gcc / clang but msvc's compiler is more conservative with inlining.
#pragma inline_recursion(on)
#endif

template <typename FuncMapped, typename FuncReserved, typename FuncUnmapped>
inline void MemoryManager::MemoryOperation(GPUVAddr gpu_src_addr, std::size_t size, bool is_big_page, FuncMapped&& func_mapped, FuncReserved&& func_reserved, FuncUnmapped&& func_unmapped) const {
    if (!IsWithinGPUAddressRange(gpu_src_addr) || size > address_space_size - gpu_src_addr) {
        return;
    }
    using FuncMappedReturn =
        typename std::invoke_result<FuncMapped, std::size_t, std::size_t, std::size_t>::type;
    using FuncReservedReturn =
        typename std::invoke_result<FuncReserved, std::size_t, std::size_t, std::size_t>::type;
    using FuncUnmappedReturn =
        typename std::invoke_result<FuncUnmapped, std::size_t, std::size_t, std::size_t>::type;
    static constexpr bool BOOL_BREAK_MAPPED = std::is_same_v<FuncMappedReturn, bool>;
    static constexpr bool BOOL_BREAK_RESERVED = std::is_same_v<FuncReservedReturn, bool>;
    static constexpr bool BOOL_BREAK_UNMAPPED = std::is_same_v<FuncUnmappedReturn, bool>;
    u64 used_page_size;
    u64 used_page_mask;
    u64 used_page_bits;
    if (is_big_page) {
        used_page_size = big_page_size;
        used_page_mask = big_page_mask;
        used_page_bits = big_page_bits;
    } else {
        used_page_size = page_size;
        used_page_mask = page_mask;
        used_page_bits = page_bits;
    }
    std::size_t remaining_size{size};
    std::size_t page_index{gpu_src_addr >> used_page_bits};
    std::size_t page_offset{gpu_src_addr & used_page_mask};
    GPUVAddr current_address = gpu_src_addr;

    while (remaining_size > 0) {
        const std::size_t copy_amount{
            (std::min)(static_cast<std::size_t>(used_page_size) - page_offset, remaining_size)};
        auto entry = GetEntry(current_address, is_big_page);
        if (entry == EntryType::Mapped) [[likely]] {
            if constexpr (BOOL_BREAK_MAPPED) {
                if (func_mapped(page_index, page_offset, copy_amount)) {
                    return;
                }
            } else {
                func_mapped(page_index, page_offset, copy_amount);
            }

        } else if (entry == EntryType::Reserved) {
            if constexpr (BOOL_BREAK_RESERVED) {
                if (func_reserved(page_index, page_offset, copy_amount)) {
                    return;
                }
            } else {
                func_reserved(page_index, page_offset, copy_amount);
            }

        } else [[unlikely]] {
            if constexpr (BOOL_BREAK_UNMAPPED) {
                if (func_unmapped(page_index, page_offset, copy_amount)) {
                    return;
                }
            } else {
                func_unmapped(page_index, page_offset, copy_amount);
            }
        }
        page_index++;
        page_offset = 0;
        remaining_size -= copy_amount;
        current_address += copy_amount;
    }
}

void MemoryManager::ReadBlockImpl(GPUVAddr gpu_src_addr, void* dest_buffer, std::size_t size, [[maybe_unused]] VideoCommon::CacheType which, bool unsafe) const {
    if (!IsWithinGPUAddressRange(gpu_src_addr) || size > address_space_size - gpu_src_addr) {
        std::memset(dest_buffer, 0, size);
        return;
    }
    auto set_to_zero = [&]([[maybe_unused]] std::size_t page_index, [[maybe_unused]] std::size_t offset, std::size_t copy_amount) {
        std::memset(dest_buffer, 0, copy_amount);
        dest_buffer = static_cast<u8*>(dest_buffer) + copy_amount;
    };
    auto mapped_normal = [&](std::size_t page_index, std::size_t offset, std::size_t copy_amount) {
        const DAddr dev_addr_base = (DAddr(page_table[page_index]) << cpu_page_bits) + offset;
        if (!unsafe) {
            rasterizer->FlushRegion(dev_addr_base, copy_amount, which);
        }
        // A device page without memory (a sparse block nobody wrote yet) reads zero.
        if (const u8* physical = memory.GetPointer<u8>(dev_addr_base)) [[likely]] {
            std::memcpy(dest_buffer, physical, copy_amount);
        } else {
            std::memset(dest_buffer, 0, copy_amount);
        }
        dest_buffer = static_cast<u8*>(dest_buffer) + copy_amount;
    };
    auto mapped_big = [&](std::size_t page_index, std::size_t offset, std::size_t copy_amount) {
        const DAddr dev_addr_base = (DAddr(big_page_table_dev[page_index]) << cpu_page_bits) + offset;
        if (!unsafe) {
            rasterizer->FlushRegion(dev_addr_base, copy_amount, which);
        }
        // A continuous big page may still have no host memory (a Switch 2 region backed as it is
        // written, while its backing is pending): the per-page copy handles that.
        u8* const physical = IsBigPageContinuous(page_index) ? memory.GetPointer<u8>(dev_addr_base)
                                                             : nullptr;
        if (physical != nullptr) [[likely]] {
            std::memcpy(dest_buffer, physical, copy_amount);
        } else {
            Common::UnmappedReadDiagnostic::Scope trace;
            trace.TranslatedGpu((page_index << big_page_bits) + offset, copy_amount);
            memory.ReadBlockUnsafe(dev_addr_base, dest_buffer, copy_amount);
        }
        dest_buffer = static_cast<u8*>(dest_buffer) + copy_amount;
    };
    auto read_short_pages = [&](std::size_t page_index, std::size_t offset, std::size_t copy_amount) {
        GPUVAddr base = (page_index << big_page_bits) + offset;
        MemoryOperation(base, copy_amount, false, mapped_normal, set_to_zero, set_to_zero);
    };
    MemoryOperation(gpu_src_addr, size, true, mapped_big, set_to_zero, read_short_pages);
}

void MemoryManager::ReadBlock(GPUVAddr gpu_src_addr, void* dest_buffer, std::size_t size,
                              VideoCommon::CacheType which, std::source_location caller) const {
    Common::UnmappedReadDiagnostic::Scope trace;
    trace.Gpu(gpu_src_addr, size, caller);
    ReadBlockImpl(gpu_src_addr, dest_buffer, size, which, false);
}

void MemoryManager::ReadBlockUnsafe(GPUVAddr gpu_src_addr, void* dest_buffer,
                                    const std::size_t size, std::source_location caller) const {
    Common::UnmappedReadDiagnostic::Scope trace;
    trace.Gpu(gpu_src_addr, size, caller);
    // Small reads within one 4 KiB page (texture and sampler descriptors, several per draw): one
    // translation and a copy, as Read<T> does, instead of the page walk (~3% of the GPU thread
    // in title 1). Anything else, unmapped or sparse pages included, takes the full path.
    constexpr std::size_t SmallPage = 0x1000;
    if (size <= SmallPage && (gpu_src_addr & (SmallPage - 1)) + size <= SmallPage) {
        if (const u8* const pointer = GetPointer(gpu_src_addr)) {
            std::memcpy(dest_buffer, pointer, size);
            return;
        }
    }
    ReadBlockImpl(gpu_src_addr, dest_buffer, size, VideoCommon::CacheType::None, true);
}

void MemoryManager::ReadWordsUnsafe(GPUVAddr gpu_src_addr, u32* dest, u32 count) const {
    if (count == 0) {
        return;
    }
    const u64 size = u64{count} * sizeof(u32);
    const auto shim = native_syncpoint_shim.load(std::memory_order_acquire);
    const bool overlaps_shim = shim &&
        (gpu_src_addr < shim ? shim - gpu_src_addr < size
                            : gpu_src_addr - shim < NumShimSyncpoints * sizeof(u32));
    if (!Common::IsAligned(gpu_src_addr, sizeof(u32)) ||
        !IsWithinGPUAddressRange(gpu_src_addr) ||
        !IsWithinGPUAddressRange(gpu_src_addr + size - 1) || overlaps_shim) {
        for (u32 i = 0; i < count; ++i) {
            dest[i] = Read<u32>(gpu_src_addr + u64{i} * sizeof(u32));
        }
        return;
    }
    ReadBlockUnsafe(gpu_src_addr, dest, size);
}

void MemoryManager::WriteBlockImpl(GPUVAddr gpu_dest_addr, const void* src_buffer, std::size_t size, [[maybe_unused]] VideoCommon::CacheType which, bool unsafe) {
    // Give Switch 2 sparse pages the write lands on their memory first (otherwise it is dropped).
    Tegra::Nx2BackSparseWrite(gpu_dest_addr, size, src_buffer);
    auto just_advance = [&]([[maybe_unused]] std::size_t page_index, [[maybe_unused]] std::size_t offset, std::size_t copy_amount) {
        src_buffer = static_cast<const u8*>(src_buffer) + copy_amount;
    };
    auto mapped_normal = [&](std::size_t page_index, std::size_t offset, std::size_t copy_amount) {
        const DAddr dev_addr_base = (DAddr(page_table[page_index]) << cpu_page_bits) + offset;
        if (!unsafe) {
            rasterizer->InvalidateRegion(dev_addr_base, copy_amount, which);
        }
        if (u8* physical = memory.GetPointer<u8>(dev_addr_base)) [[likely]] {
            std::memcpy(physical, src_buffer, copy_amount);
        }
        src_buffer = static_cast<const u8*>(src_buffer) + copy_amount;
    };
    auto mapped_big = [&](std::size_t page_index, std::size_t offset, std::size_t copy_amount) {
        const DAddr dev_addr_base = (DAddr(big_page_table_dev[page_index]) << cpu_page_bits) + offset;
        if (!unsafe) {
            rasterizer->InvalidateRegion(dev_addr_base, copy_amount, which);
        }
        u8* const physical = IsBigPageContinuous(page_index) ? memory.GetPointer<u8>(dev_addr_base)
                                                             : nullptr;
        if (physical != nullptr) [[likely]] {
            std::memcpy(physical, src_buffer, copy_amount);
        } else {
            memory.WriteBlockUnsafe(dev_addr_base, src_buffer, copy_amount);
        }
        src_buffer = static_cast<const u8*>(src_buffer) + copy_amount;
    };
    auto write_short_pages = [&](std::size_t page_index, std::size_t offset, std::size_t copy_amount) {
        GPUVAddr base = (page_index << big_page_bits) + offset;
        MemoryOperation(base, copy_amount, false, mapped_normal, just_advance, just_advance);
    };
    MemoryOperation(gpu_dest_addr, size, true, mapped_big, just_advance, write_short_pages);
}

void MemoryManager::WriteBlock(GPUVAddr gpu_dest_addr, const void* src_buffer, std::size_t size, VideoCommon::CacheType which) {
    WriteBlockImpl(gpu_dest_addr, src_buffer, size, which, false);
}

void MemoryManager::WriteBlockUnsafe(GPUVAddr gpu_dest_addr, const void* src_buffer, std::size_t size) {
    WriteBlockImpl(gpu_dest_addr, src_buffer, size, VideoCommon::CacheType::None, true);
}

void MemoryManager::WriteBlockCached(GPUVAddr gpu_dest_addr, const void* src_buffer, std::size_t size) {
    WriteBlockImpl(gpu_dest_addr, src_buffer, size, VideoCommon::CacheType::None, true);
    accumulator.Add(gpu_dest_addr, size);
}

void MemoryManager::FlushRegion(GPUVAddr gpu_addr, size_t size,
                                VideoCommon::CacheType which) const {
    auto do_nothing = [&]([[maybe_unused]] std::size_t page_index,
                          [[maybe_unused]] std::size_t offset,
                          [[maybe_unused]] std::size_t copy_amount) {};

    auto mapped_normal = [&](std::size_t page_index, std::size_t offset, std::size_t copy_amount) {
        const DAddr dev_addr_base = (DAddr(page_table[page_index]) << cpu_page_bits) + offset;
        rasterizer->FlushRegion(dev_addr_base, copy_amount, which);
    };
    auto mapped_big = [&](std::size_t page_index, std::size_t offset, std::size_t copy_amount) {
        const DAddr dev_addr_base = (DAddr(big_page_table_dev[page_index]) << cpu_page_bits) + offset;
        rasterizer->FlushRegion(dev_addr_base, copy_amount, which);
    };
    auto flush_short_pages = [&](std::size_t page_index, std::size_t offset, std::size_t copy_amount) {
        GPUVAddr base = (page_index << big_page_bits) + offset;
        MemoryOperation(base, copy_amount, false, mapped_normal, do_nothing, do_nothing);
    };
    MemoryOperation(gpu_addr, size, true, mapped_big, do_nothing, flush_short_pages);
}

bool MemoryManager::IsMemoryDirty(GPUVAddr gpu_addr, size_t size,
                                  VideoCommon::CacheType which) const {
    bool result = false;
    auto do_nothing = [&]([[maybe_unused]] std::size_t page_index,
                          [[maybe_unused]] std::size_t offset,
                          [[maybe_unused]] std::size_t copy_amount) { return false; };

    auto mapped_normal = [&](std::size_t page_index, std::size_t offset, std::size_t copy_amount) {
        const DAddr dev_addr_base =
            (static_cast<DAddr>(page_table[page_index]) << cpu_page_bits) + offset;
        result |= rasterizer->MustFlushRegion(dev_addr_base, copy_amount, which);
        return result;
    };
    auto mapped_big = [&](std::size_t page_index, std::size_t offset, std::size_t copy_amount) {
        const DAddr dev_addr_base =
            (static_cast<DAddr>(big_page_table_dev[page_index]) << cpu_page_bits) + offset;
        result |= rasterizer->MustFlushRegion(dev_addr_base, copy_amount, which);
        return result;
    };
    auto check_short_pages = [&](std::size_t page_index, std::size_t offset,
                                 std::size_t copy_amount) {
        GPUVAddr base = (page_index << big_page_bits) + offset;
        MemoryOperation(base, copy_amount, false, mapped_normal, do_nothing, do_nothing);
        return result;
    };
    MemoryOperation(gpu_addr, size, true, mapped_big, do_nothing, check_short_pages);
    return result;
}

size_t MemoryManager::MaxContinuousRange(GPUVAddr gpu_addr, size_t size) const {
    std::optional<DAddr> old_page_addr{};
    size_t range_so_far = 0;
    bool result{false};
    auto fail = [&]([[maybe_unused]] std::size_t page_index, [[maybe_unused]] std::size_t offset,
                    std::size_t copy_amount) {
        result = true;
        return true;
    };
    auto short_check = [&](std::size_t page_index, std::size_t offset, std::size_t copy_amount) {
        const DAddr dev_addr_base =
            (static_cast<DAddr>(page_table[page_index]) << cpu_page_bits) + offset;
        if (old_page_addr && *old_page_addr != dev_addr_base) {
            result = true;
            return true;
        }
        range_so_far += copy_amount;
        old_page_addr = {dev_addr_base + copy_amount};
        return false;
    };
    auto big_check = [&](std::size_t page_index, std::size_t offset, std::size_t copy_amount) {
        const DAddr dev_addr_base =
            (static_cast<DAddr>(big_page_table_dev[page_index]) << cpu_page_bits) + offset;
        if (old_page_addr && *old_page_addr != dev_addr_base) {
            return true;
        }
        range_so_far += copy_amount;
        old_page_addr = {dev_addr_base + copy_amount};
        return false;
    };
    auto check_short_pages = [&](std::size_t page_index, std::size_t offset,
                                 std::size_t copy_amount) {
        GPUVAddr base = (page_index << big_page_bits) + offset;
        MemoryOperation(base, copy_amount, false, short_check, fail, fail);
        return result;
    };
    MemoryOperation(gpu_addr, size, true, big_check, fail, check_short_pages);
    return range_so_far;
}

size_t MemoryManager::GetMemoryLayoutSize(GPUVAddr gpu_addr, size_t max_size) const {
    // Asked for every storage buffer binding and indexed draw: answered from the kind entry this
    // thread looked up last while no page kind of any address space changed since, instead of
    // taking the lock and searching the map each time.
    struct Cached {
        const MemoryManager* owner;
        u64 generation;
        GPUVAddr start;
        GPUVAddr end;
        bool null;
    };
    thread_local Cached cached{};
    const u64 generation = kind_generation.load(std::memory_order_acquire);
    if (cached.owner == this && cached.generation == generation && gpu_addr >= cached.start &&
        gpu_addr < cached.end) {
        return cached.null ? 0 : static_cast<size_t>(cached.end - gpu_addr);
    }
    std::unique_lock<std::mutex> lock(guard);
    const auto key = static_cast<s64>(gpu_addr);
    if (key < 0) {
        return 0;
    }
    const auto segment = kind_map.GetSegment(key);
    cached = {
        .owner = this,
        .generation = kind_generation.load(std::memory_order_relaxed),
        .start = static_cast<GPUVAddr>(std::max<s64>(segment.start, 0)),
        .end = static_cast<GPUVAddr>(segment.end),
        .null = segment.value == PTEKind::INVALID,
    };
    return cached.null ? 0 : static_cast<size_t>(cached.end - gpu_addr);
}

void MemoryManager::InvalidateRegion(GPUVAddr gpu_addr, size_t size,
                                     VideoCommon::CacheType which) const {
    auto do_nothing = [&]([[maybe_unused]] std::size_t page_index,
                          [[maybe_unused]] std::size_t offset,
                          [[maybe_unused]] std::size_t copy_amount) {};

    auto mapped_normal = [&](std::size_t page_index, std::size_t offset, std::size_t copy_amount) {
        const DAddr dev_addr_base = (DAddr(page_table[page_index]) << cpu_page_bits) + offset;
        rasterizer->InvalidateRegion(dev_addr_base, copy_amount, which);
    };
    auto mapped_big = [&](std::size_t page_index, std::size_t offset, std::size_t copy_amount) {
        const DAddr dev_addr_base = (DAddr(big_page_table_dev[page_index]) << cpu_page_bits) + offset;
        rasterizer->InvalidateRegion(dev_addr_base, copy_amount, which);
    };
    auto invalidate_short_pages = [&](std::size_t page_index, std::size_t offset, std::size_t copy_amount) {
        GPUVAddr base = (page_index << big_page_bits) + offset;
        MemoryOperation(base, copy_amount, false, mapped_normal, do_nothing, do_nothing);
    };
    MemoryOperation(gpu_addr, size, true, mapped_big, do_nothing, invalidate_short_pages);
}

void MemoryManager::CopyBlock(GPUVAddr gpu_dest_addr, GPUVAddr gpu_src_addr, std::size_t size,
                              VideoCommon::CacheType which) {
    Tegra::Memory::GpuGuestMemoryScoped<u8, GuestMemoryFlags::SafeReadWrite> data(
        *this, gpu_src_addr, size);
    data.SetAddressAndSize(gpu_dest_addr, size);
    FlushRegion(gpu_dest_addr, size, which);
}

bool MemoryManager::IsGranularRange(GPUVAddr gpu_addr, std::size_t size) const {
    if (!IsWithinGPUAddressRange(gpu_addr) || size > address_space_size - gpu_addr) {
        return false;
    }
    if (GetEntry(gpu_addr, true) == EntryType::Mapped) [[likely]] {
        size_t page_index = gpu_addr >> big_page_bits;
        if (IsBigPageContinuous(page_index)) [[likely]] {
            const std::size_t page{(page_index & big_page_mask) + size};
            return page <= big_page_size;
        }
        const std::size_t page{(gpu_addr & Core::DEVICE_PAGEMASK) + size};
        return page <= Core::DEVICE_PAGESIZE;
    }
    if (GetEntry(gpu_addr, false) != EntryType::Mapped) {
        return false;
    }
    const std::size_t page{(gpu_addr & Core::DEVICE_PAGEMASK) + size};
    return page <= Core::DEVICE_PAGESIZE;
}

bool MemoryManager::IsContinuousRange(GPUVAddr gpu_addr, std::size_t size) const {
    if (!IsWithinGPUAddressRange(gpu_addr) || size > address_space_size - gpu_addr) {
        return false;
    }
    std::optional<DAddr> old_page_addr{};
    bool result{true};
    auto fail = [&]([[maybe_unused]] std::size_t page_index, [[maybe_unused]] std::size_t offset,
                    std::size_t copy_amount) {
        result = false;
        return true;
    };
    auto short_check = [&](std::size_t page_index, std::size_t offset, std::size_t copy_amount) {
        const DAddr dev_addr_base =
            (static_cast<DAddr>(page_table[page_index]) << cpu_page_bits) + offset;
        if (old_page_addr && *old_page_addr != dev_addr_base) {
            result = false;
            return true;
        }
        old_page_addr = {dev_addr_base + copy_amount};
        return false;
    };
    auto big_check = [&](std::size_t page_index, std::size_t offset, std::size_t copy_amount) {
        const DAddr dev_addr_base =
            (static_cast<DAddr>(big_page_table_dev[page_index]) << cpu_page_bits) + offset;
        if (old_page_addr && *old_page_addr != dev_addr_base) {
            result = false;
            return true;
        }
        old_page_addr = {dev_addr_base + copy_amount};
        return false;
    };
    auto check_short_pages = [&](std::size_t page_index, std::size_t offset,
                                 std::size_t copy_amount) {
        GPUVAddr base = (page_index << big_page_bits) + offset;
        MemoryOperation(base, copy_amount, false, short_check, fail, fail);
        return !result;
    };
    MemoryOperation(gpu_addr, size, true, big_check, fail, check_short_pages);
    return result;
}

bool MemoryManager::IsFullyMappedRange(GPUVAddr gpu_addr, std::size_t size) const {
    if (!IsWithinGPUAddressRange(gpu_addr) || size > address_space_size - gpu_addr) {
        return false; // MemoryOperation skips such a range without calling the unmapped callback
    }
    bool result{true};
    auto fail = [&]([[maybe_unused]] std::size_t page_index, [[maybe_unused]] std::size_t offset,
                    [[maybe_unused]] std::size_t copy_amount) {
        result = false;
        return true;
    };
    auto pass = [&]([[maybe_unused]] std::size_t page_index, [[maybe_unused]] std::size_t offset,
                    [[maybe_unused]] std::size_t copy_amount) { return false; };
    auto check_short_pages = [&](std::size_t page_index, std::size_t offset, std::size_t copy_amount) {
        GPUVAddr base = (page_index << big_page_bits) + offset;
        MemoryOperation(base, copy_amount, false, pass, pass, fail);
        return !result;
    };
    MemoryOperation(gpu_addr, size, true, pass, fail, check_short_pages);
    return result;
}

boost::container::small_vector<std::pair<GPUVAddr, std::size_t>, 32>
MemoryManager::GetSubmappedRange(GPUVAddr gpu_addr, std::size_t size) const {
    boost::container::small_vector<std::pair<GPUVAddr, std::size_t>, 32> result{};
    GetSubmappedRangeImpl<true>(gpu_addr, size, result);
    return result;
}

template <bool is_gpu_address>
void MemoryManager::GetSubmappedRangeImpl(GPUVAddr gpu_addr, std::size_t size, boost::container::small_vector<std::pair<std::conditional_t<is_gpu_address, GPUVAddr, DAddr>, std::size_t>, 32>& result)
    const {
    std::optional<std::pair<std::conditional_t<is_gpu_address, GPUVAddr, DAddr>, std::size_t>> last_segment{};
    std::optional<DAddr> old_page_addr{};
    const auto split = [&last_segment, &result]([[maybe_unused]] std::size_t page_index,
                                                [[maybe_unused]] std::size_t offset,
                                                [[maybe_unused]] std::size_t copy_amount) {
        if (last_segment) {
            result.push_back(*last_segment);
            last_segment = std::nullopt;
        }
    };
    const auto extend_size_big = [this, &split, &old_page_addr,
                                  &last_segment](std::size_t page_index, std::size_t offset,
                                                 std::size_t copy_amount) {
        const DAddr dev_addr_base =
            (static_cast<DAddr>(big_page_table_dev[page_index]) << cpu_page_bits) + offset;
        if (old_page_addr) {
            if (*old_page_addr != dev_addr_base) {
                split(0, 0, 0);
            }
        }
        old_page_addr = {dev_addr_base + copy_amount};
        if (!last_segment) {
            if constexpr (is_gpu_address) {
                const GPUVAddr new_base_addr = (page_index << big_page_bits) + offset;
                last_segment = {new_base_addr, copy_amount};
            } else {
                last_segment = {dev_addr_base, copy_amount};
            }
        } else {
            last_segment->second += copy_amount;
        }
    };
    const auto extend_size_short = [this, &split, &old_page_addr,
                                    &last_segment](std::size_t page_index, std::size_t offset,
                                                   std::size_t copy_amount) {
        const DAddr dev_addr_base =
            (static_cast<DAddr>(page_table[page_index]) << cpu_page_bits) + offset;
        if (old_page_addr) {
            if (*old_page_addr != dev_addr_base) {
                split(0, 0, 0);
            }
        }
        old_page_addr = {dev_addr_base + copy_amount};
        if (!last_segment) {
            if constexpr (is_gpu_address) {
                const GPUVAddr new_base_addr = (page_index << page_bits) + offset;
                last_segment = {new_base_addr, copy_amount};
            } else {
                last_segment = {dev_addr_base, copy_amount};
            }
        } else {
            last_segment->second += copy_amount;
        }
    };
    auto do_short_pages = [&](std::size_t page_index, std::size_t offset, std::size_t copy_amount) {
        GPUVAddr base = (page_index << big_page_bits) + offset;
        MemoryOperation(base, copy_amount, false, extend_size_short, split, split);
    };
    MemoryOperation(gpu_addr, size, true, extend_size_big, split, do_short_pages);
    split(0, 0, 0);
}

void MemoryManager::FlushCaching() {
    // Flush from the invalidate accumulator
    if (accumulator.InvalidateAll([this](GPUVAddr addr, size_t size) {
        GetSubmappedRangeImpl<false>(addr, size, page_stash2);
    })) {
        rasterizer->InnerInvalidation(page_stash2);
        page_stash2.clear();
    }
}

const u8* MemoryManager::GetSpan(const GPUVAddr src_addr, const std::size_t size) const {
    if (!IsContinuousRange(src_addr, size)) {
        return nullptr;
    }
    auto dev_addr = GpuToCpuAddress(src_addr);
    if (dev_addr) {
        return memory.GetSpan(*dev_addr, size);
    }
    return nullptr;
}

u8* MemoryManager::GetSpan(const GPUVAddr src_addr, const std::size_t size) {
    if (!IsContinuousRange(src_addr, size)) {
        return nullptr;
    }
    auto dev_addr = GpuToCpuAddress(src_addr);
    if (dev_addr) {
        return memory.GetSpan(*dev_addr, size);
    }
    return nullptr;
}

} // namespace Tegra
