// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "video_core/buffer_cache/sparse_uniform_upload.h"
#include "video_core/buffer_cache/uniform_stream_reuse.h"

namespace {
template <size_t AddressBits = 16>
struct SparseMemoryImpl {
    static constexpr size_t AS_BITS = AddressBits;
    static constexpr size_t PageSize = 0x1000;
    std::array<u8, 1 << AS_BITS> bytes{};
    std::array<int, (1 << AS_BITS) / PageSize> pages{};
    u64 generation{1};
    size_t pointer_reads{};
    size_t direct_reads{};

    SparseMemoryImpl() {
        pages.fill(-1);
        for (size_t index = 0; index < bytes.size(); ++index) {
            bytes[index] = static_cast<u8>((index / PageSize) * 13 + index % 251 + 1);
        }
    }
    void Map(size_t page, int backing) {
        pages.at(page) = backing;
        ++generation;
    }
    template <class T>
    T* GetPointer(u64 address) {
        REQUIRE(address < bytes.size());
        ++pointer_reads;
        const int backing = pages[address / PageSize];
        return backing < 0
                   ? nullptr
                   : reinterpret_cast<T*>(bytes.data() + backing * PageSize + address % PageSize);
    }
    const u8* GetSpan(u64 address, size_t size) {
        REQUIRE(address < bytes.size());
        REQUIRE(size <= bytes.size() - address);
        const int backing = pages[address / PageSize];
        if (backing < 0)
            return nullptr;
        const u8* start = bytes.data() + backing * PageSize + address % PageSize;
        for (size_t offset = 0; offset < size;) {
            const size_t current = address + offset;
            const int page = pages[current / PageSize];
            if (page < 0 || bytes.data() + page * PageSize + current % PageSize != start + offset) {
                return nullptr;
            }
            offset += (std::min)(size - offset, PageSize - current % PageSize);
        }
        ++direct_reads;
        return start;
    }
};

using SparseMemory = SparseMemoryImpl<>;

template <class Memory>
void CheckBytes(const Memory& memory, u64 address, std::span<const u8> actual) {
    for (size_t index = 0; index < actual.size(); ++index) {
        u8 expected = 0;
        if (address < memory.bytes.size() && index < memory.bytes.size() - address) {
            const size_t current = address + index;
            const int backing = memory.pages[current / Memory::PageSize];
            if (backing >= 0) {
                expected = memory.bytes[backing * Memory::PageSize + current % Memory::PageSize];
            }
        }
        REQUIRE(actual[index] == expected);
    }
}

struct StreamRuntime {
    struct StreamUniformRef {
        size_t allocation;
    };
    std::vector<std::vector<u8>> allocations;
    StreamUniformRef bound{};
    u64 CurrentTick() const {
        return 1;
    }
    StreamUniformRef LastStreamUniformRef() const {
        return bound;
    }
    std::span<u8> BindMappedUniformBuffer(size_t, u32, u32 size) {
        allocations.emplace_back(size);
        bound = {allocations.size() - 1};
        return allocations.back();
    }
    void RebindStreamUniform(StreamUniformRef ref, u32 size) {
        REQUIRE(allocations.at(ref.allocation).size() == size);
        bound = ref;
    }
};
} // namespace

TEST_CASE("Sparse native uniforms preserve holes, suffixes and unaligned page edges",
          "[video_core][sparse_uniform]") {
    SparseMemory memory;
    memory.Map(0, 8);
    memory.Map(2, 3);  // An interior hole must not discard this mapped suffix.
    memory.Map(3, 11); // Host backing is deliberately not contiguous.
    std::vector<u8> destination(0x3009, 0xa5);
    VideoCommon::ReadSparseUniformMemory(memory, 0xffb, destination);
    CheckBytes(memory, 0xffb, destination);
    CHECK(memory.pointer_reads == 5);
    CHECK(memory.direct_reads == 0);
    CHECK(std::all_of(destination.begin() + 5, destination.begin() + 0x1005,
                      [](u8 value) { return value == 0; }));
}

TEST_CASE("Sparse native uniform reads keep a contiguous fast path and bound address arithmetic",
          "[video_core][sparse_uniform]") {
    SparseMemory memory;
    memory.Map(1, 4);
    memory.Map(2, 5);
    std::vector<u8> destination(0x1001, 0xa5);
    VideoCommon::ReadSparseUniformMemory(memory, 0x1fff, destination);
    CheckBytes(memory, 0x1fff, destination);
    CHECK(memory.direct_reads == 1);
    CHECK(memory.pointer_reads == 0);
    memory.Map(15, 1);
    destination.assign(32, 0xa5);
    VideoCommon::ReadSparseUniformMemory(memory, 0xfff8, destination);
    CheckBytes(memory, 0xfff8, destination);
    const size_t reads = memory.pointer_reads;
    VideoCommon::ReadSparseUniformMemory(memory, (std::numeric_limits<u64>::max)() - 3,
                                         destination);
    CHECK(memory.pointer_reads == reads);
    CHECK(std::all_of(destination.begin(), destination.end(), [](u8 value) { return value == 0; }));
    VideoCommon::ReadSparseUniformMemory(memory, (std::numeric_limits<u64>::max)(),
                                         std::span<u8>{});
}

TEST_CASE("Retained native uniform uploads refresh across holes, remaps and frames",
          "[video_core][sparse_uniform]") {
    SparseMemory memory;
    memory.Map(0, 0);
    memory.Map(1, 1);
    memory.Map(2, 2);
    VideoCommon::SparseUniformRefreshTracker refresh;
    // Model the retained host descriptor allocation: skipping an upload leaves its old bytes.
    std::vector<u8> retained(0x3000, 0xa5);
    size_t uploads = 0;
    const auto bind = [&](u64 frame) {
        if (refresh.NeedsRefresh(0, static_cast<u32>(retained.size()), frame, memory.generation)) {
            VideoCommon::ReadSparseUniformMemory(memory, 0, retained);
            ++uploads;
        }
        CheckBytes(memory, 0, retained);
    };
    bind(1);
    bind(1);
    REQUIRE(uploads == 1);
    memory.Map(1, -1);
    bind(1); // Same frame: previous nonzero host bytes must become zero.
    REQUIRE(uploads == 2);
    memory.Map(1, 7);
    bind(1); // Same frame, no CPU write: zero bytes must become the new backing's bytes.
    REQUIRE(uploads == 3);
    memory.bytes[2] ^= 0xff;
    bind(2); // The existing per-frame refresh still covers untracked CPU changes.
    CHECK(uploads == 4);
    CHECK(retained.size() == 0x3000);
    CHECK(refresh.NeedsRefresh(0, 0x2000, 2, memory.generation));
}

TEST_CASE("Small sparse uniform stream reuse observes remapped backing in the same command",
          "[video_core][sparse_uniform]") {
    SparseMemory memory;
    memory.Map(0, 0);
    memory.Map(1, 1);
    VideoCommon::SparseUniformReadView view{memory};
    StreamRuntime runtime;
    VideoCommon::ResetUniformStreamReuse<StreamRuntime>();
    const auto bind = [&] {
        REQUIRE(VideoCommon::BindSmallUniformReusing(runtime, view, 0, 0, 0, 0xff0, 64));
        CheckBytes(memory, 0xff0, runtime.allocations.at(runtime.bound.allocation));
    };
    bind();
    bind();
    REQUIRE(runtime.allocations.size() == 1);
    memory.Map(1, -1);
    bind();
    REQUIRE(runtime.allocations.size() == 2);
    memory.Map(1, 9);
    bind();
    REQUIRE(runtime.allocations.size() == 3);
    bind();
    CHECK(runtime.allocations.size() == 3);
}

namespace {
struct LegacyGpuMapping {
    u64 gpu{0x40001a00};
    u64 device{0x1a00};
    bool mapped{true};
    bool continuous{true};
    u64 generation{1};
    mutable size_t translation_checks{};
    mutable size_t continuity_checks{};
    u64 GpuMappingGeneration() const {
        return generation;
    }
    std::optional<u64> GpuToCpuAddress(u64 address) const {
        ++translation_checks;
        return mapped && address == gpu ? std::optional{device} : std::nullopt;
    }
    bool IsContinuousRange(u64 address, size_t size) const {
        ++continuity_checks;
        return mapped && continuous && address == gpu && size == 0x10000;
    }
};
} // namespace

TEST_CASE("Legacy uniform linear-bank guard requires an exact descriptor and current GPU mapping",
          "[video_core][sparse_uniform]") {
    LegacyGpuMapping gpu;
    const auto eligible = [&](u32 descriptor = 0x10000, u32 shader = 0x10000) {
        return VideoCommon::IsLegacySparseUniformBank(gpu, 0x40001a00, 0x1a00, descriptor, shader);
    };
    REQUIRE(eligible());
    CHECK_FALSE(eligible(0xffff));
    CHECK_FALSE(eligible(0x10001));
    CHECK_FALSE(eligible(0x10000, 0xffff));
    gpu.continuous = false;
    CHECK_FALSE(eligible()); // A split bank cannot be mistaken for one linear device range.
    gpu.continuous = true;
    gpu.mapped = false;
    CHECK_FALSE(eligible());
    gpu.mapped = true;
    gpu.device += 0x1000;
    CHECK_FALSE(eligible()); // Binding's retained device address no longer matches the GPU mapping.
    CHECK_FALSE(VideoCommon::IsLegacySparseUniformBank(gpu, 0, gpu.device, 0x10000, 0x10000));
    CHECK_FALSE(VideoCommon::IsLegacySparseUniformBank(gpu, gpu.gpu, 0, 0x10000, 0x10000));
}

TEST_CASE(
    "Legacy uniform eligibility caches positive and negative answers until GPU mappings change",
    "[video_core][sparse_uniform]") {
    LegacyGpuMapping gpu;
    VideoCommon::LegacyUniformEligibilityCache cache;
    const auto eligible = [&] { return cache.IsEligible(gpu, gpu.gpu, 0x1a00, 0x10000, 0x10000); };
    REQUIRE(eligible());
    REQUIRE(eligible());
    CHECK(gpu.translation_checks == 1);
    CHECK(gpu.continuity_checks == 1);
    gpu.continuous = false;
    ++gpu.generation; // Mapping became discontinuous while the retained binding stayed identical.
    REQUIRE_FALSE(eligible());
    REQUIRE_FALSE(eligible());
    CHECK(gpu.translation_checks == 2);
    CHECK(gpu.continuity_checks == 2);
    gpu.continuous = true;
    ++gpu.generation;
    REQUIRE(eligible());
    CHECK(gpu.continuity_checks == 3);
    gpu.device = 0x2a00;
    ++gpu.generation; // A remap cannot keep admission for the binding's old device address.
    REQUIRE_FALSE(eligible());
    CHECK(cache.IsEligible(gpu, gpu.gpu, 0x2a00, 0x10000, 0x10000));
    CHECK_FALSE(cache.IsEligible(gpu, gpu.gpu, 0x2a00, 0xffff, 0x10000));
    CHECK_FALSE(cache.IsEligible(gpu, gpu.gpu, 0x2a00, 0x10000, 0xffff));
    LegacyGpuMapping other;
    other.device = gpu.device;
    other.generation = gpu.generation;
    other.continuous = false;
    REQUIRE_FALSE(cache.IsEligible(other, other.gpu, other.device, 0x10000, 0x10000));
    CHECK(other.translation_checks == 1); // Identical addresses/epoch, different memory owner.
    CHECK(other.continuity_checks == 1);
    CHECK(cache.IsEligible(gpu, gpu.gpu, gpu.device, 0x10000, 0x10000));
    const size_t previous_checks = gpu.translation_checks;
    gpu.gpu += 0x10000; // Moving this slot to another GPU bank invalidates the cached key, too.
    CHECK(cache.IsEligible(gpu, gpu.gpu, gpu.device, 0x10000, 0x10000));
    CHECK(gpu.translation_checks == previous_checks + 1);
}

namespace {
struct SplitGpuMapping {
    u64 gpu{0x40001a00};
    u64 first{0x1a00};
    u64 second{0x21000};
    u64 generation{1};
    bool fully_mapped{true};
    bool virtual_gap{};
    mutable size_t range_queries{};
    static constexpr u32 FirstSize = 0xf600;
    static constexpr u32 SecondSize = 0xa00;
    u64 GpuMappingGeneration() const {
        return generation;
    }
    std::optional<u64> GpuToCpuAddress(u64 address) const {
        if (address < gpu || address - gpu >= 0x10000 || !fully_mapped) {
            return std::nullopt;
        }
        const u64 offset = address - gpu;
        return offset < FirstSize ? first + offset : second + offset - FirstSize;
    }
    bool IsContinuousRange(u64, size_t) const {
        return false;
    }
    bool IsFullyMappedRange(u64 address, size_t size) const {
        return fully_mapped && address == gpu && size == 0x10000;
    }
    std::vector<std::pair<u64, size_t>> GetSubmappedRange(u64 address, size_t size) const {
        REQUIRE(address == gpu);
        REQUIRE(size == 0x10000);
        ++range_queries;
        return {{gpu, FirstSize}, {gpu + FirstSize + (virtual_gap ? 0x1000 : 0), SecondSize}};
    }
};

struct SegmentMemory : SparseMemoryImpl<18> {
    std::vector<std::string> events;
    template <class T>
    T* GetPointer(u64 address) {
        events.emplace_back("source");
        return SparseMemoryImpl<18>::GetPointer<T>(address);
    }
    const u8* GetSpan(u64 address, size_t size) {
        events.emplace_back("source");
        return SparseMemoryImpl<18>::GetSpan(address, size);
    }
};

std::vector<std::string> ControlEvents(const SegmentMemory& memory) {
    std::vector<std::string> result;
    for (const auto& event : memory.events) {
        if (event != "source") {
            result.push_back(event);
        }
    }
    return result;
}
} // namespace

TEST_CASE("Split legacy uniform banks gather actual device segments and download GPU-owned bytes",
          "[video_core][sparse_uniform]") {
    SplitGpuMapping gpu;
    SegmentMemory memory;
    for (size_t page = 1; page < 14; ++page) {
        memory.Map(page, static_cast<int>(page));
    }
    for (size_t page = 33; page <= 36; ++page) {
        memory.Map(page, static_cast<int>(page + 4));
    }
    // Wonder geometry: the ACTUAL first segment contains three backing holes (pages 14..16).
    // Its later virtual suffix lives at another device address, not the poisoned linear tail.
    memory.Map(17, 18);
    std::fill_n(memory.bytes.begin() + 18 * memory.PageSize, memory.PageSize, 0xd3);
    VideoCommon::LegacyUniformEligibilityCache cache;
    std::vector<u8> descriptor;
    bool gpu_modified = true;
    size_t downloads{};
    const auto bind = [&] {
        const auto segments = cache.SplitSegments(gpu, gpu.gpu, gpu.first, 0x10000, 0x10000);
        REQUIRE(segments.size() == 2);
        cache.ClearPrepared();
        REQUIRE(VideoCommon::PrepareSplitUniformBank(
            segments, [&](u64 address, u32) { return gpu_modified && address == gpu.second; },
            [&](u64 address, u32 size) {
                REQUIRE(address == gpu.second);
                REQUIRE(size == SplitGpuMapping::SecondSize);
                memory.events.emplace_back("download");
                // Model DownloadMemory making the latest GPU-owned byte authoritative on CPU.
                *memory.GetPointer<u8>(address) = 0x7e;
                gpu_modified = false;
                ++downloads;
            }));
        cache.SetPrepared(segments, 0, 0, 0);
        REQUIRE(
            VideoCommon::StreamSplitUniformBank(memory, cache.Prepared().segments, [&](u32 size) {
                memory.events.emplace_back("allocate");
                descriptor.assign(size, 0xa5);
                return std::span<u8>{descriptor};
            }));
        REQUIRE(descriptor.size() == 0x10000);
        CheckBytes(memory, gpu.first,
                   std::span<const u8>{descriptor}.first(SplitGpuMapping::FirstSize));
        CheckBytes(memory, gpu.second,
                   std::span<const u8>{descriptor}.subspan(SplitGpuMapping::FirstSize));
    };
    bind();
    REQUIRE(ControlEvents(memory) == std::vector<std::string>{"download", "allocate"});
    CHECK(downloads == 1);
    CHECK(descriptor[SplitGpuMapping::FirstSize] == 0x7e);
    const size_t hole_offset = 14 * memory.PageSize - gpu.first;
    CHECK(std::all_of(descriptor.begin() + hole_offset,
                      descriptor.begin() + SplitGpuMapping::FirstSize,
                      [](u8 value) { return value == 0; }));
    CHECK(descriptor[SplitGpuMapping::FirstSize] != 0xd3); // Never copy fictitious linear tail.
    memory.Map(14, 55); // Backing installed inside an actual translated segment, no GPU map change.
    *memory.GetPointer<u8>(14 * memory.PageSize) = 0xb4;
    bind();
    CHECK(descriptor[hole_offset] == 0xb4);
    memory.Map(14, 56); // A later backing remap must replace retained bytes, without a CPU write.
    *memory.GetPointer<u8>(14 * memory.PageSize) = 0xd7;
    bind();
    CHECK(descriptor[hole_offset] == 0xd7);
    REQUIRE(gpu.range_queries == 1);
    memory.events.clear();
    bind();
    CHECK(ControlEvents(memory) == std::vector<std::string>{"allocate"});
    CHECK(downloads == 1);
    CHECK(gpu.range_queries == 1); // The GPU segment layout is reused across stable draws.
    memory.Map(33,
               47); // Device-backing remap: gather sees fresh bytes without a GPU layout change.
    bind();
    CHECK(gpu.range_queries == 1);
    gpu.second = 0x31000;
    for (size_t page = 49; page <= 52; ++page) {
        memory.Map(page, static_cast<int>(page - 40));
    }
    ++gpu.generation;
    bind();
    CHECK(gpu.range_queries == 2); // GPU remap replaces the retained segment addresses.
}

TEST_CASE("Split legacy uniform admission rejects virtual holes, stale bindings and short banks",
          "[video_core][sparse_uniform]") {
    SplitGpuMapping gpu;
    VideoCommon::LegacyUniformEligibilityCache cache;
    const auto segments = [&] {
        return cache.SplitSegments(gpu, gpu.gpu, gpu.first, 0x10000, 0x10000);
    };
    REQUIRE(segments().size() == 2);
    gpu.fully_mapped = false;
    ++gpu.generation;
    CHECK(segments().empty());
    CHECK(segments().empty());
    CHECK(gpu.range_queries == 1);
    gpu.fully_mapped = true;
    gpu.virtual_gap = true;
    ++gpu.generation;
    CHECK(segments().empty()); // Full-map claims do not replace explicit segment coverage checks.
    CHECK(segments().empty());
    CHECK(gpu.range_queries == 2);
    gpu.virtual_gap = false;
    ++gpu.generation;
    CHECK(segments().size() == 2);
    CHECK(cache.SplitSegments(gpu, gpu.gpu, gpu.first + 0x1000, 0x10000, 0x10000).empty());
    CHECK(cache.SplitSegments(gpu, gpu.gpu, gpu.first, 0xffff, 0x10000).empty());
    CHECK(cache.SplitSegments(gpu, gpu.gpu, gpu.first, 0x10000, 0xffff).empty());
}

TEST_CASE("Malformed split uniform layouts cannot trigger downloads or partial descriptor reads",
          "[video_core][sparse_uniform]") {
    SegmentMemory memory;
    size_t callbacks{};
    const auto attempt = [&](std::span<const VideoCommon::LegacyUniformSegment> segments) {
        const bool prepared = VideoCommon::PrepareSplitUniformBank(
            segments,
            [&](u64, u32) {
                ++callbacks;
                return true;
            },
            [&](u64, u32) { ++callbacks; });
        const bool streamed = VideoCommon::StreamSplitUniformBank(memory, segments, [&](u32) {
            ++callbacks;
            return std::span<u8>{};
        });
        return prepared || streamed;
    };
    const std::array short_bank{VideoCommon::LegacyUniformSegment{0x1000, 0x1000},
                                VideoCommon::LegacyUniformSegment{0x9000, 0x1000}};
    CHECK_FALSE(attempt(short_bank));
    const std::array oversized{VideoCommon::LegacyUniformSegment{0x1000, 0xffff},
                               VideoCommon::LegacyUniformSegment{0x9000, 2}};
    CHECK_FALSE(attempt(oversized));
    const std::array zero_segment{VideoCommon::LegacyUniformSegment{0x1000, 0x10000},
                                  VideoCommon::LegacyUniformSegment{0x9000, 0}};
    CHECK_FALSE(attempt(zero_segment));
    CHECK(callbacks == 0);
    CHECK(memory.events.empty());
}

TEST_CASE("All-stage split uniform preparation completes before any draw stream allocation",
          "[video_core][sparse_uniform]") {
    SegmentMemory memory;
    for (size_t page = 1; page <= 52; ++page) {
        memory.Map(page, static_cast<int>(page));
    }
    std::array<SplitGpuMapping, 2> gpu;
    gpu[1].gpu += 0x100000;
    gpu[1].first = 0x11a00;
    gpu[1].second = 0x31000;
    std::array<VideoCommon::LegacyUniformEligibilityCache, 2> cache;
    std::array<bool, 2> prior_gpu_written{true, true};
    std::array<std::vector<u8>, 2> descriptors;
    size_t allocations{};
    // This models two stages. Complete both banks' readbacks before ordinary or split UBOs
    // allocate anything, rather than preparing lazily when each individual bank is bound.
    for (size_t stage = 0; stage < 2; ++stage) {
        cache[stage].ClearPrepared();
        const auto segments = cache[stage].SplitSegments(gpu[stage], gpu[stage].gpu,
                                                         gpu[stage].first, 0x10000, 0x10000);
        REQUIRE(VideoCommon::PrepareSplitUniformBank(
            segments,
            [&](u64 address, u32) {
                return prior_gpu_written[stage] && address == gpu[stage].second;
            },
            [&](u64 address, u32) {
                REQUIRE(allocations == 0); // Finish must precede even the earliest ordinary UBO.
                memory.events.emplace_back("download" + std::to_string(stage));
                *memory.GetPointer<u8>(address) = static_cast<u8>(0xe0 + stage);
                prior_gpu_written[stage] = false;
            }));
        cache[stage].SetPrepared(segments, 1, 0, 0);
    }
    memory.events.emplace_back("ordinary-allocate");
    ++allocations;
    // A later GPU-layout query must not mutate the owning per-draw snapshot.
    gpu[1].second = 0x41000;
    ++gpu[1].generation;
    REQUIRE(cache[1].SplitSegments(gpu[1], gpu[1].gpu, gpu[1].first, 0x10000, 0x10000).size() == 2);
    for (size_t stage = 0; stage < 2; ++stage) {
        // Storage bindings can now mark pending writes. Gathering cannot download these flags.
        prior_gpu_written[stage] = true;
        REQUIRE(VideoCommon::StreamSplitUniformBank(
            memory, cache[stage].Prepared().segments, [&](u32 size) {
                memory.events.emplace_back("split-allocate" + std::to_string(stage));
                ++allocations;
                descriptors[stage].assign(size, 0xa5);
                return std::span<u8>{descriptors[stage]};
            }));
        CHECK(descriptors[stage][SplitGpuMapping::FirstSize] == static_cast<u8>(0xe0 + stage));
    }
    CHECK(ControlEvents(memory) == std::vector<std::string>{"download0", "download1",
                                                            "ordinary-allocate", "split-allocate0",
                                                            "split-allocate1"});
    cache[0].ClearPrepared();
    cache[1].ClearPrepared();
    CHECK(cache[0].Prepared().segments.empty());
    CHECK(cache[1].Prepared().segments.empty()); // No prior-draw/channel snapshot may be reused.
}
