// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <cstring>
#include <span>
#include <unordered_map>
#include <vector>

#include "common/common_types.h"

namespace VideoCommon {

// Native banks and validated split legacy uniform segments may contain device pages without
// backing. Preserve that whole descriptor range: holes read zero, and mapped pages after holes
// still count. Other buffer reads retain their strict unmapped-memory diagnostics.
template <class Memory>
void ReadSparseUniformMemory(Memory& memory, u64 address, std::span<u8> destination) {
    static_assert(Memory::AS_BITS < 64);
    constexpr u64 address_limit = u64{1} << Memory::AS_BITS;
    constexpr u64 page_size = 0x1000;
    if (destination.empty()) {
        return;
    }
    if (address >= address_limit) {
        std::memset(destination.data(), 0, destination.size());
        return;
    }
    const size_t readable =
        static_cast<size_t>((std::min<u64>)(destination.size(), address_limit - address));
    if (readable == destination.size()) {
        if (const u8* pointer = memory.GetSpan(address, readable)) {
            std::memcpy(destination.data(), pointer, readable);
            return;
        }
    }
    size_t offset = 0;
    while (offset < readable) {
        const size_t count =
            (std::min)(readable - offset, size_t{page_size - (address & (page_size - 1))});
        if (const u8* pointer = memory.template GetPointer<u8>(address)) {
            std::memcpy(destination.data() + offset, pointer, count);
        } else {
            std::memset(destination.data() + offset, 0, count);
        }
        offset += count;
        address += count;
    }
    if (readable < destination.size()) {
        std::memset(destination.data() + readable, 0, destination.size() - readable);
    }
}

// Adapter for the existing uniform stream reuse reader. Comparing freshly read zero-filled
// bytes also prevents reusing an allocation whose source page was unmapped or remapped.
template <class Memory>
struct SparseUniformReadView {
    Memory& memory;
    void ReadBlockUnsafe(u64 address, void* destination, size_t size) {
        ReadSparseUniformMemory(memory, address, std::span{static_cast<u8*>(destination), size});
    }
};

// Linear-bank guard used to distinguish ordinary legacy bindings from the split gather path.
// This checks GPU-to-device continuity, not host backing, and does not enable sparse/zero reads
// for ordinary legacy buffers. Missing GPU mappings and stale retained translations retain their
// existing diagnostics; fully mapped split banks are read only through their actual segments.
template <class GpuMemory>
bool IsLegacySparseUniformBank(GpuMemory& memory, u64 gpu_address, u64 device_address,
                               u32 descriptor_size, u32 shader_size) {
    constexpr u32 bank_size = 0x10000;
    if (descriptor_size != bank_size || shader_size != bank_size || gpu_address == 0 ||
        device_address == 0) {
        return false;
    }
    const auto translated = memory.GpuToCpuAddress(gpu_address);
    return translated && *translated == device_address &&
           memory.IsContinuousRange(gpu_address, bank_size);
}

struct LegacyUniformSegment {
    u64 device_address;
    u32 size;
};

inline bool IsCompleteSplitUniformBank(std::span<const LegacyUniformSegment> segments) {
    constexpr u32 bank_size = 0x10000;
    u32 total{};
    for (const auto& segment : segments) {
        if (segment.size == 0 || segment.size > bank_size - total) {
            return false;
        }
        total += segment.size;
    }
    return total == bank_size && segments.size() >= 2;
}

// Run for ALL stages before allocating any per-draw stream/descriptor payload. Download may
// finish a scheduler tick, which would retire already allocated, not-yet-consumed stream data.
template <class IsGpuModified, class Download>
bool PrepareSplitUniformBank(std::span<const LegacyUniformSegment> segments,
                             IsGpuModified&& is_gpu_modified, Download&& download) {
    if (!IsCompleteSplitUniformBank(segments)) {
        return false;
    }
    for (const auto& segment : segments) {
        if (is_gpu_modified(segment.device_address, segment.size)) {
            download(segment.device_address, segment.size);
        }
    }
    return true;
}

// Only allocate and read the prepared actual device segments here. Their backing holes read zero;
// each later mapped segment still contributes its bytes. No linear device-bank range is inferred.
// This phase cannot download/Finish: earlier banks/stages may already have stream descriptors.
template <class Memory, class Allocate>
bool StreamSplitUniformBank(Memory& memory, std::span<const LegacyUniformSegment> segments,
                            Allocate&& allocate) {
    constexpr u32 bank_size = 0x10000;
    if (!IsCompleteSplitUniformBank(segments)) {
        return false;
    }
    const std::span<u8> destination = allocate(bank_size);
    if (destination.size() < bank_size) {
        return false;
    }
    u32 offset{};
    for (const auto& segment : segments) {
        ReadSparseUniformMemory(memory, segment.device_address,
                                destination.subspan(offset, segment.size));
        offset += segment.size;
    }
    return true;
}

// One cache per uniform slot avoids walking the GPU page table on every draw. Both positive
// and negative answers are reusable until a mapping changes, the owner changes, or the binding
// moves. Descriptor-size checks stay outside the cache so a short bank can never reuse admission.
class LegacyUniformEligibilityCache {
public:
    struct PreparedBank {
        std::vector<LegacyUniformSegment> segments;
        u32 downloaded_segments{};
        u64 gpu_generation{};
        s64 begin_ns{};
        s64 end_ns{};
    };

    void ClearPrepared() {
        prepared.segments.clear();
    }

    void SetPrepared(std::span<const LegacyUniformSegment> segments, u32 downloaded_segments,
                     s64 begin_ns, s64 end_ns) {
        prepared.segments.assign(segments.begin(), segments.end());
        prepared.downloaded_segments = downloaded_segments;
        prepared.gpu_generation = mapping_generation;
        prepared.begin_ns = begin_ns;
        prepared.end_ns = end_ns;
    }

    const PreparedBank& Prepared() const {
        return prepared;
    }

    template <class GpuMemory>
    bool IsEligible(GpuMemory& memory, u64 gpu_address, u64 device_address, u32 descriptor_size,
                    u32 shader_size) {
        if (descriptor_size != 0x10000 || shader_size != 0x10000 || gpu_address == 0 ||
            device_address == 0) {
            return false;
        }
        const u64 generation = memory.GpuMappingGeneration();
        if (owner == &memory && gpu == gpu_address && device == device_address &&
            mapping_generation == generation) {
            return eligible;
        }
        const bool result = IsLegacySparseUniformBank(memory, gpu_address, device_address,
                                                      descriptor_size, shader_size);
        owner = &memory;
        gpu = gpu_address;
        device = device_address;
        mapping_generation = generation;
        eligible = result;
        split_checked = false;
        split_diagnostic_claimed = false;
        split_segments.clear();
        return result;
    }

    template <class GpuMemory>
    std::span<const LegacyUniformSegment> SplitSegments(GpuMemory& memory, u64 gpu_address,
                                                        u64 device_address, u32 descriptor_size,
                                                        u32 shader_size) {
        constexpr u32 bank_size = 0x10000;
        if (descriptor_size != bank_size || shader_size != bank_size || gpu_address == 0 ||
            device_address == 0 ||
            IsEligible(memory, gpu_address, device_address, descriptor_size, shader_size)) {
            return {};
        }
        if (split_checked) {
            return split_segments;
        }
        split_checked = true;
        const auto first_device = memory.GpuToCpuAddress(gpu_address);
        if (!first_device || *first_device != device_address ||
            !memory.IsFullyMappedRange(gpu_address, bank_size)) {
            return {};
        }
        const auto ranges = memory.GetSubmappedRange(gpu_address, bank_size);
        u32 covered{};
        for (const auto& [address, size] : ranges) {
            if (address != gpu_address + covered || size == 0 || size > bank_size - covered) {
                split_segments.clear();
                return {};
            }
            const auto translated = memory.GpuToCpuAddress(address);
            if (!translated || *translated == 0) {
                split_segments.clear();
                return {};
            }
            split_segments.push_back({*translated, static_cast<u32>(size)});
            covered += static_cast<u32>(size);
        }
        if (covered != bank_size || split_segments.size() < 2) {
            split_segments.clear();
        }
        return split_segments;
    }

    bool ClaimSplitDiagnostic() {
        if (split_diagnostic_claimed) {
            return false;
        }
        split_diagnostic_claimed = true;
        return true;
    }

private:
    const void* owner{};
    u64 gpu{};
    u64 device{};
    u64 mapping_generation{};
    bool eligible{};
    bool split_checked{};
    bool split_diagnostic_claimed{};
    std::vector<LegacyUniformSegment> split_segments;
    PreparedBank prepared;
};

// Each bank is refreshed once per frame, and again if device mappings change in that frame.
// Map() does not mark buffer-cache pages CPU modified, so remembering only the frame can retain
// zeros uploaded for a hole after that hole acquires real backing.
class SparseUniformRefreshTracker {
public:
    bool NeedsRefresh(u64 address, u32 size, u64 frame, u64 mapping_generation) {
        if (frame != current_frame) {
            banks.clear();
            current_frame = frame;
        }
        const auto [it, inserted] = banks.try_emplace(address, Entry{size, mapping_generation});
        if (!inserted && it->second.size == size &&
            it->second.mapping_generation == mapping_generation) {
            return false;
        }
        it->second = {size, mapping_generation};
        return true;
    }

private:
    struct Entry {
        u32 size;
        u64 mapping_generation;
    };
    u64 current_frame{~u64{0}};
    std::unordered_map<u64, Entry> banks;
};

} // namespace VideoCommon
