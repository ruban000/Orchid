// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <algorithm>
#include <cstddef>
#include <span>
#include <utility>
#include "common/common_types.h"

namespace VideoCommon {

// Native compute pointers have no following size word.
[[nodiscard]] constexpr u32 StorageBufferSizeLimit(u64 kind_extent, u64 next_word,
                                                   bool native_compute) {
    constexpr u32 mib = 1024 * 1024;
    if (native_compute) {
        return static_cast<u32>((std::min)(kind_extent, u64{8U * mib}));
    }
    const u32 packed_size = static_cast<u32>(next_word);
    if ((next_word >> 32) == 0 && packed_size != 0 && packed_size <= kind_extent) {
        return packed_size;
    }
    return static_cast<u32>((std::min)(kind_extent, u64{8U * mib}));
}

// The mapping API splits on both physical discontinuities and unmapped holes. Adjacent
// virtual ranges still form one usable prefix for the multi-range storage path.
[[nodiscard]] inline u32 StorageBufferMappedPrefix(
    u64 start, u32 limit, std::span<const std::pair<u64, std::size_t>> ranges) {
    u32 mapped = 0;
    for (const auto& [address, length] : ranges) {
        if (address != start + mapped || length > limit - mapped) {
            break;
        }
        mapped += static_cast<u32>(length);
    }
    return mapped;
}

} // namespace VideoCommon
