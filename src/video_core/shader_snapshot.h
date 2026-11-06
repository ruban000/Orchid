// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <cstring>
#include <limits>
#include <span>

#include "common/common_types.h"
#include "video_core/textures/texture.h"

namespace VideoCommon {

enum class ShaderSnapshotFailure {
    None,
    TexturePoolEntryLimit,
    TexturePoolNotMapped,
    Count,
};

// Preserve the existing admission policy. Report the actual reason instead of calling every
// failure a non-contiguous texture pool (the mapping test accepts discontiguous mapped pages).
template <class Memory>
[[nodiscard]] ShaderSnapshotFailure CheckTextureSnapshotPool(Memory& memory, GPUVAddr address,
                                                             u32 limit) {
    constexpr u32 MaxTexturePoolEntries = 1U << 15;
    if (limit >= MaxTexturePoolEntries) {
        return ShaderSnapshotFailure::TexturePoolEntryLimit;
    }
    const size_t bytes = (size_t{limit} + 1) * sizeof(Tegra::Texture::TICEntry);
    return memory.IsFullyMappedRange(address, bytes) ? ShaderSnapshotFailure::None
                                                     : ShaderSnapshotFailure::TexturePoolNotMapped;
}

// Constant-buffer capture previously used ReadBlockUnsafe (no GPU flush). Preserve that
// behavior, including zero-filled holes, while avoiding a bulk read across pending device
// backing. GetPointer resolves both GPU mapping and device backing; its pointer is copied only
// to the next 4 KiB boundary and never retained by a worker.
template <class Memory>
void ReadConstantBufferSnapshot(Memory& memory, GPUVAddr address, std::span<u8> destination) {
    if (destination.empty()) {
        return;
    }
    if (destination.size() - 1 > std::numeric_limits<GPUVAddr>::max() - address ||
        !memory.IsWithinGPUAddressRange(address) ||
        !memory.IsWithinGPUAddressRange(address + destination.size() - 1)) {
        std::memset(destination.data(), 0, destination.size());
        return;
    }
    constexpr size_t PageSize = 0x1000;
    size_t offset{};
    while (offset < destination.size()) {
        const GPUVAddr current = address + offset;
        const size_t count = (std::min)(destination.size() - offset,
                                        PageSize - static_cast<size_t>(current & (PageSize - 1)));
        if (const u8* pointer = memory.GetPointer(current)) {
            std::memcpy(destination.data() + offset, pointer, count);
        } else {
            std::memset(destination.data() + offset, 0, count);
        }
        offset += count;
    }
}

} // namespace VideoCommon
