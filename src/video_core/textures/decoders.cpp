// SPDX-FileCopyrightText: Copyright 2025 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2018 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>

#include "common/alignment.h"
#include "common/assert.h"
#include "common/bit_util.h"
#include "common/div_ceil.h"
#include "video_core/gpu.h"
#include "video_core/textures/decoders.h"

namespace Tegra::Texture {
namespace {
constexpr u32 NativeGobOffset(u32 offset, bool native_tiling) {
    if (!native_tiling) return offset;
    const u32 difference = ((offset >> 5) ^ (offset >> 6)) & 1;
    return offset ^ (difference << 5) ^ (difference << 6);
}
template <u32 mask>
constexpr u32 pdep(u32 value) {
    u32 result = 0;
    u32 m = mask;
    for (u32 bit = 1; m; bit += bit) {
        if (value & bit)
            result |= m & (~m + 1);
        m &= m - 1;
    }
    return result;
}

template <u32 mask, u32 incr_amount>
void incrpdep(u32& value) {
    static constexpr u32 swizzled_incr = pdep<mask>(incr_amount);
    value = ((value | ~mask) + swizzled_incr) & mask;
}

// A GOB stores each row in four contiguous 16-byte runs. Traverse eight-row GOB bands,
// avoiding per-pixel address arithmetic while keeping only eight output rows active.
// This avoids evicting unfinished output rows when textures have large row strides.
// Native tiling only exchanges two address bits above those runs.
template <bool NATIVE>
void UnswizzleGobs2D(std::span<u8> output, std::span<const u8> input, u32 row_bytes, u32 height,
                     u32 block_height, u32 gobs_in_x) {
    static constexpr std::array<u32, 8> row_offsets =
        NATIVE ? std::array<u32, 8>{0, 16, 32, 48, 128, 144, 160, 176}
               : std::array<u32, 8>{0, 16, 64, 80, 128, 144, 192, 208};
    static constexpr std::array<u32, 4> column_offsets{0, NATIVE ? 64U : 32U, 256,
                                                       NATIVE ? 320U : 288U};
    const u32 block_rows = GOB_SIZE_Y << block_height;
    const u32 block_bytes = GOB_SIZE << block_height;
    const u32 blocks_y = Common::DivCeil(height, block_rows);
    const u32 used_gobs_x = static_cast<u32>((u64{row_bytes} + GOB_SIZE_X - 1) / GOB_SIZE_X);
    for (u32 block_y = 0; block_y < blocks_y; ++block_y) {
        const u32 first_y = block_y * block_rows;
        const u32 rows = (std::min)(block_rows, height - first_y);
        const u32 used_gobs_y = Common::DivCeil(rows, GOB_SIZE_Y);
        for (u32 gob_y = 0; gob_y < used_gobs_y; ++gob_y) {
            const u32 first_row = gob_y * GOB_SIZE_Y;
            const u32 lines = (std::min)(GOB_SIZE_Y, rows - first_row);
            for (u32 gob_x = 0; gob_x < used_gobs_x; ++gob_x) {
                const u32 first_x = gob_x * GOB_SIZE_X;
                const u32 bytes = (std::min)(GOB_SIZE_X, row_bytes - first_x);
                const size_t block_offset = (size_t{block_y} * gobs_in_x + gob_x) * block_bytes;
                const u8* const gob = input.data() + block_offset + gob_y * GOB_SIZE;
                for (u32 line = 0; line < lines; ++line) {
                    u8* const dst =
                        output.data() + size_t{first_y + first_row + line} * row_bytes + first_x;
                    const u8* const src = gob + row_offsets[line];
                    if (bytes == GOB_SIZE_X) {
                        std::memcpy(dst, src, 16);
                        std::memcpy(dst + 16, src + column_offsets[1], 16);
                        std::memcpy(dst + 32, src + column_offsets[2], 16);
                        std::memcpy(dst + 48, src + column_offsets[3], 16);
                    } else {
                        for (u32 column = 0; column < bytes; column += 16) {
                            std::memcpy(dst + column, src + column_offsets[column / 16],
                                        (std::min)(16U, bytes - column));
                        }
                    }
                }
            }
        }
    }
}

bool TryUnswizzleGobs2D(std::span<u8> output, std::span<const u8> input, u32 bytes_per_pixel,
                        u32 width, u32 height, u32 depth, u32 block_height, u32 block_depth,
                        u32 stride, bool native_tiling) {
    // Keep unusual formats, 3D block layouts and short source spans on the original path.
    if (depth != 1 || block_depth != 0 || block_height > 5 || height == 0 ||
        (bytes_per_pixel != 1 && bytes_per_pixel != 2 && bytes_per_pixel != 4 &&
         bytes_per_pixel != 8 && bytes_per_pixel != 16)) {
        return false;
    }
    constexpr u64 MaxOffset = (std::numeric_limits<u32>::max)();
    const u64 row_bytes = u64{width} * bytes_per_pixel;
    if (row_bytes < GOB_SIZE_X || row_bytes > MaxOffset || stride < row_bytes) {
        return false;
    }
    const u64 linear_bytes = row_bytes * height;
    if (linear_bytes > MaxOffset || output.size() < linear_bytes) {
        return false;
    }
    const u64 gobs_in_x = (u64{stride} + GOB_SIZE_X - 1) >> GOB_SIZE_X_SHIFT;
    const u64 block_rows = u64{GOB_SIZE_Y} << block_height;
    const u64 blocks_y = (u64{height} + block_rows - 1) / block_rows;
    const u64 block_bytes = u64{GOB_SIZE} << block_height;
    const u64 block_row_bytes = gobs_in_x * block_bytes;
    if (block_row_bytes > MaxOffset || blocks_y > MaxOffset / block_row_bytes) {
        return false;
    }
    const u64 tiled_bytes = blocks_y * block_row_bytes;
    if (input.size() < tiled_bytes) {
        return false;
    }
    // The original traversal remains available for callers with overlapping spans.
    const auto src = reinterpret_cast<std::uintptr_t>(input.data());
    const auto dst = reinterpret_cast<std::uintptr_t>(output.data());
    if (src <= dst ? dst - src < tiled_bytes : src - dst < linear_bytes) {
        return false;
    }
    if (native_tiling) {
        UnswizzleGobs2D<true>(output, input, static_cast<u32>(row_bytes), height, block_height,
                              static_cast<u32>(gobs_in_x));
    } else {
        UnswizzleGobs2D<false>(output, input, static_cast<u32>(row_bytes), height, block_height,
                               static_cast<u32>(gobs_in_x));
    }
    return true;
}

template <bool TO_LINEAR, u32 BYTES_PER_PIXEL>
void SwizzleImpl(std::span<u8> output, std::span<const u8> input, u32 width, u32 height, u32 depth,
                 u32 block_height, u32 block_depth, u32 stride, bool native_tiling) {
    // The origin of the transformation can be configured here, leave it as zero as the current API
    // doesn't expose it.
    static constexpr u32 origin_x = 0;
    static constexpr u32 origin_y = 0;
    static constexpr u32 origin_z = 0;

    // We can configure here a custom pitch
    // As it's not exposed 'width * BYTES_PER_PIXEL' will be the expected pitch.
    const u32 pitch = width * BYTES_PER_PIXEL;

    const u32 gobs_in_x = Common::DivCeilLog2(stride, GOB_SIZE_X_SHIFT);
    const u32 block_size = gobs_in_x << (GOB_SIZE_SHIFT + block_height + block_depth);
    const u32 slice_size =
        Common::DivCeilLog2(height, block_height + GOB_SIZE_Y_SHIFT) * block_size;

    const u32 block_height_mask = (1U << block_height) - 1;
    const u32 block_depth_mask = (1U << block_depth) - 1;
    const u32 x_shift = GOB_SIZE_SHIFT + block_height + block_depth;

    for (u32 slice = 0; slice < depth; ++slice) {
        const u32 z = slice + origin_z;
        const u32 offset_z = (z >> block_depth) * slice_size +
                             ((z & block_depth_mask) << (GOB_SIZE_SHIFT + block_height));
        for (u32 line = 0; line < height; ++line) {
            const u32 y = line + origin_y;
            const u32 swizzled_y = pdep<SWIZZLE_Y_BITS>(y);

            const u32 block_y = y >> GOB_SIZE_Y_SHIFT;
            const u32 offset_y = (block_y >> block_height) * block_size +
                                 ((block_y & block_height_mask) << GOB_SIZE_SHIFT);

            u32 swizzled_x = pdep<SWIZZLE_X_BITS>(origin_x * BYTES_PER_PIXEL);
            for (u32 column = 0; column < width;
                 ++column, incrpdep<SWIZZLE_X_BITS, BYTES_PER_PIXEL>(swizzled_x)) {
                const u32 x = (column + origin_x) * BYTES_PER_PIXEL;
                const u32 offset_x = (x >> GOB_SIZE_X_SHIFT) << x_shift;

                const u32 base_swizzled_offset = offset_z + offset_y + offset_x;
                const u32 swizzled_offset = NativeGobOffset(base_swizzled_offset + (swizzled_x | swizzled_y), native_tiling);

                const u32 unswizzled_offset =
                    slice * pitch * height + line * pitch + column * BYTES_PER_PIXEL;

                u8* const dst = &output[TO_LINEAR ? swizzled_offset : unswizzled_offset];
                const u8* const src = &input[TO_LINEAR ? unswizzled_offset : swizzled_offset];

                std::memcpy(dst, src, BYTES_PER_PIXEL);
            }
        }
    }
}

template <bool TO_LINEAR, u32 BYTES_PER_PIXEL>
void SwizzleSubrectImpl(std::span<u8> output, std::span<const u8> input, u32 width, u32 height,
                        u32 depth, u32 origin_x, u32 origin_y, u32 extent_x, u32 num_lines,
                        u32 block_height, u32 block_depth, u32 pitch_linear, bool native_tiling) {
    // The origin of the transformation can be configured here, leave it as zero as the current API
    // doesn't expose it.
    static constexpr u32 origin_z = 0;

    // We can configure here a custom pitch
    // As it's not exposed 'width * BYTES_PER_PIXEL' will be the expected pitch.
    const u32 pitch = pitch_linear;
    const u32 stride = Common::AlignUpLog2(width * BYTES_PER_PIXEL, GOB_SIZE_X_SHIFT);

    const u32 gobs_in_x = Common::DivCeilLog2(stride, GOB_SIZE_X_SHIFT);
    const u32 block_size = gobs_in_x << (GOB_SIZE_SHIFT + block_height + block_depth);
    const u32 slice_size =
        Common::DivCeilLog2(height, block_height + GOB_SIZE_Y_SHIFT) * block_size;

    const u32 block_height_mask = (1U << block_height) - 1;
    const u32 block_depth_mask = (1U << block_depth) - 1;
    const u32 x_shift = GOB_SIZE_SHIFT + block_height + block_depth;

    u32 unprocessed_lines = num_lines;
    u32 extent_y = (std::min)(num_lines, height - origin_y);

    for (u32 slice = 0; slice < depth; ++slice) {
        const u32 z = slice + origin_z;
        const u32 offset_z = (z >> block_depth) * slice_size +
                             ((z & block_depth_mask) << (GOB_SIZE_SHIFT + block_height));
        const u32 lines_in_y = (std::min)(unprocessed_lines, extent_y);
        for (u32 line = 0; line < lines_in_y; ++line) {
            const u32 y = line + origin_y;
            const u32 swizzled_y = pdep<SWIZZLE_Y_BITS>(y);

            const u32 block_y = y >> GOB_SIZE_Y_SHIFT;
            const u32 offset_y = (block_y >> block_height) * block_size +
                                 ((block_y & block_height_mask) << GOB_SIZE_SHIFT);

            u32 swizzled_x = pdep<SWIZZLE_X_BITS>(origin_x * BYTES_PER_PIXEL);
            for (u32 column = 0; column < extent_x;
                 ++column, incrpdep<SWIZZLE_X_BITS, BYTES_PER_PIXEL>(swizzled_x)) {
                const u32 x = (column + origin_x) * BYTES_PER_PIXEL;
                const u32 offset_x = (x >> GOB_SIZE_X_SHIFT) << x_shift;

                const u32 base_swizzled_offset = offset_z + offset_y + offset_x;
                const u32 swizzled_offset = NativeGobOffset(base_swizzled_offset + (swizzled_x | swizzled_y), native_tiling);

                const u32 unswizzled_offset =
                    slice * pitch * height + line * pitch + column * BYTES_PER_PIXEL;

                u8* const dst = &output[TO_LINEAR ? swizzled_offset : unswizzled_offset];
                const u8* const src = &input[TO_LINEAR ? unswizzled_offset : swizzled_offset];

                std::memcpy(dst, src, BYTES_PER_PIXEL);
            }
        }
        unprocessed_lines -= lines_in_y;
        if (unprocessed_lines == 0) {
            return;
        }
    }
}

template <bool TO_LINEAR>
void Swizzle(std::span<u8> output, std::span<const u8> input, u32 bytes_per_pixel, u32 width,
             u32 height, u32 depth, u32 block_height, u32 block_depth, u32 stride_alignment, bool native_tiling) {
    switch (bytes_per_pixel) {
#define BPP_CASE(x)                                                                                \
    case x:                                                                                        \
        return SwizzleImpl<TO_LINEAR, x>(output, input, width, height, depth, block_height,        \
                                         block_depth, stride_alignment, native_tiling);
        BPP_CASE(1)
        BPP_CASE(2)
        BPP_CASE(3)
        BPP_CASE(4)
        BPP_CASE(6)
        BPP_CASE(8)
        BPP_CASE(12)
        BPP_CASE(16)
#undef BPP_CASE
    default:
        ASSERT_MSG(false, "Invalid bytes_per_pixel={}", bytes_per_pixel);
        break;
    }
}

} // Anonymous namespace

void UnswizzleTexture(std::span<u8> output, std::span<const u8> input, u32 bytes_per_pixel,
                      u32 width, u32 height, u32 depth, u32 block_height, u32 block_depth,
                      u32 stride_alignment, bool native_tiling) {
    const u32 stride = Common::AlignUpLog2(width, stride_alignment) * bytes_per_pixel;
    if (TryUnswizzleGobs2D(output, input, bytes_per_pixel, width, height, depth, block_height,
                            block_depth, stride, native_tiling)) {
        return;
    }
    const u32 new_bpp = (std::min)(4U, static_cast<u32>(std::countr_zero(width * bytes_per_pixel)));
    width = (width * bytes_per_pixel) >> new_bpp;
    bytes_per_pixel = 1U << new_bpp;
    Swizzle<false>(output, input, bytes_per_pixel, width, height, depth, block_height, block_depth,
                   stride, native_tiling);
}

void SwizzleTexture(std::span<u8> output, std::span<const u8> input, u32 bytes_per_pixel, u32 width,
                    u32 height, u32 depth, u32 block_height, u32 block_depth,
                    u32 stride_alignment, bool native_tiling) {
    const u32 stride = Common::AlignUpLog2(width, stride_alignment) * bytes_per_pixel;
    const u32 new_bpp = (std::min)(4U, static_cast<u32>(std::countr_zero(width * bytes_per_pixel)));
    width = (width * bytes_per_pixel) >> new_bpp;
    bytes_per_pixel = 1U << new_bpp;
    Swizzle<true>(output, input, bytes_per_pixel, width, height, depth, block_height, block_depth,
                  stride, native_tiling);
}

void SwizzleSubrect(std::span<u8> output, std::span<const u8> input, u32 bytes_per_pixel, u32 width,
                    u32 height, u32 depth, u32 origin_x, u32 origin_y, u32 extent_x, u32 extent_y,
                    u32 block_height, u32 block_depth, u32 pitch_linear, bool native_tiling) {
    switch (bytes_per_pixel) {
#define BPP_CASE(x)                                                                                \
    case x:                                                                                        \
        return SwizzleSubrectImpl<true, x>(output, input, width, height, depth, origin_x,          \
                                           origin_y, extent_x, extent_y, block_height,             \
                                           block_depth, pitch_linear, native_tiling);
        BPP_CASE(1)
        BPP_CASE(2)
        BPP_CASE(3)
        BPP_CASE(4)
        BPP_CASE(6)
        BPP_CASE(8)
        BPP_CASE(12)
        BPP_CASE(16)
#undef BPP_CASE
    default:
        ASSERT_MSG(false, "Invalid bytes_per_pixel={}", bytes_per_pixel);
        break;
    }
}

void UnswizzleSubrect(std::span<u8> output, std::span<const u8> input, u32 bytes_per_pixel,
                      u32 width, u32 height, u32 depth, u32 origin_x, u32 origin_y, u32 extent_x,
                      u32 extent_y, u32 block_height, u32 block_depth, u32 pitch_linear, bool native_tiling) {
    switch (bytes_per_pixel) {
#define BPP_CASE(x)                                                                                \
    case x:                                                                                        \
        return SwizzleSubrectImpl<false, x>(output, input, width, height, depth, origin_x,         \
                                            origin_y, extent_x, extent_y, block_height,            \
                                            block_depth, pitch_linear, native_tiling);
        BPP_CASE(1)
        BPP_CASE(2)
        BPP_CASE(3)
        BPP_CASE(4)
        BPP_CASE(6)
        BPP_CASE(8)
        BPP_CASE(12)
        BPP_CASE(16)
#undef BPP_CASE
    default:
        ASSERT_MSG(false, "Invalid bytes_per_pixel={}", bytes_per_pixel);
        break;
    }
}

std::size_t CalculateSize(bool tiled, u32 bytes_per_pixel, u32 width, u32 height, u32 depth,
                          u32 block_height, u32 block_depth) {
    if (tiled) {
        const u32 aligned_width = Common::AlignUpLog2(width * bytes_per_pixel, GOB_SIZE_X_SHIFT);
        const u32 aligned_height = Common::AlignUpLog2(height, GOB_SIZE_Y_SHIFT + block_height);
        const u32 aligned_depth = Common::AlignUpLog2(depth, GOB_SIZE_Z_SHIFT + block_depth);
        return aligned_width * aligned_height * aligned_depth;
    } else {
        return width * height * depth * bytes_per_pixel;
    }
}

u64 GetGOBOffset(u32 width, u32 height, u32 dst_x, u32 dst_y, u32 block_height,
                 u32 bytes_per_pixel) {
    auto div_ceil = [](const u32 x, const u32 y) { return ((x + y - 1) / y); };
    const u32 gobs_in_block = 1 << block_height;
    const u32 y_blocks = GOB_SIZE_Y << block_height;
    const u32 x_per_gob = GOB_SIZE_X / bytes_per_pixel;
    const u32 x_blocks = div_ceil(width, x_per_gob);
    const u32 block_size = GOB_SIZE * gobs_in_block;
    const u32 stride = block_size * x_blocks;
    const u32 base = (dst_y / y_blocks) * stride + (dst_x / x_per_gob) * block_size;
    const u32 relative_y = dst_y % y_blocks;
    return base + (relative_y / GOB_SIZE_Y) * GOB_SIZE;
}

} // namespace Tegra::Texture
