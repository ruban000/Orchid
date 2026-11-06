// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <span>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "video_core/textures/decoders.h"

namespace {
struct Layout {
    u32 width, height, depth, bytes_per_pixel;
    u32 block_height, block_depth, stride_alignment;
};

u64 DivideUp(u64 value, u64 divisor) {
    return (value + divisor - 1) / divisor;
}

u64 RowBytes(const Layout& layout) {
    return u64{layout.width} * layout.bytes_per_pixel;
}

u64 StrideBytes(const Layout& layout) {
    const u64 alignment = 1ULL << layout.stride_alignment;
    return DivideUp(layout.width, alignment) * alignment * layout.bytes_per_pixel;
}

u64 TiledBytes(const Layout& layout) {
    return DivideUp(StrideBytes(layout), 64) *
           (512ULL << layout.block_height << layout.block_depth) *
           DivideUp(layout.height, 8ULL << layout.block_height) *
           DivideUp(layout.depth, 1ULL << layout.block_depth);
}

// Independent byte-address oracle: address a block, then a GOB, then its 16-byte sector.
// No production pdep, decoder, NativeGobOffset or swizzle helper is used to construct the input.
u64 ReferenceOffset(const Layout& layout, u32 x_byte, u32 y, u32 z, bool native) {
    const u64 block_height = 1ULL << layout.block_height;
    const u64 block_depth = 1ULL << layout.block_depth;
    const u64 gobs_x = DivideUp(StrideBytes(layout), 64);
    const u64 blocks_y = DivideUp(layout.height, 8 * block_height);
    const u64 block_bytes = 512 * block_height * block_depth;
    const u64 block =
        ((z / block_depth) * blocks_y + y / (8 * block_height)) * gobs_x + x_byte / 64;
    const u64 gob = (z % block_depth) * block_height + (y / 8) % block_height;
    const u32 x = x_byte % 64;
    const u32 row = y % 8;
    const u32 sector =
        native ? (x / 32) * 256 + (row / 4) * 128 + ((x % 32) / 16) * 64 + ((row % 4) / 2) * 32 +
                     (row % 2) * 16
               : (x / 32) * 256 + (row / 2) * 64 + ((x % 32) / 16) * 32 + (row % 2) * 16;
    return block * block_bytes + gob * 512 + sector + x % 16;
}

u8 Pattern(size_t index, u32 layer) {
    u64 value = (index + u64{layer} * 31337 + 1) * 0x9e3779b185ebca87ULL;
    value ^= value >> 29;
    return static_cast<u8>(value >> 17);
}

void CheckLayout(const Layout& layout, bool native, bool clipped_source = false) {
    CAPTURE(layout.width, layout.height, layout.depth, layout.bytes_per_pixel, layout.block_height,
            layout.block_depth, layout.stride_alignment, native, clipped_source);
    constexpr size_t InputGuard = 7;
    constexpr size_t OutputGuard = 13;
    constexpr u8 Canary = 0xea;
    const size_t linear_size = RowBytes(layout) * layout.height * layout.depth;
    std::vector<u8> linear(linear_size);
    std::vector<u8> tiled(TiledBytes(layout) + 2 * InputGuard, Canary);
    size_t last_byte{};
    for (u32 z = 0; z < layout.depth; ++z) {
        for (u32 y = 0; y < layout.height; ++y) {
            for (u32 x = 0; x < RowBytes(layout); ++x) {
                const size_t offset = (size_t{z} * layout.height + y) * RowBytes(layout) + x;
                const auto address = ReferenceOffset(layout, x, y, z, native);
                last_byte = (std::max)(last_byte, static_cast<size_t>(address));
                linear[offset] = Pattern(offset, 0);
                tiled[InputGuard + address] = linear[offset];
            }
        }
    }
    const auto original_tiled = tiled;
    std::vector<u8> result(linear_size + 2 * OutputGuard, Canary);
    const size_t source_size = clipped_source ? last_byte + 1 : TiledBytes(layout);
    Tegra::Texture::UnswizzleTexture(std::span{result}.subspan(OutputGuard, linear_size),
                                     std::span<const u8>{tiled}.subspan(InputGuard, source_size),
                                     layout.bytes_per_pixel, layout.width, layout.height,
                                     layout.depth, layout.block_height, layout.block_depth,
                                     layout.stride_alignment, native);
    CHECK(std::equal(linear.begin(), linear.end(), result.begin() + OutputGuard));
    CHECK(std::all_of(result.begin(), result.begin() + OutputGuard,
                      [](u8 value) { return value == Canary; }));
    CHECK(std::all_of(result.end() - OutputGuard, result.end(),
                      [](u8 value) { return value == Canary; }));
    CHECK(tiled == original_tiled);
}
} // namespace

TEST_CASE("Texture unswizzle matches independent GOB layouts and partial edges",
          "[video_core][texture][unswizzle]") {
    constexpr std::array layouts{
        Layout{1, 1, 1, 1, 0, 0, 0},    Layout{15, 7, 1, 1, 0, 0, 0},
        Layout{16, 8, 1, 1, 0, 0, 0},   Layout{63, 9, 1, 1, 1, 0, 0},
        Layout{64, 8, 1, 1, 0, 0, 0},   Layout{65, 9, 1, 1, 2, 0, 0},
        Layout{127, 17, 1, 1, 3, 0, 0}, Layout{129, 19, 1, 2, 2, 0, 0},
        Layout{17, 9, 1, 4, 1, 0, 0},   Layout{19, 13, 1, 8, 2, 0, 3},
        Layout{9, 17, 1, 16, 3, 0, 4},  Layout{257, 65, 1, 4, 5, 0, 3},
    };
    for (const bool native : {false, true}) {
        for (const auto& layout : layouts) {
            CheckLayout(layout, native);
        }
        // Every admitted block height and element width, including partial right/bottom GOBs
        // and padded input strides. Unaligned guarded spans also exercise portable memcpy use.
        for (const u32 bpp : {1U, 2U, 4U, 8U, 16U}) {
            for (u32 block_height = 0; block_height <= 5; ++block_height) {
                for (const u32 width : {17U, 65U, 129U}) {
                    CheckLayout(Layout{width, 9 + block_height * 7, 1, bpp, block_height, 0, 3},
                                native);
                }
            }
        }
    }
}

TEST_CASE("Texture unswizzle keeps scalar fallback for 3D and unusual formats",
          "[video_core][texture][unswizzle]") {
    constexpr std::array layouts{
        Layout{8, 16, 5, 16, 1, 2, 1}, Layout{33, 7, 3, 4, 2, 1, 2}, Layout{65, 3, 7, 2, 0, 2, 4},
        Layout{17, 7, 3, 3, 1, 1, 3},  Layout{33, 9, 2, 6, 2, 1, 2}, Layout{11, 17, 1, 12, 2, 0, 4},
        Layout{65, 9, 1, 1, 6, 0, 2},
    };
    for (const bool native : {false, true}) {
        for (const auto& layout : layouts) {
            CheckLayout(layout, native);
        }
        // Only actually referenced source bytes are present: do not widen reads into GOB padding.
        CheckLayout(Layout{64, 1, 1, 1, 3, 0, 0}, native, true);
    }
}

TEST_CASE("Texture unswizzle preserves independent array layer boundaries",
          "[video_core][texture][unswizzle]") {
    constexpr Layout layout{17, 9, 1, 4, 1, 0, 2};
    constexpr size_t Layers = 3;
    const size_t layer_size = RowBytes(layout) * layout.height * layout.depth;
    const size_t tiled_stride = TiledBytes(layout);
    for (const bool native : {false, true}) {
        std::vector<u8> tiled(tiled_stride * Layers, 0xcc);
        std::vector<u8> expected(layer_size * Layers);
        std::vector<u8> result(expected.size(), 0xff);
        for (u32 layer = 0; layer < Layers; ++layer) {
            for (u32 z = 0; z < layout.depth; ++z) {
                for (u32 y = 0; y < layout.height; ++y) {
                    for (u32 x = 0; x < RowBytes(layout); ++x) {
                        const size_t offset =
                            (size_t{z} * layout.height + y) * RowBytes(layout) + x;
                        const u8 value = Pattern(offset, layer);
                        expected[layer * layer_size + offset] = value;
                        tiled[layer * tiled_stride + ReferenceOffset(layout, x, y, z, native)] =
                            value;
                    }
                }
            }
            Tegra::Texture::UnswizzleTexture(
                std::span{result}.subspan(layer * layer_size, layer_size),
                std::span<const u8>{tiled}.subspan(layer * tiled_stride, tiled_stride),
                layout.bytes_per_pixel, layout.width, layout.height, layout.depth,
                layout.block_height, layout.block_depth, layout.stride_alignment, native);
        }
        CHECK(result == expected);
    }
}
