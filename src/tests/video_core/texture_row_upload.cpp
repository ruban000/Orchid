// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <cstring>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "video_core/texture_cache/util.h"

TEST_CASE("Direct row uploads match whole-image bytes across mip and layer boundaries",
          "[texture_row_upload]") {
    using VideoCore::Surface::PixelFormat;
    const auto format =
        GENERATE(PixelFormat::R8_UNORM, PixelFormat::R16_UNORM, PixelFormat::A8B8G8R8_UNORM,
                 PixelFormat::R32G32_FLOAT, PixelFormat::R32G32B32A32_FLOAT,
                 PixelFormat::BC1_RGBA_UNORM, PixelFormat::BC3_UNORM);
    const bool native = GENERATE(false, true);
    const auto block_height = GENERATE(0U, 2U, 4U);
    const auto spacing = GENERATE(0U, 1U);
    const bool partial = GENERATE(false, true);
    CAPTURE(format, native, block_height, spacing, partial);
    VideoCommon::ImageInfo info;
    info.type = VideoCommon::ImageType::e2D;
    info.format = format;
    info.size = {258, 137, 1};
    info.resources = {5, 3};
    info.block.height = block_height;
    info.tile_width_spacing = spacing;
    info.layer_stride = VideoCommon::CalculateLayerStride(info);
    std::vector<u8> guest(VideoCommon::CalculateGuestSizeInBytes(info));
    u32 random = 0x9128ebf;
    for (auto& byte : guest) {
        random ^= random << 13;
        random ^= random >> 17;
        random ^= random << 5;
        byte = static_cast<u8>(random);
    }
    std::vector<u8> whole(VideoCommon::CalculateUnswizzledSizeBytes(info));
    const auto reference = VideoCommon::UnswizzleBlockLinearImage(info, guest, whole, native);
    const u64 middle = guest.size() / 2;
    const std::vector<std::pair<u64, u64>> ranges =
        partial ? std::vector<std::pair<u64, u64>>{{511, 1025},
                                                   {middle - 1, middle + 513},
                                                   {guest.size() - 517, guest.size()}}
                : std::vector<std::pair<u64, u64>>{{0, guest.size()}};
    constexpr GPUVAddr address = 0x123450000;
    constexpr size_t guard = 32;
    std::vector<u8> storage;
    size_t allocations = 0, reads = 0;
    const auto copies = VideoCommon::UnswizzleImageRows(
        [&](GPUVAddr source, std::span<u8> bytes) {
            REQUIRE(source >= address);
            REQUIRE(source - address + bytes.size() <= guest.size());
            std::memcpy(bytes.data(), guest.data() + source - address, bytes.size());
            ++reads;
        },
        address, info, ranges,
        [&](size_t bytes) {
            ++allocations;
            storage.assign(bytes + 2 * guard, 0xa5);
            return std::span<u8>{storage}.subspan(guard, bytes);
        },
        native);
    REQUIRE_FALSE(copies.empty());
    CHECK(allocations == 1);
    CHECK(reads == copies.size());
    CHECK(std::ranges::all_of(std::span{storage}.first(guard), [](u8 b) { return b == 0xa5; }));
    CHECK(std::ranges::all_of(std::span{storage}.last(guard), [](u8 b) { return b == 0xa5; }));
    const VideoCommon::Extent2D block{VideoCore::Surface::DefaultBlockWidth(format),
                                      VideoCore::Surface::DefaultBlockHeight(format)};
    const auto bpp = VideoCore::Surface::BytesPerBlock(format);
    size_t end = 0;
    for (const auto& copy : copies) {
        REQUIRE(copy.buffer_offset == end);
        end += copy.buffer_size;
        REQUIRE(end <= storage.size() - 2 * guard);
        const auto& ref = reference[copy.image_subresource.base_level];
        const auto row_bytes = ((copy.image_extent.width + block.width - 1) / block.width) * bpp;
        const auto rows = (copy.image_extent.height + block.height - 1) / block.height;
        const auto host_pitch = (copy.buffer_row_length / block.width) * bpp;
        const auto ref_pitch = ((ref.buffer_row_length + block.width - 1) / block.width) * bpp;
        const auto ref_rows = (ref.buffer_image_height + block.height - 1) / block.height;
        const auto ref_offset =
            ref.buffer_offset +
            static_cast<size_t>(copy.image_subresource.base_layer) * ref_pitch * ref_rows +
            static_cast<size_t>(copy.image_offset.y / static_cast<s32>(block.height)) * ref_pitch +
            static_cast<size_t>(copy.image_offset.x / static_cast<s32>(block.width)) * bpp;
        for (u32 row = 0; row < rows; ++row) {
            REQUIRE(ref_offset + row * ref_pitch + row_bytes <= whole.size());
            CHECK(std::memcmp(storage.data() + guard + copy.buffer_offset + row * host_pitch,
                              whole.data() + ref_offset + row * ref_pitch, row_bytes) == 0);
        }
    }
    CHECK(end == storage.size() - 2 * guard);
}

TEST_CASE("Unsupported and untouched row uploads do not allocate or read", "[texture_row_upload]") {
    VideoCommon::ImageInfo info;
    info.type = VideoCommon::ImageType::e2D;
    info.format = VideoCore::Surface::PixelFormat::R8_UNORM;
    info.size = {64, 64, 1};
    info.resources = {1, 1};
    const std::array<std::pair<u64, u64>, 1> outside{{{1ULL << 32, (1ULL << 32) + 1}}};
    const auto read = [](GPUVAddr, std::span<u8>) { FAIL("Unexpected read"); };
    const auto allocate = [](size_t) -> std::span<u8> {
        FAIL("Unexpected allocation");
        return {};
    };
    CHECK(VideoCommon::UnswizzleImageRows(read, 0, info, outside, allocate, false).empty());
    CHECK(VideoCommon::UnswizzleImageRows(read, 0, info, {}, allocate, false).empty());
    info.type = VideoCommon::ImageType::e3D;
    const std::array<std::pair<u64, u64>, 1> all{{{0, 4096}}};
    CHECK(VideoCommon::UnswizzleImageRows(read, 0, info, all, allocate, false).empty());
}

TEST_CASE("Short row-upload storage is rejected before reading guest memory",
          "[texture_row_upload]") {
    VideoCommon::ImageInfo info;
    info.type = VideoCommon::ImageType::e2D;
    info.format = VideoCore::Surface::PixelFormat::R8_UNORM;
    info.size = {64, 64, 1};
    info.resources = {1, 1};
    const std::array<std::pair<u64, u64>, 1> all{{{0, 4096}}};
    std::array<u8, 32> guard;
    guard.fill(0xa5);
    size_t allocations = 0;
    CHECK(VideoCommon::UnswizzleImageRows(
              [](GPUVAddr, std::span<u8>) { FAIL("Unexpected guest read"); }, 0, info, all,
              [&](size_t bytes) {
                  CHECK(bytes > guard.size());
                  ++allocations;
                  return std::span{guard};
              },
              false)
              .empty());
    CHECK(allocations == 1);
    CHECK(std::ranges::all_of(guard, [](u8 byte) { return byte == 0xa5; }));
}
