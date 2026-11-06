// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <bit>
#include <cstddef>
#include <limits>
#include <tuple>
#include <utility>
#include <catch2/catch_test_macros.hpp>

#include "video_core/buffer_cache/buffer_cache_base.h"
#include "video_core/buffer_cache/storage_buffer_size.h"

TEST_CASE("Native compute pointers use their mapped extent, not the next parameter",
          "[video_core][compute_storage]") {
    constexpr u32 mib = 1024 * 1024;
    using VideoCommon::StorageBufferSizeLimit;
    // The following value can be a texture handle, as in Fortune's Weave.
    CHECK(StorageBufferSizeLimit(4 * mib, 0x10a, true) == 4 * mib);
    CHECK(StorageBufferSizeLimit(16 * mib, 0x10a, true) == 8 * mib);
    // NVN and graphics descriptors still use their explicit valid size.
    CHECK(StorageBufferSizeLimit(4 * mib, 0x10a, false) == 0x10aU);
    CHECK(StorageBufferSizeLimit(4 * mib, 0, false) == 4 * mib);
    CHECK(StorageBufferSizeLimit(4 * mib, 0x10000010aULL, false) == 4 * mib);
    // Cap before narrowing: a kind range of 4 GiB must not wrap to zero.
    constexpr u64 four_gib = 1ULL << 32;
    CHECK(StorageBufferSizeLimit(four_gib, 0x10a, true) == 8 * mib);
    CHECK(StorageBufferSizeLimit(0, 0x10a, true) == 0U);
    CHECK(StorageBufferSizeLimit(0x1000, 0x10a, true) == 0x1000U);
}

TEST_CASE("Native storage mapped prefix joins adjacent virtual segments and stops at gaps",
          "[video_core][compute_storage]") {
    using VideoCommon::StorageBufferMappedPrefix;
    constexpr u64 base = 0x10000;
    constexpr std::array adjacent{std::pair<u64, std::size_t>{base, 0x1000},
                                  std::pair<u64, std::size_t>{base + 0x1000, 0x1000}};
    constexpr std::array gap{std::pair<u64, std::size_t>{base, 0x1000},
                             std::pair<u64, std::size_t>{base + 0x2000, 0x1000}};
    constexpr std::array missing_start{std::pair<u64, std::size_t>{base + 0x1000, 0x1000}};
    constexpr std::array unaligned{std::pair<u64, std::size_t>{base + 0x80, 0xf80},
                                   std::pair<u64, std::size_t>{base + 0x1000, 0x140}};
    CHECK(StorageBufferMappedPrefix(base, 0x2000, adjacent) == 0x2000U);
    CHECK(StorageBufferMappedPrefix(base, 0x3000, gap) == 0x1000U);
    CHECK(StorageBufferMappedPrefix(base, 0x2000, missing_start) == 0U);
    CHECK(StorageBufferMappedPrefix(base, 0x1800, adjacent) == 0x1000U);
    CHECK(StorageBufferMappedPrefix(base + 0x80, 0x10c0, unaligned) == 0x10c0U);
}

TEST_CASE("Compute storage bindings cover every shader descriptor and mask bit",
          "[video_core][compute_storage]") {
    using State = VideoCommon::BufferCacheChannelInfo;
    constexpr size_t count = std::tuple_size_v<decltype(State::compute_storage_buffers)>;
    STATIC_REQUIRE(count == Shader::Info::MAX_SSBOS);
    STATIC_REQUIRE(count <= std::numeric_limits<decltype(State::enabled_compute_storage_buffers)>::digits);
    STATIC_REQUIRE(count <= std::numeric_limits<decltype(State::written_compute_storage_buffers)>::digits);
    std::array<VideoCommon::Binding, count> bindings{};
    u32 enabled{};
    u32 written{};
    for (u32 index = 0; index < count; ++index) {
        bindings[index].size = index + 1;
        enabled |= 1U << index;
        written |= (index % 2) << index;
    }
    size_t visited{};
    while (enabled != 0) {
        const u32 index = std::countr_zero(enabled);
        REQUIRE(bindings[index].size == index + 1);
        REQUIRE(((written >> index) & 1) == index % 2);
        enabled &= enabled - 1;
        ++visited;
    }
    REQUIRE(visited == count);
}
