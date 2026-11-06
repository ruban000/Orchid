// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <map>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "video_core/shader_snapshot.h"

namespace {
constexpr GPUVAddr Pool = 0x10000;
} // namespace

namespace {
struct ConstantBufferMemory {
    static constexpr size_t PageSize = 0x1000;
    std::map<GPUVAddr, std::array<u8, PageSize>> pages;
    std::vector<GPUVAddr> lookups;
    u64 address_limit = 1ULL << 40;

    void Map(GPUVAddr address, u8 value) {
        pages[address].fill(value);
    }
    bool IsWithinGPUAddressRange(GPUVAddr address) const {
        return address < address_limit;
    }
    const u8* GetPointer(GPUVAddr address) {
        lookups.push_back(address);
        const auto found = pages.find(address & ~(GPUVAddr{PageSize} - 1));
        return found == pages.end() ? nullptr : found->second.data() + (address & (PageSize - 1));
    }
};
} // namespace

TEST_CASE("Constant buffer snapshots copy backed pages after holes and bounded padding",
          "[video_core][texture][snapshot]") {
    ConstantBufferMemory memory;
    memory.Map(Pool, 0x11);
    memory.Map(Pool + 0x2000, 0x33);
    std::vector<u8> bytes(0x10000 + sizeof(u32), 0xff);
    VideoCommon::ReadConstantBufferSnapshot(memory, Pool, bytes);
    for (size_t offset = 0; offset < bytes.size(); ++offset) {
        const u8 expected = offset < 0x1000 ? 0x11 : offset >= 0x2000 && offset < 0x3000 ? 0x33 : 0;
        REQUIRE(bytes[offset] == expected);
    }
    const auto retained = bytes;
    memory.Map(Pool, 0x99);
    memory.Map(Pool + 0x10000, 0x77);
    std::vector<u8> updated(bytes.size());
    VideoCommon::ReadConstantBufferSnapshot(memory, Pool, updated);
    CHECK(bytes == retained);
    CHECK(updated.front() == 0x99);
    CHECK(std::all_of(updated.end() - sizeof(u32), updated.end(),
                      [](u8 byte) { return byte == 0x77; }));
}

TEST_CASE("Constant buffer snapshots bound unaligned copies and reject invalid address spans",
          "[video_core][texture][snapshot]") {
    ConstantBufferMemory memory;
    memory.Map(Pool, 0x11);
    memory.Map(Pool + 0x2000, 0x33);
    std::array<u8, 0x2010> bytes{};
    VideoCommon::ReadConstantBufferSnapshot(memory, Pool + 0xff8, bytes);
    for (size_t offset = 0; offset < bytes.size(); ++offset) {
        const u8 expected = offset < 8 ? 0x11 : offset >= 0x1008 && offset < 0x2008 ? 0x33 : 0;
        REQUIRE(bytes[offset] == expected);
    }
    CHECK(memory.lookups ==
          std::vector<GPUVAddr>{Pool + 0xff8, Pool + 0x1000, Pool + 0x2000, Pool + 0x3000});
    memory.lookups.clear();
    bytes.fill(0xff);
    VideoCommon::ReadConstantBufferSnapshot(memory, std::numeric_limits<GPUVAddr>::max() - 1,
                                            bytes);
    CHECK(std::all_of(bytes.begin(), bytes.end(), [](u8 byte) { return byte == 0; }));
    memory.address_limit = Pool + bytes.size() - 1;
    bytes.fill(0xff);
    VideoCommon::ReadConstantBufferSnapshot(memory, Pool, bytes);
    CHECK(std::all_of(bytes.begin(), bytes.end(), [](u8 byte) { return byte == 0; }));
    CHECK(memory.lookups.empty());
}

TEST_CASE("Graphics snapshot rejection distinguishes the unchanged pool cap from mapping holes",
          "[video_core][texture][snapshot]") {
    struct PoolMemory {
        bool mapped{true};
        u32 checks{};
        size_t last_size{};
        bool IsFullyMappedRange(GPUVAddr, size_t size) {
            ++checks;
            last_size = size;
            return mapped;
        }
    } memory;
    using VideoCommon::CheckTextureSnapshotPool;
    using VideoCommon::ShaderSnapshotFailure;
    CHECK(CheckTextureSnapshotPool(memory, Pool, 32767) == ShaderSnapshotFailure::None);
    CHECK(memory.checks == 1);
    CHECK(memory.last_size == 1024 * 1024);
    CHECK(CheckTextureSnapshotPool(memory, Pool, 32768) ==
          ShaderSnapshotFailure::TexturePoolEntryLimit);
    CHECK(CheckTextureSnapshotPool(memory, Pool, std::numeric_limits<u32>::max()) ==
          ShaderSnapshotFailure::TexturePoolEntryLimit);
    CHECK(memory.checks == 1);
    memory.mapped = false;
    CHECK(CheckTextureSnapshotPool(memory, Pool, 5) == ShaderSnapshotFailure::TexturePoolNotMapped);
    CHECK(memory.last_size == 6 * sizeof(Tegra::Texture::TICEntry));
    CHECK(memory.checks == 2);
}
