// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later
#include <atomic>
#include <stdexcept>
#include <thread>
#include <vector>
#include <catch2/catch_test_macros.hpp>
#include "video_core/compute_binding_diagnostic.h"

namespace diag = VideoCommon::ComputeBindingDiagnostic;

TEST_CASE("Compute binding trace selector is explicit and exact",
          "[video_core][compute_binding_trace]") {
    REQUIRE(diag::ParseSelector("9694a462ae7e186c") == 0x9694a462ae7e186cULL);
    REQUIRE(diag::ParseSelector("0000000000000000") == 0);
    REQUIRE_FALSE(diag::ParseSelector("9694a462ae7e186"));
    REQUIRE_FALSE(diag::ParseSelector("0x9694a462ae7e186c"));
    REQUIRE_FALSE(diag::ParseSelector("9694a462ae7e186z"));
}

TEST_CASE("Compute binding trace claims only eight concurrent records",
          "[video_core][compute_binding_trace]") {
    diag::Limit limit;
    std::atomic<u32> successes{};
    std::vector<std::thread> threads;
    for (u32 i = 0; i < 32; ++i) {
        threads.emplace_back([&] {
            for (u32 j = 0; j < 128; ++j)
                successes += limit.Claim();
        });
    }
    for (auto& thread : threads)
        thread.join();
    REQUIRE(successes == 8);
    REQUIRE_FALSE(limit.Claim());
    diag::Limit separate;
    REQUIRE(separate.Claim());
}

TEST_CASE("Compute binding trace restores nested thread contexts and distinguishes zero",
          "[video_core][compute_binding_trace]") {
    REQUIRE(diag::current == nullptr);
    diag::Record compile{};
    compile.phase = diag::Phase::Compile;
    REQUIRE_FALSE(compile.compile_low.has_value());
    {
        diag::Scope scope{compile};
        diag::CompileWord(0, 0x50, 0);
        REQUIRE(compile.compile_low.has_value());
        REQUIRE(*compile.compile_low == 0);
        REQUIRE_FALSE(compile.compile_high.has_value());
        diag::CompileWord(1, 0x54, 7);
        REQUIRE_FALSE(compile.compile_high.has_value());
        REQUIRE_FALSE(compile.pointer_mapped.has_value());
        diag::CompileMapping(0, 0x50, false);
        REQUIRE(compile.pointer_mapped.has_value());
        REQUIRE_FALSE(*compile.pointer_mapped);
        try {
            diag::Record dispatch{};
            dispatch.phase = diag::Phase::Dispatch;
            diag::Scope nested{dispatch};
            diag::CompileWord(0, 0x54, 9);
            REQUIRE_FALSE(compile.compile_high.has_value());
            REQUIRE_FALSE(diag::IsTarget(0, 0, 0x50));
            dispatch.descriptor_seen = true;
            dispatch.descriptor_index = 2;
            REQUIRE(diag::IsTarget(2, 0, 0x50));
            dispatch.duplicate_descriptor = true;
            REQUIRE_FALSE(diag::IsTarget(2, 0, 0x50));
            std::atomic<bool> isolated{};
            std::thread worker([&] { isolated = diag::current == nullptr; });
            worker.join();
            REQUIRE(isolated);
            throw std::runtime_error("restore");
        } catch (const std::runtime_error&) {
        }
        REQUIRE(diag::current == &compile);
    }
    REQUIRE(diag::current == nullptr);
}
