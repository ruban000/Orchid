// SPDX-FileCopyrightText: Copyright 2026 Orchid Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <barrier>
#include <cstdlib>
#include <stdexcept>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "video_core/stall_diagnostics.h"

namespace {
using namespace VideoCommon::StallDiagnostics;
using namespace std::chrono_literals;
constexpr Clock::time_point Origin{};
} // namespace

TEST_CASE("Stall diagnostic session is disabled without an explicit opt in",
          "[video_core][stall_diagnostics]") {
    if (std::getenv("EDEN_NX2_STALL_DIAGNOSTICS")) {
        SKIP("Run with EDEN_NX2_STALL_DIAGNOSTICS unset to test the default");
    }
    Start();
    CHECK_FALSE(IsActive());
    Scope scope{{Phase::QueueSubmit, "test"}};
    CHECK_FALSE(IsActive());
}

TEST_CASE("Stall diagnostics restore nested phases on normal and exceptional exit",
          "[video_core][stall_diagnostics]") {
    Recorder recorder{Origin};
    {
        Scope outer{recorder, {Phase::SchedulerChunk, "scheduler-worker", 11}, 123, Origin};
        CHECK(recorder.Sample(Origin + 999ms).count == 0);
        auto first = recorder.Sample(Origin + 1s);
        REQUIRE(first.count == 1);
        CHECK(first.records[0].context.key == 11);
        CHECK(first.records[0].tid == 123);
        {
            Scope inner{recorder, {Phase::ComputeCreate, "compute-build", 42}, 123, Origin + 1s};
            auto sample = recorder.Sample(Origin + 2s);
            REQUIRE(sample.count == 1);
            CHECK(sample.records[0].context.phase == Phase::ComputeCreate);
        }
        try {
            Scope inner{
                recorder, {Phase::PipelineBuildWait, "scheduler-worker", 77}, 123, Origin + 2s};
            throw std::runtime_error{"test unwind"};
        } catch (const std::runtime_error&) {
        }
        // The parent's operation ID, start and report throttle survive both nested scopes.
        CHECK(recorder.Sample(Origin + 3s).count == 0);
        auto restored = recorder.Sample(Origin + 6s);
        REQUIRE(restored.count == 1);
        CHECK(restored.records[0].operation == first.records[0].operation);
        CHECK(restored.records[0].since == Origin);
        CHECK(restored.records[0].context.phase == Phase::SchedulerChunk);
    }
    CHECK(recorder.Sample(Origin + 7s).count == 0);
}

TEST_CASE("Stall diagnostics bound concurrent slots, reports and overflow notices",
          "[video_core][stall_diagnostics]") {
    Recorder recorder{Origin};
    std::barrier entered{static_cast<std::ptrdiff_t>(Recorder::MaxThreads + 1)};
    std::barrier release{static_cast<std::ptrdiff_t>(Recorder::MaxThreads + 1)};
    std::vector<std::jthread> workers;
    for (std::size_t i = 0; i < Recorder::MaxThreads; ++i) {
        workers.emplace_back([&, i] {
            Scope scope{
                recorder, {Phase::GraphicsCreate, "graphics-build", i + 1}, i + 100, Origin};
            entered.arrive_and_wait();
            release.arrive_and_wait();
        });
    }
    entered.arrive_and_wait();
    {
        Scope overflow{recorder, {Phase::NativeCapture, "test-overflow"}, 999, Origin};
        std::size_t reports = 0;
        for (std::size_t i = 0; i < Recorder::MaxReports / Recorder::MaxThreads; ++i) {
            const auto sample = recorder.Sample(Origin + 1s + i * 5s);
            CHECK(sample.overflow == (i == 0));
            CHECK(sample.count == Recorder::MaxThreads);
            for (std::size_t j = 0; j < sample.count; ++j) {
                CHECK(sample.records[j].tid == sample.records[j].context.key + 99);
            }
            reports += sample.count;
            CHECK(sample.saturated == (reports == Recorder::MaxReports));
        }
        const auto after = recorder.Sample(Origin + 60s);
        CHECK(after.count == 0);
        CHECK(after.finished);
        CHECK_FALSE(after.overflow);
    }
    // Always release before a fatal assertion or joining; no sleeps or timing races.
    release.arrive_and_wait();
    workers.clear();
}

TEST_CASE("Stall diagnostic window expires and slots can be reused",
          "[video_core][stall_diagnostics]") {
    Recorder recorder{Origin};
    for (std::size_t i = 0; i < 100; ++i) {
        Scope scope{recorder, {Phase::SemaphoreWait, "wait-caller", 0, 0, -1, i}, 123, Origin};
    }
    CHECK(recorder.Sample(Origin + 2s).count == 0);
    Scope scope{recorder, {Phase::SemaphoreWait, "wait-caller"}, 123, Origin};
    const auto expired = recorder.Sample(Origin + Recorder::Window);
    CHECK(expired.finished);
    CHECK_FALSE(expired.saturated);
    CHECK_FALSE(expired.overflow);
    CHECK(expired.count == 0);
    CHECK_FALSE(recorder.Enter({}, 123, Origin + Recorder::Window));
}
