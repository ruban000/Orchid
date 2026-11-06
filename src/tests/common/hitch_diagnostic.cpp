// SPDX-License-Identifier: GPL-3.0-or-later
#include <catch2/catch_test_macros.hpp>
#include "common/hitch_diagnostic.h"

TEST_CASE("Frame gap attribution separates the prior service from current fence work", "[hitch_trace]") {
    using namespace Common::HitchDiagnostic;
    const FrameTimes previous{10'000'000, 40'000'000, 41'000'000, 42'000'000};
    const FrameTimes current{45'000'000, 49'000'000, 50'000'000, 62'000'000};
    const auto metrics = MeasureFrame(current, &previous);
    CHECK(metrics.gap_us == 35'000);
    CHECK(metrics.external_us == 3'000);
    CHECK(metrics.previous_service_us == 32'000);
    CHECK(metrics.fence_us == 4'000);
    CHECK(metrics.composite_us == 1'000);
    CHECK(metrics.pace_us == 12'000);
    CHECK(metrics.service_us == 17'000);
    CHECK(metrics.gap_us == metrics.external_us + metrics.previous_service_us);
    const auto first = MeasureFrame(current, nullptr);
    CHECK(first.gap_us == 0);
    CHECK(first.external_us == 0);
    CHECK(first.previous_service_us == 0);
    CHECK(first.service_us == 17'000);
}
