// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <chrono>
#include <cstdlib>
#include "common/common_types.h"

namespace Common::HitchDiagnostic {

inline bool Enabled() noexcept {
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_NX2_HITCH_TRACE");
        return value && value[0] == '1' && value[1] == '\0';
    }();
    return enabled;
}

inline s64 NowNs() noexcept {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct FrameTimes {
    s64 entered{}, waited{}, composited{}, paced{};
};
struct FrameMetrics {
    s64 gap_us{}, external_us{}, previous_service_us{};
    s64 fence_us{}, composite_us{}, pace_us{}, service_us{};
};
constexpr s64 ElapsedUs(s64 begin, s64 end) noexcept {
    return end >= begin ? (end - begin) / 1000 : 0;
}

// The interval entering frame N contains frame N-1's service and the game's intervening work.
// Current-frame fence time belongs to frame N, not to that already elapsed interval.
constexpr FrameMetrics MeasureFrame(FrameTimes current, const FrameTimes* previous) noexcept {
    return {
        .gap_us = previous ? ElapsedUs(previous->entered, current.entered) : 0,
        .external_us = previous ? ElapsedUs(previous->paced, current.entered) : 0,
        .previous_service_us = previous ? ElapsedUs(previous->entered, previous->paced) : 0,
        .fence_us = ElapsedUs(current.entered, current.waited),
        .composite_us = ElapsedUs(current.waited, current.composited),
        .pace_us = ElapsedUs(current.composited, current.paced),
        .service_us = ElapsedUs(current.entered, current.paced),
    };
}
} // namespace Common::HitchDiagnostic
