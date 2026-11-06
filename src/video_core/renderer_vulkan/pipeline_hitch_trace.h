// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <atomic>

#include "common/common_types.h"
#include "common/hitch_diagnostic.h"
#include "common/logging.h"

namespace Vulkan::PipelineHitchTrace {

// All timestamps use the same steady-clock nanoseconds as present-service hitch traces.
// Reserve a budget for each phase so boot-cache builds cannot consume all first-use wait
// records. Three phases times 512 records bounds the combined graphics/compute trace at 1536.
enum class Phase { DriverCreate, BuildQueue, FirstUseWait, Count };

inline void Report(Phase phase, const char* kind, u64 hash, s64 start_ns, s64 end_ns) {
    constexpr s64 ThresholdNs = 20'000'000;
    constexpr u32 MaxRecords = 512;
    if (start_ns == 0 || end_ns <= start_ns || end_ns - start_ns <= ThresholdNs) {
        return;
    }
    static std::array<std::atomic<u32>, static_cast<size_t>(Phase::Count)> counters{};
    auto& reported = counters[static_cast<size_t>(phase)];
    u32 count = reported.load(std::memory_order_relaxed);
    do {
        if (count >= MaxRecords) {
            // Emit one marker on the first suppressed event so an empty later interval is not
            // mistaken for proof that no more slow calls occurred.
            if (count == MaxRecords && reported.compare_exchange_strong(
                                           count, MaxRecords + 1, std::memory_order_relaxed)) {
                static constexpr std::array names{"driver-create", "build-queue", "first-use-wait"};
                LOG_INFO(Render_Vulkan,
                         "Hitch trace: pipeline record cap reached phase={} cap={} "
                         "additional_slow_records_suppressed=true",
                         names[static_cast<size_t>(phase)], MaxRecords);
            }
            return;
        }
    } while (!reported.compare_exchange_weak(count, count + 1, std::memory_order_relaxed));
    LOG_INFO(Render_Vulkan,
             "Hitch trace: pipeline {} hash={:016x} start_ns={} end_ns={} duration_us={} "
             "record={}/{}",
             kind, hash, start_ns, end_ns, (end_ns - start_ns) / 1000, count + 1, MaxRecords);
}

// No clock read on the disabled path. Finish immediately after a driver call to exclude later
// statistics/logging; the destructor also records a slow call that exits via an exception.
class Scope {
public:
    Scope(Phase phase_, const char* kind_, u64 hash_, bool enabled)
        : phase{phase_}, kind{kind_}, hash{hash_},
          start_ns{enabled ? Common::HitchDiagnostic::NowNs() : 0} {}

    ~Scope() {
        Finish();
    }

    void Finish() {
        if (start_ns != 0) {
            const s64 begin = start_ns;
            start_ns = 0;
            Report(phase, kind, hash, begin, Common::HitchDiagnostic::NowNs());
        }
    }

    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;

private:
    Phase phase;
    const char* kind;
    u64 hash;
    s64 start_ns;
};

} // namespace Vulkan::PipelineHitchTrace
