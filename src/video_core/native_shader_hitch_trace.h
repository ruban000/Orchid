// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <exception>

#include "common/common_types.h"
#include "common/hitch_diagnostic.h"
#include "common/logging.h"
#include "shader_recompiler/stage.h"

namespace VideoCommon::NativeShaderHitchTrace {

inline const char* StageName(u32 stage) {
    static constexpr std::array names{"vertex-b", "tess-control", "tess-eval",
                                      "geometry", "fragment",     "compute",
                                      "vertex-a", "task",         "mesh"};
    return stage < names.size() ? names[stage] : "unknown";
}

enum class RecordKind { Lowering, SlowOuter, Count };

// A full lowering gets one composite record, including fast phases, so many small captures can
// be correlated with a long frame. Keep a separate budget for slow enclosing operations.
inline u32 Claim(RecordKind kind) {
    static std::array<std::atomic<u32>, static_cast<size_t>(RecordKind::Count)> counts{};
    constexpr std::array<u32, 2> limits{512, 128};
    const size_t index = static_cast<size_t>(kind);
    const u32 limit = limits[index];
    auto& counter = counts[index];
    u32 count = counter.load(std::memory_order_relaxed);
    do {
        if (count >= limit) {
            if (count == limit &&
                counter.compare_exchange_strong(count, limit + 1, std::memory_order_relaxed)) {
                LOG_INFO(Render_Vulkan,
                         "Hitch trace: native-shader record cap reached kind={} cap={} "
                         "additional_records_suppressed=true",
                         kind == RecordKind::Lowering ? "lowering" : "slow-outer", limit);
            }
            return 0;
        }
    } while (!counter.compare_exchange_weak(count, count + 1, std::memory_order_relaxed));
    return count + 1;
}

enum class Phase { Capture, Primary, Upstream, Count };

class Lowering {
public:
    template <class Instructions>
    Lowering(u64 guest_, Shader::Stage stage, const Instructions& input_, u32 header_bytes_)
        : guest{guest_}, stage_id{static_cast<u32>(stage)}, input{&input_},
          input_count{
              [](const void* value) { return static_cast<const Instructions*>(value)->size(); }},
          header_bytes{header_bytes_},
          start_ns{Common::HitchDiagnostic::Enabled() ? Common::HitchDiagnostic::NowNs() : 0},
          exceptions{start_ns != 0 ? std::uncaught_exceptions() : 0} {}

    ~Lowering() {
        if (start_ns != 0) {
            Finish(std::uncaught_exceptions() > exceptions ? "exception" : "returned");
        }
    }

    void Begin(Phase phase) {
        if (start_ns != 0) {
            active = static_cast<size_t>(phase);
            phases[active].start_ns = Common::HitchDiagnostic::NowNs();
        }
    }

    void End(bool accepted, size_t output_instructions, size_t error_offset = 0) {
        if (start_ns != 0 && active < phases.size()) {
            auto& phase = phases[active];
            phase.end_ns = Common::HitchDiagnostic::NowNs();
            phase.status = accepted ? "ok" : "rejected";
            phase.output_instructions = static_cast<s64>(output_instructions);
            phase.error_offset = accepted ? -1 : static_cast<s64>(error_offset);
            active = phases.size();
        }
    }

    void Finish(const char* status, s64 source_bytes = -1, s64 output_instructions = -1) {
        if (start_ns == 0) {
            return;
        }
        const s64 end_ns = Common::HitchDiagnostic::NowNs();
        const s64 begin_ns = start_ns;
        start_ns = 0;
        if (active < phases.size()) {
            phases[active].end_ns = end_ns;
            phases[active].status = status;
        }
        const u32 record = Claim(RecordKind::Lowering);
        if (record == 0) {
            return;
        }
        const auto& capture = phases[static_cast<size_t>(Phase::Capture)];
        const auto& primary = phases[static_cast<size_t>(Phase::Primary)];
        const auto& upstream = phases[static_cast<size_t>(Phase::Upstream)];
        LOG_INFO(
            Render_Vulkan,
            "Hitch trace: native-shader lowering guest={:#x} stage={} input_insns={} "
            "output_insns={} header_bytes={} source_bytes={} start_ns={} end_ns={} "
            "duration_us={} capture_start_ns={} capture_end_ns={} capture_status={} "
            "primary_start_ns={} primary_end_ns={} primary_status={} primary_output_insns={} "
            "primary_error_offset={} upstream_start_ns={} upstream_end_ns={} upstream_status={} "
            "upstream_output_insns={} upstream_error_offset={} status={} record={}/512",
            guest, StageName(stage_id), input_count(input), output_instructions, header_bytes,
            source_bytes, begin_ns, end_ns, (end_ns - begin_ns) / 1000, capture.start_ns,
            capture.end_ns, capture.status, primary.start_ns, primary.end_ns, primary.status,
            primary.output_instructions, primary.error_offset, upstream.start_ns, upstream.end_ns,
            upstream.status, upstream.output_instructions, upstream.error_offset, status, record);
    }

    Lowering(const Lowering&) = delete;
    Lowering& operator=(const Lowering&) = delete;

private:
    struct Interval {
        s64 start_ns{}, end_ns{};
        const char* status{"not-run"};
        s64 output_instructions{-1}, error_offset{-1};
    };
    u64 guest;
    u32 stage_id;
    const void* input;
    size_t (*input_count)(const void*);
    u32 header_bytes;
    s64 start_ns;
    int exceptions;
    std::array<Interval, static_cast<size_t>(Phase::Count)> phases{};
    size_t active{phases.size()};
};

class SlowScope {
public:
    SlowScope(const char* kind_, u64 guest_, u32 stage_, bool enabled)
        : kind{kind_}, guest{guest_}, stage_id{stage_},
          start_ns{enabled ? Common::HitchDiagnostic::NowNs() : 0},
          exceptions{start_ns != 0 ? std::uncaught_exceptions() : 0} {}

    ~SlowScope() {
        if (start_ns != 0) {
            Finish(std::uncaught_exceptions() > exceptions ? "exception" : "returned");
        }
    }

    void Stage(Shader::Stage stage) {
        stage_id = static_cast<u32>(stage);
    }
    void SourceBytes(size_t size) {
        source_bytes = static_cast<s64>(size);
    }

    void Finish(const char* status) {
        if (start_ns == 0) {
            return;
        }
        const s64 end_ns = Common::HitchDiagnostic::NowNs();
        const s64 begin_ns = start_ns;
        start_ns = 0;
        if (end_ns - begin_ns <= 20'000'000) {
            return;
        }
        const u32 record = Claim(RecordKind::SlowOuter);
        if (record != 0) {
            LOG_INFO(Render_Vulkan,
                     "Hitch trace: native-shader {} guest={:#x} stage={} source_bytes={} "
                     "start_ns={} end_ns={} duration_us={} status={} record={}/128",
                     kind, guest, StageName(stage_id), source_bytes, begin_ns, end_ns,
                     (end_ns - begin_ns) / 1000, status, record);
        }
    }

    SlowScope(const SlowScope&) = delete;
    SlowScope& operator=(const SlowScope&) = delete;

private:
    const char* kind;
    u64 guest;
    u32 stage_id;
    s64 source_bytes{-1};
    s64 start_ns;
    int exceptions;
};

} // namespace VideoCommon::NativeShaderHitchTrace
