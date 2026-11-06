// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <atomic>

#include "common/hitch_diagnostic.h"
#include "common/logging.h"

namespace Vulkan::HitchTrace {

enum class Event : size_t {
    WorkerChunk,
    WorkerWait,
    Submit,
    GpuWait,
    Composite,
    PresentFrameFree,
    PresentFrameFence,
    PresentCopy,
    AcquireImage,
    SchedulerTickWait,
    SchedulerPacing,
    PresentSubmitMutex,
    PresentSubmit,
    QueuePresentMutex,
    QueuePresent,
    Count,
};

inline constexpr std::array names{
    "worker_chunk",
    "worker_wait",
    "submit",
    "gpu_wait",
    "renderer_composite",
    "present_frame_free",
    "present_frame_fence",
    "present_copy",
    "acquire_image",
    "scheduler_tick_wait",
    "scheduler_pacing",
    "present_submit_mutex",
    "present_submit",
    "queue_present_mutex",
    "queue_present",
};
static_assert(names.size() == static_cast<size_t>(Event::Count));

// Each category has a separate cap so a burst of queued chunks cannot consume all GPU-wait logs.
inline bool Claim(Event event, u64 duration_ns) {
    constexpr u64 threshold_ns = 20'000'000;
    if (duration_ns < threshold_ns) {
        return false;
    }
    static std::array<std::atomic<u32>, static_cast<size_t>(Event::Count)> events{};
    const u32 limit = event < Event::Composite ? 128 : 64;
    const u32 count = events[static_cast<size_t>(event)].fetch_add(1, std::memory_order_relaxed);
    if (count == limit) {
        LOG_INFO(Render_Vulkan, "VulkanHitch truncated category={} max_events={} at_ns={}",
                 names[static_cast<size_t>(event)], limit, Common::HitchDiagnostic::NowNs());
    }
    return count < limit;
}

// Wall intervals on the executing thread, not CPU-time measurements: driver waits and callbacks
// inside a broad scope are deliberately included. Matching monotonic timestamps allow overlap
// with the service-thread frame trace and the more specific scheduler/pipeline phases.
class Scope {
public:
    explicit Scope(Event event_, u64 item_ = 0)
        : event{event_}, item{item_}, enabled{Common::HitchDiagnostic::Enabled()},
          begin_ns{enabled ? Common::HitchDiagnostic::NowNs() : 0} {}
    ~Scope() {
        if (!enabled) {
            return;
        }
        const s64 end_ns = Common::HitchDiagnostic::NowNs();
        if (Claim(event, static_cast<u64>(end_ns - begin_ns))) {
            LOG_INFO(Render_Vulkan, "VulkanHitch {} begin_ns={} end_ns={} duration_ns={} item={}",
                     names[static_cast<size_t>(event)], begin_ns, end_ns, end_ns - begin_ns, item);
        }
    }
    void SetItem(u64 value) {
        item = value;
    }
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;

private:
    Event event;
    u64 item;
    bool enabled;
    s64 begin_ns;
};

} // namespace Vulkan::HitchTrace
