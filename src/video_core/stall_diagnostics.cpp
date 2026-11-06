// SPDX-FileCopyrightText: Copyright 2026 Orchid Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <ctime>
#include <string_view>

#ifdef _WIN32
#include <windows.h>
#elif defined(__unix__) || defined(__APPLE__)
#include <unistd.h>
#ifdef __linux__
#include <sys/syscall.h>
#elif defined(__APPLE__)
#include <pthread.h>
#endif
#endif

#include "common/logging.h"
#include "common/polyfill_thread.h"
#include "video_core/stall_diagnostics.h"

namespace VideoCommon::StallDiagnostics {
namespace {
std::atomic<Recorder*> active{};

std::uint64_t NativeTid() {
#ifdef _WIN32
    return GetCurrentThreadId();
#elif defined(__linux__)
    return static_cast<std::uint64_t>(syscall(SYS_gettid));
#elif defined(__APPLE__)
    std::uint64_t tid{};
    pthread_threadid_np(nullptr, &tid);
    return tid;
#else
    return 0; // Unknown, not a hash that could be mistaken for a native TID.
#endif
}

std::uint64_t HostPid() {
#ifdef _WIN32
    return GetCurrentProcessId();
#elif defined(__unix__) || defined(__APPLE__)
    return static_cast<std::uint64_t>(getpid());
#else
    return 0;
#endif
}

const char* Name(Phase phase) {
    switch (phase) {
#define STALL_PHASE(name)                                                                          \
    case Phase::name:                                                                              \
        return #name
        STALL_PHASE(NativeCapture);
        STALL_PHASE(NativePrimaryLower);
        STALL_PHASE(NativeFallbackLower);
        STALL_PHASE(PipelinePrepare);
        STALL_PHASE(GraphicsCreate);
        STALL_PHASE(ComputeCreate);
        STALL_PHASE(PipelineBuildWait);
        STALL_PHASE(PipelineLoadJoin);
        STALL_PHASE(SchedulerQueueDrain);
        STALL_PHASE(SchedulerExecutionAcquire);
        STALL_PHASE(SchedulerChunk);
        STALL_PHASE(SubmitLockAcquire);
        STALL_PHASE(QueueSubmit);
        STALL_PHASE(CachedTickWait);
        STALL_PHASE(SemaphoreCounter);
        STALL_PHASE(SemaphoreWait);
        STALL_PHASE(FenceWait);
#undef STALL_PHASE
    }
    return "unknown";
}

class Session {
public:
    Session() : origin{Clock::now()}, recorder{new Recorder{origin}} {
        // Deliberately process-lifetime storage (bounded to one 32-slot recorder).
        // A blocked driver thread can unwind its Scope after the static Session is destroyed.
        // Clearing active stops new entries; it cannot revoke already-issued scope tokens.
        const auto wall = std::chrono::system_clock::now();
        const auto utc = std::chrono::system_clock::to_time_t(wall);
        std::tm calendar{};
#ifdef _WIN32
        gmtime_s(&calendar, &utc);
#else
        gmtime_r(&utc, &calendar);
#endif
        char date[32]{};
        std::strftime(date, sizeof(date), "%Y-%m-%dT%H:%M:%S", &calendar);
        const auto unix_us =
            std::chrono::duration_cast<std::chrono::microseconds>(wall.time_since_epoch()).count();
        LOG_INFO(Render_Vulkan,
                 "Host stall diagnostics origin: pid={}, tid={}, utc={}.{:06}Z, unix_us={}, "
                 "steady_us={}, window_s=180, slots=32, max_reports=256; diagnostic origin, "
                 "not process creation; no GPU polling; key/address 0 = not recorded, stage -1 = "
                 "unknown",
                 HostPid(), NativeTid(), date, unix_us % 1'000'000, unix_us,
                 std::chrono::duration_cast<std::chrono::microseconds>(origin.time_since_epoch())
                     .count());
        Common::Log::Flush();
        active.store(recorder, std::memory_order_release);
        worker = std::jthread([this](std::stop_token stop) {
            std::mutex wait_mutex;
            std::unique_lock lock{wait_mutex};
            while (!stop.stop_requested()) {
                wake.wait_for(lock, stop, std::chrono::seconds{1}, [] { return false; });
                if (stop.stop_requested()) {
                    break;
                }
                const auto now = Clock::now();
                const auto batch = recorder->Sample(now);
                if (batch.overflow) {
                    LOG_WARNING(
                        Render_Vulkan,
                        "Host stall diagnostics: thread slots exhausted; some phases unknown");
                }
                for (std::size_t i = 0; i < batch.count; ++i) {
                    const auto& record = batch.records[i];
                    const auto& c = record.context;
                    LOG_INFO(
                        Render_Vulkan,
                        "Host stall snapshot: pid={}, tid={}, op={}, role={}, phase={}, "
                        "key_or_0={:016x}, address_or_0={:#x}, stage_or_minus1={}, tick={}, "
                        "submitted_at_entry={}, cached_completed_at_entry={}, age_ms={}, "
                        "origin_elapsed_ms={}; active host call, not GPU attribution",
                        HostPid(), record.tid, record.operation, c.role, Name(c.phase), c.key,
                        c.address, c.stage, c.tick, c.submitted_at_entry, c.completed_at_entry,
                        std::chrono::duration_cast<std::chrono::milliseconds>(now - record.since)
                            .count(),
                        std::chrono::duration_cast<std::chrono::milliseconds>(now - origin)
                            .count());
                }
                if (batch.finished) {
                    LOG_INFO(Render_Vulkan,
                             "Host stall diagnostics ended: {}; later phases unknown",
                             batch.saturated ? "report limit" : "window expired");
                    Common::Log::Flush();
                    break;
                }
                // FileBackend buffers writes. Preserve a bounded snapshot even if the
                // instrumented thread never returns or the reporter terminates the process.
                if (batch.count != 0 || batch.overflow) {
                    Common::Log::Flush();
                }
            }
            active.store(nullptr, std::memory_order_release);
        });
    }

    ~Session() {
        active.store(nullptr, std::memory_order_release);
        worker.request_stop();
        wake.notify_all();
        // Join before wake destruction. Recorder survives even scopes on blocked driver threads.
        worker.join();
    }

private:
    Clock::time_point origin;
    Recorder* const recorder;
    std::condition_variable_any wake;
    std::jthread worker;
};
} // namespace

void Start() {
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_NX2_STALL_DIAGNOSTICS");
        return value && std::string_view{value} == "1";
    }();
    if (enabled) {
        static Session session;
    }
}

bool IsActive() {
    return active.load(std::memory_order_acquire) != nullptr;
}

Scope::Scope(Context context) {
    if (auto* current = active.load(std::memory_order_acquire)) {
        recorder = current;
        token = recorder->Enter(context, NativeTid(), Clock::now());
    }
}

} // namespace VideoCommon::StallDiagnostics
