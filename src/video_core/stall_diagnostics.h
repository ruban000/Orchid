// SPDX-FileCopyrightText: Copyright 2026 Orchid Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <thread>

namespace VideoCommon::StallDiagnostics {

using Clock = std::chrono::steady_clock;

// These are host call boundaries, not claims about the GPU's current shader or command.
enum class Phase {
    NativeCapture,
    NativePrimaryLower,
    NativeFallbackLower,
    PipelinePrepare,
    GraphicsCreate,
    ComputeCreate,
    PipelineBuildWait,
    PipelineLoadJoin,
    SchedulerQueueDrain,
    SchedulerExecutionAcquire,
    SchedulerChunk,
    SubmitLockAcquire,
    QueueSubmit,
    CachedTickWait,
    SemaphoreCounter,
    SemaphoreWait,
    FenceWait,
};

struct Context {
    Phase phase{};
    const char* role{"unknown"}; // Static strings only; no borrowed guest/worker data.
    std::uint64_t key{};         // Pipeline key, or compute shader key (matching its build waiter).
    std::uint64_t address{}; // Only populated where the native program address is already known.
    std::int32_t stage{-1};  // Shader::Stage, -1 when not known at this boundary.
    std::uint64_t tick{};
    std::uint64_t submitted_at_entry{};
    std::uint64_t completed_at_entry{}; // Cached value only, never a diagnostic Vulkan query.
};

// Fixed storage, deterministic sampling, and no logging under this mutex. Also used by CPU tests.
class Recorder {
public:
    static constexpr std::size_t MaxThreads = 32;
    static constexpr std::size_t MaxReports = 256;
    static constexpr auto Window = std::chrono::seconds{180};
    static constexpr auto MinimumAge = std::chrono::seconds{1};
    static constexpr auto RepeatInterval = std::chrono::seconds{5};

    struct Record {
        Context context{};
        std::uint64_t tid{};
        std::uint64_t operation{};
        Clock::time_point since{};
        std::optional<Clock::time_point> last_report;
    };
    struct Batch {
        std::array<Record, MaxThreads> records{};
        std::size_t count{};
        bool overflow{};
        bool finished{};
        bool saturated{};
    };
    struct Token {
        std::size_t slot{};
        Record previous{};
    };

    explicit Recorder(Clock::time_point origin_) : origin{origin_} {}

    std::optional<Token> Enter(Context context, std::uint64_t tid, Clock::time_point now) {
        std::scoped_lock lock{mutex};
        if (finished || now - origin >= Window) {
            return std::nullopt;
        }
        const auto owner = std::this_thread::get_id();
        std::size_t index = MaxThreads;
        for (std::size_t i = 0; i < MaxThreads; ++i) {
            if (slots[i].record.operation != 0 && slots[i].owner == owner) {
                index = i;
                break;
            }
            if (index == MaxThreads && slots[i].record.operation == 0) {
                index = i;
            }
        }
        if (index == MaxThreads) {
            overflow = true;
            return std::nullopt;
        }
        Token token{index, slots[index].record};
        slots[index] = {owner, {context, tid, ++next_operation, now, std::nullopt}};
        return token;
    }

    void Leave(const Token& token) {
        std::scoped_lock lock{mutex};
        slots[token.slot].record = token.previous;
    }

    Batch Sample(Clock::time_point now) {
        std::scoped_lock lock{mutex};
        Batch batch;
        batch.overflow = overflow && !overflow_reported;
        overflow_reported |= overflow;
        if (finished || now - origin >= Window) {
            finished = batch.finished = true;
            return batch;
        }
        for (auto& slot : slots) {
            auto& record = slot.record;
            if (record.operation == 0 || now - record.since < MinimumAge ||
                (record.last_report && now - *record.last_report < RepeatInterval)) {
                continue;
            }
            record.last_report = now;
            batch.records[batch.count++] = record;
            if (++reports == MaxReports) {
                finished = batch.finished = batch.saturated = true;
                break;
            }
        }
        return batch;
    }

private:
    struct Slot {
        std::thread::id owner{};
        Record record{};
    };
    std::mutex mutex;
    std::array<Slot, MaxThreads> slots{};
    Clock::time_point origin;
    std::uint64_t next_operation{};
    std::size_t reports{};
    bool overflow{};
    bool overflow_reported{};
    bool finished{};
};

// Starts at most one bounded window per process, only for EDEN_NX2_STALL_DIAGNOSTICS=1.
void Start();
bool IsActive();

class Scope {
public:
    explicit Scope(Context context);
    Scope(Recorder& recorder_, Context context, std::uint64_t tid, Clock::time_point now)
        : recorder{&recorder_}, token{recorder->Enter(context, tid, now)} {}
    ~Scope() {
        if (token) {
            recorder->Leave(*token);
        }
    }
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;

private:
    Recorder* recorder{};
    std::optional<Recorder::Token> token;
};

} // namespace VideoCommon::StallDiagnostics
