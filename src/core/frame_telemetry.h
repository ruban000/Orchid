// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <sstream>
#include <string>
#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

namespace Core {

/// Diagnostic system-frame work intervals, NOT GPU execution or actual display presentation.
/// steady_clock is CLOCK_MONOTONIC on Linux, allowing Python monotonic_ns window selection.
/// Disabled runs allocate nothing and reuse PerfStats' existing timestamps without new clock reads.
class FrameTelemetry {
public:
    using Clock = std::chrono::steady_clock;

    static std::unique_ptr<FrameTelemetry> FromEnvironment(std::uint64_t title_id) {
        const char* directory = std::getenv("EDEN_FRAME_TELEMETRY_DIR");
        if (!directory || !*directory) {
            return {};
        }
        auto result = std::make_unique<FrameTelemetry>(std::filesystem::path{directory}, title_id);
        return result->file.is_open() ? std::move(result) : nullptr;
    }

    FrameTelemetry(const std::filesystem::path& directory, std::uint64_t title_id) {
        std::error_code error;
        std::filesystem::create_directories(directory, error);
        if (error) {
            return;
        }
        static std::atomic<std::uint64_t> sequence{0};
#ifdef _WIN32
        const auto pid = _getpid();
#else
        const auto pid = getpid();
#endif
        std::ostringstream name;
        name << "system-frames-" << std::hex << std::uppercase << std::setfill('0')
             << std::setw(16) << title_id << std::dec << '-' << pid << '-'
             << Nanoseconds(Clock::now()) << '-' << sequence.fetch_add(1) << ".csv";
        file.open(directory / name.str(), std::ios::out | std::ios::trunc);
        if (file) {
            file << "frame,start_ns,end_ns,frame_ms\n" << std::setprecision(17);
        }
    }

    ~FrameTelemetry() {
        Flush();
    }

    void Record(Clock::time_point start, Clock::time_point end) {
        pending[count++] = {Nanoseconds(start), Nanoseconds(end)};
        if (count == pending.size()) {
            Flush();
        }
    }

private:
    static std::int64_t Nanoseconds(Clock::time_point time) {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(time.time_since_epoch()).count();
    }

    void Flush() {
        // Batch all formatting and stream I/O; no per-frame stream/system call.
        for (std::size_t i = 0; i < count; ++i) {
            const auto [start, end] = pending[i];
            file << frame++ << ',' << start << ',' << end << ','
                 << static_cast<double>(end - start) / 1'000'000.0 << '\n';
        }
        if (count != 0) {
            file.flush();
            count = 0;
        }
    }

    std::ofstream file;
    std::array<std::array<std::int64_t, 2>, 256> pending{};
    std::size_t count{};
    std::uint64_t frame{};
};

} // namespace Core
