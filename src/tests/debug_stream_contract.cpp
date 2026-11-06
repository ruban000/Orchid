// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// Standalone contract test, no emulator/GPU needed:
// c++ -std=c++20 -Isrc src/tests/debug_stream_contract.cpp -o <test>
// <test> <new-empty-test-directory> [disabled]
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>
#include "core/frame_telemetry.h"
#include "video_core/texture_handle_trace.h"

namespace {
void Require(bool value) {
    if (!value) throw std::runtime_error("debug stream contract failed");
}

void Environment(const char* key, const std::string& value) {
#ifdef _WIN32
    _putenv_s(key, value.c_str());
#else
    setenv(key, value.c_str(), 1);
#endif
}

std::vector<std::filesystem::path> Files(const std::filesystem::path& directory) {
    std::vector<std::filesystem::path> result;
    if (std::filesystem::exists(directory)) {
        for (const auto& entry : std::filesystem::directory_iterator{directory}) {
            if (entry.is_regular_file()) result.push_back(entry.path());
        }
    }
    return result;
}

std::string Contents(const std::filesystem::path& path) {
    std::ifstream file{path};
    return {std::istreambuf_iterator<char>{file}, std::istreambuf_iterator<char>{}};
}

void Frames(const std::filesystem::path& directory, std::size_t count) {
    using Clock = Core::FrameTelemetry::Clock;
    {
        Core::FrameTelemetry stream{directory, 0x1234};
        for (std::size_t i = 0; i < count; ++i) {
            const auto start = Clock::time_point{std::chrono::nanoseconds{
                static_cast<std::int64_t>(1'000'000 + i * 3'000'000)}};
            stream.Record(start, start + std::chrono::nanoseconds{1'000'001});
        }
        if (count == 257) {
            const auto files = Files(directory);
            Require(files.size() == 1);
            const auto text = Contents(files[0]);
            Require(std::count(text.begin(), text.end(), '\n') == 257); // header + batch256
        }
    }
    const auto files = Files(directory);
    Require(files.size() == 1);
    const auto text = Contents(files[0]);
    Require(text.starts_with("frame,start_ns,end_ns,frame_ms\n"));
    Require(std::count(text.begin(), text.end(), '\n') == count + 1);
    if (count) {
        const std::string prefix = "0,1000000,2000001,";
        const auto row = text.find(prefix);
        Require(row != std::string::npos);
        Require(std::abs(std::stod(text.substr(row + prefix.size())) - 1.000001) < 1e-12);
    }
}
} // namespace

int main(int argc, char** argv) {
    Require(argc >= 2);
    const std::filesystem::path root{argv[1]};
    Require(!std::filesystem::exists(root));
    if (argc > 2 && std::string{argv[2]} == "disabled") {
        Environment("EDEN_FRAME_TELEMETRY_DIR", "");
        Environment("EDEN_HANDLE_TRACE_DIR", "");
        Require(!Core::FrameTelemetry::FromEnvironment(1));
        Require(!VideoCommon::TextureHandleTrace::Enabled());
        Require(!std::filesystem::exists(root));
        return 0;
    }
    std::filesystem::create_directories(root);
    Frames(root / "zero", 0);
    Frames(root / "one", 1);
    Frames(root / "short", 4);
    Frames(root / "batch", 257);
    Environment("EDEN_FRAME_TELEMETRY_DIR", (root / "environment").string());
    Require(Core::FrameTelemetry::FromEnvironment(1) != nullptr);
    const auto traces = root / "traces";
    Environment("EDEN_HANDLE_TRACE_DIR", traces.string());
    Require(VideoCommon::TextureHandleTrace::Enabled());
    for (u32 i = 0; i < 8; ++i) {
        VideoCommon::TextureHandleTrace success;
        success.Add({.kind = "cbuf_word", .value = 42, .width = 4, .performed = true,
                     .mapped_at_trace = true});
        const std::array<u32, 1> code{i};
        success.Finish(code, i, 42, std::source_location::current());
    }
    Require(Files(traces).empty());
    std::vector<u32> code(700, 123);
    for (u32 program = 0; program < 7; ++program) {
        // The repeat at program1 must not consume another distinct fault slot.
        code[0] = program == 1 ? 0 : program;
        VideoCommon::TextureHandleTrace fault;
        for (std::size_t i = 0; i < 129; ++i) {
            fault.Add({.kind = "storage_word", .storage_binding = 3, .byte_offset = 16,
                       .raw_pointer = 0x123, .address = 0x133, .value = 42, .width = 4,
                       .performed = true, .mapped_at_trace = false});
        }
        fault.Finish(code, program, 42, std::source_location::current());
        if (program == 0) {
            const auto files = Files(traces);
            Require(files.size() == 1);
            const auto text = Contents(files[0]);
            Require(text.find("\"code_truncated\":true") != std::string::npos);
            Require(text.find("\"reads_truncated\":true") != std::string::npos);
            Require(text.find("\"order\":127,") != std::string::npos);
            Require(text.find("\"order\":128,") == std::string::npos);
            Require(text.find("\"final_texture_handle\":42") != std::string::npos);
        }
    }
    Require(Files(traces).size() == 4);
}
