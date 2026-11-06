// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <optional>
#include <source_location>
#include <span>
#include <string>
#include "common/common_types.h"
#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

namespace VideoCommon {

/// Opt-in, bounded read ledger. It observes existing reads; it never rereads or flushes memory.
/// Mapping is sampled after each read and is explicitly racy, not proof of the mapping at read time.
class TextureHandleTrace {
public:
    static constexpr std::size_t MaxPrograms = 4;
    static constexpr std::size_t MaxReads = 128;
    static constexpr std::size_t MaxCodeWords = 512;

    struct Read {
        const char* kind;
        u32 cbuf_index{};
        u32 cbuf_offset{};
        u32 storage_binding{};
        u32 byte_offset{};
        u64 cbuf_base{};
        u64 raw_pointer{};
        u64 address{};
        u64 value{};
        u32 width{};
        bool performed{};
        bool mapped_at_trace{};
        bool address_wrapped{};
    };

    static bool Enabled() {
        return !State().directory.empty();
    }

    void Add(Read read) {
        fault |= read.performed && !read.mapped_at_trace;
        if (count < reads.size()) {
            reads[count++] = read;
        } else {
            reads_truncated = true;
        }
    }

    void Finish(std::span<const u32> code, u32 program, std::optional<u32> result,
                std::source_location caller) const {
        if (!fault) {
            return; // Successful programs never consume the fault quota or write files.
        }
        u64 fingerprint = 14695981039346656037ULL;
        for (const u32 word : code) {
            for (unsigned int shift = 0; shift < 32; shift += 8) {
                fingerprint = (fingerprint ^ ((word >> shift) & 255U)) * 1099511628211ULL;
            }
        }
        auto& state = State();
        std::scoped_lock lock{state.mutex};
        for (std::size_t i = 0; i < state.count; ++i) {
            if (state.programs[i] == fingerprint) {
                return;
            }
        }
        if (state.count == state.programs.size()) {
            return;
        }
        state.programs[state.count++] = fingerprint;
        std::error_code error;
        std::filesystem::create_directories(state.directory, error);
        if (error) {
            return;
        }
#ifdef _WIN32
        const auto pid = _getpid();
#else
        const auto pid = getpid();
#endif
        const auto now = std::chrono::duration_cast<std::chrono::nanoseconds>(
                             std::chrono::steady_clock::now().time_since_epoch()).count();
        const auto name = "texture-handle-" + std::to_string(pid) + '-' +
                          std::to_string(now) + '-' + std::to_string(state.count) + ".json";
        std::ofstream file{state.directory / name};
        if (!file) {
            return;
        }
        file << "{\"schema\":\"eden.texture_handle_trace.v1\",\"pid\":" << pid
             << ",\"monotonic_ns\":" << now << ",\"program_index\":" << program
             << ",\"program_fingerprint_fnv1a64\":\"" << std::hex << fingerprint << std::dec
             << "\",\"fault_basis\":\"unmapped_range_at_trace_racy\","
                "\"dependency_analysis\":\"ordered_reads_only_select_is_eager\",\"caller\":{\"file\":";
        String(file, caller.file_name());
        file << ",\"line\":" << caller.line() << ",\"function\":";
        String(file, caller.function_name());
        file << "},\"evaluation_valid\":" << (result ? "true" : "false")
             << ",\"final_texture_handle\":" << result.value_or(0U)
             << ",\"code_words_total\":" << code.size() << ",\"code_truncated\":"
             << (code.size() > MaxCodeWords ? "true" : "false") << ",\"code\":[";
        for (std::size_t i = 0; i < code.size() && i < MaxCodeWords; ++i) {
            if (i) file << ',';
            file << code[i];
        }
        file << "],\"reads_truncated\":" << (reads_truncated ? "true" : "false")
             << ",\"reads\":[";
        for (std::size_t i = 0; i < count; ++i) {
            const auto& read = reads[i];
            if (i) file << ',';
            file << "{\"order\":" << i << ",\"kind\":";
            String(file, read.kind);
            file << ",\"cbuf_index\":" << read.cbuf_index
                 << ",\"cbuf_offset\":" << read.cbuf_offset
                 << ",\"storage_binding\":" << read.storage_binding
                 << ",\"byte_offset\":" << read.byte_offset
                 << ",\"cbuf_base\":" << read.cbuf_base
                 << ",\"raw_pointer\":" << read.raw_pointer
                 << ",\"computed_address\":" << read.address
                 << ",\"already_read_value\":" << read.value
                 << ",\"width_bytes\":" << read.width
                 << ",\"read_performed\":" << (read.performed ? "true" : "false")
                 << ",\"mapped_range_at_trace_racy\":";
            if (read.performed) file << (read.mapped_at_trace ? "true" : "false");
            else file << "null";
            file << ",\"address_wrapped\":" << (read.address_wrapped ? "true" : "false") << '}';
        }
        file << "]}\n";
    }

private:
    struct TraceState {
        TraceState() {
            const char* value = std::getenv("EDEN_HANDLE_TRACE_DIR");
            if (value && *value) directory = value;
        }
        std::filesystem::path directory;
        std::mutex mutex;
        std::array<u64, MaxPrograms> programs{};
        std::size_t count{};
    };

    static TraceState& State() {
        static TraceState state;
        return state;
    }

    static void String(std::ostream& file, const char* value) {
        file << '"';
        for (const unsigned char* p = reinterpret_cast<const unsigned char*>(value); *p; ++p) {
            if (*p == '"' || *p == '\\') {
                file << '\\' << static_cast<char>(*p);
            } else if (*p < 32) {
                file << "\\u00" << "0123456789abcdef"[*p >> 4]
                     << "0123456789abcdef"[*p & 15];
            } else {
                file << static_cast<char>(*p);
            }
        }
        file << '"';
    }

    std::array<Read, MaxReads> reads{};
    std::size_t count{};
    bool fault{};
    bool reads_truncated{};
};

} // namespace VideoCommon
