// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <atomic>
#include <charconv>
#include <cstddef>
#include <cstdlib>
#include <optional>
#include <string_view>

#include "common/common_types.h"

namespace VideoCommon::ComputeBindingDiagnostic {

// Only an explicitly supplied native FNV key enables the trace. This is not the pipeline hash.
inline std::optional<u64> ParseSelector(std::string_view text) noexcept {
    if (text.size() != 16)
        return {};
    u64 key{};
    const auto result = std::from_chars(text.data(), text.data() + text.size(), key, 16);
    if (result.ec != std::errc{} || result.ptr != text.data() + text.size())
        return {};
    return key;
}
inline std::optional<u64> Selector() noexcept {
    static const std::optional<u64> selector = []() -> std::optional<u64> {
        const char* const value = std::getenv("EDEN_NX2_TRACE_COMPUTE_KEY");
        return value ? ParseSelector(value) : std::nullopt;
    }();
    return selector;
}

inline bool Matches(std::optional<u64> native_key) noexcept {
    const auto selector = Selector();
    return selector && native_key && *selector == *native_key;
}

class Limit {
public:
    bool Claim() noexcept {
        u32 count = seen.load(std::memory_order_relaxed);
        while (count < 8) {
            if (seen.compare_exchange_weak(count, count + 1, std::memory_order_relaxed))
                return true;
        }
        return false;
    }

private:
    std::atomic<u32> seen{};
};
inline Limit compile_limit, dispatch_limit;
inline std::atomic<u64> sequence{};

enum class Phase : u8 { Compile, Dispatch };
enum class BindingState : u8 {
    NotObserved,
    Descriptor,
    Disabled,
    Indirect,
    IndexOutOfRange,
    LimitSkipped,
    NullPointer,
    EmptyRange,
    UnmappedAligned,
    UnmappedPointer,
    NoPhysicalPrefix,
    Normal,
    MultiRange
};

struct Record {
    Phase phase{};
    u64 native_key{}, shader_hash{}, serial{}, program{};
    bool cbuf_enabled{};
    std::optional<u64> cbuf_base;
    std::optional<u32> cbuf_size;
    std::optional<u32> compile_low, compile_high;
    std::optional<bool> pointer_mapped;
    bool descriptor_seen{};
    bool duplicate_descriptor{};
    u32 descriptor_index{};
    bool is_written{};
    bool descriptor_indirect{};
    std::optional<u64> indirect_holder;
    BindingState binding_state{BindingState::NotObserved};
    std::optional<u64> pointer;
    std::optional<u64> layout_size;
    std::optional<u32> selected_size, alignment, logical_size, physical_size;
    std::optional<u64> aligned_gpu, device_address, descriptor_offset;
    std::optional<u32> descriptor_range, descriptor_binding, cache_buffer_id;
};
inline thread_local Record* current{};

// This is defined in the .cpp so tests may exercise scopes without invoking logging.
void Emit(const Record& record) noexcept;

class Scope {
public:
    explicit Scope(Record& record) noexcept : previous{current} {
        current = &record;
    }
    ~Scope() {
        current = previous;
    }
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;

private:
    Record* previous;
};

inline void CompileWord(u32 bank, u32 offset, u32 value) noexcept {
    if (!current || current->phase != Phase::Compile || bank != 0)
        return;
    if (offset == 0x50)
        current->compile_low = value;
    if (offset == 0x54)
        current->compile_high = value;
}
inline void CompileMapping(u32 bank, u32 offset, bool mapped) noexcept {
    if (current && current->phase == Phase::Compile && bank == 0 && offset == 0x50)
        current->pointer_mapped = mapped;
}
inline bool IsTarget(u32 index, u32 bank, u32 offset) noexcept {
    return current && current->phase == Phase::Dispatch && current->descriptor_seen &&
           !current->duplicate_descriptor && current->descriptor_index == index && bank == 0 &&
           offset == 0x50;
}
} // namespace VideoCommon::ComputeBindingDiagnostic
