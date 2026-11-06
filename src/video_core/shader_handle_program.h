// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "common/common_types.h"
#include "shader_recompiler/shader_info.h"
#include "video_core/memory_manager.h"
#include "video_core/texture_handle_trace.h"

namespace VideoCommon {

namespace detail {

template <typename CbufAddress>
u32 EvaluateTextureHandleTraced(const Shader::Info& info, u32 program,
                               Tegra::MemoryManager& gpu_memory, CbufAddress&& cbuf_address,
                               std::source_location caller) {
    TextureHandleTrace trace;
    const auto result = Shader::EvaluateHandleProgram(
        info.handle_programs[program],
        [&](u32 index, u32 offset) -> u32 {
            const GPUVAddr base{cbuf_address(index)};
            const GPUVAddr address{base + offset};
            const u32 value = base != 0 ? gpu_memory.Read<u32>(address) : 0U;
            trace.Add({.kind = "cbuf_word", .cbuf_index = index, .cbuf_offset = offset,
                       .byte_offset = offset, .cbuf_base = base, .raw_pointer = base,
                       .address = address, .value = value, .width = 4, .performed = base != 0,
                       .mapped_at_trace = base != 0 && gpu_memory.IsFullyMappedRange(address, 4),
                       .address_wrapped = address < base});
            return value;
        },
        [&](u32 binding, u32 offset) -> u32 {
            if (binding >= info.storage_buffers_descriptors.size()) {
                trace.Add({.kind = "storage_binding_missing", .storage_binding = binding,
                           .byte_offset = offset});
                return 0U;
            }
            const auto& descriptor{info.storage_buffers_descriptors[binding]};
            const GPUVAddr cbuf{cbuf_address(descriptor.cbuf_index)};
            if (cbuf == 0) {
                trace.Add({.kind = "storage_cbuf_unbound", .cbuf_index = descriptor.cbuf_index,
                           .cbuf_offset = descriptor.cbuf_offset, .storage_binding = binding,
                           .byte_offset = offset});
                return 0U;
            }
            const GPUVAddr pointer_address{cbuf + descriptor.cbuf_offset};
            const GPUVAddr base{gpu_memory.Read<u64>(pointer_address)};
            trace.Add({.kind = "storage_pointer", .cbuf_index = descriptor.cbuf_index,
                       .cbuf_offset = descriptor.cbuf_offset, .storage_binding = binding,
                       .byte_offset = offset, .cbuf_base = cbuf, .raw_pointer = base,
                       .address = pointer_address, .value = base, .width = 8, .performed = true,
                       .mapped_at_trace = gpu_memory.IsFullyMappedRange(pointer_address, 8),
                       .address_wrapped = pointer_address < cbuf});
            const GPUVAddr address{base + offset};
            const u32 value = base != 0 ? gpu_memory.Read<u32>(address) : 0U;
            trace.Add({.kind = "storage_word", .cbuf_index = descriptor.cbuf_index,
                       .cbuf_offset = descriptor.cbuf_offset, .storage_binding = binding,
                       .byte_offset = offset, .cbuf_base = cbuf, .raw_pointer = base,
                       .address = address, .value = value, .width = 4, .performed = base != 0,
                       .mapped_at_trace = base != 0 && gpu_memory.IsFullyMappedRange(address, 4),
                       .address_wrapped = address < base});
            return value;
        });
    trace.Finish(info.handle_programs[program], program, result, caller);
    return result.value_or(0U);
}

} // namespace detail

/// The texture handle a shader reads from a storage buffer, computed from the current constant
/// buffer and storage buffer words (Shader::Info::handle_programs). cbuf_address(index) is the
/// bound constant buffer's address, 0 when none is bound.
template <typename CbufAddress>
u32 EvaluateTextureHandle(const Shader::Info& info, u32 program, Tegra::MemoryManager& gpu_memory,
                          CbufAddress&& cbuf_address,
                          std::source_location caller = std::source_location::current()) {
    if (TextureHandleTrace::Enabled()) {
        return detail::EvaluateTextureHandleTraced(info, program, gpu_memory, cbuf_address, caller);
    }
    return Shader::EvaluateHandleProgram(
               info.handle_programs[program],
               [&](u32 index, u32 offset) -> u32 {
                   const GPUVAddr base{cbuf_address(index)};
                   return base != 0 ? gpu_memory.Read<u32>(base + offset) : 0U;
               },
               [&](u32 binding, u32 offset) -> u32 {
                   if (binding >= info.storage_buffers_descriptors.size()) {
                       return 0U;
                   }
                   const auto& descriptor{info.storage_buffers_descriptors[binding]};
                   const GPUVAddr cbuf{cbuf_address(descriptor.cbuf_index)};
                   if (cbuf == 0) {
                       return 0U;
                   }
                   const GPUVAddr base{gpu_memory.Read<u64>(cbuf + descriptor.cbuf_offset)};
                   return base != 0 ? gpu_memory.Read<u32>(base + offset) : 0U;
               })
        .value_or(0U);
}

} // namespace VideoCommon
