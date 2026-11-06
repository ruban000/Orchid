// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later
#include "video_core/compute_binding_diagnostic.h"

#include "common/logging.h"

namespace VideoCommon::ComputeBindingDiagnostic {
static const char* StateName(BindingState state) noexcept {
    switch (state) {
    case BindingState::NotObserved:
        return "not_observed";
    case BindingState::Descriptor:
        return "descriptor";
    case BindingState::Disabled:
        return "disabled";
    case BindingState::Indirect:
        return "indirect";
    case BindingState::IndexOutOfRange:
        return "index_out_of_range";
    case BindingState::LimitSkipped:
        return "limit_skipped";
    case BindingState::NullPointer:
        return "null_pointer";
    case BindingState::EmptyRange:
        return "empty_range";
    case BindingState::UnmappedAligned:
        return "unmapped_aligned";
    case BindingState::UnmappedPointer:
        return "unmapped_pointer";
    case BindingState::NoPhysicalPrefix:
        return "no_physical_prefix";
    case BindingState::Normal:
        return "normal";
    case BindingState::MultiRange:
        return "multi_range";
    }
    return "unknown";
}
void Emit(const Record& r) noexcept {
    LOG_INFO(
        HW_GPU,
        "Compute binding trace phase={} native={:016x} shader={:016x} seq={} program={:#x} "
        "cb0_enabled={} cb0_base_known={} cb0_base={:#x} cb0_size_known={} cb0_size={} "
        "compile_lo_known={} compile_lo={:#x} compile_hi_known={} compile_hi={:#x} "
        "compile_mapping_known={} compile_mapping={} descriptor_seen={} descriptor_duplicate={} "
        "descriptor_index={} written={} indirect={} indirect_holder_known={} indirect_holder={:#x} "
        "state={} pointer_known={} pointer={:#x} "
        "layout_known={} layout={} size_known={} size={} alignment_known={} alignment={} "
        "aligned_gpu_known={} aligned_gpu={:#x} device_known={} device={:#x} "
        "logical_known={} logical={} physical_known={} physical={} "
        "host_offset_known={} host_offset={} host_range_known={} host_range={} "
        "host_binding_known={} host_binding={} cache_buffer_known={} cache_buffer_id={} "
        "next_qword=not_read disk_file_native_key=unavailable",
        r.phase == Phase::Compile ? "compile" : "dispatch", r.native_key, r.shader_hash, r.serial,
        r.program, r.cbuf_enabled, r.cbuf_base.has_value(), r.cbuf_base.value_or(0),
        r.cbuf_size.has_value(), r.cbuf_size.value_or(0), r.compile_low.has_value(),
        r.compile_low.value_or(0), r.compile_high.has_value(), r.compile_high.value_or(0),
        r.pointer_mapped.has_value(), r.pointer_mapped.value_or(false), r.descriptor_seen,
        r.duplicate_descriptor, r.descriptor_index, r.is_written, r.descriptor_indirect,
        r.indirect_holder.has_value(), r.indirect_holder.value_or(0), StateName(r.binding_state),
        r.pointer.has_value(), r.pointer.value_or(0), r.layout_size.has_value(),
        r.layout_size.value_or(0), r.selected_size.has_value(), r.selected_size.value_or(0),
        r.alignment.has_value(), r.alignment.value_or(0), r.aligned_gpu.has_value(),
        r.aligned_gpu.value_or(0), r.device_address.has_value(), r.device_address.value_or(0),
        r.logical_size.has_value(), r.logical_size.value_or(0), r.physical_size.has_value(),
        r.physical_size.value_or(0), r.descriptor_offset.has_value(),
        r.descriptor_offset.value_or(0), r.descriptor_range.has_value(),
        r.descriptor_range.value_or(0), r.descriptor_binding.has_value(),
        r.descriptor_binding.value_or(0), r.cache_buffer_id.has_value(),
        r.cache_buffer_id.value_or(0));
}
} // namespace VideoCommon::ComputeBindingDiagnostic
