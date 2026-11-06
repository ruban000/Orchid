# SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
# SPDX-License-Identifier: GPL-3.0-or-later

# SPDX-FileCopyrightText: 2018 yuzu Emulator Project
# SPDX-License-Identifier: GPL-2.0-or-later

add_executable(tests
    common/bit_field.cpp
    common/cityhash.cpp
    common/container_hash.cpp
    common/event.cpp
    common/fibers.cpp
    common/game_presets.cpp
    common/host_memory.cpp
    common/host_memory_fallback.cpp
    common/page_table.cpp
    common/paged_scalar.cpp
    common/param_package.cpp
    common/queue_lifetime.cpp
    common/range_map.cpp
    common/range_set_move.cpp
    common/ring_buffer.cpp
    common/scratch_buffer.cpp
    common/slot_vector.cpp
    common/undefined_fix.cpp
    common/unique_function.cpp
    common/zstd_compression.cpp
    core/address_arbiter_read.cpp
    core/address_space.cpp
    core/core_timing.cpp
    core/page_heap.cpp
    core/read_traps.cpp
    core/nx2_readback_cache.cpp
    core/nx2_core_readback_capacity.cpp
    core/nso.cpp
    core/nrr.cpp
    core/ncz_decompressor.cpp
    core/ldn.cpp
    core/internal_network/socket_proxy.cpp
    core/nup_writer.cpp
    core/priority_queue.cpp
    core/vfs_vector.cpp
    core/internal_network/network.cpp
    video_core/aftermath_dump.cpp
    video_core/oot_visual_workaround.cpp
    video_core/storage_buffer_pointer_tracking.cpp
    video_core/storage_buffer_control_flow.cpp
    video_core/memory_tracker.cpp
    video_core/buffer_cache_download.cpp
    video_core/unmapped_read_diagnostic.cpp
    video_core/memory_budget.cpp
    video_core/astc_recompression_override.cpp
    video_core/nx2_video_configuration.cpp
    video_core/nx2_decoder_image.cpp
    video_core/nx2_present.cpp
    video_core/nx2_gpu_mappings.cpp
    video_core/nx2_sparse_mappings.cpp
    video_core/nx2_sparse_region.cpp
    video_core/nx2_sparse_tiles.cpp
    video_core/nx2_sparse_images.cpp
    video_core/nx2_sparse_image_layout.cpp
    video_core/nx2_sparse_image_write.cpp
    video_core/nx2_command_sync.cpp
    video_core/nx2_command_ranges.cpp
    video_core/physical_alias.cpp
    video_core/h264_packet.cpp
    video_core/framebuffer_clear_layers.cpp
    video_core/vertex_repack.cpp
    video_core/query_copy.cpp
    video_core/shader_reuse_environment.cpp
    video_core/ampere_lowering.cpp
    video_core/ampere_matrix.cpp
    video_core/ampere_returns.cpp
    video_core/ampere_f2i.cpp
    video_core/ampere_i2f64.cpp
    video_core/ampere_uniform_fset.cpp
    video_core/compute_storage_buffers.cpp
    video_core/cooperative_matrix.cpp
    video_core/texture_depth_float.cpp
    video_core/texture_descriptor_bounds.cpp
    video_core/texture_r2ur_handle.cpp
    video_core/ampere_kirby.cpp
    video_core/ampere_geometry.cpp
    video_core/ampere_geometry_lowering.cpp
    video_core/geometry_layer_viewport_relative.cpp
    video_core/demote_nested_returns.cpp
    video_core/ampere_geometry_exit.cpp
    video_core/ampere_uniform_lea.cpp
    video_core/ampere_compute_forms.cpp
    video_core/ampere_integer_multiply.cpp
    video_core/ampere_program_override.cpp
    video_core/x8d24_conversion.cpp
    video_core/ampere_bindless_tex.cpp
    video_core/ampere_sured.cpp
    video_core/ampere_yoshi_surface.cpp
    video_core/ampere_yoshi_shift.cpp
    video_core/ampere_yoshi_uniform_shift.cpp
    video_core/ampere_yoshi_cbuf_carry.cpp
    video_core/ampere_yoshi_uniform_carry.cpp
    video_core/ampere_yoshi_lea_carry.cpp
    video_core/maxwell_shr.cpp
    video_core/ampere_release_guards.cpp
    video_core/ampere_release_f2i.cpp
    video_core/ampere_uniform_immediate_shift.cpp
    video_core/ampere_bindless_surfaces.cpp
    video_core/syncpoint_manager.cpp
    video_core/native_macro_fifo.cpp
    video_core/uniform_stream_reuse.cpp
    video_core/shader_invalidation.cpp
    video_core/descriptor_table.cpp
    video_core/usage_tracker.cpp
    input_common/calibration_configuration_job.cpp
)

create_target_directory_groups(tests)

if (WIN32)
    target_sources(tests PRIVATE common/host_memory_commit.cpp)
endif()

target_link_libraries(tests PRIVATE common core input_common video_core)
target_link_libraries(tests PRIVATE ${PLATFORM_LIBRARIES} Catch2::Catch2WithMain Threads::Threads)

add_test(NAME tests COMMAND tests)

# needed for vma
if (NOT MSVC)
    target_compile_options(tests PRIVATE
        $<$<COMPILE_LANGUAGE:C,CXX>:-Wno-conversion>
        $<$<COMPILE_LANGUAGE:C,CXX>:-Wno-unused-variable>
        $<$<COMPILE_LANGUAGE:C,CXX>:-Wno-unused-parameter>
        $<$<COMPILE_LANGUAGE:C,CXX>:-Wno-missing-field-initializers>)
endif()
