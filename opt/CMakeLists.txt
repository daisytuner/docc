# Find SDFG
if(NOT TARGET sdfg::sdfg)
    find_package(sdfg CONFIG REQUIRED)
endif()

# Find ISL
find_package(PkgConfig REQUIRED)
pkg_check_modules(ISL REQUIRED IMPORTED_TARGET isl)

set(SOURCE_FILES
    src/passes/einsum.cpp
    src/passes/normalization/loop_normal_form.cpp
    src/passes/normalization/perfect_loop_distribution.cpp
    src/passes/normalization/stride_minimization.cpp
    src/passes/normalization/map_fusion.cpp
    src/passes/targets/target_mapping_pass.cpp
    src/passes/offloading/cuda_library_node_rewriter_pass.cpp
    src/passes/offloading/code_motion/block_hoisting.cpp
    src/passes/offloading/code_motion/block_sorting.cpp
    src/passes/offloading/data_transfer_minimization_pass.cpp
    src/passes/offloading/gpu_loop_reordering_pass.cpp
    src/passes/offloading/gpu_nested_parallelization_pass.cpp
    src/passes/offloading/gpu_tiling_pass.cpp
    src/passes/offloading/remove_redundant_transfers_pass.cpp
    src/passes/offloading/rocm_library_node_rewriter_pass.cpp
    src/passes/offloading/cuda_library_node_transfer_extraction_pass.cpp
    src/passes/offloading/rocm_library_node_transfer_extraction_pass.cpp
    src/passes/offloading/sync_condition_propagation.cpp
    src/passes/collapse_pass.cpp
    src/passes/extended_data_transfer_minimization_pass.cpp
    src/passes/readonly_transfer_hoisting_pass.cpp
    src/passes/dot_expansion_pass.cpp
    src/passes/gemm_expansion_pass.cpp
    src/passes/opt_pipeline.cpp
    src/passes/scheduler/cuda_scheduler.cpp
    src/passes/scheduler/rocm_scheduler.cpp
    src/passes/scheduler/highway_scheduler.cpp
    src/passes/scheduler/loop_scheduling_pass.cpp
    src/passes/scheduler/omp_scheduler.cpp
    src/passes/scheduler/scheduler_registry.cpp
    src/targets/cuda/blas/dot.cpp
    src/targets/cuda/blas/gemm.cpp
    src/targets/cuda/blas/utils.cpp
    src/targets/cuda/stdlib/memset.cpp
    src/targets/cuda/cuda.cpp
    src/targets/cuda/cuda_data_offloading_node.cpp
    src/targets/cuda/cuda_map_dispatcher.cpp
    src/targets/rocm/blas/dot.cpp
    src/targets/rocm/blas/gemm.cpp
    src/targets/rocm/blas/utils.cpp
    src/targets/rocm/stdlib/memset.cpp
    src/targets/rocm/rocm.cpp
    src/targets/rocm/rocm_data_offloading_node.cpp
    src/targets/rocm/rocm_map_dispatcher.cpp
    src/targets/gpu/gpu_map_utils.cpp
    src/targets/highway/codegen/highway_map_dispatcher.cpp
    src/targets/offloading/external_offloading_node.cpp
    src/targets/offloading/utils.cpp
    src/targets/omp/codegen/omp_map_dispatcher.cpp
    src/transformations/offloading/cublas_data_transfer_extraction.cpp
    src/transformations/offloading/cuda_stdlib_data_transfer_extraction.cpp
    src/transformations/offloading/rocblas_data_transfer_extraction.cpp
    src/transformations/offloading/rocm_stdlib_data_transfer_extraction.cpp
    src/transformations/offloading/cuda_parallelize_nested_map.cpp
    src/transformations/offloading/cuda_transform.cpp
    src/transformations/offloading/gpu_condition_propagation.cpp
    src/transformations/offloading/gpu_loop_reordering.cpp
    src/transformations/offloading/gpu_tiling.cpp
    src/transformations/offloading/rocm_parallelize_nested_map.cpp
    src/transformations/offloading/rocm_transform.cpp
    src/transformations/offloading/kernel_local_storage.cpp
    src/transformations/collapse_to_depth.cpp
    src/transformations/highway_transform.cpp
    src/transformations/omp_transform.cpp
    src/transformations/offloading/offload_transform.cpp
    src/transformations/map_collapse.cpp
    src/transformations/loop_distribute.cpp
    src/transformations/einsum2dot.cpp
    src/transformations/einsum2gemm.cpp
    src/transformations/einsum_expand.cpp
    src/transformations/einsum_extend.cpp
    src/transformations/einsum_lift.cpp
    src/transformations/loop_condition_normalize.cpp
    src/transformations/loop_indvar_finalize.cpp
    src/transformations/loop_interchange.cpp
    src/transformations/loop_rotate.cpp
    src/transformations/loop_shift.cpp
    src/transformations/loop_skewing.cpp
    src/transformations/loop_tiling.cpp
    src/transformations/loop_unit_stride.cpp
    src/transformations/map_fusion.cpp
    src/transformations/tile_fusion.cpp
    src/transformations/out_local_storage.cpp
    src/transformations/in_local_storage.cpp
    src/transformations/accumulator_tile.cpp
    src/transformations/recorder.cpp
    src/transformations/replayer.cpp
    src/transformations/utils.cpp
    src/passes/code_motion/extended_block_hoisting.cpp
    src/passes/code_motion/extended_block_sorting.cpp
)
add_library(sdfgopt
    ${SOURCE_FILES}
)
add_library(sdfg::opt ALIAS sdfgopt)
add_library(docc::opt ALIAS sdfgopt)
add_library(sdfgopt::sdfgopt ALIAS sdfgopt)
target_include_directories(sdfgopt
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# Sanitizers
if(SDFG_ENABLE_SANITIZER)
  target_compile_options(sdfgopt PUBLIC ${_SAN_COMPILE_OPTS})
  target_link_options   (sdfgopt PUBLIC ${_SAN_LINK_OPTS})
endif()

# Coverage
if(SDFG_ENABLE_COVERAGE)
  target_compile_options(sdfgopt PRIVATE
    -fprofile-instr-generate
    -fcoverage-mapping
    -O0
    -g
    -fno-inline
  )
  target_link_options(sdfgopt PRIVATE
    -fprofile-instr-generate
  )
endif()

target_link_libraries(sdfgopt PUBLIC sdfg::sdfg PkgConfig::ISL)
set_target_properties(sdfgopt PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)

## INSTALLATION

install(TARGETS sdfgopt
    EXPORT sdfgoptTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
install(DIRECTORY include/sdfg DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

## CMAKE CONFIGURATION

install(EXPORT sdfgoptTargets
    NAMESPACE sdfg::
    FILE sdfgoptTargets.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/sdfg
)

include(CMakePackageConfigHelpers)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/sdfgoptConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
    ${CMAKE_CURRENT_LIST_DIR}/cmake/sdfgoptConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/sdfgoptConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/sdfg
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/sdfgoptConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/sdfgoptConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/sdfg
)

if(SDFG_BUILD_TESTS)
    set(INSTALL_GTEST OFF)
    add_subdirectory(tests)
endif()
