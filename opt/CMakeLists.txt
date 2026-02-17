# Find SDFG
if(NOT TARGET sdfg::sdfg)
    find_package(sdfg CONFIG REQUIRED)
endif()

# Find ISL
find_package(PkgConfig REQUIRED)
pkg_check_modules(ISL REQUIRED IMPORTED_TARGET isl)

set(SOURCE_FILES
    src/analysis/scop_analysis.cpp
    src/passes/normalization/perfect_loop_distribution.cpp
    src/passes/normalization/stride_minimization.cpp
    src/passes/offloading/cuda_library_node_rewriter_pass.cpp
    src/passes/offloading/code_motion/block_hoisting.cpp
    src/passes/offloading/code_motion/block_sorting.cpp
    src/passes/offloading/data_transfer_minimization_pass.cpp
    src/passes/offloading/onnx_library_node_rewriter_pass.cpp
    src/passes/offloading/remove_redundant_transfers_pass.cpp
    src/passes/offloading/sync_condition_propagation.cpp
    src/passes/dot_expansion_pass.cpp
    src/passes/gemm_expansion_pass.cpp
    src/passes/opt_pipeline.cpp
    src/passes/scheduler/cuda_scheduler.cpp
    src/passes/scheduler/hip_scheduler.cpp
    src/passes/scheduler/highway_scheduler.cpp
    src/passes/scheduler/loop_scheduling_pass.cpp
    src/passes/scheduler/omp_scheduler.cpp
    src/passes/scheduler/polly_scheduler.cpp
    src/passes/scheduler/scheduler_registry.cpp
    src/targets/cuda/blas/dot.cpp
    src/targets/cuda/blas/gemm.cpp
    src/targets/cuda/blas/utils.cpp
    src/targets/cuda/cuda.cpp
    src/targets/cuda/cuda_data_offloading_node.cpp
    src/targets/cuda/cuda_map_dispatcher.cpp
    src/targets/hip/blas/dot.cpp
    src/targets/hip/blas/gemm.cpp
    src/targets/hip/blas/utils.cpp
    src/targets/hip/hip.cpp
    src/targets/hip/hip_data_offloading_node.cpp
    src/targets/hip/hip_map_dispatcher.cpp
    src/targets/gpu/gpu_map_utils.cpp
    src/targets/highway/codegen/highway_map_dispatcher.cpp
    src/targets/offloading/data_offloading_node.cpp
    src/targets/offloading/utils.cpp
    src/targets/omp/codegen/omp_map_dispatcher.cpp
    src/targets/onnx/blas/dot_dispatcher.cpp
    src/targets/onnx/blas/gemm_dispatcher.cpp
    src/targets/onnx/tensor/broadcast_dispatcher.cpp
    src/targets/onnx/tensor/conv_dispatcher.cpp
    src/targets/onnx/tensor/elementwise_dispatcher.cpp
    src/targets/onnx/tensor/reduce_dispatcher.cpp
    src/targets/onnx/tensor/transpose_dispatcher.cpp
    src/transformations/offloading/cublas_offloading_expansion.cpp
    src/transformations/offloading/cuda_parallelize_nested_map.cpp
    src/transformations/offloading/cuda_transform.cpp
    src/transformations/offloading/gpu_condition_propagation.cpp
    src/transformations/offloading/gpu_loop_reordering.cpp
    src/transformations/offloading/gpu_tiling.cpp
    src/transformations/offloading/hip_parallelize_nested_map.cpp
    src/transformations/offloading/hip_transform.cpp
    src/transformations/offloading/kernel_local_storage.cpp
    src/transformations/highway_transform.cpp
    src/transformations/omp_transform.cpp
    src/transformations/offloading/offload_transform.cpp
    src/transformations/polly_transform.cpp
)
add_library(sdfgopt
    ${SOURCE_FILES}
)
add_library(sdfg::opt ALIAS sdfgopt)
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
