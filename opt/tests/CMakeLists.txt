include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/5376968f6948923e2411081fd9372e71a59d8e77.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

set(TEST_FILES
    analysis/scop_analysis_test.cpp
    cuda/blas/cublas_test.cpp
    cuda/cuda_kernel_test.cpp
    cuda/cuda_library_nodes_test.cpp
    passes/normalization/perfect_loop_distribution_test.cpp
    passes/normalization/stride_minimization_test.cpp
    passes/offloading/code_motion/block_hoisting_test.cpp
    passes/offloading/code_motion/block_sorting_test.cpp
    passes/offloading/data_transfer_minimization_pass_test.cpp
    passes/offloading/remove_redundant_data_transfers_pass_test.cpp
    transformations/offloading/cuda_parallelize_nested_map_test.cpp
    transformations/offloading/gpu_tiling_test.cpp
    transformations/offloading/kernel_local_storage_test.cpp
    transformations/polly_transform_test.cpp
    passes/scheduler/cuda_scheduler_test.cpp
    passes/scheduler/highway_scheduler_test.cpp
    passes/scheduler/omp_scheduler_test.cpp
    passes/scheduler/polly_scheduler_test.cpp
    targets/omp/codegen/omp_map_dispatcher_test.cpp
    targets/omp/schedule_test.cpp
    targets/highway/codegen/highway_map_dispatcher_test.cpp
    transformations/collapse_to_depth_test.cpp
    transformations/highway_transform_test.cpp
    transformations/omp_transform_test.cpp
    transformations/polly_transform_test.cpp
    transformations/map_collapse_test.cpp
    transformations/loop_distribute_test.cpp
    transformations/loop_interchange_test.cpp
    transformations/loop_skewing_test.cpp
    transformations/loop_tiling_test.cpp
    transformations/map_fusion_test.cpp
    transformations/tile_fusion_test.cpp
    transformations/out_local_storage_test.cpp
    transformations/recorder_test.cpp
    test.cpp
)

# Now simply link against gtest or gtest_main as needed. Eg
add_executable(sdfgopt_test ${TEST_FILES})
target_include_directories(sdfgopt_test PRIVATE ./)
target_link_libraries(sdfgopt_test gtest_main sdfg::opt)
if(SDFG_ENABLE_COVERAGE)
  target_compile_options(sdfgopt_test PRIVATE
    -fprofile-instr-generate
    -fcoverage-mapping
  )
  target_link_options(sdfgopt_test PRIVATE
    -fprofile-instr-generate
  )
endif()

add_test(NAME sdfgopt_test_test COMMAND sdfgopt_test)

include(GoogleTest)
gtest_discover_tests(sdfgopt_test)
