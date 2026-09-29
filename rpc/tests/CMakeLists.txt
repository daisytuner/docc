include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/5376968f6948923e2411081fd9372e71a59d8e77.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

set(TEST_FILES
    passes/rpc/rpc_loop_opt_test.cpp
    transformations/rpc_node_transform_test.cpp
    test.cpp
)

# Now simply link against gtest or gtest_main as needed. Eg
add_executable(sdfgrpc_test ${TEST_FILES})
target_include_directories(sdfgrpc_test PRIVATE ./ ${CMAKE_CURRENT_SOURCE_DIR}/../../opt/tests)
target_link_libraries(sdfgrpc_test gtest_main sdfgrpc)

# Daisytuner RPC tests (polybench transfer tuning regression tests)
set(DAISYTUNER_RPC_TEST_FILES
    passes/rpc/polybench_transfertuning_test.cpp
    test_daisytuner_rpc.cpp
)

add_executable(sdfgrpc_daisytuner_rpc_test ${DAISYTUNER_RPC_TEST_FILES})
target_include_directories(sdfgrpc_daisytuner_rpc_test PRIVATE ./ ${CMAKE_CURRENT_SOURCE_DIR}/../../opt/tests)
target_link_libraries(sdfgrpc_daisytuner_rpc_test gtest_main sdfgrpc)
if(SDFG_ENABLE_COVERAGE)
  target_compile_options(sdfgrpc_test PRIVATE
    -fprofile-instr-generate
    -fcoverage-mapping
  )
  target_link_options(sdfgrpc_test PRIVATE
    -fprofile-instr-generate
  )
  target_compile_options(sdfgrpc_daisytuner_rpc_test PRIVATE
    -fprofile-instr-generate
    -fcoverage-mapping
  )
  target_link_options(sdfgrpc_daisytuner_rpc_test PRIVATE
    -fprofile-instr-generate
  )
endif()

add_test(NAME sdfgrpc_test_test COMMAND sdfgrpc_test)
set_tests_properties(sdfgrpc_test_test PROPERTIES LABELS "sdfgrpc")

include(GoogleTest)
gtest_discover_tests(sdfgrpc_test
  PROPERTIES LABELS "sdfgrpc"
)
gtest_discover_tests(sdfgrpc_daisytuner_rpc_test
  PROPERTIES LABELS "sdfgrpc_daisytuner_rpc"
)
