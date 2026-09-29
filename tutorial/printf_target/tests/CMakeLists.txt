include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/5376968f6948923e2411081fd9372e71a59d8e77.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

set(TEST_FILES
    printf_map_dispatcher_test.cpp
    printf_data_offloading_node_test.cpp
    printf_transform_test.cpp
    test_main.cpp
)

add_executable(printf_target_test ${TEST_FILES})
target_include_directories(printf_target_test PRIVATE ./)
target_link_libraries(printf_target_test
    gtest_main
    sdfg::tutorial_printf_target
    sdfg::opt
)

if(SDFG_ENABLE_COVERAGE)
  target_compile_options(printf_target_test PRIVATE
    -fprofile-instr-generate
    -fcoverage-mapping
  )
  target_link_options(printf_target_test PRIVATE
    -fprofile-instr-generate
  )
endif()

add_test(NAME printf_target_test COMMAND printf_target_test)

include(GoogleTest)
gtest_discover_tests(printf_target_test
  PROPERTIES LABELS "printf_target"
)
