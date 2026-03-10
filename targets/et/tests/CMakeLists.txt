include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/5376968f6948923e2411081fd9372e71a59d8e77.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

set(TEST_FILES
    test.cpp
)

# Now simply link against gtest or gtest_main as needed. Eg
add_executable(docc-target-et_test ${TEST_FILES})
target_include_directories(docc-target-et_test PRIVATE ./)
target_link_libraries(docc-target-et_test gtest_main sdfg::opt)
if(SDFG_ENABLE_COVERAGE)
  target_compile_options(docc-target-et_test PRIVATE
    -fprofile-instr-generate
    -fcoverage-mapping
  )
  target_link_options(docc-target-et_test PRIVATE
    -fprofile-instr-generate
  )
endif()

add_test(NAME docc-target-et_test_test COMMAND docc-target-et_test)

include(GoogleTest)
gtest_discover_tests(docc-target-et_test)
