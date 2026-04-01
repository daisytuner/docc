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
add_executable(c-compile_test ${TEST_FILES})
target_include_directories(c-compile_test PRIVATE ./)
target_link_libraries(c-compile_test gtest_main sdfg::opt)
if(SDFG_ENABLE_COVERAGE)
  target_compile_options(c-compile_test PRIVATE
    -fprofile-instr-generate
    -fcoverage-mapping
  )
  target_link_options(c-compile_test PRIVATE
    -fprofile-instr-generate
  )
endif()

add_test(NAME c-compile_test_test COMMAND c-compile_test)

include(GoogleTest)
gtest_discover_tests(c-compile_test)
