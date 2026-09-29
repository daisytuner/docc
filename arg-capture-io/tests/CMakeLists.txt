include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/5376968f6948923e2411081fd9372e71a59d8e77.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

set(TEST_FILES
    capture_tests.cpp
    ser_deser_tests.cpp
    test.cpp
)

# Now simply link against gtest or gtest_main as needed. Eg
add_executable(capture_io_test ${TEST_FILES})
target_include_directories(capture_io_test PRIVATE ./)
target_link_libraries(capture_io_test gtest_main arg_capture_io)

add_test(NAME capture_io_test_test COMMAND capture_io_test)

include(GoogleTest)
gtest_discover_tests(capture_io_test
  PROPERTIES LABELS "capture_io"
)
