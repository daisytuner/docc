include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/5376968f6948923e2411081fd9372e71a59d8e77.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(docc-target-axelera_test axscript_test.cpp)
target_link_libraries(docc-target-axelera_test gtest docc::target::axelera)

include(GoogleTest)
gtest_discover_tests(docc-target-axelera_test)
