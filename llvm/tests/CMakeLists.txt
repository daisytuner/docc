include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/5376968f6948923e2411081fd9372e71a59d8e77.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

set(TEST_FILES
    lifting/blas_lifting_test.cpp
    lifting/function_lifting_test.cpp
    lifting/intrinsic_lifting_test.cpp
    lifting/libfunc_lifting_test.cpp
    lifting/lifting_test.cpp
    passes/code_generation/cfg_dump_test.cpp
    passes/code_generation/docc_paths_test.cpp
    test.cpp
)

# Now simply link against gtest or gtest_main as needed. Eg
add_executable(docc_llvm_pass_test ${TEST_FILES})
target_include_directories(docc_llvm_pass_test PRIVATE ./)
target_link_libraries(docc_llvm_pass_test gtest_main docc_llvm_pass_internal)

if(SDFG_ENABLE_COVERAGE)
target_compile_options(docc_llvm_pass_test PRIVATE
  -fprofile-instr-generate
  -fcoverage-mapping
)
target_link_options(docc_llvm_pass_test PRIVATE
  -fprofile-instr-generate
)
endif()

add_test(NAME docc_llvm_pass_test_test COMMAND docc_llvm_pass_test)

include(GoogleTest)
gtest_discover_tests(docc_llvm_pass_test)
