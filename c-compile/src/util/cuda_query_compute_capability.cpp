#include "docc/util/cuda_query_compute_capability.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <string>

namespace docc::util {

void clang_21_set_cuda_forward_compatible_options(
    compile::SrcFileCompilerBuilder& builder, compile::SrcFileCompilerBuilder& snippet_builder
) {
    builder.add_compile_option("--cuda-gpu-arch=sm_70");
    builder.add_compile_option("--cuda-include-ptx=all");
    snippet_builder.add_compile_option("--cuda-gpu-arch=sm_70");
    snippet_builder.add_compile_option("--cuda-include-ptx=all");
}

void clang_21_set_cuda_forward_compatible_options(std::vector<std::string>& compiler_args) {
    compiler_args.emplace_back("--cuda-gpu-arch=sm_70");
    compiler_args.emplace_back("--cuda-include-ptx=all");
}

void clang_21_set_cuda_specific_compute_cap(
    compile::SrcFileCompilerBuilder& builder, compile::SrcFileCompilerBuilder& snippet_builder, uint32_t sm_cap
) {
    auto str = std::to_string(sm_cap);
    builder.add_compile_option("--cuda-gpu-arch=sm_" + str);
    snippet_builder.add_compile_option("--cuda-gpu-arch=sm_" + str);
}

void clang_21_set_cuda_specific_compute_cap(std::vector<std::string>& compiler_args, uint32_t sm_cap) {
    auto str = std::to_string(sm_cap);
    compiler_args.emplace_back("--cuda-gpu-arch=sm_" + str);
}

} // namespace docc::util
