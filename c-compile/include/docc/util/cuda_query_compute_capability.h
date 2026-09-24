#pragma once

#include <string>
#include <vector>
#include "docc/compile/src_file_compiler_builder.h"
#include "sdfg/targets/cuda/cuda_arch.h"

namespace docc::util {

/// Set what options work best for any CUDA card (set an old SM version, try to make it JIT ready)
void clang_21_set_cuda_forward_compatible_options(
    compile::SrcFileCompilerBuilder& builder, compile::SrcFileCompilerBuilder& snippet_builder
);

/// What above does, but for legacy llvm frontend compiler args
void clang_21_set_cuda_forward_compatible_options(std::vector<std::string>& compiler_args);

/// Only need to build for this specific compute capability
void clang_21_set_cuda_specific_compute_cap(
    compile::SrcFileCompilerBuilder& builder, compile::SrcFileCompilerBuilder& snippet_builder, uint32_t sm_cap
);

/// Same as above, for legacy llvm frontend
void clang_21_set_cuda_specific_compute_cap(std::vector<std::string>& compiler_args, uint32_t sm_cap);


[[deprecated("moved to sdfg::gpu::cuda::query_cuda_compute_capabilities()")]]
inline std::vector<sdfg::gpu::cuda::CudaComputeCapability> query_cuda_compute_capabilities() {
    return sdfg::gpu::cuda::query_cuda_compute_capabilities();
}
} // namespace docc::util
