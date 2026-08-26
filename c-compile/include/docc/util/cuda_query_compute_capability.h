#pragma once

#include <string>
#include <vector>
#include "docc/compile/src_file_compiler_builder.h"

namespace docc::util {

/// A CUDA compute capability together with the name(s) of the device(s) that
/// report it.
struct CudaComputeCapability {
    /// Compute capability in clang's integer convention (e.g. 8.6 -> 86,
    /// 12.0 -> 120).
    uint32_t compute_cap;
    /// Distinct device names that share this compute capability, useful for
    /// logging.
    std::vector<std::string> device_names;
};

/// Queries the CUDA compute capabilities of the GPUs available on this machine.
///
/// Internally this invokes `nvidia-smi --query-gpu=name,compute_cap` and parses
/// its output. Each device's compute capability is reported as an integer using
/// the same convention as clang (e.g. `8.6` -> `86`, `12.0` -> `120`).
///
/// The returned list is uniqued by compute capability (the names of all devices
/// sharing a capability are collected together) and sorted from the highest to
/// the lowest compute capability.
///
/// @return A descending, duplicate-free list of the available compute
///         capabilities, or an empty vector if no CUDA device could be queried
///         (e.g. `nvidia-smi` is not available).
std::vector<CudaComputeCapability> query_cuda_compute_capabilities();

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

} // namespace docc::util
