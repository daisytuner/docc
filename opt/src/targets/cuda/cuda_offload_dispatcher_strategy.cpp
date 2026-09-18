#include "sdfg/targets/cuda/cuda_offload_dispatcher_strategy.h"

#include <string>
#include <vector>

#include <sdfg/analysis/analysis.h>
#include <sdfg/analysis/loop_analysis.h>
#include <sdfg/helpers/helpers.h>

#include "sdfg/targets/cuda/cuda.h"
#include "sdfg/targets/gpu/gpu_offload_base_dispatcher.h"

namespace sdfg {
namespace cuda {

CUDAOffloadDispatcherStrategy::CUDAOffloadDispatcherStrategy(StructuredSDFG& sdfg)
    : gpu::GPUOffloadDispatcherStrategy(), kernel_language_extension_(sdfg) {};

codegen::LanguageExtension& CUDAOffloadDispatcherStrategy::create_kernel_language_extension() {
    return kernel_language_extension_;
}

void CUDAOffloadDispatcherStrategy::dispatch_kernel_call(
    codegen::PrettyPrinter& main_stream,
    const std::string& kernel_name,
    codegen::LanguageExtension& host_language_ext,
    symbolic::Expression& num_blocks_x,
    symbolic::Expression& num_blocks_y,
    symbolic::Expression& num_blocks_z,
    symbolic::Expression& block_size_x,
    symbolic::Expression& block_size_y,
    symbolic::Expression& block_size_z,
    std::vector<std::string>& arguments_device
) {
    main_stream << "{" << std::endl;
    main_stream.setIndent(main_stream.indent() + 4);

    // Kernel launch
    main_stream << kernel_name << "<<<";
    main_stream << "dim3((int)(" << host_language_ext.expression(num_blocks_x) << "), ";
    main_stream << "(int)(" << host_language_ext.expression(num_blocks_y) << "), ";
    main_stream << "(int)(" << host_language_ext.expression(num_blocks_z) << ")), ";
    main_stream << "dim3((int)(" << host_language_ext.expression(block_size_x) << "), ";
    main_stream << "(int)(" << host_language_ext.expression(block_size_y) << "), ";
    main_stream << "(int)(" << host_language_ext.expression(block_size_z) << "))";
    main_stream << ">>>(";
    main_stream << helpers::join(arguments_device, ", ");
    main_stream << ")";
    main_stream << ";" << std::endl;

    // Synchronize / check launch errors
    this->dispatch_kernel_launch_error_check(main_stream, host_language_ext);

    main_stream.setIndent(main_stream.indent() - 4);
    main_stream << "}" << std::endl;
}

void CUDAOffloadDispatcherStrategy::dispatch_kernel_launch_error_check(
    codegen::PrettyPrinter& stream, const codegen::LanguageExtension& language_extension
) {
    check_cuda_kernel_launch_errors(stream, language_extension, false);
}

int CUDAOffloadDispatcherStrategy::get_warp_size() const { return CUDA_WARP_SIZE; }

bool CUDAOffloadDispatcherStrategy::is_device_pointer_storage(const types::StorageType& storage) const {
    return storage.is_nv_generic();
}

std::string CUDAOffloadDispatcherStrategy::kernel_file_extension() const { return KERNEL_SNIPPET_FILE_EXT; }
std::string CUDAOffloadDispatcherStrategy::kernel_header_file_extension() const { return KERNEL_SNIPPET_HEADER_EXT; }

void CUDAOffloadDispatcherStrategy::emit_target_header_declarations(codegen::PrettyPrinter& kernel_header_stream) {
    // fp16/bf16 atomics (e.g. split-K accumulate) use the __half / __nv_bfloat16
    // struct overloads of atomicAdd, declared in these headers.
    kernel_header_stream << "#include <cuda_fp16.h>" << std::endl;
    kernel_header_stream << "#include <cuda_bf16.h>" << std::endl;
    // cp.async pipeline primitives for software-pipelined cooperative copies.
    kernel_header_stream << "#include <cuda_pipeline.h>" << std::endl;
}

std::string CUDAOffloadDispatcherStrategy::warp_shuffle_xor(const std::string& value, const std::string& lane_mask)
    const {
    return "__shfl_xor_sync(0xffffffff, " + value + ", " + lane_mask + ")";
}

codegen::TargetType CUDAOffloadDispatcherStrategy::get_instrumentation_kernel_target_type() const {
    return TargetType_CUDA;
}

} // namespace cuda
} // namespace sdfg
