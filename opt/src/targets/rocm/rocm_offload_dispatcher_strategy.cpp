#include "sdfg/targets/rocm/rocm_offload_dispatcher_strategy.h"

#include <string>
#include <vector>

#include <sdfg/analysis/analysis.h>
#include <sdfg/analysis/loop_analysis.h>
#include <sdfg/helpers/helpers.h>

#include "sdfg/targets/rocm/rocm.h"

namespace sdfg {
namespace rocm {

ROCMOffloadDispatcherStrategy::ROCMOffloadDispatcherStrategy(StructuredSDFG& sdfg)
    : kernel_language_extension_(sdfg) {};

codegen::LanguageExtension& ROCMOffloadDispatcherStrategy::create_kernel_language_extension() {
    return kernel_language_extension_;
}

void ROCMOffloadDispatcherStrategy::dispatch_kernel_call(
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
    main_stream << "hipLaunchKernelGGL(" << kernel_name << ", ";
    main_stream << "dim3((int)(" << host_language_ext.expression(num_blocks_x) << "), ";
    main_stream << "(int)(" << host_language_ext.expression(num_blocks_y) << "), ";
    main_stream << "(int)(" << host_language_ext.expression(num_blocks_z) << ")), ";
    main_stream << "dim3((int)(" << host_language_ext.expression(block_size_x) << "), ";
    main_stream << "(int)(" << host_language_ext.expression(block_size_y) << "), ";
    main_stream << "(int)(" << host_language_ext.expression(block_size_z) << ")), ";
    main_stream << "0, 0, "; // shared memory size and stream
    main_stream << helpers::join(arguments_device, ", ");
    main_stream << ")";
    main_stream << ";" << std::endl;

    // Synchronize / check launch errors
    this->dispatch_kernel_launch_error_check(main_stream, host_language_ext);

    main_stream.setIndent(main_stream.indent() - 4);
    main_stream << "}" << std::endl;
}

void ROCMOffloadDispatcherStrategy::emit_target_header_declarations(codegen::PrettyPrinter& kernel_header_stream) {
    // fp16/bf16 atomics (e.g. split-K accumulate) use the __half / __hip_bfloat16
    // struct overloads of atomicAdd, declared in these HIP headers.
    kernel_header_stream << "#include <hip/hip_fp16.h>" << std::endl;
    kernel_header_stream << "#include <hip/hip_bf16.h>" << std::endl;
}

void ROCMOffloadDispatcherStrategy::dispatch_kernel_launch_error_check(
    codegen::PrettyPrinter& stream, const codegen::LanguageExtension& language_extension
) {
    check_rocm_kernel_launch_errors(stream, language_extension);
}

int ROCMOffloadDispatcherStrategy::get_warp_size() const { return rocm_wavefront_size(); }

bool ROCMOffloadDispatcherStrategy::is_device_pointer_storage(const types::StorageType& storage) const {
    return storage.is_amd_generic();
}

std::string ROCMOffloadDispatcherStrategy::kernel_file_extension() const { return KERNEL_SNIPPET_FILE_EXT; }

std::string ROCMOffloadDispatcherStrategy::kernel_header_file_extension() const { return KERNEL_SNIPPET_HEADER_EXT; }

std::string ROCMOffloadDispatcherStrategy::warp_shuffle_xor(const std::string& value, const std::string& lane_mask)
    const {
    // HIP's __shfl_xor_sync requires a 64-bit membership mask (static_assert on
    // sizeof(mask) == 8), so the literal is always ULL-typed. The value covers
    // exactly the physical wavefront: the low 32 bits for wave32 (RDNA), all 64
    // bits for wave64 (CDNA/GCN).
    const std::string mask = rocm_wavefront_size() > 32 ? "0xffffffffffffffffULL" : "0x00000000ffffffffULL";
    return "__shfl_xor_sync(" + mask + ", " + value + ", " + lane_mask + ")";
}

codegen::TargetType ROCMOffloadDispatcherStrategy::get_instrumentation_kernel_target_type() const {
    return TargetType_ROCM;
}

} // namespace rocm
} // namespace sdfg
