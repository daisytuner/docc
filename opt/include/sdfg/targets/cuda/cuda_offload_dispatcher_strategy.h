#pragma once

#include <string>
#include <vector>

#include "codegen/cuda_language_extension.h"
#include "sdfg/codegen/language_extension.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/targets/gpu/gpu_offload_base_dispatcher.h"


namespace sdfg {
namespace cuda {

/**
 * @brief CUDA specialization of @ref gpu::GPUOffloadMapDispatcher.
 *
 * Supplies the CUDA-specific policy for the shared coverage-loop offload lowering:
 * the device language extension, the `<<<...>>>` kernel launch and launch-error check.
 */
class CUDAOffloadDispatcherStrategy : public gpu::GPUOffloadDispatcherStrategy {
public:
    void dispatch_kernel_call(
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
    ) override;

    void emit_target_header_declarations(codegen::PrettyPrinter& kernel_header_stream) override;
    void emit_target_global_includes(codegen::CodeSnippetFactory& library_snippet_factory) override;

    CUDAOffloadDispatcherStrategy();

    void dispatch_kernel_launch_error_check(
        codegen::PrettyPrinter& stream, const codegen::LanguageExtension& language_extension
    );

    int get_warp_size() const override;

    bool is_device_pointer_storage(const types::StorageType& storage) const override;

    constexpr static const char* KERNEL_SNIPPET_FILE_EXT = "cu";
    constexpr static const char* KERNEL_SNIPPET_HEADER_EXT = "cu.h";

    std::string kernel_file_extension() const override;
    std::string kernel_header_file_extension() const override;

    std::string warp_shuffle_xor(const std::string& value, const std::string& lane_mask) const override;

    codegen::TargetType get_instrumentation_kernel_target_type() const override;

    std::unique_ptr<codegen::LanguageExtension>
    create_kernel_language_extension(StructuredSDFG& sdfg, const StructuredLoop& loop) override;

    bool is_matching_kernel_language_extension(
        const StructuredSDFG& sdfg, const StructuredLoop& loop, codegen::LanguageExtension& language_extension
    ) override;
};

} // namespace cuda
} // namespace sdfg
