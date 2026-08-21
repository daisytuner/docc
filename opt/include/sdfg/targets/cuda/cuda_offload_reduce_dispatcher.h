#pragma once

#include <string>
#include <vector>

#include "sdfg/analysis/analysis.h"
#include "sdfg/codegen/instrumentation/instrumentation_info.h"
#include "sdfg/codegen/language_extensions/cuda_language_extension.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/targets/gpu/gpu_offload_reduce_dispatcher.h"


namespace sdfg {
namespace cuda {

/**
 * @brief CUDA specialization of @ref gpu::GPUOffloadReduceDispatcher.
 *
 * Supplies the CUDA-specific policy for the multi-level reduction offload lowering:
 * the device language extension, the `<<<...>>>` kernel launch and launch-error check.
 */
class CUDAOffloadReduceDispatcher : public gpu::GPUOffloadReduceDispatcher {
protected:
    codegen::CUDALanguageExtension kernel_language_extension_;

    codegen::LanguageExtension& create_kernel_language_extension() override;

    void dispatch_kernel_call(
        codegen::PrettyPrinter& main_stream,
        const std::string& kernel_name,
        symbolic::Expression& num_blocks_x,
        symbolic::Expression& num_blocks_y,
        symbolic::Expression& num_blocks_z,
        symbolic::Expression& block_size_x,
        symbolic::Expression& block_size_y,
        symbolic::Expression& block_size_z,
        std::vector<std::string>& arguments_device
    ) override;

public:
    CUDAOffloadReduceDispatcher(
        codegen::LanguageExtension& language_extension,
        StructuredSDFG& sdfg,
        analysis::AnalysisManager& analysis_manager,
        structured_control_flow::Reduce& node,
        codegen::InstrumentationPlan& instrumentation_plan,
        codegen::ArgCapturePlan& arg_capture_plan
    );

    void dispatch_kernel_launch_error_check(
        codegen::PrettyPrinter& stream, const codegen::LanguageExtension& language_extension, bool instrumented
    ) override;

    codegen::InstrumentationInfo instrumentation_info() const override;

    int get_warp_size() const override;

    bool is_device_pointer_storage(const types::StorageType& storage) const override;

    std::string kernel_file_extension() const override;

    std::string warp_shuffle_xor(const std::string& value, const std::string& lane_mask) const override;
};

} // namespace cuda
} // namespace sdfg
