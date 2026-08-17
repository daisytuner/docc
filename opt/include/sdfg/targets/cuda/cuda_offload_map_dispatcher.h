#pragma once

#include "sdfg/analysis/analysis.h"
#include "sdfg/codegen/instrumentation/instrumentation_info.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/targets/gpu/gpu_offload_map_dispatcher.h"


namespace sdfg {
namespace cuda {

class CUDAMapDispatcher : public gpu::GPUOffloadMapDispatcher {
protected:
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

    codegen::LanguageExtension create_kernel_language_extension() override;

public:
    CUDAMapDispatcher(
        codegen::LanguageExtension& language_extension,
        StructuredSDFG& sdfg,
        analysis::AnalysisManager& analysis_manager,
        structured_control_flow::Map& node,
        codegen::InstrumentationPlan& instrumentation_plan,
        codegen::ArgCapturePlan& arg_capture_plan
    );

    void dispatch_kernel_launch_error_check(
        codegen::PrettyPrinter& stream, const codegen::LanguageExtension& language_extension, bool instrumented
    ) override;

    codegen::InstrumentationInfo instrumentation_info() const override = 0;
};

} // namespace cuda
} // namespace sdfg
