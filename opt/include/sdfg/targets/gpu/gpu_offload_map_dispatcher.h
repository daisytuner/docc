#pragma once

#include <string>
#include <vector>

#include "sdfg/analysis/analysis.h"
#include "sdfg/codegen/code_snippet_factory.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/targets/gpu/gpu_offload_base_dispatcher.h"

namespace sdfg {
namespace gpu {

/**
 * @brief Map node policy for the GPU offload dispatcher.
 *
 * Supplies the coverage-loop kernel body for an offloaded @ref structured_control_flow::Map.
 * The kernel scaffolding and the device-target launch policy come from
 * @ref GPUOffloadBaseDispatcher (inherited virtually so a concrete dispatcher can mix this
 * with a Cuda/Rocm kernel policy).
 */
class GPUOffloadMapDispatcher : public GPUOffloadBaseDispatcher {
protected:
    void dispatch_kernel_body(
        codegen::NestedCodeSnippetFactory& kernel_snippet_factory,
        codegen::PrettyPrinter& kernel_source_stream,
        codegen::PrettyPrinter& kernel_header_stream,
        symbolic::Symbol indvar,
        std::vector<std::string>& scope_variables,
        symbolic::Expression& num_iterations
    ) override;

public:
    GPUOffloadMapDispatcher(
        codegen::LanguageExtension& language_extension,
        StructuredSDFG& sdfg,
        analysis::AnalysisManager& analysis_manager,
        structured_control_flow::Map& node,
        codegen::InstrumentationPlan& instrumentation_plan,
        codegen::ArgCapturePlan& arg_capture_plan,
        std::unique_ptr<GPUOffloadDispatcherStrategy> strategy
    );
};

} // namespace gpu
} // namespace sdfg
