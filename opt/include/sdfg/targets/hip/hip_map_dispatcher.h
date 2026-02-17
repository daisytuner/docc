#pragma once

#include "sdfg/analysis/analysis.h"
#include "sdfg/codegen/dispatchers/node_dispatcher.h"
#include "sdfg/codegen/instrumentation/instrumentation_info.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/targets/gpu/gpu_map_utils.h"
#include "sdfg/targets/hip/hip.h"


namespace sdfg {
namespace hip {

class HIPMapDispatcher : public codegen::NodeDispatcher {
private:
    structured_control_flow::Map& node_;

    void dispatch_kernel_body(
        codegen::CodeSnippetFactory& library_snippet_factory,
        codegen::PrettyPrinter& globals_stream,
        symbolic::Symbol indvar,
        std::vector<std::string>& scope_variables,
        symbolic::Expression& num_iterations
    );

    void dispatch_header(
        codegen::PrettyPrinter& globals_stream,
        const std::string& kernel_name,
        std::vector<std::string>& arguments_declaration
    );

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
    );

    void dispatch_kernel_preamble(
        codegen::PrettyPrinter& library_stream,
        analysis::AnalysisManager& analysis_manager,
        const std::string& kernel_name,
        symbolic::SymbolSet& x_vars,
        symbolic::SymbolSet& y_vars,
        symbolic::SymbolSet& z_vars,
        std::vector<std::string>& arguments_declaration
    );

public:
    HIPMapDispatcher(
        codegen::LanguageExtension& language_extension,
        StructuredSDFG& sdfg,
        analysis::AnalysisManager& analysis_manager,
        structured_control_flow::Map& node,
        codegen::InstrumentationPlan& instrumentation_plan,
        codegen::ArgCapturePlan& arg_capture_plan
    );

    void dispatch_node(
        codegen::PrettyPrinter& main_stream,
        codegen::PrettyPrinter& globals_stream,
        codegen::CodeSnippetFactory& library_snippet_factory
    ) override;

    virtual codegen::InstrumentationInfo instrumentation_info() const override;
};

} // namespace hip
} // namespace sdfg
