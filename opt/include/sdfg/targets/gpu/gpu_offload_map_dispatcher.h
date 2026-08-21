#pragma once

#include "sdfg/analysis/analysis.h"
#include "sdfg/codegen/dispatchers/node_dispatcher.h"
#include "sdfg/codegen/instrumentation/instrumentation_info.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/type.h"

namespace sdfg {
namespace gpu {

class GPUOffloadMapDispatcher : public codegen::NodeDispatcher {
protected:
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

    bool is_outermost_map(analysis::AnalysisManager& analysis_manager);

    void dispatch_kernel_params(
        codegen::PrettyPrinter& main_stream,
        symbolic::Expression& num_blocks_x,
        symbolic::Expression& num_blocks_y,
        symbolic::Expression& num_blocks_z,
        symbolic::Expression& block_size_x,
        symbolic::Expression& block_size_y,
        symbolic::Expression& block_size_z
    );

    virtual void dispatch_kernel_call(
        codegen::PrettyPrinter& main_stream,
        const std::string& kernel_name,
        symbolic::Expression& num_blocks_x,
        symbolic::Expression& num_blocks_y,
        symbolic::Expression& num_blocks_z,
        symbolic::Expression& block_size_x,
        symbolic::Expression& block_size_y,
        symbolic::Expression& block_size_z,
        std::vector<std::string>& arguments_device
    ) = 0;

    void dispatch_kernel_preamble(
        codegen::PrettyPrinter& library_stream,
        analysis::AnalysisManager& analysis_manager,
        const std::string& kernel_name,
        std::vector<std::string>& arguments_declaration
    );

    virtual codegen::LanguageExtension& create_kernel_language_extension() = 0;

    virtual int get_warp_size() const = 0;

    virtual bool is_device_pointer_storage(const types::StorageType& storage) const = 0;

    /// File extension for the kernel translation unit ("cu"/"rocm.cpp").
    virtual std::string kernel_file_extension() const = 0;

public:
    GPUOffloadMapDispatcher(
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

    virtual void dispatch_kernel_launch_error_check(
        codegen::PrettyPrinter& stream, const codegen::LanguageExtension& language_extension, bool instrumented
    ) = 0;

    virtual codegen::InstrumentationInfo instrumentation_info() const override = 0;
};

} // namespace gpu
} // namespace sdfg
