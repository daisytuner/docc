#pragma once

#include "sdfg/analysis/analysis.h"
#include "sdfg/codegen/dispatchers/node_dispatcher.h"
#include "sdfg/codegen/instrumentation/instrumentation_info.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/targets/gpu/gpu_schedule_type.h"


namespace sdfg {
namespace gpu {

class GPUOffloadReduceDispatcher : public codegen::NodeDispatcher {
protected:
    structured_control_flow::Reduce& node_;

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

    // Whether a WARP-level reduction nested below this node reduces @p container,
    // in which case this (block) level combines per-warp partials instead of per-thread.
    bool has_nested_warp_reduction(const std::string& container);

    // Whether a BLOCK-level reduction enclosing this node reduces @p container. A WARP
    // reduce relies on such an enclosing block reduce to own the shared buffer its
    // per-warp partials are published into; without one, the warp must flush directly
    // to the global accumulator instead.
    bool has_enclosing_block_reduction(const std::string& container);

    // Linearized flat thread index within the block: threadIdx.x + threadIdx.y * blockDim.x
    // + threadIdx.z * blockDim.x * blockDim.y. Used to address per-thread shared slots so
    // that every thread of a multi-dimensional block owns a distinct accumulator entry.
    std::string reduce_linear_thread_index(codegen::LanguageExtension& language_extension);

    // Element stride, in the flat shared layout, between consecutive threads along the
    // reduce node's own axis: 1 for x, blockDim.x for y, blockDim.x * blockDim.y for z.
    std::string reduce_axis_stride(codegen::LanguageExtension& language_extension, TargetLevel target_level);

    // Compile-time product of all block-level parallel sizes enclosing/within this node
    // (x * y * z). Sizes the per-thread shared buffer for a multi-dimensional block.
    symbolic::Expression reduce_block_size_product();

    void dispatch_reduction_declarations(
        codegen::LanguageExtension& language_extension,
        codegen::PrettyPrinter& stream,
        codegen::CodeSnippetFactory& library_snippet_factory,
        TargetLevel target_level
    );

    void dispatch_reduction_shadow(
        codegen::LanguageExtension& language_extension, codegen::PrettyPrinter& stream, TargetLevel target_level
    );

    void dispatch_reduction_combine(
        codegen::LanguageExtension& language_extension,
        codegen::PrettyPrinter& stream,
        codegen::CodeSnippetFactory& library_snippet_factory,
        TargetLevel target_level
    );

    virtual codegen::LanguageExtension& create_kernel_language_extension() = 0;

    virtual int get_warp_size() const = 0;

public:
    GPUOffloadReduceDispatcher(
        codegen::LanguageExtension& language_extension,
        StructuredSDFG& sdfg,
        analysis::AnalysisManager& analysis_manager,
        structured_control_flow::Reduce& node,
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
