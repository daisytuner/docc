#pragma once

#include "sdfg/analysis/analysis.h"
#include "sdfg/codegen/dispatchers/node_dispatcher.h"
#include "sdfg/codegen/instrumentation/instrumentation_info.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/targets/gpu/gpu_offload_schedule_type.h"
#include "sdfg/types/type.h"


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

    // Whether a BLOCK-level reduction nested below this node reduces @p container. When a
    // grid reduce owns such a nested block reduce of the same container, its register is
    // populated only by the block's axis leaders (via the block combine), so every other
    // thread holds the operator identity and the grid commit must stay unguarded.
    bool has_nested_block_reduction(const std::string& container);

    // Whether a BLOCK-level reduction enclosing this node reduces @p container. A WARP
    // reduce relies on such an enclosing block reduce to own the shared buffer its
    // per-warp partials are published into; without one, the warp must flush directly
    // to the global accumulator instead.
    bool has_enclosing_block_reduction(const std::string& container);

    // Whether a GRID-level reduction enclosing this node reduces @p container. When a
    // block reduce is nested under such a grid reduce, its per-block result is folded
    // into the grid's register accumulator, which persists across the grid coverage
    // loop; the block level must therefore combine (accumulate) into that target rather
    // than overwrite it, so partials from every coverage-loop iteration are retained.
    bool has_enclosing_grid_reduction(const std::string& container);

    // Whether this block reduce's per-block result written to @p index collides across the
    // grid: true when an enclosing grid-level offloaded loop's induction variable does not
    // appear in the accumulator index, so multiple grid blocks / coverage-loop iterations
    // target the same global slot and must combine atomically rather than overwrite.
    bool block_result_collides_across_grid(const symbolic::Expression& index);

    // Predicate (as a C expression) selecting the single thread that commits the folded
    // result of @p container to global memory: the leader across every reduced block axis,
    // i.e. this level's axis index and every nested block-reduce level's axis index are 0.
    // Any remaining (mapped) block dimensions stay free so each of their slots is written.
    std::string block_reduce_leader_condition(codegen::LanguageExtension& language_extension, const std::string& container);

    // Linearized flat thread index within the block: threadIdx.x + threadIdx.y * blockDim.x
    // + threadIdx.z * blockDim.x * blockDim.y. Used to address per-thread shared slots so
    // that every thread of a multi-dimensional block owns a distinct accumulator entry.
    std::string reduce_linear_thread_index(codegen::LanguageExtension& language_extension);

    // Element stride, in the flat shared layout, between consecutive threads along the
    // reduce node's own axis: 1 for x, blockDim.x for y, blockDim.x * blockDim.y for z.
    std::string reduce_axis_stride(codegen::LanguageExtension& language_extension, TargetLevel target_level);

    // Static size of a single block dimension, taken from the corresponding block level's
    // schedule parallel_size in the enclosing/nested loop nest (1 if that level is absent).
    // The launched blockDim equals these constants, so using them instead of the dynamic
    // blockDim.x/y/z keeps the flat thread-index layout a compile-time constant expression
    // that matches the statically sized shared buffer.
    symbolic::Expression reduce_block_dim(TargetLevel block_level);

    // Compile-time product of all block-level parallel sizes enclosing/within this node
    // (x * y * z). Sizes the per-thread shared buffer for a multi-dimensional block.
    symbolic::Expression reduce_block_size_product();

    // Name of the shared partials buffer for @p container: the schedule's
    // partial_container property when set, else the invented __daisy_reduce_smem_<c>.
    std::string partials_buffer_name(const std::string& container);

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

    // Whether the body of a block-level shared reduction for @p container accumulates into
    // a per-thread register partial (published to shared once after the coverage loop)
    // instead of read-modify-writing shared each iteration. Applies only to a standalone
    // block level that solely owns the body (no enclosing block reduce, no nested block
    // reduce, no nested warp reduce). This keeps the hot accumulation in a register so the
    // FMA chain is not serialized through shared memory; the existing shared tree/shuffle
    // combine is unchanged.
    bool uses_register_partial(TargetLevel target_level, const std::string& container);

    // Publish each register partial to its shared slot once, after the coverage loop and
    // before the combine (a single st.shared per thread instead of one per element).
    void dispatch_reduction_publish(
        codegen::LanguageExtension& language_extension, codegen::PrettyPrinter& stream, TargetLevel target_level
    );

    virtual codegen::LanguageExtension& create_kernel_language_extension() = 0;

    virtual int get_warp_size() const = 0;

    virtual bool is_device_pointer_storage(const types::StorageType& storage) const = 0;

    /// File extension for the kernel translation unit ("cu"/"rocm.cpp").
    virtual std::string kernel_file_extension() const = 0;

    /// Cross-lane XOR butterfly shuffle of `value` by `lane_mask` (CUDA/HIP __shfl_xor_sync).
    virtual std::string warp_shuffle_xor(const std::string& value, const std::string& lane_mask) const = 0;

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
