#pragma once

#include <string>
#include <vector>

#include "sdfg/analysis/analysis.h"
#include "sdfg/codegen/code_snippet_factory.h"
#include "sdfg/codegen/dispatchers/node_dispatcher.h"
#include "sdfg/codegen/instrumentation/instrumentation_info.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/targets/gpu/gpu_offload_schedule_type.h"
#include "sdfg/types/type.h"

namespace sdfg {
namespace gpu {

// Interface to handle everything cuda/rocm specific
class GPUOffloadDispatcherStrategy {
public:
    virtual ~GPUOffloadDispatcherStrategy() = default;

    /// Target-specific includes/declarations emitted into the kernel include header
    /// (device fp16/bf16 intrinsics, lib-dependency includes, ...).
    virtual void emit_target_header_declarations(codegen::PrettyPrinter& kernel_header_stream) = 0;

    virtual void dispatch_kernel_call(
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
    ) = 0;

    virtual codegen::TargetType get_instrumentation_kernel_target_type() const = 0;

    virtual int get_warp_size() const = 0;

    virtual codegen::LanguageExtension& create_kernel_language_extension() = 0;

    virtual bool is_device_pointer_storage(const types::StorageType& storage) const = 0;

    /// File extension for the kernel translation unit ("cu"/"rocm.cpp").
    virtual std::string kernel_file_extension() const = 0;
    /// File extension for the kernel's include header ("cu.h"/"rocm.h").
    virtual std::string kernel_header_file_extension() const = 0;

    /// Cross-lane XOR butterfly shuffle of `value` by `lane_mask` (CUDA/HIP __shfl_xor_sync).
    virtual std::string warp_shuffle_xor(const std::string& value, const std::string& lane_mask) const = 0;
};

/**
 * @brief Shared scaffolding for the GPU offload dispatchers (Map and Reduce).
 *
 * Owns the code common to every offloaded kernel: collecting kernel arguments and
 * scope variables, computing the launch grid/block sizes, emitting the kernel
 * declaration, opening the kernel translation unit + its include header, and
 * calling the (device-)target launch. The node-specific body and the
 * target-specific launch/language policy are supplied by subclasses.
 *
 * The target specific parts are handled by the DispatcherStrategy across Map and Reduce kernels
 */
class GPUOffloadBaseDispatcher : public codegen::NodeDispatcher {
protected:
    structured_control_flow::StructuredLoop& node_;
    std::unique_ptr<GPUOffloadDispatcherStrategy> strategy_;

    GPUOffloadBaseDispatcher(
        codegen::LanguageExtension& language_extension,
        StructuredSDFG& sdfg,
        analysis::AnalysisManager& analysis_manager,
        structured_control_flow::StructuredLoop& node,
        codegen::InstrumentationPlan& instrumentation_plan,
        codegen::ArgCapturePlan& arg_capture_plan,
        std::unique_ptr<GPUOffloadDispatcherStrategy> strategy
    );

    bool is_outermost_map(analysis::AnalysisManager& analysis_manager);

    void dispatch_header(
        codegen::PrettyPrinter& globals_stream,
        const std::string& kernel_name,
        std::vector<std::string>& arguments_declaration
    );

    void dispatch_kernel_preamble(
        codegen::PrettyPrinter& library_stream,
        const std::string& kernel_name,
        std::vector<std::string>& arguments_declaration
    );

    // Emit the includes for every kernel-local library dependency (e.g. rocwmma)
    // tracked by the nested snippet factory, into the kernel's include header.
    void emit_lib_dependency_includes(
        codegen::PrettyPrinter& kernel_header_stream, codegen::NestedCodeSnippetFactory& nested_snippet_factory
    );

    /// Node-specific preconditions checked before any code is emitted (default: none).
    virtual void validate_before_dispatch(analysis::AnalysisManager& analysis_manager) {}

    virtual void dispatch_kernel_body(
        codegen::NestedCodeSnippetFactory& kernel_snippet_factory,
        codegen::PrettyPrinter& kernel_source_stream,
        codegen::PrettyPrinter& kernel_header_stream,
        symbolic::Symbol indvar,
        std::vector<std::string>& scope_variables,
        symbolic::Expression& num_iterations
    ) = 0;

public:
    void dispatch_node(
        codegen::PrettyPrinter& main_stream,
        codegen::PrettyPrinter& globals_stream,
        codegen::CodeSnippetFactory& library_snippet_factory
    ) override;

    codegen::InstrumentationInfo instrumentation_info() const override;
};

} // namespace gpu
} // namespace sdfg
