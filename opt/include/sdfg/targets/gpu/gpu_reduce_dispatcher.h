#pragma once

#include <memory>
#include <string>
#include <vector>

#include "sdfg/analysis/analysis.h"
#include "sdfg/codegen/dispatchers/node_dispatcher.h"
#include "sdfg/codegen/instrumentation/instrumentation_info.h"
#include "sdfg/codegen/language_extension.h"
#include "sdfg/codegen/utils.h"
#include "sdfg/structured_control_flow/reduce.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/targets/gpu/gpu_types.h"
#include "sdfg/types/type.h"

namespace sdfg {
namespace gpu {

/**
 * @brief Atomics-based GPU code generator for @ref structured_control_flow::Reduce
 *
 * This is the target-agnostic core shared by the CUDA and ROCm reduce
 * dispatchers. A Reduce loop carries one or more associative/commutative
 * reductions (@ref structured_control_flow::ReductionInfo) into a loop-invariant
 * accumulator; this dispatcher handles a Reduce whose own schedule is a GPU
 * schedule, i.e. the reduction dimension is parallelized across threads on its
 * configured GPU dimension.
 *
 * Correctness strategy ("atomics baseline"):
 * - The reduction dimension is parallelized with a *grid-stride* loop on the
 *   schedule's dimension: thread `t` starts at `init + t*stride` and advances
 *   by `num_threads*stride`. This is correct for any launch geometry (even a
 *   single thread), so the lowering composes whether the Reduce is top-level or
 *   nested inside a GPU map.
 * - For every reduction the accumulator is *privatized*: a thread-local partial
 *   is declared and initialized to the operator's identity, and the accumulator
 *   container is shadowed to point at that private storage so the in-body
 *   combine (`acc = acc OP x`) accumulates into the register.
 * - After the loop, each thread merges its private partial into the real
 *   device-resident accumulator with an atomic combine (native `atomicAdd`
 *   where available, otherwise a compare-and-swap loop). Associativity and
 *   commutativity (guaranteed by the Reduce node) make the merge order-free.
 *
 * Subclasses supply only the backend-specific policy (language extension,
 * device-pointer storage, kernel launch syntax and includes).
 */
class GPUReduceDispatcher : public codegen::NodeDispatcher {
protected:
    structured_control_flow::Reduce& node_;

    // --- Backend policy hooks -------------------------------------------------

    /// Schedule value string identifying this backend's GPU schedule ("CUDA"/"ROCM").
    virtual std::string schedule_value() const = 0;

    /// Target type used for instrumentation info.
    virtual codegen::TargetType target_type() const = 0;

    /// Create the backend device-code language extension (CUDA/HIP).
    virtual std::unique_ptr<codegen::LanguageExtension> create_device_language_extension() const = 0;

    /// True if @p storage designates a device-resident generic pointer for this backend.
    virtual bool is_device_pointer_storage(const types::StorageType& storage) const = 0;

    /// File extension for the kernel translation unit ("cu"/"rocm.cpp").
    virtual std::string kernel_file_extension() const = 0;

    /// Emit the backend include directives required by the generated kernel.
    virtual void emit_kernel_includes(codegen::CodeSnippetFactory& library_snippet_factory) const = 0;

    /// Emit any extra preamble at the top of the kernel translation unit.
    virtual void emit_library_preamble(codegen::PrettyPrinter& library_stream) const = 0;

    /// Emit the host-side kernel launch.
    virtual void emit_kernel_call(
        codegen::PrettyPrinter& main_stream,
        const std::string& kernel_name,
        symbolic::Expression& num_blocks,
        symbolic::Expression& block_size,
        std::vector<std::string>& arguments_device
    ) = 0;

    // --- Shared implementation ------------------------------------------------

    /**
     * @brief True if this Reduce is nested inside an enclosing GPU kernel
     *        (an ancestor Map with this backend's GPU schedule).
     */
    bool is_nested_in_gpu_kernel();

    void dispatch_header(
        codegen::PrettyPrinter& stream, const std::string& kernel_name, std::vector<std::string>& arguments_declaration
    );

    /**
     * @brief Emit the privatize + grid-stride reduce loop + atomic merge into
     *        the given (device) stream, parallelized on @p dimension.
     *
     * @param declare_scope_variables  When true (top-level kernel), declare the
     *        loop-local scope variables here. When false (nested), the enclosing
     *        map already declared them.
     */
    void dispatch_reduce_core(
        codegen::CodeSnippetFactory& library_snippet_factory,
        codegen::PrettyPrinter& library_stream,
        GPUDimension dimension,
        bool declare_scope_variables,
        std::vector<std::string>& scope_variables
    );

    /**
     * @brief Emit the device-side helper(s) and the atomic merge statement that
     *        combines a thread's private partial into the global accumulator.
     */
    void dispatch_atomic_merge(
        codegen::PrettyPrinter& library_stream,
        codegen::CodeSnippetFactory& library_snippet_factory,
        const structured_control_flow::ReductionInfo& reduction
    );

public:
    GPUReduceDispatcher(
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

    codegen::InstrumentationInfo instrumentation_info() const override;
};

} // namespace gpu
} // namespace sdfg
