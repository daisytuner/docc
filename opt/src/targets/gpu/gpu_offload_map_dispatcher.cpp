#include "sdfg/targets/gpu/gpu_offload_map_dispatcher.h"

#include <iostream>
#include <limits>
#include <sdfg/analysis/analysis.h>
#include <sdfg/analysis/assumptions_analysis.h>
#include <sdfg/analysis/loop_analysis.h>
#include <sdfg/builder/structured_sdfg_builder.h>
#include <sdfg/codegen/dispatchers/sequence_dispatcher.h>
#include <sdfg/exceptions.h>
#include <sdfg/helpers/helpers.h>
#include <sdfg/structured_control_flow/control_flow_node.h>
#include <sdfg/structured_control_flow/map.h>
#include <sdfg/symbolic/extreme_values.h>
#include <sdfg/symbolic/symbolic.h>
#include <sdfg/types/scalar.h>
#include <sdfg/types/type.h>
#include <string>
#include <unordered_map>
#include <unordered_set>


#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/targets/gpu/gpu_map_utils.h"
#include "sdfg/targets/gpu/gpu_offload_schedule_type.h"

namespace sdfg {
namespace gpu {

GPUOffloadMapDispatcher::GPUOffloadMapDispatcher(
    codegen::LanguageExtension& language_extension,
    StructuredSDFG& sdfg,
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::Map& node,
    codegen::InstrumentationPlan& instrumentation_plan,
    codegen::ArgCapturePlan& arg_capture_plan,
    std::unique_ptr<GPUOffloadDispatcherStrategy> strategy
)
    : GPUOffloadBaseDispatcher(
          language_extension, sdfg, analysis_manager, node, instrumentation_plan, arg_capture_plan, std::move(strategy)
      ) {}

void GPUOffloadMapDispatcher::dispatch_kernel_body(
    codegen::NestedCodeSnippetFactory& library_snippet_factory,
    codegen::PrettyPrinter& kernel_source_stream,
    codegen::PrettyPrinter& kernel_header_stream,
    symbolic::Symbol indvar,
    std::vector<std::string>& scope_variables,
    symbolic::Expression& num_iterations
) {
    codegen::LanguageExtension& kernel_language_extension = strategy_->create_kernel_language_extension();
    if (is_outermost_map(analysis_manager_)) {
        // Declare and optionally allocate scope variables
        for (auto& local : scope_variables) {
            if (local.starts_with("__daisy_gpu")) {
                continue;
            }
            std::string val = kernel_language_extension.declaration(local, sdfg_.type(local), false, true);
            if (!val.empty()) {
                kernel_source_stream << val;
                kernel_source_stream << ";" << std::endl;
            }
            auto& type = sdfg_.type(local);
            if (type.storage_type().allocation() == types::StorageType::AllocationType::Managed) {
                kernel_source_stream << local << " = ";
                kernel_source_stream
                    << "malloc(" << kernel_language_extension.expression(type.storage_type().allocation_size()) << ")";
                kernel_source_stream << ";" << std::endl;
            }
        }
    }
    // generate coverage loop
    TargetLevel target_level = gpu::ScheduleType_GPU_Offload::target_level(node_.schedule_type());
    std::string coverage_loop_var = "__daisy_gpu_coverage_loop_" + gpu::to_string(target_level);
    std::string size = kernel_language_extension.expression(node_.num_iterations());
    auto warp_size = strategy_->get_warp_size();
    if (target_level == TargetLevel::WARP) {
        // Thread helpers (num_warps/warp_id/lane) are bounded by the hardware
        // block-size cap (<=1024), so uint32 always suffices.
        const std::string type_thread = kernel_language_extension.primitive_type(types::PrimitiveType::UInt32);
        std::string warp_dim = kernel_language_extension.expression(get_target_level_dim(target_level, warp_size));
        kernel_source_stream << type_thread << " num_warps = ("
                             << kernel_language_extension.expression(symbolic::blockDim_x()) << " + " << warp_dim
                             << " - 1) / " << warp_dim << ";" << std::endl;
        kernel_source_stream
            << type_thread << " warp_id = " << kernel_language_extension.expression(symbolic::threadIdx_x()) << " / "
            << kernel_language_extension.expression(get_target_level_dim(target_level, warp_size)) << ";" << std::endl;
        kernel_source_stream << type_thread
                             << " lane = " << kernel_language_extension.expression(symbolic::threadIdx_x()) << " & ("
                             << kernel_language_extension.expression(get_target_level_dim(target_level, warp_size))
                             << " - 1);" << std::endl;
    }

    // The indvar keeps its declared (container) width.
    const auto& indvar_dtype = sdfg_.type(indvar->get_name());
    const std::string type_index = kernel_language_extension.primitive_type(indvar_dtype.primitive_type());

    std::string coverage_dim = kernel_language_extension.expression(get_target_level_dim(target_level, warp_size));
    // For the WARP level each thread iterates sequentially over the warp-level
    // iteration space (the cross-lane reduction is performed by the reduce
    // dispatcher via __shfl_xor_sync over the enclosing X_BLOCK lanes), so the
    // coverage loop must run once per iteration rather than once per warp_size.
    std::string coverage_count_dim = (target_level == TargetLevel::WARP) ? std::string("1") : coverage_dim;

    // Resolve the map trip and parallel size to constants (via assumptions for the
    // symbolic min(N, base+block) tile trips) so the coverage-loop bound and the
    // boundary guard become compile-time constants. Emitting the raw symbolic trip
    // makes every worker iteration recompute a min/max/imod chain, which dominates
    // a multi-tile persistent (Stream-K) walk; a constant bound lets it fold away.
    long long resolved_trip = -1;
    if (target_level != TargetLevel::WARP) {
        auto trip = node_.num_iterations();
        if (!trip.is_null() && SymEngine::is_a<SymEngine::Integer>(*trip)) {
            resolved_trip = SymEngine::rcp_static_cast<const SymEngine::Integer>(trip)->as_int();
        } else if (!trip.is_null()) {
            auto& aa = analysis_manager_.get<analysis::AssumptionsAnalysis>();
            const auto& assums = aa.get(node_.root(), true);
            const auto& params = aa.parameters();
            auto mx = symbolic::maximum(trip, params, assums, true);
            auto mn = symbolic::minimum(trip, params, assums, true);
            if (!mx.is_null() && !mn.is_null() && symbolic::eq(mx, mn) && SymEngine::is_a<SymEngine::Integer>(*mx)) {
                resolved_trip = SymEngine::rcp_static_cast<const SymEngine::Integer>(mx)->as_int();
            }
        }
    }
    long long psize_int = -1;
    {
        auto psize = gpu::ScheduleType_GPU_Offload::parallel_size(node_.schedule_type());
        if (!psize.is_null() && SymEngine::is_a<SymEngine::Integer>(*psize)) {
            psize_int = SymEngine::rcp_static_cast<const SymEngine::Integer>(psize)->as_int();
        }
    }

    // Coverage counter type + bound. When the trip and parallel size are known
    // constants the wave count is exact, so emit it as a literal (the loop folds to a
    // single pass) and size the counter to that count -- a tiny int32 in practice.
    // Otherwise fall back to the indvar's type (coverage in [0, ceil(trip/dim)), so
    // coverage <= trip <= what the indvar holds) and cast both max() operands to it:
    // blockDim/gridDim are unsigned and the bare 1 is signed, so max(1, <expr>) is
    // otherwise ambiguous under clang-cuda.
    std::string coverage_bound;
    std::string type_coverage;
    if (resolved_trip >= 0 && psize_int > 0) {
        long long cov = (resolved_trip + psize_int - 1) / psize_int;
        if (cov < 1) cov = 1;
        coverage_bound = std::to_string(cov);
        type_coverage = kernel_language_extension.primitive_type(
            cov <= std::numeric_limits<int>::max() ? types::PrimitiveType::Int32 : types::PrimitiveType::Int64
        );
    } else {
        type_coverage = type_index;
        coverage_bound = "max((" + type_coverage + ")1, (" + type_coverage + ")((" + size + " + " + coverage_count_dim +
                         " - 1) / " + coverage_count_dim + "))";
    }
    kernel_source_stream << "for (" << type_coverage << " " << coverage_loop_var << " = 0; " << coverage_loop_var
                         << " < " << coverage_bound << "; " << coverage_loop_var << "++) {" << std::endl;
    kernel_source_stream.setIndent(kernel_source_stream.indent() + 4);

    std::string indvar_name = indvar->get_name();
    if (target_level == TargetLevel::WARP) {
        auto x_block_parent = find_x_block_owning_warp_level(node_, analysis_manager_);
        if (!x_block_parent) {
            throw InvalidSDFGException("WARP level map must be nested within an X_BLOCK level map");
        }

        // Sequential per-thread iteration over the warp-level space.
        kernel_source_stream << type_index << " " << indvar_name << " = "
                             << kernel_language_extension.expression(node_.init()) << " + " << coverage_loop_var
                             << " * " << kernel_language_extension.expression(node_.stride()) << ";" << std::endl;
    } else {
        // 0-based parallel index across this dimension: `coverage` sweeps of `dim`
        // units plus this thread/block's index. The map's induction variable is
        // then init + stride * parallel_index, so the stride applies to BOTH the
        // coverage and the index terms (tiled offload maps have stride != 1).
        std::string dim_expr = kernel_language_extension.expression(get_target_level_dim(target_level, warp_size));
        std::string idx_expr = kernel_language_extension.expression(get_target_level_idx(target_level));
        std::string parallel_index = coverage_loop_var + " * " + dim_expr + " + " + idx_expr;

        std::string offset;
        if (target_level == TargetLevel::X_BLOCK && nested_warp_dim(node_, analysis_manager_)) {
            // Warp handles the sub-stride; the block index is used directly.
            offset = "(" + parallel_index + ")";
        } else {
            offset = kernel_language_extension.expression(node_.stride()) + " * (" + parallel_index + ")";
        }

        kernel_source_stream << type_index << " " << indvar_name << " = "
                             << kernel_language_extension.expression(node_.init()) << " + " << offset << ";"
                             << std::endl;
    }


    // Boundary Conditions
    // The coverage loop maps indvar to `init + stride*(coverage*dim + idx)`. A
    // partial final wave (some threads overshoot the trip) exists ONLY when the
    // trip count is not a whole multiple of the parallel size. When
    // `trip % parallel_size == 0` every generated index is in range, so the loop
    // condition holds for all threads and the per-thread guard is redundant --
    // dropping it lets the body's cooperative loads/stores vectorize instead of
    // sitting under a predicate. WARP level is excluded (its sequential coverage
    // is handled separately). Sound: only fires on compile-time-constant trips.
    bool guard_redundant =
        (target_level != TargetLevel::WARP && resolved_trip >= 0 && psize_int > 0 && resolved_trip % psize_int == 0);
    bool emit_guard = !gpu::ScheduleType_GPU_Offload::nested_sync(node_.schedule_type()) && !guard_redundant;
    if (emit_guard) {
        kernel_source_stream << "if (" << kernel_language_extension.expression(node_.condition()) << ") {" << std::endl;
        kernel_source_stream.setIndent(kernel_source_stream.indent() + 4);
    }

    // Body
    codegen::SequenceDispatcher dispatcher(
        kernel_language_extension, sdfg_, analysis_manager_, node_.root(), instrumentation_plan_, arg_capture_plan_
    );
    dispatcher.dispatch(kernel_source_stream, kernel_header_stream, library_snippet_factory);

    // Free managed scope variables
    for (auto& local : scope_variables) {
        auto& type = sdfg_.type(local);
        if (type.storage_type().deallocation() == types::StorageType::AllocationType::Managed) {
            kernel_source_stream << "free(" << local << ")";
            kernel_source_stream << ";" << std::endl;
        }
    }

    if (emit_guard) {
        kernel_source_stream.setIndent(kernel_source_stream.indent() - 4);
        kernel_source_stream << "}" << std::endl;
    }

    kernel_source_stream.setIndent(kernel_source_stream.indent() - 4);
    kernel_source_stream << "}" << std::endl;
}


} // namespace gpu
} // namespace sdfg
