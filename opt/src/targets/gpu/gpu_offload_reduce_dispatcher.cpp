#include "sdfg/targets/gpu/gpu_offload_reduce_dispatcher.h"

#include <iostream>
#include <sdfg/analysis/analysis.h>
#include <sdfg/analysis/loop_analysis.h>
#include <sdfg/analysis/users.h>
#include <sdfg/builder/structured_sdfg_builder.h>
#include <sdfg/codegen/dispatchers/sequence_dispatcher.h>
#include <sdfg/exceptions.h>
#include <sdfg/helpers/helpers.h>
#include <sdfg/serializer/json_serializer.h>
#include <sdfg/structured_control_flow/control_flow_node.h>
#include <sdfg/structured_control_flow/if_else.h>
#include <sdfg/structured_control_flow/map.h>
#include <sdfg/structured_control_flow/while.h>
#include <sdfg/symbolic/extreme_values.h>
#include <sdfg/symbolic/symbolic.h>
#include <sdfg/types/type.h>
#include <sdfg/visitor/structured_sdfg_visitor.h>
#include <string>
#include <unordered_set>


#include "sdfg/element.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/targets/gpu/gpu_map_utils.h"
#include "sdfg/targets/gpu/gpu_offload_schedule_type.h"
#include "sdfg/tiles/analysis/reduction_buffer_analysis.h"

#include <algorithm>
#include <sdfg/data_flow/access_node.h>
#include <sdfg/data_flow/memlet.h>
#include <sdfg/structured_control_flow/block.h>
#include <sdfg/structured_control_flow/reduce.h>
#include <sdfg/types/pointer.h>
#include <sdfg/types/scalar.h>
#include <sdfg/types/utils.h>

namespace sdfg {
namespace gpu {

namespace {

using structured_control_flow::ReductionOperation;

std::string op_tag(ReductionOperation op) {
    switch (op) {
        case ReductionOperation::Add:
            return "add";
        case ReductionOperation::Mul:
            return "mul";
        case ReductionOperation::Min:
            return "min";
        case ReductionOperation::Max:
            return "max";
    }
    throw InvalidSDFGException("GPUOffloadReduceDispatcher: unknown reduction operation");
}

// Identity element of the operator for the given primitive type, as a C literal.
std::string identity_literal(ReductionOperation op, types::PrimitiveType prim) {
    if (op == ReductionOperation::Add) {
        return "0";
    }
    if (op == ReductionOperation::Mul) {
        return "1";
    }
    if (types::is_floating_point(prim)) {
        return op == ReductionOperation::Min ? "INFINITY" : "-INFINITY";
    }

    // Bool is neither is_signed nor is_unsigned; OR(max)/AND(min) identities are false/true.
    if (prim == types::PrimitiveType::Bool) {
        return op == ReductionOperation::Min ? "1" : "0";
    }

    const size_t width = types::bit_width(prim);
    const bool is_unsigned = types::is_unsigned(prim);
    if (op == ReductionOperation::Min) {
        if (is_unsigned) {
            if (width == 8) return "UINT8_MAX";
            if (width == 16) return "UINT16_MAX";
            if (width == 32) return "UINT32_MAX";
            if (width == 64) return "UINT64_MAX";
        } else {
            if (width == 8) return "INT8_MAX";
            if (width == 16) return "INT16_MAX";
            if (width == 32) return "INT32_MAX";
            if (width == 64) return "INT64_MAX";
        }
    } else {
        if (is_unsigned) {
            return "0";
        }
        if (width == 8) return "INT8_MIN";
        if (width == 16) return "INT16_MIN";
        if (width == 32) return "INT32_MIN";
        if (width == 64) return "INT64_MIN";
    }
    throw InvalidSDFGException("GPUOffloadReduceDispatcher: unsupported integer width for min/max reduction");
}

// `cur OP val` as a C expression string, header-free (ternaries for min/max).
std::string combine_expr(ReductionOperation op, const std::string& a, const std::string& b) {
    switch (op) {
        case ReductionOperation::Add:
            return "(" + a + ") + (" + b + ")";
        case ReductionOperation::Mul:
            return "(" + a + ") * (" + b + ")";
        case ReductionOperation::Min:
            return "((" + a + ") < (" + b + ") ? (" + a + ") : (" + b + "))";
        case ReductionOperation::Max:
            return "((" + a + ") < (" + b + ") ? (" + b + ") : (" + a + "))";
    }
    throw InvalidSDFGException("GPUOffloadReduceDispatcher: unknown reduction operation");
}

// Whether the runtime provides a native atomicAdd overload for this primitive.
bool has_native_atomic_add(types::PrimitiveType prim) {
    const size_t width = types::bit_width(prim);
    if (types::is_floating_point(prim)) {
        return width == 32 || width == 64;
    }
    if (width == 32) {
        return true;
    }
    if (width == 64 && types::is_unsigned(prim)) {
        return true;
    }
    return false;
}

} // namespace

GPUOffloadReduceDispatcher::GPUOffloadReduceDispatcher(
    codegen::LanguageExtension& language_extension,
    StructuredSDFG& sdfg,
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::Reduce& node,
    codegen::InstrumentationPlan& instrumentation_plan,
    codegen::ArgCapturePlan& arg_capture_plan,
    std::unique_ptr<GPUOffloadDispatcherStrategy> strategy
)
    : GPUOffloadBaseDispatcher(
          language_extension, sdfg, analysis_manager, node, instrumentation_plan, arg_capture_plan, std::move(strategy)
      ),
      node_(node) {

      };


void GPUOffloadReduceDispatcher::validate_before_dispatch(analysis::AnalysisManager& analysis_manager) {
    auto& buffers = analysis_manager.get<tiles::ReductionBufferAnalysis>();
    reduction_buffers_.clear();
    for (const auto& reduction : node_.reductions()) {
        auto result = buffers.require(node_, reduction.container);
        if (!result.materialized) {
            throw InvalidSDFGException("GPU reduction requires materialized partial buffers");
        }
        reduction_buffers_.emplace(reduction.container, std::move(result));
    }
}


void GPUOffloadReduceDispatcher::dispatch_kernel_body(
    codegen::NestedCodeSnippetFactory& kernel_snippet_factory,
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

    // The partial-storage strategy fixes the combine mechanism; only these (scope, mechanism)
    // pairs are supported: warp+Register (shuffle), block+Shared (tree), block/grid+Global
    // (atomics). Others (e.g. a shared tree at grid scope, a shuffle off a warp) have no
    // lowering and must fail loudly rather than emit undefined code.
    ReduceStrategy strategy = gpu::ScheduleType_GPU_Offload::partial_storage(node_.schedule_type());
    if (strategy == ReduceStrategy::Register && target_level != TargetLevel::WARP) {
        throw InvalidSDFGException("GPUOffloadReduceDispatcher: Register partial storage is only valid at WARP level");
    }
    if (strategy == ReduceStrategy::Shared && !is_block_level(target_level)) {
        throw InvalidSDFGException("GPUOffloadReduceDispatcher: Shared partial storage is only valid at block levels");
    }
    if (strategy == ReduceStrategy::Global && target_level == TargetLevel::WARP) {
        throw InvalidSDFGException("GPUOffloadReduceDispatcher: Global partial storage is not valid at WARP level");
    }

    // A placed partials container (partial_container) is a shared buffer; it only applies to
    // the shared-tree mechanism, names a single reduction's buffer, and — when it already
    // exists in the SDFG — must be NV_Shared storage.
    std::string placed_partials = gpu::ScheduleType_GPU_Offload::partial_container(node_.schedule_type());
    if (!placed_partials.empty()) {
        if (strategy != ReduceStrategy::Shared) {
            throw InvalidSDFGException("GPUOffloadReduceDispatcher: partial_container requires Shared partial storage");
        }
        if (node_.reductions().size() != 1) {
            throw InvalidSDFGException(
                "GPUOffloadReduceDispatcher: partial_container names a single buffer but the reduce carries "
                "multiple reductions"
            );
        }
        if (sdfg_.exists(placed_partials) && !sdfg_.type(placed_partials).storage_type().is_nv_shared()) {
            throw InvalidSDFGException(
                "GPUOffloadReduceDispatcher: placed partials container '" + placed_partials + "' must be NV_Shared"
            );
        }
    }

    // Declare this level's reduction partials (registers for WARP/GRID, shared memory for
    // BLOCK) and initialize them to each operator's identity element.
    this->dispatch_reduction_declarations(
        kernel_language_extension, kernel_source_stream, kernel_snippet_factory, target_level
    );


    auto warp_size = strategy_->get_warp_size();
    if (target_level == TargetLevel::WARP) {
        std::string warp_dim = kernel_language_extension.expression(get_target_level_dim(target_level, warp_size));
        kernel_source_stream << "uint32_t num_warps = (" << kernel_language_extension.expression(symbolic::blockDim_x())
                             << " + " << warp_dim << " - 1) / " << warp_dim << ";" << std::endl;
        kernel_source_stream
            << "uint32_t warp_id = " << kernel_language_extension.expression(symbolic::threadIdx_x()) << " / "
            << kernel_language_extension.expression(get_target_level_dim(target_level, warp_size)) << ";" << std::endl;
        kernel_source_stream << "uint32_t lane = " << kernel_language_extension.expression(symbolic::threadIdx_x())
                             << " & ("
                             << kernel_language_extension.expression(get_target_level_dim(target_level, warp_size))
                             << " - 1);" << std::endl;
    }

    std::string coverage_dim = kernel_language_extension.expression(get_target_level_dim(target_level, warp_size));
    // For the WARP level each thread iterates sequentially over the warp-level
    // iteration space and accumulates into its per-thread register; the
    // cross-lane reduction is performed afterwards via __shfl_xor_sync over the
    // enclosing X_BLOCK lanes. The coverage loop therefore runs once per
    // iteration rather than once per warp_size.
    std::string coverage_count_dim = (target_level == TargetLevel::WARP) ? std::string("1") : coverage_dim;
    // Cast the ceil-div to int: blockDim/gridDim are unsigned, and CUDA 12.9's max()
    // overload set makes max(1, <unsigned>) ambiguous under clang-cuda.
    kernel_source_stream << "for (int " << coverage_loop_var << " = 0; " << coverage_loop_var << " < "
                         << "max(1, (int)((" << size << " + " << coverage_count_dim << " - 1) / " << coverage_count_dim
                         << ")); " << coverage_loop_var << "++) {" << std::endl;
    kernel_source_stream.changeIndent(+4);

    if (target_level == TargetLevel::WARP) {
        std::string indvar_name = indvar->get_name();
        auto x_block_parent = find_x_block_owning_warp_level(node_, analysis_manager_);
        if (!x_block_parent) {
            throw InvalidSDFGException("WARP level map must be nested within an X_BLOCK level map");
        }

        // Sequential per-thread iteration over the warp-level space.
        kernel_source_stream << "size_t " << indvar_name << " = " << kernel_language_extension.expression(node_.init())
                             << " + " << coverage_loop_var << " * "
                             << kernel_language_extension.expression(node_.stride()) << ";" << std::endl;
    } else {
        std::string target_level_idx_access = kernel_language_extension.expression(node_.stride()) + " * " +
                                              kernel_language_extension.expression(get_target_level_idx(target_level));

        if (target_level == TargetLevel::X_BLOCK && nested_warp_dim(node_, analysis_manager_)) {
            target_level_idx_access = kernel_language_extension.expression(get_target_level_idx(target_level));
        }

        // compute the effective indvar for this coverage loop iteration
        kernel_source_stream << "size_t " << indvar->get_name() << " = "
                             << kernel_language_extension.expression(node_.init()) << " + " << coverage_loop_var
                             << " * "
                             << kernel_language_extension.expression(get_target_level_dim(target_level, warp_size))
                             << " + " << target_level_idx_access << ";" << std::endl;
    }


    // Boundary Conditions
    if (!gpu::ScheduleType_GPU_Offload::nested_sync(node_.schedule_type())) {
        kernel_source_stream << "if (" << kernel_language_extension.expression(node_.condition()) << ") {" << std::endl;
        kernel_source_stream.changeIndent(+4);
    }


    // Body
    codegen::SequenceDispatcher dispatcher(
        kernel_language_extension, sdfg_, analysis_manager_, node_.root(), instrumentation_plan_, arg_capture_plan_
    );
    dispatcher.dispatch(kernel_source_stream, kernel_header_stream, kernel_snippet_factory);

    // Free managed scope variables
    for (auto& local : scope_variables) {
        auto& type = sdfg_.type(local);
        if (type.storage_type().deallocation() == types::StorageType::AllocationType::Managed) {
            kernel_source_stream << "free(" << local << ")";
            kernel_source_stream << ";" << std::endl;
        }
    }

    if (!gpu::ScheduleType_GPU_Offload::nested_sync(node_.schedule_type())) {
        kernel_source_stream.changeIndent(-4);
        kernel_source_stream << "}" << std::endl;
    }

    kernel_source_stream.changeIndent(-4);
    kernel_source_stream << "}" << std::endl;

    // Publish per-thread register partials to their shared slots once, before the combine.
    this->dispatch_reduction_publish(kernel_language_extension, kernel_source_stream, target_level);

    // Combine the per-thread / per-warp partials for this level into the accumulator.
    this->dispatch_reduction_combine(kernel_language_extension, kernel_source_stream, kernel_snippet_factory, target_level);
}

bool GPUOffloadReduceDispatcher::has_nested_warp_reduction(const std::string& container) {
    auto& loop_analysis = analysis_manager_.get<analysis::LoopAnalysis>();
    for (auto* loop : loop_analysis.descendants(&node_)) {
        auto* reduce = dynamic_cast<structured_control_flow::Reduce*>(loop);
        if (reduce == nullptr) {
            continue;
        }
        if (reduce->schedule_type().category() != structured_control_flow::ScheduleTypeCategory::Offloader) {
            continue;
        }
        if (gpu::ScheduleType_GPU_Offload::target_level(reduce->schedule_type()) != TargetLevel::WARP) {
            continue;
        }
        for (const auto& r : reduce->reductions()) {
            if (r.container == container) {
                return true;
            }
        }
    }
    return false;
}

bool GPUOffloadReduceDispatcher::has_enclosing_block_reduction(const std::string& container) {
    auto& loop_analysis = analysis_manager_.get<analysis::LoopAnalysis>();
    for (auto* loop : loop_analysis.ancestors(&node_)) {
        auto* reduce = dynamic_cast<structured_control_flow::Reduce*>(loop);
        if (reduce == nullptr) {
            continue;
        }
        if (reduce->schedule_type().category() != structured_control_flow::ScheduleTypeCategory::Offloader) {
            continue;
        }
        if (!is_block_level(gpu::ScheduleType_GPU_Offload::target_level(reduce->schedule_type()))) {
            continue;
        }
        for (const auto& r : reduce->reductions()) {
            if (r.container == container) {
                return true;
            }
        }
    }
    return false;
}

bool GPUOffloadReduceDispatcher::has_nested_block_reduction(const std::string& container) {
    auto& loop_analysis = analysis_manager_.get<analysis::LoopAnalysis>();
    for (auto* loop : loop_analysis.descendants(&node_)) {
        auto* reduce = dynamic_cast<structured_control_flow::Reduce*>(loop);
        if (reduce == nullptr) {
            continue;
        }
        if (reduce->schedule_type().category() != structured_control_flow::ScheduleTypeCategory::Offloader) {
            continue;
        }
        if (!is_block_level(gpu::ScheduleType_GPU_Offload::target_level(reduce->schedule_type()))) {
            continue;
        }
        for (const auto& r : reduce->reductions()) {
            if (r.container == container) {
                return true;
            }
        }
    }
    return false;
}

bool GPUOffloadReduceDispatcher::has_enclosing_grid_reduction(const std::string& container) {
    auto& loop_analysis = analysis_manager_.get<analysis::LoopAnalysis>();
    for (auto* loop : loop_analysis.ancestors(&node_)) {
        auto* reduce = dynamic_cast<structured_control_flow::Reduce*>(loop);
        if (reduce == nullptr) {
            continue;
        }
        if (reduce->schedule_type().category() != structured_control_flow::ScheduleTypeCategory::Offloader) {
            continue;
        }
        if (!is_grid_level(gpu::ScheduleType_GPU_Offload::target_level(reduce->schedule_type()))) {
            continue;
        }
        for (const auto& r : reduce->reductions()) {
            if (r.container == container) {
                return true;
            }
        }
    }
    return false;
}

bool GPUOffloadReduceDispatcher::block_result_collides_across_grid(const symbolic::Expression& index) {
    auto& loop_analysis = analysis_manager_.get<analysis::LoopAnalysis>();

    // Free symbols of the accumulator index.
    std::unordered_set<std::string> index_symbols;
    for (auto& atom : symbolic::atoms(index)) {
        index_symbols.insert(atom->get_name());
    }

    for (auto* loop : loop_analysis.ancestors(&node_)) {
        auto* struc_loop = dynamic_cast<structured_control_flow::StructuredLoop*>(loop);
        if (struc_loop == nullptr) {
            continue;
        }
        if (struc_loop->schedule_type().category() != structured_control_flow::ScheduleTypeCategory::Offloader) {
            continue;
        }
        if (!is_grid_level(gpu::ScheduleType_GPU_Offload::target_level(struc_loop->schedule_type()))) {
            continue;
        }
        // If this grid loop's induction variable does not index the accumulator, every
        // grid block (and coverage-loop iteration) writes the same global slot.
        if (index_symbols.find(struc_loop->indvar()->get_name()) == index_symbols.end()) {
            return true;
        }
    }
    return false;
}

std::string GPUOffloadReduceDispatcher::
    block_reduce_leader_condition(codegen::LanguageExtension& language_extension, const std::string& container) {
    auto& loop_analysis = analysis_manager_.get<analysis::LoopAnalysis>();
    std::vector<std::string> conditions;

    // This level's own axis leader.
    TargetLevel my_level = gpu::ScheduleType_GPU_Offload::target_level(node_.schedule_type());
    conditions.push_back("(" + language_extension.expression(get_target_level_idx(my_level)) + " == 0)");

    // Plus every nested block-level reduce of the same container: their axes have been
    // folded into flat-index 0, so only that slot holds the fully combined result.
    for (auto* loop : loop_analysis.descendants(&node_)) {
        auto* reduce = dynamic_cast<structured_control_flow::Reduce*>(loop);
        if (reduce == nullptr) {
            continue;
        }
        if (reduce->schedule_type().category() != structured_control_flow::ScheduleTypeCategory::Offloader) {
            continue;
        }
        TargetLevel level = gpu::ScheduleType_GPU_Offload::target_level(reduce->schedule_type());
        if (!is_block_level(level)) {
            continue;
        }
        bool reduces_container = false;
        for (const auto& r : reduce->reductions()) {
            if (r.container == container) {
                reduces_container = true;
                break;
            }
        }
        if (!reduces_container) {
            continue;
        }
        conditions.push_back("(" + language_extension.expression(get_target_level_idx(level)) + " == 0)");
    }

    return helpers::join(conditions, " && ");
}

std::string GPUOffloadReduceDispatcher::reduce_linear_thread_index(codegen::LanguageExtension& language_extension) {
    // threadIdx.x + threadIdx.y * BX + threadIdx.z * BX * BY, where BX/BY are the static
    // block dimensions from the schedule (== launched blockDim.x/y), so the flat layout is
    // a constant expression consistent with the statically sized shared buffer.
    symbolic::Expression bx = reduce_block_dim(TargetLevel::X_BLOCK);
    symbolic::Expression by = reduce_block_dim(TargetLevel::Y_BLOCK);
    symbolic::Expression lin = symbolic::
        add(symbolic::threadIdx_x(),
            symbolic::
                add(symbolic::mul(symbolic::threadIdx_y(), bx),
                    symbolic::mul(symbolic::threadIdx_z(), symbolic::mul(bx, by))));
    return language_extension.expression(lin);
}

std::string GPUOffloadReduceDispatcher::
    reduce_axis_stride(codegen::LanguageExtension& language_extension, TargetLevel target_level) {
    symbolic::Expression bx = reduce_block_dim(TargetLevel::X_BLOCK);
    symbolic::Expression by = reduce_block_dim(TargetLevel::Y_BLOCK);
    switch (target_level) {
        case TargetLevel::Y_BLOCK:
        case TargetLevel::Y_GRID:
            return language_extension.expression(bx);
        case TargetLevel::Z_BLOCK:
        case TargetLevel::Z_GRID:
            return language_extension.expression(symbolic::mul(bx, by));
        default:
            return language_extension.expression(symbolic::one());
    }
}

symbolic::Expression GPUOffloadReduceDispatcher::reduce_block_dim(TargetLevel block_level) {
    auto& loop_analysis = analysis_manager_.get<analysis::LoopAnalysis>();

    symbolic::Expression dim = symbolic::one();
    auto collect = [&](structured_control_flow::StructuredLoop* loop) {
        if (loop->schedule_type().category() != structured_control_flow::ScheduleTypeCategory::Offloader) {
            return;
        }
        if (gpu::ScheduleType_GPU_Offload::target_level(loop->schedule_type()) == block_level) {
            dim = gpu::ScheduleType_GPU_Offload::parallel_size(loop->schedule_type());
        }
    };

    collect(&node_);
    for (auto* loop : loop_analysis.ancestors(&node_)) {
        if (auto* struc_loop = dyn_cast<structured_control_flow::StructuredLoop*>(loop)) {
            collect(struc_loop);
        }
    }
    for (auto* loop : loop_analysis.descendants(&node_)) {
        if (auto* struc_loop = dyn_cast<structured_control_flow::StructuredLoop*>(loop)) {
            collect(struc_loop);
        }
    }
    return dim;
}

symbolic::Expression GPUOffloadReduceDispatcher::reduce_block_size_product() {
    return symbolic::
        mul(reduce_block_dim(TargetLevel::X_BLOCK),
            symbolic::mul(reduce_block_dim(TargetLevel::Y_BLOCK), reduce_block_dim(TargetLevel::Z_BLOCK)));
}

std::string GPUOffloadReduceDispatcher::partials_buffer_name(const std::string& container) {
    return reduction_buffers_.at(container).shared_buffer;
}

bool GPUOffloadReduceDispatcher::is_scalar_accumulator(const std::string& container) {
    return sdfg_.type(container).type_id() == types::TypeID::Scalar;
}

std::string GPUOffloadReduceDispatcher::
    reduce_base_slot(codegen::LanguageExtension& language_extension, const std::string& container) {
    // Which block axes are reduced for this container: this level's own axis plus every
    // nested block reduce of the same container. Zeroing these axes in the flat thread
    // index maps every thread of a reduced group onto the group's flat slot 0, where the
    // halving tree left the combined value.
    bool reduced_x = false, reduced_y = false, reduced_z = false;
    auto mark = [&](TargetLevel lvl) {
        if (lvl == TargetLevel::X_BLOCK) {
            reduced_x = true;
        } else if (lvl == TargetLevel::Y_BLOCK) {
            reduced_y = true;
        } else if (lvl == TargetLevel::Z_BLOCK) {
            reduced_z = true;
        }
    };
    TargetLevel my = gpu::ScheduleType_GPU_Offload::target_level(node_.schedule_type());
    if (is_block_level(my)) {
        mark(my);
    }
    auto& loop_analysis = analysis_manager_.get<analysis::LoopAnalysis>();
    for (auto* loop : loop_analysis.descendants(&node_)) {
        auto* reduce = dynamic_cast<structured_control_flow::Reduce*>(loop);
        if (reduce == nullptr) {
            continue;
        }
        if (reduce->schedule_type().category() != structured_control_flow::ScheduleTypeCategory::Offloader) {
            continue;
        }
        TargetLevel nested = gpu::ScheduleType_GPU_Offload::target_level(reduce->schedule_type());
        if (!is_block_level(nested)) {
            continue;
        }
        for (const auto& rr : reduce->reductions()) {
            if (rr.container == container) {
                mark(nested);
                break;
            }
        }
    }
    symbolic::Expression bx = reduce_block_dim(TargetLevel::X_BLOCK);
    symbolic::Expression by = reduce_block_dim(TargetLevel::Y_BLOCK);
    symbolic::Expression slot = symbolic::zero();
    if (!reduced_x) {
        slot = symbolic::add(slot, symbolic::threadIdx_x());
    }
    if (!reduced_y) {
        slot = symbolic::add(slot, symbolic::mul(symbolic::threadIdx_y(), bx));
    }
    if (!reduced_z) {
        slot = symbolic::add(slot, symbolic::mul(symbolic::threadIdx_z(), symbolic::mul(bx, by)));
    }
    return language_extension.expression(slot);
}

bool GPUOffloadReduceDispatcher::uses_register_partial(TargetLevel target_level, const std::string& container) {
    if (!is_block_level(target_level)) {
        return false;
    }
    if (gpu::ScheduleType_GPU_Offload::partial_storage(node_.schedule_type()) != ReduceStrategy::Shared) {
        return false;
    }
    // The register partial requires this to be the sole block level owning the container's
    // body: no nested warp reduce (which emits the body itself), no nested block reduce (which
    // owns the body and shadows the accumulator onto shared, leaving the register at identity
    // so the publish would clobber the shared result), and no enclosing block reduce (which
    // already declares the shared buffer, so a register partial here would redeclare it).
    return !has_nested_warp_reduction(container) && !has_nested_block_reduction(container) &&
           !has_enclosing_block_reduction(container);
}

void GPUOffloadReduceDispatcher::dispatch_reduction_publish(
    codegen::LanguageExtension& language_extension, codegen::PrettyPrinter& stream, TargetLevel target_level
) {
    std::string lin_tid = reduce_linear_thread_index(language_extension);
    for (const auto& r : node_.reductions()) {
        if (!uses_register_partial(target_level, r.container)) {
            continue;
        }
        const auto& buffer = reduction_buffers_.at(r.container);
        const auto& reg_name = buffer.private_buffer;
        std::string smem_name = partials_buffer_name(r.container);
        if (!buffer.multi_output) {
            stream << smem_name << "[" << lin_tid << "] = " << reg_name << "[0];" << std::endl;
        } else {
            std::string slot = "__daisy_reduce_slot_" + r.container;
            stream << "for (int " << slot << " = 0; " << slot << " < " << buffer.layout->extent << "; ++" << slot
                   << ") " << smem_name << "[(" << lin_tid << ") * " << buffer.layout->extent << " + " << slot
                   << "] = " << reg_name << "[" << slot << "];" << std::endl;
        }
    }
    // No sync here: the combine's leading __syncthreads() (emit_block_tree) makes every
    // thread's published slot visible before any neighbour slot is read.
}

void GPUOffloadReduceDispatcher::dispatch_reduction_declarations(
    codegen::LanguageExtension& language_extension,
    codegen::PrettyPrinter& stream,
    codegen::CodeSnippetFactory& library_snippet_factory,
    TargetLevel target_level
) {
    std::string lin_tid = reduce_linear_thread_index(language_extension);
    bool declared_shared = false;
    for (const auto& entry : node_.reductions()) {
        const auto& buffer = reduction_buffers_.at(entry.container);
        auto ctype = language_extension.primitive_type(*buffer.primitive);
        auto identity = identity_literal(entry.operation, *buffer.primitive);
        auto slot = "__daisy_reduce_slot_" + entry.container;
        if (*buffer.private_bytes) {
            stream << ctype << " " << buffer.private_buffer << "[" << buffer.layout->extent << "];" << std::endl;
            stream << "for (int " << slot << " = 0; " << slot << " < " << buffer.layout->extent << "; ++" << slot
                   << ") " << buffer.private_buffer << "[" << slot << "] = " << identity << ";" << std::endl;
        }
        if (!*buffer.shared_bytes) {
            continue;
        }
        stream << "__shared__ " << ctype << " " << buffer.shared_buffer << "["
               << *buffer.shared_bytes / *buffer.element_bytes << "];" << std::endl;
        if (!*buffer.private_bytes) {
            declared_shared = true;
            stream << "for (int " << slot << " = 0; " << slot << " < " << buffer.layout->extent << "; ++" << slot
                   << ") " << buffer.shared_buffer << "[(" << lin_tid << ") * " << buffer.layout->extent << " + "
                   << slot << "] = " << identity << ";" << std::endl;
        }
    }
    if (declared_shared) {
        stream << "__syncthreads();" << std::endl;
    }
}

std::string GPUOffloadReduceDispatcher::reduction_target(
    codegen::LanguageExtension& language_extension, const std::string& container, symbolic::Expression index
) {
    auto& loops = analysis_manager_.get<analysis::LoopAnalysis>();
    auto& buffer_analysis = analysis_manager_.get<tiles::ReductionBufferAnalysis>();
    std::vector<structured_control_flow::Reduce*> parents;
    for (auto* node : loops.ancestors(&node_)) {
        auto* parent = dyn_cast<structured_control_flow::Reduce*>(node);
        if (!parent || parent->schedule_type().category() != structured_control_flow::ScheduleTypeCategory::Offloader) {
            continue;
        }
        for (const auto& entry : parent->reductions()) {
            if (entry.container == container) {
                parents.push_back(parent);
            }
        }
    }
    std::sort(parents.begin(), parents.end(), [&](auto* left, auto* right) {
        return loops.ancestors(left).size() > loops.ancestors(right).size();
    });
    for (auto* parent : parents) {
        const auto& buffer = buffer_analysis.require(*parent, container);
        if (!buffer.private_buffer.empty()) {
            return buffer.private_buffer + "[" + language_extension.expression(buffer.layout->pack(index)) + "]";
        }
        if (*buffer.shared_bytes) {
            auto slot = symbolic::
                add(symbolic::mul(buffer.linear_thread_index, symbolic::integer(buffer.layout->extent)),
                    buffer.layout->pack(index));
            return buffer.shared_buffer + "[" + language_extension.expression(slot) + "]";
        }
    }
    auto primitive = *reduction_buffers_.at(container).primitive;
    return "reinterpret_cast<" + language_extension.primitive_type(primitive) + " *>(" + container + ")[" +
           language_extension.expression(index) + "]";
}

void GPUOffloadReduceDispatcher::dispatch_reduction_combine(
    codegen::LanguageExtension& language_extension,
    codegen::PrettyPrinter& stream,
    codegen::CodeSnippetFactory& library_snippet_factory,
    TargetLevel target_level
) {
    const ReduceStrategy strategy = gpu::ScheduleType_GPU_Offload::partial_storage(node_.schedule_type());

    // A reduce reduces only along its own axis; every other block dimension is an
    // independent "row". Shared slots are addressed by the flat thread index, so the
    // halving tree walks the reduce axis with the stride that separates its neighbours
    // in the flat layout, while the axis-local index bounds the loop and selects writers.
    std::string lin_tid = reduce_linear_thread_index(language_extension);

    std::string warp_size =
        language_extension.expression(get_target_level_dim(TargetLevel::WARP, strategy_->get_warp_size()));

    for (const auto& r : node_.reductions()) {
        if (strategy == ReduceStrategy::Shared && has_enclosing_block_reduction(r.container)) continue;
        const auto& buffer = reduction_buffers_.at(r.container);
        auto prim = *buffer.primitive;
        std::string ctype = language_extension.primitive_type(prim);
        std::string reg_name = buffer.private_buffer;
        std::string smem_name = partials_buffer_name(r.container);
        auto index = buffer.accumulator_index;
        const auto& layout = *buffer.layout;
        bool multi_output = buffer.multi_output;
        if (!multi_output) {
            reg_name += "[0]";
        }
        std::string shared_index = lin_tid;
        std::string shared_stride = "1";
        if (multi_output) {
            std::string slot = "__daisy_reduce_slot_" + r.container;
            stream << "for (int " << slot << " = 0; " << slot << " < " << layout.extent << "; ++" << slot << ") {"
                   << std::endl;
            stream.changeIndent(+4);
            reg_name += "[" + slot + "]";
            index = layout.unpack(symbolic::symbol(slot));
            shared_stride = std::to_string(layout.extent);
            shared_index = "(" + lin_tid + ") * " + shared_stride + " + " + slot;
        }
        std::string target = reduction_target(language_extension, r.container, index);

        // A scalar accumulator is a per-thread private reduction target; it has no global
        // slot and grid-level combine (cross-block atomics) is meaningless for it. The
        // nested-loop transform keeps such reduces off grid levels, so this only fires if an
        // SDFG bypasses it.
        if (is_scalar_accumulator(r.container) && is_grid_level(target_level)) {
            throw InvalidSDFGException(
                "GPUOffloadReduceDispatcher: scalar accumulator '" + r.container +
                "' cannot be reduced at a grid level; fold it into a block or warp level instead"
            );
        }

        if (strategy == ReduceStrategy::Register) {
            // Reduce the per-lane partials into every lane's register. The shuffle is read
            // once into a temporary before combining: emitting the __shfl_xor_sync inside a
            // combine expression (e.g. a min/max ternary) would duplicate it into divergent
            // branches, and on Volta+ a masked shuffle stalls until every named lane reaches
            // that exact instruction, deadlocking the warp.
            std::string other = "__daisy_reduce_shfl_" + r.container;
            stream << "for (int __daisy_reduce_mask = " << warp_size << " / 2; __daisy_reduce_mask > 0; "
                   << "__daisy_reduce_mask >>= 1) {" << std::endl;
            stream.setIndent(stream.indent() + 4);
            stream << ctype << " " << other << " = " << strategy_->warp_shuffle_xor(reg_name, "__daisy_reduce_mask")
                   << ";" << std::endl;
            stream << reg_name << " = " << combine_expr(r.operation, reg_name, other) << ";" << std::endl;
            stream.setIndent(stream.indent() - 4);
            stream << "}" << std::endl;

            if (has_enclosing_block_reduction(r.container)) {
                // Publish this warp's result into the enclosing block's per-thread shared
                // buffer at the lane-0 flat slot; every other slot keeps the operator
                // identity set at declaration time. The block-level per-thread reduction
                // tree then folds these partials (across all remaining block dimensions)
                // exactly as it folds ordinary per-thread partials.
                //
                // Combine (rather than overwrite) into the slot: when the enclosing block
                // owner's count exceeds its parallel_size its coverage loop runs multiple
                // tiles, re-executing this publish once per tile, and each tile's warp
                // result must fold into the identity-initialised slot instead of clobbering
                // the previous tiles. For a single tile combine(identity, reg) == reg.
                std::string slot = target;
                stream << "if (lane == 0) {" << std::endl;
                stream.setIndent(stream.indent() + 4);
                stream << slot << " = " << combine_expr(r.operation, slot, reg_name) << ";" << std::endl;
                stream.setIndent(stream.indent() - 4);
                stream << "}" << std::endl;
            } else if (is_scalar_accumulator(r.container)) {
                // Standalone warp scalar: the shuffle left the full result in every lane's
                // register, so broadcast it to each lane's private scalar. There is no global
                // slot to commit to (a scalar accumulator is thread-private).
                stream << r.container << " = " << reg_name << ";" << std::endl;
            } else {
                // No block level owns this container, so the per-warp result must reach
                // the global accumulator directly. The warp leader atomically merges its
                // register (cross-warp and cross-block combine) into acc[index].
                stream << "if (lane == 0) {" << std::endl;
                stream.setIndent(stream.indent() + 4);
                if (r.operation == ReductionOperation::Add && has_native_atomic_add(prim)) {
                    stream << "atomicAdd(&" << target << ", " << reg_name << ");" << std::endl;
                } else {
                    std::string type_tag = ctype;
                    std::replace(type_tag.begin(), type_tag.end(), ' ', '_');
                    std::string helper = "__daisy_reduce_combine_" + op_tag(r.operation) + "_" + type_tag;
                    stream << helper << "(&" << target << ", " << reg_name << ");" << std::endl;
                }
                stream.setIndent(stream.indent() - 4);
                stream << "}" << std::endl;
            }
        } else if (strategy == ReduceStrategy::Shared) {
            // Inner block levels only accumulate their per-thread partials into the single
            // shared buffer via the body; they emit no fold here. A block coverage loop of
            // an enclosing level iterates multiple times when count > parallel_size, so
            // folding a nested axis at the inner level would collapse the buffer between
            // coverage passes and corrupt slots that later passes still accumulate into.
            // Instead, the outermost block level folds every reduced block axis exactly
            // once, after all coverage-loop iterations have finished accumulating.
            // Emit one halving tree over a block axis. Neighbours are `half * stride` flat
            // slots apart (stride 1/bx/bx*by for x/y/z); ceil-half + bound guard handles
            // non-power-of-two sizes. A nested warp publishes its per-warp result into the
            // lane-0 slot and leaves every other slot at the operator identity, so the same
            // per-thread tree passes those partials through unchanged.
            auto emit_block_tree = [&](TargetLevel lvl, ReductionOperation op) {
                std::string a_dim = language_extension.expression(reduce_block_dim(lvl));
                std::string a_idx = language_extension.expression(get_target_level_idx(lvl));
                std::string a_stride = reduce_axis_stride(language_extension, lvl);
                std::string tag = gpu::to_string(lvl);
                std::string mvar = "__daisy_reduce_m_" + r.container + "_" + tag;
                std::string hvar = "__daisy_reduce_half_" + r.container + "_" + tag;
                std::string a = smem_name + "[" + shared_index + "]";
                std::string b = smem_name + "[" + shared_index + " + " + hvar + " * " + a_stride + " * " +
                                shared_stride + "]";
                stream << "__syncthreads();" << std::endl;
                stream << "for (int " << mvar << " = " << a_dim << "; " << mvar << " > 1; ) {" << std::endl;
                stream.setIndent(stream.indent() + 4);
                stream << "int " << hvar << " = (" << mvar << " + 1) / 2;" << std::endl;
                stream << "if (" << a_idx << " < " << mvar << " - " << hvar << ") {" << std::endl;
                stream.setIndent(stream.indent() + 4);
                stream << a << " = " << combine_expr(op, a, b) << ";" << std::endl;
                stream.setIndent(stream.indent() - 4);
                stream << "}" << std::endl;
                stream << "__syncthreads();" << std::endl;
                stream << mvar << " = " << hvar << ";" << std::endl;
                stream.setIndent(stream.indent() - 4);
                stream << "}" << std::endl;
            };

            // Fold every nested block axis of this container first, then this (outermost)
            // level's axis, so all block dimensions collapse into flat slot 0.
            auto& loop_analysis = analysis_manager_.get<analysis::LoopAnalysis>();
            for (auto* loop : loop_analysis.descendants(&node_)) {
                auto* reduce = dynamic_cast<structured_control_flow::Reduce*>(loop);
                if (reduce == nullptr) {
                    continue;
                }
                if (reduce->schedule_type().category() != structured_control_flow::ScheduleTypeCategory::Offloader) {
                    continue;
                }
                TargetLevel nested = gpu::ScheduleType_GPU_Offload::target_level(reduce->schedule_type());
                if (!is_block_level(nested)) {
                    continue;
                }
                for (const auto& rr : reduce->reductions()) {
                    if (rr.container == r.container) {
                        emit_block_tree(nested, rr.operation);
                        break;
                    }
                }
            }
            emit_block_tree(target_level, r.operation);

            // The outermost block level commits to the global accumulator. The writer is the
            // leader across every reduced block axis (this level plus all nested block
            // reduces), so exactly one thread per remaining (mapped) slot commits
            // smem[lin_tid]. The block result is always combined into the accumulator's
            // current value rather than overwriting it (see the non-colliding branch below).
            if (is_scalar_accumulator(r.container)) {
                // Scalar accumulator: broadcast the block-reduced value from flat slot 0 of
                // each reduced group to every thread's private scalar, so the by-name reads
                // after the reduce (e.g. an in-place pooling writeback that runs on every
                // thread of the reduced axis) all observe the combined result rather than
                // only the leader.
                stream << "__syncthreads();" << std::endl;
                stream << r.container << " = " << smem_name << "[" << reduce_base_slot(language_extension, r.container)
                       << "];" << std::endl;
            } else {
                std::string block_src = smem_name + "[" + shared_index + "]";
                std::string leader = block_reduce_leader_condition(language_extension, r.container);
                bool enclosed_by_reduction = has_enclosing_grid_reduction(r.container);
                bool collides = !enclosed_by_reduction && (multi_output || block_result_collides_across_grid(index));
                stream << "if (" << leader << ") {" << std::endl;
                stream.setIndent(stream.indent() + 4);
                if (collides) {
                    if (r.operation == ReductionOperation::Add && has_native_atomic_add(prim)) {
                        stream << "atomicAdd(&" << target << ", " << block_src << ");" << std::endl;
                    } else {
                        std::string type_tag = ctype;
                        std::replace(type_tag.begin(), type_tag.end(), ' ', '_');
                        std::string helper = "__daisy_reduce_combine_" + op_tag(r.operation) + "_" + type_tag;
                        stream << helper << "(&" << target << ", " << block_src << ");" << std::endl;
                    }
                } else {
                    // Always fold the block result into the accumulator's existing value
                    // instead of overwriting it. The partials are seeded with the operator
                    // identity, so when the source intends to overwrite, the SDFG initialises
                    // the accumulator before the reduce (target holds the identity) and
                    // combine(identity, block_src) == block_src. When the accumulator is a
                    // genuine live-in (read-modify-write, e.g. x[i] = x[i] + sum(...)), this
                    // preserves the incoming value rather than dropping it. The leader is the
                    // sole writer of this non-colliding slot, so no atomic is required.
                    stream << target << " = " << combine_expr(r.operation, target, block_src) << ";" << std::endl;
                }
                stream.setIndent(stream.indent() - 4);
                stream << "}" << std::endl;
            }
        } else if (strategy == ReduceStrategy::Global) {
            // A grid level nested inside another grid reduction of the same accumulator folds
            // into the enclosing level's shadowed *thread-local* register, not the real global
            // slot. Its target lives in local memory, where atomics are illegal (NVPTX cannot
            // select an atomic in address space 5) and unnecessary — only the outermost grid
            // level races across blocks. Combine plainly into that register; the outermost
            // level then atomically commits the folded result to global memory.
            if ((is_grid_level(target_level) || multi_output) && has_enclosing_grid_reduction(r.container)) {
                stream << target << " = " << combine_expr(r.operation, target, reg_name) << ";" << std::endl;
                if (multi_output) {
                    stream.changeIndent(-4);
                    stream << "}" << std::endl;
                }
                continue;
            }

            // Atomic commit of each thread's register straight to the global accumulator.
            // At a grid level with no nested block/warp reduce, the reduce body is replicated
            // verbatim across all block threads and each holds an identical partial; committing
            // all of them would multiply the result by blockDim, so a single thread commits.
            // When fed by a nested block/warp reduction, only the axis leaders hold a
            // non-identity value (every other thread holds the operator identity), so every
            // thread may commit. At a block level (block+Global storage) each thread instead
            // holds a *distinct* reduce-axis partial, so all of them must commit.
            bool fed_by_nested_reduction = has_nested_block_reduction(r.container) ||
                                           has_nested_warp_reduction(r.container);
            bool redundant_threads = is_grid_level(target_level) && !fed_by_nested_reduction;
            if (redundant_threads) {
                std::string leader = lin_tid + " == 0";
                if (multi_output) {
                    std::unordered_set<TargetLevel> mapped_axes;
                    for (auto* loop : analysis_manager_.get<analysis::LoopAnalysis>().descendants(&node_)) {
                        auto* map = dynamic_cast<structured_control_flow::Map*>(loop);
                        if (map && map->schedule_type().category() ==
                                       structured_control_flow::ScheduleTypeCategory::Offloader) {
                            mapped_axes.insert(gpu::ScheduleType_GPU_Offload::target_level(map->schedule_type()));
                        }
                    }
                    std::vector<std::string> conditions;
                    for (auto level : {TargetLevel::X_BLOCK, TargetLevel::Y_BLOCK, TargetLevel::Z_BLOCK}) {
                        if (!mapped_axes.contains(level)) {
                            conditions
                                .push_back("(" + language_extension.expression(get_target_level_idx(level)) + " == 0)");
                        }
                    }
                    leader = conditions.empty() ? "true" : helpers::join(conditions, " && ");
                }
                stream << "if (" << leader << ") {" << std::endl;
                stream.setIndent(stream.indent() + 4);
            }
            if (r.operation == ReductionOperation::Add && has_native_atomic_add(prim)) {
                stream << "atomicAdd(&" << target << ", " << reg_name << ");" << std::endl;
            } else {
                std::string type_tag = ctype;
                std::replace(type_tag.begin(), type_tag.end(), ' ', '_');
                std::string helper = "__daisy_reduce_combine_" + op_tag(r.operation) + "_" + type_tag;
                stream << helper << "(&" << target << ", " << reg_name << ");" << std::endl;
            }
            if (redundant_threads) {
                stream.setIndent(stream.indent() - 4);
                stream << "}" << std::endl;
            }
        }
        if (multi_output) {
            stream.changeIndent(-4);
            stream << "}" << std::endl;
        }
    }
}


} // namespace gpu
} // namespace sdfg
