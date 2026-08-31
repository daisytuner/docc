#include "sdfg/passes/scheduler/cuda_offload_scheduler.h"

#include "sdfg/passes/dataflow/dead_data_elimination.h"
#include "sdfg/passes/dataflow/memlet_simplification.h"
#include "sdfg/passes/offloading/cuda_library_node_transfer_extraction_pass.h"
#include "sdfg/passes/offloading/gpu_nested_offload_pass.h"
#include "sdfg/passes/offloading/gpu_nested_parallelization_pass.h"
#include "sdfg/passes/scheduler/loop_scheduler.h"
#include "sdfg/passes/structured_control_flow/dead_cfg_elimination.h"
#include "sdfg/passes/symbolic/symbol_propagation.h"
#include "sdfg/passes/tiling_pass.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/structured_control_flow/reduce.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/targets/gpu/gpu_map_utils.h"
#include "sdfg/transformations/offloading/cuda_offload_transform.h"

namespace sdfg {
namespace passes {
namespace scheduler {

SchedulerAction CUDAOffloadScheduler::find(
    builder::StructuredSDFGBuilder& builder,
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::StructuredLoop& loop,
    bool offload_unknown_sizes
) {
    if (dyn_cast<structured_control_flow::Map*>(&loop)) {
        return APPLY;
    }

    auto& loop_analysis = analysis_manager.get<analysis::LoopAnalysis>();
    auto loop_info = loop_analysis.loop_info(&loop);

    if (loop_info.loopnest_index == -1 || loop_info.num_maps <= 1 || loop_info.is_perfectly_nested ||
        loop_info.has_side_effects) {
        return NEXT;
    } else {
        return CHILDREN;
    }
}

SchedulerAction CUDAOffloadScheduler::find(
    builder::StructuredSDFGBuilder& builder,
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::While& loop,
    bool offload_unknown_sizes
) {
    auto& loop_analysis = analysis_manager.get<analysis::LoopAnalysis>();
    auto loop_info = loop_analysis.loop_info(&loop);
    if (loop_info.loopnest_index == -1 || loop_info.has_side_effects) {
        return NEXT;
    } else {
        return CHILDREN;
    }
}

symbolic::Integer CUDAOffloadScheduler::get_parallel_size(structured_control_flow::StructuredLoop& loop) {
    auto iteration_count = loop.num_iterations();
    if (SymEngine::is_a<SymEngine::Integer>(*iteration_count)) {
        return symbolic::integer(SymEngine::rcp_static_cast<const SymEngine::Integer>(iteration_count)->as_int());
    } else {
        return symbolic::integer(128);
    }
}

bool CUDAOffloadScheduler::can_apply_schedule(
    builder::StructuredSDFGBuilder& builder,
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::StructuredLoop& loop,
    bool offload_unknown_sizes
) {
    cuda::CUDAOffloadTransform
        cuda_transform(loop, get_parallel_size(loop), gpu::TargetLevel::X_GRID, offload_unknown_sizes);
    return cuda_transform.can_be_applied(builder, analysis_manager);
}

void CUDAOffloadScheduler::apply_schedule(
    builder::StructuredSDFGBuilder& builder,
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::StructuredLoop& loop,
    bool offload_unknown_sizes
) {
    if (recorder_ != nullptr) {
        recorder_->apply<cuda::CUDAOffloadTransform>(
            builder,
            analysis_manager,
            false,
            loop,
            get_parallel_size(loop),
            gpu::TargetLevel::X_GRID,
            offload_unknown_sizes
        );
    } else {
        cuda::CUDAOffloadTransform
            cuda_transform(loop, get_parallel_size(loop), gpu::TargetLevel::X_GRID, offload_unknown_sizes);
        cuda_transform.apply(builder, analysis_manager);
    }
}

void CUDAOffloadScheduler::pre_schedule(
    builder::StructuredSDFGBuilder& builder,
    analysis::AnalysisManager& analysis_manager,
    std::vector<structured_control_flow::StructuredLoop*>& applicable_loops
) {
    if (applicable_loops.empty()) {
        return;
    }

    // Split loops by number of perfectly nested loops: single loop vs. 2 or more loops.
    std::vector<structured_control_flow::StructuredLoop*> single_loops;
    std::vector<structured_control_flow::StructuredLoop*> nested_loops;
    for (auto* loop : applicable_loops) {
        if (gpu::perfectly_nested_depth(loop) <= 1) {
            single_loops.push_back(loop);
        } else {
            nested_loops.push_back(loop);
        }
    }

    // Tile the single loops to expose an outer parallel loop.
    TilingPass tiling_pass(single_loops, 128);
    tiling_pass.run(builder, analysis_manager);
    analysis_manager.invalidate_all();

    passes::SymbolPropagation symbol_propagation_pass;
    symbol_propagation_pass.run(builder, analysis_manager);
    passes::DeadDataElimination ddead_pass;
    ddead_pass.run(builder, analysis_manager);
    passes::DeadCFGElimination dcfg_pass;
    dcfg_pass.run(builder, analysis_manager);
    passes::MemletSimplificationPass subset_simplification_pass;
    subset_simplification_pass.run(builder, analysis_manager);
    analysis_manager.invalidate_all();

    applicable_loops.clear();
    for (auto* loop : nested_loops) {
        applicable_loops.push_back(loop);
    }
    for (auto* loop : single_loops) {
        applicable_loops.push_back(loop);
    }
}

void CUDAOffloadScheduler::post_schedule(
    builder::StructuredSDFGBuilder& builder,
    analysis::AnalysisManager& analysis_manager,
    std::vector<structured_control_flow::StructuredLoop*>& scheduled_loops
) {
    std::vector<structured_control_flow::StructuredLoop*> gpu_loops;
    for (auto* loop : scheduled_loops) {
        if (auto* sloop = dyn_cast<structured_control_flow::StructuredLoop*>(loop)) {
            gpu_loops.push_back(sloop);
        }
    }

    if (!gpu_loops.empty()) {
        GPUNestedOffloadPass nested_offload_pass(gpu_loops, GPUTarget::CUDA);
        nested_offload_pass.run(builder, analysis_manager);
        analysis_manager.invalidate_all();
    }

    cuda::CudaLibraryNodeTransferExtractionPass transfer_extraction_pass;
    transfer_extraction_pass.run(builder, analysis_manager);
    analysis_manager.invalidate_all();
}

std::unordered_set<ScheduleTypeCategory> CUDAOffloadScheduler::compatible_types() {
    return {ScheduleTypeCategory::None};
}

} // namespace scheduler
} // namespace passes
} // namespace sdfg
