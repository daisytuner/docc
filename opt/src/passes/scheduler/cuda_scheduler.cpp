#include "sdfg/passes/scheduler/cuda_scheduler.h"

#include "sdfg/passes/collapse_pass.h"
#include "sdfg/passes/dataflow/dead_data_elimination.h"
#include "sdfg/passes/dataflow/memlet_simplification.h"
#include "sdfg/passes/offloading/cuda_library_node_transfer_extraction_pass.h"
#include "sdfg/passes/offloading/gpu_loop_reordering_pass.h"
#include "sdfg/passes/offloading/gpu_nested_parallelization_pass.h"
#include "sdfg/passes/offloading/gpu_tiling_pass.h"
#include "sdfg/passes/scheduler/loop_scheduler.h"
#include "sdfg/passes/structured_control_flow/dead_cfg_elimination.h"
#include "sdfg/passes/symbolic/symbol_propagation.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/transformations/offloading/cuda_transform.h"

namespace sdfg {
namespace passes {
namespace scheduler {

SchedulerAction CUDAScheduler::find(
    builder::StructuredSDFGBuilder& builder,
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::StructuredLoop& loop,
    bool offload_unknown_sizes
) {
    if (dynamic_cast<structured_control_flow::Map*>(&loop)) {
        return NEXT;
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

SchedulerAction CUDAScheduler::find(
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

bool CUDAScheduler::can_apply_schedule(
    builder::StructuredSDFGBuilder& builder,
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::StructuredLoop& loop,
    bool offload_unknown_sizes
) {
    auto* map = dynamic_cast<structured_control_flow::Map*>(&loop);
    if (!map) {
        return false;
    }
    cuda::CUDATransform cuda_transform(*map, 32, offload_unknown_sizes);
    return cuda_transform.can_be_applied(builder, analysis_manager);
}

void CUDAScheduler::apply_schedule(
    builder::StructuredSDFGBuilder& builder,
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::StructuredLoop& loop,
    bool offload_unknown_sizes
) {
    auto* map = dynamic_cast<structured_control_flow::Map*>(&loop);
    cuda::CUDATransform cuda_transform(*map, 32, offload_unknown_sizes);
    cuda_transform.apply(builder, analysis_manager);
}

void CUDAScheduler::pre_schedule(
    builder::StructuredSDFGBuilder& builder,
    analysis::AnalysisManager& analysis_manager,
    std::vector<structured_control_flow::StructuredLoop*>& applicable_loops
) {
    std::vector<structured_control_flow::Map*> applicable_maps;
    for (auto* loop : applicable_loops) {
        if (auto* map = dynamic_cast<structured_control_flow::Map*>(loop)) {
            applicable_maps.push_back(map);
        }
    }

    if (applicable_maps.empty()) {
        return;
    }

    CollapsePass collapse_pass(applicable_maps, 2);
    collapse_pass.run(builder, analysis_manager);
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
    for (auto* map : applicable_maps) {
        applicable_loops.push_back(map);
    }
}

void CUDAScheduler::post_schedule(
    builder::StructuredSDFGBuilder& builder,
    analysis::AnalysisManager& analysis_manager,
    std::vector<structured_control_flow::StructuredLoop*>& scheduled_loops
) {
    std::vector<structured_control_flow::Map*> gpu_maps;
    for (auto* loop : scheduled_loops) {
        if (auto* map = dynamic_cast<structured_control_flow::Map*>(loop)) {
            gpu_maps.push_back(map);
        }
    }

    if (gpu_maps.empty()) {
        return;
    }

    GPULoopReorderingPass reordering_pass(gpu_maps);
    reordering_pass.run(builder, analysis_manager);
    analysis_manager.invalidate_all();

    GPUNestedParallelizationPass nested_pass(gpu_maps, GPUTarget::CUDA, 8);
    nested_pass.run(builder, analysis_manager);
    analysis_manager.invalidate_all();

    GPUTilingPass tiling_pass(gpu_maps, 8);
    tiling_pass.run(builder, analysis_manager);
    analysis_manager.invalidate_all();

    cuda::CudaLibraryNodeTransferExtractionPass transfer_extraction_pass;
    transfer_extraction_pass.run(builder, analysis_manager);
    analysis_manager.invalidate_all();
}

std::unordered_set<ScheduleTypeCategory> CUDAScheduler::compatible_types() { return {ScheduleTypeCategory::None}; }

} // namespace scheduler
} // namespace passes
} // namespace sdfg
