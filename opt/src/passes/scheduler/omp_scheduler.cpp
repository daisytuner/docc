#include "sdfg/passes/scheduler/omp_scheduler.h"

#include "sdfg/loops/passes/collapse_pass.h"
#include "sdfg/loops/passes/trivial_loop_elimination_pass.h"
#include "sdfg/loops/transformations/vectorize_transform.h"
#include "sdfg/passes/dataflow/dead_data_elimination.h"
#include "sdfg/passes/dataflow/memlet_simplification.h"
#include "sdfg/passes/structured_control_flow/dead_cfg_elimination.h"
#include "sdfg/passes/symbolic/symbol_propagation.h"
#include "sdfg/transformations/omp_transform.h"

namespace sdfg {
namespace passes {
namespace scheduler {

SchedulerAction OMPScheduler::find(
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

SchedulerAction OMPScheduler::find(
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

bool OMPScheduler::can_apply_schedule(
    builder::StructuredSDFGBuilder& builder,
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::StructuredLoop& loop,
    bool offload_unknown_sizes
) {
    auto* map = dyn_cast<structured_control_flow::Map*>(&loop);
    if (!map) {
        return false;
    }
    transformations::OMPTransform omp_transform(*map);
    return omp_transform.can_be_applied(builder, analysis_manager);
}

void OMPScheduler::apply_schedule(
    builder::StructuredSDFGBuilder& builder,
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::StructuredLoop& loop,
    bool offload_unknown_sizes
) {
    auto* map = dyn_cast<structured_control_flow::Map*>(&loop);
    if (recorder_ != nullptr) {
        recorder_->apply<transformations::OMPTransform>(builder, analysis_manager, false, *map);
    } else {
        transformations::OMPTransform omp_transform(*map);
        omp_transform.apply(builder, analysis_manager);
    }
}

void OMPScheduler::pre_schedule(
    builder::StructuredSDFGBuilder& builder,
    analysis::AnalysisManager& analysis_manager,
    std::vector<structured_control_flow::StructuredLoop*>& applicable_loops
) {
    std::vector<structured_control_flow::Map*> applicable_maps;
    for (auto* loop : applicable_loops) {
        if (auto* map = dyn_cast<structured_control_flow::Map*>(loop)) {
            applicable_maps.push_back(map);
        }
    }

    if (applicable_maps.empty()) {
        return;
    }

    loops::TrivialLoopEliminationPass trivial_loop_elimination_pass(applicable_maps);
    trivial_loop_elimination_pass.run(builder, analysis_manager);
    analysis_manager.invalidate_all();
    if (applicable_maps.empty()) {
        applicable_loops.clear();
        return;
    }

    loops::CollapsePass collapse_pass(applicable_maps, 1);
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

bool OMPScheduler::post_schedule(
    builder::StructuredSDFGBuilder& builder,
    analysis::AnalysisManager& analysis_manager,
    std::vector<structured_control_flow::StructuredLoop*>& scheduled_loops
) {
    // Mark the innermost sequential Maps/Reduces for vectorization.
    auto& loop_analysis = analysis_manager.get<analysis::LoopAnalysis>();
    std::vector<structured_control_flow::StructuredLoop*> candidates;
    for (auto* node : loop_analysis.innermost_loops()) {
        candidates.push_back(dyn_cast<structured_control_flow::StructuredLoop*>(node));
    }

    bool changed = false;
    for (auto* loop : candidates) {
        loops::transformations::VectorizeTransform transform(*loop);
        if (!transform.can_be_applied(builder, analysis_manager)) {
            continue;
        }
        if (recorder_ != nullptr) {
            recorder_->record(transform, builder, analysis_manager);
        } else {
            transform.apply(builder, analysis_manager);
        }
        changed = true;
    }
    return changed;
}

std::unordered_set<ScheduleTypeCategory> OMPScheduler::compatible_types() {
    return {ScheduleTypeCategory::None};
}

} // namespace scheduler
} // namespace passes
} // namespace sdfg
