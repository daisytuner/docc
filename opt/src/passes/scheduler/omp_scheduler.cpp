#include "sdfg/passes/scheduler/omp_scheduler.h"

#include "sdfg/passes/dataflow/dead_data_elimination.h"
#include "sdfg/passes/structured_control_flow/dead_cfg_elimination.h"
#include "sdfg/passes/symbolic/symbol_propagation.h"
#include "sdfg/transformations/collapse_to_depth.h"
#include "sdfg/transformations/omp_transform.h"

namespace sdfg {
namespace passes {
namespace scheduler {

SchedulerAction OMPScheduler::schedule(
    builder::StructuredSDFGBuilder& builder,
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::StructuredLoop& loop,
    bool offload_unknown_sizes
) {
    if (auto map_node = dynamic_cast<structured_control_flow::Map*>(&loop)) {
        // Apply OpenMP parallelization to the loop
        transformations::CollapseToDepth collapse_to_depth(*map_node, 1);
        if (collapse_to_depth.can_be_applied(builder, analysis_manager)) {
            collapse_to_depth.apply(builder, analysis_manager);
            analysis_manager.invalidate_all();
        }

        passes::SymbolPropagation symbol_propagation_pass;
        symbol_propagation_pass.run(builder, analysis_manager);
        passes::DeadDataElimination ddead_pass;
        ddead_pass.run(builder, analysis_manager);
        passes::DeadCFGElimination dcfg_pass;
        dcfg_pass.run(builder, analysis_manager);
        analysis_manager.invalidate_all();

        auto collapsed_map = collapse_to_depth.outer_loop();
        transformations::OMPTransform omp_transform(*collapsed_map);
        if (omp_transform.can_be_applied(builder, analysis_manager)) {
            omp_transform.apply(builder, analysis_manager);
            return NEXT;
        }
    }

    // Check if in not outermost loop
    auto& loop_analysis = analysis_manager.get<analysis::LoopAnalysis>();
    auto loop_info = loop_analysis.loop_info(&loop);
    if (loop_info.loopnest_index == -1 || loop_info.num_maps <= 1 || loop_info.is_perfectly_nested ||
        loop_info.has_side_effects) {
        return NEXT;
    } else {
        // Visit 1st-level children
        return CHILDREN;
    }
}

SchedulerAction OMPScheduler::schedule(
    builder::StructuredSDFGBuilder& builder,
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::While& loop,
    bool offload_unknown_sizes
) {
    // Check if in not outermost loop
    auto& loop_analysis = analysis_manager.get<analysis::LoopAnalysis>();
    auto loop_info = loop_analysis.loop_info(&loop);
    if (loop_info.loopnest_index == -1 || loop_info.has_side_effects) {
        return NEXT;
    } else {
        // Visit 1st-level children
        return CHILDREN;
    }
}


std::unordered_set<ScheduleTypeCategory> OMPScheduler::compatible_types() {
    return {ScheduleTypeCategory::None, ScheduleTypeCategory::Vectorizer};
}


} // namespace scheduler
} // namespace passes
} // namespace sdfg
