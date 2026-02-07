#include "sdfg/passes/scheduler/highway_scheduler.h"

#include "sdfg/passes/scheduler/loop_scheduler.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/transformations/highway_transform.h"

namespace sdfg {
namespace passes {
namespace scheduler {

SchedulerAction HighwayScheduler::schedule(
    builder::StructuredSDFGBuilder& builder,
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::StructuredLoop& loop
) {
    auto& loop_analysis = analysis_manager.get<analysis::LoopAnalysis>();

    bool is_innermost = loop_analysis.children(&loop).empty();
    if (!is_innermost) {
        return CHILDREN;
    }

    if (auto map_node = dynamic_cast<structured_control_flow::Map*>(&loop)) {
        // Apply Highway vectorization to the loop
        transformations::HighwayTransform highway_transform(*map_node);
        if (highway_transform.can_be_applied(builder, analysis_manager)) {
            highway_transform.apply(builder, analysis_manager);
        }
    }

    return NEXT;
}

SchedulerAction HighwayScheduler::schedule(
    builder::StructuredSDFGBuilder& builder,
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::While& loop
) {
    auto& loop_analysis = analysis_manager.get<analysis::LoopAnalysis>();

    bool is_innermost = loop_analysis.children(&loop).empty();
    if (!is_innermost) {
        return CHILDREN;
    }
    // HighwayScheduler does not handle while loops
    return NEXT;
}

std::unordered_set<ScheduleTypeCategory> HighwayScheduler::compatible_types() {
    return {ScheduleTypeCategory::None, ScheduleTypeCategory::Parallelizer};
}


} // namespace scheduler
} // namespace passes
} // namespace sdfg
