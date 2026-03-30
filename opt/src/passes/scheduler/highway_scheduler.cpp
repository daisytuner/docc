#include "sdfg/passes/scheduler/highway_scheduler.h"

#include "sdfg/passes/scheduler/loop_scheduler.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/transformations/highway_transform.h"

namespace sdfg {
namespace passes {
namespace scheduler {

SchedulerAction HighwayScheduler::find(
    builder::StructuredSDFGBuilder& builder,
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::StructuredLoop& loop,
    bool offload_unknown_sizes
) {
    auto& loop_analysis = analysis_manager.get<analysis::LoopAnalysis>();

    bool is_innermost = loop_analysis.children(&loop).empty();
    if (!is_innermost) {
        return CHILDREN;
    }

    if (dynamic_cast<structured_control_flow::Map*>(&loop)) {
        return NEXT;
    }

    return NEXT;
}

SchedulerAction HighwayScheduler::find(
    builder::StructuredSDFGBuilder& builder,
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::While& loop,
    bool offload_unknown_sizes
) {
    auto& loop_analysis = analysis_manager.get<analysis::LoopAnalysis>();

    bool is_innermost = loop_analysis.children(&loop).empty();
    if (!is_innermost) {
        return CHILDREN;
    }
    return NEXT;
}

bool HighwayScheduler::can_apply_schedule(
    builder::StructuredSDFGBuilder& builder,
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::StructuredLoop& loop,
    bool offload_unknown_sizes
) {
    auto* map = dynamic_cast<structured_control_flow::Map*>(&loop);
    if (!map) {
        return false;
    }
    transformations::HighwayTransform highway_transform(*map);
    return highway_transform.can_be_applied(builder, analysis_manager);
}

void HighwayScheduler::apply_schedule(
    builder::StructuredSDFGBuilder& builder,
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::StructuredLoop& loop,
    bool offload_unknown_sizes
) {
    auto* map = dynamic_cast<structured_control_flow::Map*>(&loop);
    transformations::HighwayTransform highway_transform(*map);
    highway_transform.apply(builder, analysis_manager);
}

std::unordered_set<ScheduleTypeCategory> HighwayScheduler::compatible_types() {
    return {ScheduleTypeCategory::None, ScheduleTypeCategory::Parallelizer};
}

} // namespace scheduler
} // namespace passes
} // namespace sdfg
