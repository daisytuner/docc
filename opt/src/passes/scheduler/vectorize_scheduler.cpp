#include "sdfg/passes/scheduler/vectorize_scheduler.h"

#include "sdfg/passes/scheduler/loop_scheduler.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/transformations/vectorize_transform.h"

namespace sdfg {
namespace passes {
namespace scheduler {

SchedulerAction VectorizeScheduler::find(
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

    return NEXT;
}

SchedulerAction VectorizeScheduler::find(
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

bool VectorizeScheduler::can_apply_schedule(
    builder::StructuredSDFGBuilder& builder,
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::StructuredLoop& loop,
    bool offload_unknown_sizes
) {
    transformations::VectorizeTransform vectorize_transform(loop);
    return vectorize_transform.can_be_applied(builder, analysis_manager);
}

void VectorizeScheduler::apply_schedule(
    builder::StructuredSDFGBuilder& builder,
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::StructuredLoop& loop,
    bool offload_unknown_sizes
) {
    transformations::VectorizeTransform vectorize_transform(loop);
    vectorize_transform.apply(builder, analysis_manager);
}

std::unordered_set<ScheduleTypeCategory> VectorizeScheduler::compatible_types() {
    return {ScheduleTypeCategory::None, ScheduleTypeCategory::Parallelizer};
}

} // namespace scheduler
} // namespace passes
} // namespace sdfg
