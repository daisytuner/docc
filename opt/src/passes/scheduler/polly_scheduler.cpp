#include "sdfg/passes/scheduler/polly_scheduler.h"

#include "sdfg/passes/scheduler/scheduler_registry.h"
#include "sdfg/transformations/polly_transform.h"

namespace sdfg {
namespace passes {
namespace scheduler {

SchedulerAction PollyScheduler::find(
    builder::StructuredSDFGBuilder& builder,
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::StructuredLoop& loop,
    bool offload_unknown_sizes
) {
    // Polly tries to apply to the loop; if it can, return NEXT, otherwise descend
    transformations::PollyTransform polly_transform(loop, this->tile_);
    if (polly_transform.can_be_applied(builder, analysis_manager)) {
        return NEXT;
    }

    return CHILDREN;
}

SchedulerAction PollyScheduler::find(
    builder::StructuredSDFGBuilder& builder,
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::While& loop,
    bool offload_unknown_sizes
) {
    return CHILDREN;
}

bool PollyScheduler::can_apply_schedule(
    builder::StructuredSDFGBuilder& builder,
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::StructuredLoop& loop,
    bool offload_unknown_sizes
) {
    transformations::PollyTransform polly_transform(loop, this->tile_);
    return polly_transform.can_be_applied(builder, analysis_manager);
}

void PollyScheduler::apply_schedule(
    builder::StructuredSDFGBuilder& builder,
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::StructuredLoop& loop,
    bool offload_unknown_sizes
) {
    transformations::PollyTransform polly_transform(loop, this->tile_);
    if (polly_transform.can_be_applied(builder, analysis_manager)) {
        polly_transform.apply(builder, analysis_manager);
    }
}

PollyScheduler::PollyScheduler(bool tile) : tile_(tile) {};

void register_polly_scheduler(bool tile) {
    SchedulerRegistry::instance().register_loop_scheduler<PollyScheduler>(PollyScheduler::target(), tile);
}

std::unordered_set<ScheduleTypeCategory> PollyScheduler::compatible_types() { return {ScheduleTypeCategory::None}; }

} // namespace scheduler
} // namespace passes
} // namespace sdfg
