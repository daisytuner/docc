#include "sdfg/passes/scheduler/polly_scheduler.h"

#include "sdfg/passes/scheduler/scheduler_registry.h"
#include "sdfg/transformations/polly_transform.h"

namespace sdfg {
namespace passes {
namespace scheduler {

SchedulerAction PollyScheduler::schedule(
    builder::StructuredSDFGBuilder& builder,
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::StructuredLoop& loop
) {
    transformations::PollyTransform polly_transform(loop, this->tile_);
    if (polly_transform.can_be_applied(builder, analysis_manager)) {
        polly_transform.apply(builder, analysis_manager);
        return NEXT;
    }

    return CHILDREN;
}

SchedulerAction PollyScheduler::schedule(
    builder::StructuredSDFGBuilder& builder,
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::While& loop
) {
    return CHILDREN;
}

PollyScheduler::PollyScheduler(bool tile) : tile_(tile) {};

void register_polly_scheduler(bool tile) {
    SchedulerRegistry::instance().register_loop_scheduler<PollyScheduler>(PollyScheduler::target(), tile);
}

std::unordered_set<ScheduleTypeCategory> PollyScheduler::compatible_types() { return {ScheduleTypeCategory::None}; }


} // namespace scheduler
} // namespace passes
} // namespace sdfg
