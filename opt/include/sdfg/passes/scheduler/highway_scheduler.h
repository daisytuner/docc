#pragma once

#include "sdfg/passes/scheduler/loop_scheduler.h"
#include "sdfg/targets/highway/schedule.h"

namespace sdfg {
namespace passes {
namespace scheduler {

class HighwayScheduler : public LoopScheduler {
public:
    SchedulerAction schedule(
        builder::StructuredSDFGBuilder& builder,
        analysis::AnalysisManager& analysis_manager,
        structured_control_flow::StructuredLoop& loop
    ) override;

    SchedulerAction schedule(
        builder::StructuredSDFGBuilder& builder,
        analysis::AnalysisManager& analysis_manager,
        structured_control_flow::While& loop
    ) override;


    static std::string target() { return "highway"; };

    std::string name() override { return "HighwayScheduler"; };

    std::unordered_set<ScheduleTypeCategory> compatible_types() override;
};

} // namespace scheduler
} // namespace passes
} // namespace sdfg
