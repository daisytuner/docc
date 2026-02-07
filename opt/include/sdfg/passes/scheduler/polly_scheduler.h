#pragma once

#include "sdfg/passes/scheduler/loop_scheduler.h"

namespace sdfg {
namespace passes {
namespace scheduler {

class PollyScheduler : public LoopScheduler {
private:
    bool tile_;

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

    PollyScheduler(bool tile = true);

    static std::string target() { return "polly"; };

    std::string name() override { return "PollyScheduler"; };

    std::unordered_set<ScheduleTypeCategory> compatible_types() override;
};


void register_polly_scheduler(bool tile = true);

} // namespace scheduler
} // namespace passes
} // namespace sdfg
