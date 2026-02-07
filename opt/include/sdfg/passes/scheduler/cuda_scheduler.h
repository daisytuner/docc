#pragma once

#include <unordered_set>
#include "sdfg/passes/scheduler/loop_scheduler.h"
#include "sdfg/structured_control_flow/map.h"

namespace sdfg {
namespace passes {
namespace scheduler {

class CUDAScheduler : public LoopScheduler {
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

    static std::string target() { return "cuda"; };

    std::string name() override { return "CUDAScheduler"; };

    std::unordered_set<ScheduleTypeCategory> compatible_types() override;
};

} // namespace scheduler
} // namespace passes
} // namespace sdfg
