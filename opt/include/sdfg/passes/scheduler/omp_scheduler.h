#pragma once

#include "sdfg/passes/scheduler/loop_scheduler.h"

namespace sdfg {
namespace passes {
namespace scheduler {

class OMPScheduler : public LoopScheduler {
public:
    SchedulerAction find(
        builder::StructuredSDFGBuilder& builder,
        analysis::AnalysisManager& analysis_manager,
        structured_control_flow::StructuredLoop& loop,
        bool offload_unknown_sizes = false
    ) override;

    SchedulerAction find(
        builder::StructuredSDFGBuilder& builder,
        analysis::AnalysisManager& analysis_manager,
        structured_control_flow::While& loop,
        bool offload_unknown_sizes = false
    ) override;

    bool can_apply_schedule(
        builder::StructuredSDFGBuilder& builder,
        analysis::AnalysisManager& analysis_manager,
        structured_control_flow::StructuredLoop& loop,
        bool offload_unknown_sizes = false
    ) override;

    void apply_schedule(
        builder::StructuredSDFGBuilder& builder,
        analysis::AnalysisManager& analysis_manager,
        structured_control_flow::StructuredLoop& loop,
        bool offload_unknown_sizes = false
    ) override;

    void pre_schedule(
        builder::StructuredSDFGBuilder& builder,
        analysis::AnalysisManager& analysis_manager,
        std::vector<structured_control_flow::StructuredLoop*>& applicable_loops
    ) override;

    static std::string target() { return "openmp"; };

    std::unordered_set<ScheduleTypeCategory> compatible_types() override;
};


} // namespace scheduler
} // namespace passes
} // namespace sdfg
