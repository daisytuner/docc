#pragma once

#include <sdfg/passes/scheduler/loop_scheduler.h>

namespace sdfg {
namespace passes {
namespace scheduler {

/**
 * Maps innermost elementwise maps onto Axelera datapaths: the map is tiled by the PWORD width, the tile loop
 * becomes the datapath grid and the element loop the lanes.
 */
class AxeleraScheduler : public LoopScheduler {
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

    static std::string target() {
        return "axelera";
    }

    std::unordered_set<ScheduleTypeCategory> compatible_types() override;
};

} // namespace scheduler
} // namespace passes
} // namespace sdfg
