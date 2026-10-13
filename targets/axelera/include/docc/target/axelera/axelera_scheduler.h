#pragma once

#include <sdfg/options.h>
#include <sdfg/passes/scheduler/loop_scheduler.h>
#include <sdfg/structured_control_flow/map.h>

namespace sdfg {
namespace axelera {

/// AI cores every datapath loop is split across, in contiguous chunks.
inline constexpr OptionKey<int64_t> NUM_CORES{"axelera.num_cores"};

/// Why @p map cannot become a datapath loop split over @p cores; empty if it can.
std::string datapath_obstacle(
    const StructuredSDFG& sdfg,
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::Map& map,
    int64_t cores
);

} // namespace axelera

namespace passes {
namespace scheduler {

/**
 * Maps innermost elementwise maps onto Axelera datapaths: the map is tiled by the PWORD width into a Stream loop
 * over a Lane loop, and with more than one core the Stream loop is tiled again into a Core loop.
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
