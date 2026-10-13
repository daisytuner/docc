#pragma once

#include <sdfg/options.h>
#include <sdfg/passes/scheduler/loop_scheduler.h>
#include <sdfg/structured_control_flow/map.h>

namespace sdfg {
namespace axelera {

/// AI cores every datapath loop is split across, in contiguous chunks.
inline constexpr OptionKey<int64_t> NUM_CORES{"axelera.num_cores"};

/**
 * Replaces the perfect nest of maps at @p outer with one map over their linear index if every memlet in its body
 * addresses memory by an offset plus a multiple of that index, as `x[i * n + j]` does under maps over i < m and
 * j < n. Datapaths stream a single dimension.
 * @return the new map, or nullptr if the nest stays
 */
structured_control_flow::Map* flatten_dense_nest(
    builder::StructuredSDFGBuilder& builder,
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::Map& outer
);

/// Why @p loop (a map or a reduction) cannot become a datapath loop split over @p cores; empty if it can.
std::string datapath_obstacle(
    const StructuredSDFG& sdfg,
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::StructuredLoop& loop,
    int64_t cores
);

} // namespace axelera

namespace passes {
namespace scheduler {

/**
 * Maps innermost elementwise maps and reductions onto Axelera datapaths: the loop is tiled by the PWORD width into
 * a Stream loop over a Lane loop, and with more than one core it is first tiled into a Core loop of chunks. Dense
 * nests of maps are flattened into one map first.
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

    void pre_schedule(
        builder::StructuredSDFGBuilder& builder,
        analysis::AnalysisManager& analysis_manager,
        std::vector<structured_control_flow::StructuredLoop*>& applicable_loops
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
