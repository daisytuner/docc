#pragma once

#include <sdfg/analysis/flop_analysis.h>
#include <sdfg/analysis/loop_analysis.h>
#include <sdfg/passes/pass.h>
#include <unordered_set>
#include <vector>
#include "sdfg/optimization_report/pass_report_consumer.h"
#include "sdfg/structured_control_flow/map.h"

namespace sdfg {
namespace passes {
namespace scheduler {

enum SchedulerAction {
    NEXT,
    CHILDREN,
};

struct SchedulerLoopInfo {
    // Static Properties
    analysis::LoopInfo loop_info = analysis::LoopInfo();

    // Static Analysis
    symbolic::Expression flop = SymEngine::null;
};


class LoopScheduler {
protected:
    PassReportConsumer* report_ = nullptr;

public:
    virtual ~LoopScheduler() = default;

    /**
     * @brief Phase 1: Determine the action for a loop during loop discovery.
     *
     * Returns NEXT if the loop should be scheduled (added to the applicable set),
     * or CHILDREN if the scheduler should descend into child loops instead.
     */
    virtual SchedulerAction find(
        builder::StructuredSDFGBuilder& builder,
        analysis::AnalysisManager& analysis_manager,
        structured_control_flow::StructuredLoop& loop,
        bool offload_unknown_sizes = false
    ) = 0;

    virtual SchedulerAction find(
        builder::StructuredSDFGBuilder& builder,
        analysis::AnalysisManager& analysis_manager,
        structured_control_flow::While& loop,
        bool offload_unknown_sizes = false
    ) = 0;

    /**
     * @brief Pre-scheduling phase: collapse and cleanup on the applicable loops.
     *
     * Called after loop discovery and before can_apply_schedule/apply_schedule.
     * Implementations may collapse loop nests, run symbol propagation,
     * dead data/CFG elimination, etc. The vector is modified in-place to
     * reflect any pointer changes (e.g. after collapse).
     *
     * Default implementation is a no-op.
     */
    virtual void pre_schedule(
        builder::StructuredSDFGBuilder& builder,
        analysis::AnalysisManager& analysis_manager,
        std::vector<structured_control_flow::StructuredLoop*>& applicable_loops
    ) {}

    /**
     * @brief Check if the scheduling transform can be applied to a loop.
     *
     * Called after pre_schedule. Used to filter the applicable loops before applying.
     */
    virtual bool can_apply_schedule(
        builder::StructuredSDFGBuilder& builder,
        analysis::AnalysisManager& analysis_manager,
        structured_control_flow::StructuredLoop& loop,
        bool offload_unknown_sizes = false
    ) = 0;

    /**
     * @brief Apply the scheduling transform to a single loop.
     *
     * Called only on loops that passed can_apply_schedule().
     */
    virtual void apply_schedule(
        builder::StructuredSDFGBuilder& builder,
        analysis::AnalysisManager& analysis_manager,
        structured_control_flow::StructuredLoop& loop,
        bool offload_unknown_sizes = false
    ) = 0;

    /**
     * @brief Post-scheduling phase: additional transforms on the scheduled loops.
     *
     * Called after apply_schedule on all loops. GPU schedulers use this for
     * loop reordering, nested parallelization, and tiling.
     *
     * Default implementation is a no-op.
     */
    virtual void post_schedule(
        builder::StructuredSDFGBuilder& builder,
        analysis::AnalysisManager& analysis_manager,
        std::vector<structured_control_flow::StructuredLoop*>& scheduled_loops
    ) {}

    virtual void set_report(PassReportConsumer* report) { report_ = report; }

    virtual std::unordered_set<ScheduleTypeCategory> compatible_types() = 0;
};

} // namespace scheduler
} // namespace passes
} // namespace sdfg
