#pragma once

#include <sdfg/analysis/flop_analysis.h>
#include <sdfg/analysis/loop_analysis.h>
#include <sdfg/passes/pass.h>
#include <unordered_set>
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

    virtual SchedulerAction schedule(
        builder::StructuredSDFGBuilder& builder,
        analysis::AnalysisManager& analysis_manager,
        structured_control_flow::StructuredLoop& loop,
        bool offload_unknown_sizes = false
    ) = 0;

    virtual SchedulerAction schedule(
        builder::StructuredSDFGBuilder& builder,
        analysis::AnalysisManager& analysis_manager,
        structured_control_flow::While& loop,
        bool offload_unknown_sizes = false
    ) = 0;

    virtual void set_report(PassReportConsumer* report) { report_ = report; }

    virtual std::unordered_set<ScheduleTypeCategory> compatible_types() = 0;
};

} // namespace scheduler
} // namespace passes
} // namespace sdfg
