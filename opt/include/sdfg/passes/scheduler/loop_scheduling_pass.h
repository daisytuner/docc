#pragma once

#include <sdfg/analysis/flop_analysis.h>
#include <sdfg/analysis/loop_analysis.h>
#include <sdfg/passes/pass.h>
#include <string>
#include "sdfg/optimization_report/pass_report_consumer.h"
#include "sdfg/passes/scheduler/loop_scheduler.h"

namespace sdfg {
namespace transformations {
class Recorder;
}
namespace passes {
namespace scheduler {

class LoopSchedulingPass : public Pass {
private:
    std::vector<LoopScheduler*> targets_;
    sdfg::PassReportConsumer* report_;
    bool offload_unknown_sizes_;
    sdfg::transformations::Recorder* recorder_ = nullptr;

    bool run_pass_target(
        builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager, LoopScheduler& scheduler
    );

public:
    LoopSchedulingPass(
        const std::vector<LoopScheduler*>& targets, sdfg::PassReportConsumer* report, bool offload_unknown_sizes = false
    )
        : targets_(targets), report_(report), offload_unknown_sizes_(offload_unknown_sizes) {}
    ~LoopSchedulingPass() override = default;

    /**
     * @brief Attach an optional Recorder that captures each scheduling transform.
     *
     * When set, the recorder accumulates the transformations applied by the
     * target schedulers.
     */
    void set_recorder(sdfg::transformations::Recorder* recorder) { recorder_ = recorder; }

    bool run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    std::string name() override { return "LoopSchedulingPass"; }
};


} // namespace scheduler
} // namespace passes
} // namespace sdfg
