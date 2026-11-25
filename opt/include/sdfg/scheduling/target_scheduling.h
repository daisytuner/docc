#pragma once

#include <functional>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/optimization_report/pass_report_consumer.h"
#include "sdfg/passes/pass.h"
#include "sdfg/plugins/plugins.h"

namespace sdfg {
namespace transformations {
class Recorder;
}
namespace scheduling {

struct ScheduleConfig {
    /// Whether to bring the SDFG into the target's normal form. Without, the SDFG is scheduled as it is (see
    /// PassthroughScheduler), e.g. after optimizing it by hand.
    bool normalize = true;
    /// Whether to run the loop schedulers. Without, the SDFG is only brought into the target's normal form and
    /// tuned, as argument capture needs it while keeping all execution on the host.
    bool schedule_loops = true;
    bool offload_unknown_sizes = false;
    /// Called with the SDFG in the target's normal form, before anything maps or schedules it, e.g. to record it.
    std::function<void(const StructuredSDFG&)> on_normal_form;
    /// Remote tuning, run between the SDFG scheduler and the loop schedulers; nullptr for none.
    passes::Pass* remote_tuning = nullptr;
    transformations::Recorder* recorder = nullptr;
    PassReportConsumer* report = nullptr;
};

/**
 * @brief Schedules an SDFG for the target of @p options, as registered in @p context.
 *
 * Runs, in order: the target's SDFGScheduler (unless normalization is disabled), its sched-time library node mapping,
 * remote tuning and the target's loop schedulers (see docc::target::TargetScheduling).
 *
 * @throws std::invalid_argument if @p context knows no such target
 * @return true if the SDFG was modified
 */
bool schedule(
    const plugins::Context& context,
    builder::StructuredSDFGBuilder& builder,
    analysis::AnalysisManager& analysis_manager,
    const docc::target::TargetOptions& options,
    const ScheduleConfig& config = {}
);

} // namespace scheduling
} // namespace sdfg
