#include "sdfg/scheduling/target_scheduling.h"

#include <stdexcept>

#include "sdfg/scheduling/loops/loop_scheduling_pass.h"
#include "sdfg/scheduling/passthrough_scheduler.h"

namespace sdfg {
namespace scheduling {

bool schedule(
    const plugins::Context& context,
    builder::StructuredSDFGBuilder& builder,
    analysis::AnalysisManager& analysis_manager,
    const docc::target::TargetOptions& options,
    const ScheduleConfig& config
) {
    auto* target = context.get_target_handler(options.target);
    if (target == nullptr) {
        throw std::invalid_argument("Unknown target '" + options.target + "'");
    }

    std::unique_ptr<passes::scheduler::SDFGScheduler> sdfg_scheduler =
        config.normalize ? target->scheduling.create_sdfg_scheduler(options) : std::make_unique<PassthroughScheduler>();
    bool changed = sdfg_scheduler->schedule(builder, analysis_manager);
    if (config.on_normal_form) {
        config.on_normal_form(builder.subject());
    }

    if (target->apply_sched_time_mapping) {
        changed |= target->apply_sched_time_mapping(builder, analysis_manager, options);
    }

    if (config.remote_tuning) {
        changed |= config.remote_tuning->run(builder, analysis_manager);
    }

    if (!config.schedule_loops) {
        return changed;
    }

    auto loop_schedulers = target->scheduling.create_loop_schedulers(options);
    std::vector<passes::scheduler::LoopScheduler*> schedulers;
    for (auto& loop_scheduler : loop_schedulers) {
        schedulers.push_back(loop_scheduler.get());
    }
    loops::LoopSchedulingPass loop_scheduling_pass(schedulers, config.report, config.offload_unknown_sizes);
    loop_scheduling_pass.set_recorder(config.recorder);
    changed |= loop_scheduling_pass.run(builder, analysis_manager);
    return changed;
}

} // namespace scheduling
} // namespace sdfg
