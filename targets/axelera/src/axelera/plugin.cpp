#include "docc/target/axelera/plugin.h"

#include <sdfg/plugins/targets.h>

#include "docc/target/axelera/axelera_scheduler.h"
#include "docc/target/axelera/datapath_dispatcher.h"
#include "docc/target/axelera/schedule.h"
#include "docc/target/axelera/tile_target.h"

namespace docc::target::axelera {

namespace {

DoccTarget axelera_target = {
    .short_name = "axelera",
    .apply_additional_compile_options = nullptr,
    .apply_expand_time_mapping = nullptr,
    .apply_sched_time_mapping = nullptr,
    .get_target_loop_schedulers =
        [](const TargetOptions& options) -> std::vector<std::shared_ptr<sdfg::passes::scheduler::LoopScheduler>> {
        return {std::make_shared<sdfg::passes::scheduler::AxeleraScheduler>()};
    }
};

} // namespace

void register_plugin(sdfg::plugins::Context& context) {
    context.option_registry().register_option(
        sdfg::axelera::NUM_CORES.spec(1, "AI cores each Axelera datapath loop is split across in contiguous chunks")
    );
    context.map_dispatcher_registry.register_map_dispatcher(
        sdfg::axelera::ScheduleType_Axelera::value(),
        [](sdfg::codegen::LanguageExtension& language_extension,
           sdfg::StructuredSDFG& sdfg,
           sdfg::analysis::AnalysisManager& analysis_manager,
           sdfg::structured_control_flow::Map& node,
           sdfg::codegen::InstrumentationPlan& instrumentation_plan,
           sdfg::codegen::ArgCapturePlan& arg_capture_plan) {
            return std::make_unique<sdfg::axelera::AxeleraMapDispatcher>(
                language_extension, sdfg, analysis_manager, node, instrumentation_plan, arg_capture_plan
            );
        }
    );
    context.reduce_dispatcher_registry.register_reduce_dispatcher(
        sdfg::axelera::ScheduleType_Axelera::value(),
        [](sdfg::codegen::LanguageExtension& language_extension,
           sdfg::StructuredSDFG& sdfg,
           sdfg::analysis::AnalysisManager& analysis_manager,
           sdfg::structured_control_flow::Reduce& node,
           sdfg::codegen::InstrumentationPlan& instrumentation_plan,
           sdfg::codegen::ArgCapturePlan& arg_capture_plan) {
            return std::make_unique<sdfg::axelera::AxeleraReduceDispatcher>(
                language_extension, sdfg, analysis_manager, node, instrumentation_plan, arg_capture_plan
            );
        }
    );
    context.scheduler_registry.register_loop_scheduler<
        sdfg::passes::scheduler::AxeleraScheduler>(sdfg::passes::scheduler::AxeleraScheduler::target());
    context.get_tile_target_registry().register_target(
        sdfg::axelera::ScheduleType_Axelera::value(), std::make_shared<sdfg::axelera::AxeleraTileTarget>()
    );
    context.add_target(&axelera_target);
}

} // namespace docc::target::axelera
