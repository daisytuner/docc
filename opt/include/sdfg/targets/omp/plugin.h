#pragma once

#include <string>

#include "sdfg/codegen/dispatchers/node_dispatcher_registry.h"
#include "sdfg/passes/scheduler/omp_scheduler.h"
#include "sdfg/passes/scheduler/scheduler_registry.h"
#include "sdfg/targets/omp/codegen/omp_map_dispatcher.h"
#include "sdfg/targets/omp/schedule.h"

namespace sdfg {
namespace omp {

inline void register_omp_plugin(sdfg::plugins::Context& context) {
    context.get_map_dispatcher_registry().register_map_dispatcher(
        ScheduleType_OMP::value(),
        [](codegen::LanguageExtension& language_extension,
           StructuredSDFG& sdfg,
           analysis::AnalysisManager& analysis_manager,
           structured_control_flow::Map& node,
           codegen::InstrumentationPlan& instrumentation_plan,
           codegen::ArgCapturePlan& arg_capture_plan) {
            return std::make_unique<OMPMapDispatcher>(
                language_extension, sdfg, analysis_manager, node, instrumentation_plan, arg_capture_plan
            );
        }
    );

    passes::scheduler::SchedulerRegistry::instance()
        .register_loop_scheduler<passes::scheduler::OMPScheduler>(passes::scheduler::OMPScheduler::target());
}

inline void register_omp_plugin() {
    auto ctx = plugins::Context::global_context();
    register_omp_plugin(ctx);
}

} // namespace omp
} // namespace sdfg
