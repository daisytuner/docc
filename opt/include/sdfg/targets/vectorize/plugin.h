#pragma once

#include <string>

#include "sdfg/codegen/dispatchers/node_dispatcher_registry.h"
#include "sdfg/passes/scheduler/scheduler_registry.h"
#include "sdfg/passes/scheduler/vectorize_scheduler.h"
#include "sdfg/targets/vectorize/codegen/vectorize_map_dispatcher.h"
#include "sdfg/targets/vectorize/schedule.h"

namespace sdfg {
namespace vectorize {

inline void register_vectorize_plugin() {
    codegen::MapDispatcherRegistry::instance().register_map_dispatcher(
        ScheduleType_Vectorize::value(),
        [](codegen::LanguageExtension& language_extension,
           StructuredSDFG& sdfg,
           analysis::AnalysisManager& analysis_manager,
           structured_control_flow::Map& node,
           codegen::InstrumentationPlan& instrumentation_plan,
           codegen::ArgCapturePlan& arg_capture_plan) {
            return std::make_unique<VectorizeMapDispatcher>(
                language_extension, sdfg, analysis_manager, node, instrumentation_plan, arg_capture_plan
            );
        }
    );

    passes::scheduler::SchedulerRegistry::instance()
        .register_loop_scheduler<passes::scheduler::VectorizeScheduler>(passes::scheduler::VectorizeScheduler::target()
        );
}

} // namespace vectorize
} // namespace sdfg
