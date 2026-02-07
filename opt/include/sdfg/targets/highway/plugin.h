#pragma once

#include <string>

#include "sdfg/codegen/dispatchers/node_dispatcher_registry.h"
#include "sdfg/passes/scheduler/highway_scheduler.h"
#include "sdfg/passes/scheduler/scheduler_registry.h"
#include "sdfg/targets/highway/codegen/highway_map_dispatcher.h"
#include "sdfg/targets/highway/schedule.h"

namespace sdfg {
namespace highway {

inline void register_highway_plugin() {
    codegen::MapDispatcherRegistry::instance().register_map_dispatcher(
        ScheduleType_Highway::value(),
        [](codegen::LanguageExtension& language_extension,
           StructuredSDFG& sdfg,
           analysis::AnalysisManager& analysis_manager,
           structured_control_flow::Map& node,
           codegen::InstrumentationPlan& instrumentation_plan,
           codegen::ArgCapturePlan& arg_capture_plan) {
            return std::make_unique<HighwayMapDispatcher>(
                language_extension, sdfg, analysis_manager, node, instrumentation_plan, arg_capture_plan
            );
        }
    );

    passes::scheduler::SchedulerRegistry::instance()
        .register_loop_scheduler<passes::scheduler::HighwayScheduler>(passes::scheduler::HighwayScheduler::target());
}

} // namespace highway
} // namespace sdfg
