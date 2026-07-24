#pragma once

#include <string>

#include "sdfg/codegen/dispatchers/node_dispatcher_registry.h"
#include "sdfg/passes/scheduler/scheduler_registry.h"
#include "sdfg/passes/scheduler/vectorize_scheduler.h"
#include "sdfg/targets/vectorize/codegen/vectorize_dispatcher.h"
#include "sdfg/targets/vectorize/schedule.h"

namespace sdfg {
namespace vectorize {

inline void register_vectorize_plugin(plugins::Context& context) {
    context.get_map_dispatcher_registry().register_map_dispatcher(
        ScheduleType_Vectorize::value(),
        [](codegen::LanguageExtension& language_extension,
           StructuredSDFG& sdfg,
           analysis::AnalysisManager& analysis_manager,
           structured_control_flow::Map& node,
           codegen::InstrumentationPlan& instrumentation_plan,
           codegen::ArgCapturePlan& arg_capture_plan) {
            return std::make_unique<VectorizeDispatcher>(
                language_extension, sdfg, analysis_manager, node, instrumentation_plan, arg_capture_plan
            );
        }
    );

    context.get_reduce_dispatcher_registry().register_reduce_dispatcher(
        ScheduleType_Vectorize::value(),
        [](codegen::LanguageExtension& language_extension,
           StructuredSDFG& sdfg,
           analysis::AnalysisManager& analysis_manager,
           structured_control_flow::Reduce& node,
           codegen::InstrumentationPlan& instrumentation_plan,
           codegen::ArgCapturePlan& arg_capture_plan) {
            return std::make_unique<VectorizeDispatcher>(
                language_extension, sdfg, analysis_manager, node, instrumentation_plan, arg_capture_plan
            );
        }
    );

    context.get_scheduler_registry()
        .register_loop_scheduler<passes::scheduler::VectorizeScheduler>(passes::scheduler::VectorizeScheduler::target()
        );
}

inline void register_vectorize_plugin() {
    auto ctx = sdfg::plugins::Context::global_context();
    register_vectorize_plugin(ctx);
}

} // namespace vectorize
} // namespace sdfg
