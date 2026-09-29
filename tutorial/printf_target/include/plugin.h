#pragma once

#include <sdfg/codegen/dispatchers/node_dispatcher_registry.h>
#include <sdfg/serializer/json_serializer.h>

#include "printf_data_offloading_node.h"
#include "printf_map_dispatcher.h"
#include "printf_target.h"

namespace sdfg {
namespace printf_target {

/**
 * @brief Registers all printf target components with sdfglib
 *
 * This function must be called before using any printf target features.
 * It registers:
 * - Map dispatcher for Printf schedule type
 * - Library node dispatcher for PrintfOffloading nodes
 * - Serializer for PrintfDataOffloadingNode
 */
inline void register_printf_plugin() {
    // 1. Register Map Dispatcher
    // Associates the Printf schedule type with the PrintfMapDispatcher
    codegen::MapDispatcherRegistry::instance().register_map_dispatcher(
        ScheduleType_Printf::value(),
        [](codegen::LanguageExtension& language_extension,
           StructuredSDFG& sdfg,
           analysis::AnalysisManager& analysis_manager,
           structured_control_flow::Map& node,
           codegen::InstrumentationPlan& instrumentation_plan,
           codegen::ArgCapturePlan& arg_capture_plan) {
            return std::make_unique<PrintfMapDispatcher>(
                language_extension, sdfg, analysis_manager, node, instrumentation_plan, arg_capture_plan
            );
        }
    );

    // 2. Register Library Node Dispatcher
    // Associates PrintfOffloading library nodes with their code generator
    codegen::LibraryNodeDispatcherRegistry::instance().register_library_node_dispatcher(
        LibraryNodeType_Printf_Offloading,
        data_flow::ImplementationType_NONE,
        [](codegen::LanguageExtension& language_extension,
           const Function& function,
           const data_flow::DataFlowGraph& data_flow_graph,
           const data_flow::LibraryNode& node) {
            return std::make_unique<
                PrintfDataOffloadingNodeDispatcher>(language_extension, function, data_flow_graph, node);
        }
    );

    // 3. Register Serializer
    // Enables saving/loading SDFGs with PrintfDataOffloadingNodes
    serializer::LibraryNodeSerializerRegistry::instance()
        .register_library_node_serializer(LibraryNodeType_Printf_Offloading.value(), []() {
            return std::make_unique<PrintfDataOffloadingNodeSerializer>();
        });
}

} // namespace printf_target
} // namespace sdfg
