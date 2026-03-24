#pragma once

#include "sdfg/codegen/dispatchers/node_dispatcher_registry.h"
#include "sdfg/codegen/language_extension.h"
#include "sdfg/data_flow/data_flow_graph.h"
#include "sdfg/data_flow/library_node.h"
#include "sdfg/function.h"
#include "sdfg/targets/offloading/external_offloading_node.h"
#include "sdfg/serializer/json_serializer.h"

namespace sdfg {
namespace offloading {

inline void register_external_data_transfers_plugin() {
    codegen::LibraryNodeDispatcherRegistry::instance().register_library_node_dispatcher(
        LibraryNodeType_External_Offloading.value() + "::" + data_flow::ImplementationType_NONE.value(),
        [](codegen::LanguageExtension& language_extension,
           const Function& function,
           const data_flow::DataFlowGraph& data_flow_graph,
           const data_flow::LibraryNode& node) {
            return std::make_unique<
                ExternalDataOffloadingNodeDispatcher>(language_extension, function, data_flow_graph, node);
        }
    );

    serializer::LibraryNodeSerializerRegistry::instance()
        .register_library_node_serializer(LibraryNodeType_External_Offloading.value(), []() {
            return std::make_unique<ExternalDataOffloadingNodeSerializer>();
        });
}

} // namespace offloading
} // namespace sdfg
