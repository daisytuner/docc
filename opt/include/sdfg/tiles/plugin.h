#pragma once

#include <sdfg/plugins/plugins.h>
#include "sdfg/serializer/json_serializer.h"
#include "sdfg/tiles/library_nodes/pipeline_node.h"
#include "sdfg/tiles/library_nodes/tile_copy_node.h"

namespace sdfg {
namespace tiles {

inline void register_tiles_plugin(plugins::Context& context) {
    auto& libNodeSerRegistry = context.library_node_serializer_registry;

    // Pipeline primitives
    libNodeSerRegistry.register_library_node_serializer(LibraryNodeType_PipelineCommit.value(), []() {
        return std::make_unique<PipelineCommitNodeSerializer>();
    });
    libNodeSerRegistry.register_library_node_serializer(LibraryNodeType_PipelineWait.value(), []() {
        return std::make_unique<PipelineWaitNodeSerializer>();
    });
    libNodeSerRegistry.register_library_node_serializer(LibraryNodeType_TileCopy.value(), []() {
        return std::make_unique<TileCopyNodeSerializer>();
    });

    // Reference (host / sequential) dispatcher for the whole-copy TileCopyNode. GPU
    // targets register their own impl-typed dispatcher for the cooperative/async atoms.
    auto& libNodeDispRegistry = context.library_node_dispatcher_registry;
    libNodeDispRegistry.register_library_node_dispatcher(
        LibraryNodeType_TileCopy,
        data_flow::ImplementationType_NONE,
        [](codegen::LanguageExtension& language_extension,
           const Function& function,
           const data_flow::DataFlowGraph& data_flow_graph,
           const data_flow::LibraryNode& node) {
            return std::make_unique<TileCopyNodeDispatcher>(
                language_extension, function, data_flow_graph, static_cast<const TileCopyNode&>(node)
            );
        }
    );
}

/**
 * @deprecated use the variant with explicit context
 */
inline void register_tiles_plugin() {
    auto ctx = sdfg::plugins::Context::global_context();
    register_tiles_plugin(ctx);
}

} // namespace tiles
} // namespace sdfg
