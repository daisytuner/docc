#pragma once

#include <sdfg/plugins/plugins.h>
#include "sdfg/serializer/json_serializer.h"
#include "sdfg/tiles/library_nodes/async_copy_node.h"

namespace sdfg {
namespace tiles {

inline void register_tiles_plugin(plugins::Context& context) {
    auto& libNodeSerRegistry = context.library_node_serializer_registry;

    // Async copy / pipeline primitives
    libNodeSerRegistry.register_library_node_serializer(LibraryNodeType_CpAsyncCopy.value(), []() {
        return std::make_unique<CpAsyncCopyNodeSerializer>();
    });
    libNodeSerRegistry.register_library_node_serializer(LibraryNodeType_VectorCopy.value(), []() {
        return std::make_unique<VectorCopyNodeSerializer>();
    });
    libNodeSerRegistry.register_library_node_serializer(LibraryNodeType_PipelineCommit.value(), []() {
        return std::make_unique<PipelineCommitNodeSerializer>();
    });
    libNodeSerRegistry.register_library_node_serializer(LibraryNodeType_PipelineWait.value(), []() {
        return std::make_unique<PipelineWaitNodeSerializer>();
    });
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
