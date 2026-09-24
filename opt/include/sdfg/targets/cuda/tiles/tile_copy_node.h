#pragma once

#include "sdfg/tiles/library_nodes/tile_copy_node.h"

namespace sdfg {
namespace cuda {
namespace tiles {

/**
 * @brief Cooperative dispatcher for the whole-copy @ref sdfg::tiles::TileCopyNode.
 *
 * Emits the copy as a block-cooperative, thread-strided loop over the flat tile:
 * every thread of the enclosing kernel block strides the tile from its flat thread
 * index by the block thread count, so the shared tile is staged cooperatively. The
 * per-element move is @ref sdfg::tiles::emit_tile_copy_stmt. The visibility barrier
 * is emitted separately by the producer (as today).
 */
class TileCopyNodeDispatcher : public codegen::LibraryNodeDispatcher {
public:
    TileCopyNodeDispatcher(
        codegen::LanguageExtension& language_extension,
        const Function& function,
        const data_flow::DataFlowGraph& data_flow_graph,
        const ::sdfg::tiles::TileCopyNode& node
    );

    void dispatch_code_with_edges(
        codegen::CodegenOutput& out,
        std::vector<codegen::DispatchInput>& inputs,
        std::vector<codegen::DispatchOutput>& outputs
    ) override;
};

} // namespace tiles
} // namespace cuda
} // namespace sdfg
