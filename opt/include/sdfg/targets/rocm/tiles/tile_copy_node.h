#pragma once

#include "sdfg/tiles/library_nodes/tile_copy_node.h"

namespace sdfg {
namespace rocm {
namespace tiles {

/**
 * @brief Cooperative dispatcher for the whole-copy @ref sdfg::tiles::TileCopyNode.
 *
 * HIP mirror of the CUDA dispatcher: a block-cooperative, thread-strided loop over
 * the flat tile (same `threadIdx`/`blockDim` intrinsics), each thread striding from
 * its flat index by the block thread count. The per-element move is
 * @ref sdfg::tiles::emit_tile_copy_stmt; the barrier is emitted by the producer.
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
} // namespace rocm
} // namespace sdfg
