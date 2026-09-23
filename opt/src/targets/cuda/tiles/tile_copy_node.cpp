#include "sdfg/targets/cuda/tiles/tile_copy_node.h"

namespace sdfg {
namespace cuda {
namespace tiles {

TileCopyNodeDispatcher::TileCopyNodeDispatcher(
    codegen::LanguageExtension& language_extension,
    const Function& function,
    const data_flow::DataFlowGraph& data_flow_graph,
    const ::sdfg::tiles::TileCopyNode& node
)
    : codegen::LibraryNodeDispatcher(language_extension, function, data_flow_graph, node) {}

void TileCopyNodeDispatcher::dispatch_code_with_edges(
    codegen::CodegenOutput& out,
    std::vector<codegen::DispatchInput>& inputs,
    std::vector<codegen::DispatchOutput>& outputs
) {
    const auto& node = static_cast<const ::sdfg::tiles::TileCopyNode&>(node_);
    // Connector order {"_dst", "_src"} — both are the bare base pointers.
    const std::string& dst = inputs.at(0).expr;
    const std::string& src = inputs.at(1).expr;
    const auto pt = inputs.at(0).edge.base_type().primitive_type();

    // CpAsync lowers each step to a cp.async global->shared transfer; the pipeline's
    // commit/wait nodes fence it. Scalar/vector stay synchronous.
    auto async_stmt = [](const std::string& dst_addr, const std::string& src_addr, size_t bytes) {
        return "__pipeline_memcpy_async(" + dst_addr + ", " + src_addr + ", " + std::to_string(bytes) + ");";
    };
    ::sdfg::tiles::emit_cooperative_copy_loop(language_extension_, out.stream, node, dst, src, pt, async_stmt);
}

} // namespace tiles
} // namespace cuda
} // namespace sdfg
