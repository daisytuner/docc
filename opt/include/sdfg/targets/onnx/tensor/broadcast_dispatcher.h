#pragma once

#include "sdfg/codegen/dispatchers/block_dispatcher.h"
#include "sdfg/data_flow/library_nodes/math/tensor/broadcast_node.h"
#include "sdfg/targets/onnx/tensor/utils.h"

namespace sdfg {
namespace onnx {
namespace tensor {

class BroadcastNodeDispatcher_ONNX : public codegen::LibraryNodeDispatcher {
public:
    BroadcastNodeDispatcher_ONNX(
        codegen::LanguageExtension& language_extension,
        const Function& function,
        const data_flow::DataFlowGraph& data_flow_graph,
        const math::tensor::BroadcastNode& node
    );

    void dispatch_code(
        codegen::PrettyPrinter& stream,
        codegen::PrettyPrinter& globals_stream,
        codegen::CodeSnippetFactory& library_snippet_factory
    ) override;

protected:
    const math::tensor::BroadcastNode& broadcast_node_;
};

} // namespace tensor
} // namespace onnx
} // namespace sdfg
