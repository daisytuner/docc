#pragma once

#include "sdfg/codegen/dispatchers/block_dispatcher.h"
#include "sdfg/data_flow/library_nodes/math/tensor/reduce_node.h"
#include "sdfg/targets/onnx/tensor/utils.h"

namespace sdfg {
namespace onnx {
namespace tensor {

class ReduceNodeDispatcher_ONNX : public codegen::LibraryNodeDispatcher {
public:
    ReduceNodeDispatcher_ONNX(
        codegen::LanguageExtension& language_extension,
        const Function& function,
        const data_flow::DataFlowGraph& data_flow_graph,
        const math::tensor::ReduceNode& node
    );

    void dispatch_code(
        codegen::PrettyPrinter& stream,
        codegen::PrettyPrinter& globals_stream,
        codegen::CodeSnippetFactory& library_snippet_factory
    ) override;

protected:
    const math::tensor::ReduceNode& reduce_node_;
};

} // namespace tensor
} // namespace onnx
} // namespace sdfg
