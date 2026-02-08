#pragma once

#include "sdfg/codegen/dispatchers/block_dispatcher.h"
#include "sdfg/data_flow/library_nodes/math/tensor/transpose_node.h"
#include "sdfg/targets/onnx/tensor/utils.h"

namespace sdfg {
namespace onnx {
namespace tensor {

class TransposeNodeDispatcher_ONNX : public codegen::LibraryNodeDispatcher {
public:
    TransposeNodeDispatcher_ONNX(
        codegen::LanguageExtension& language_extension,
        const Function& function,
        const data_flow::DataFlowGraph& data_flow_graph,
        const math::tensor::TransposeNode& node
    );

    void dispatch_code(
        codegen::PrettyPrinter& stream,
        codegen::PrettyPrinter& globals_stream,
        codegen::CodeSnippetFactory& library_snippet_factory
    ) override;

protected:
    const math::tensor::TransposeNode& transpose_node_;
};

} // namespace tensor
} // namespace onnx
} // namespace sdfg
