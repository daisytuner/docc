#pragma once

#include "sdfg/codegen/dispatchers/block_dispatcher.h"
#include "sdfg/data_flow/library_nodes/math/tensor/elementwise_node.h"
#include "sdfg/targets/onnx/tensor/utils.h"

namespace sdfg {
namespace onnx {
namespace tensor {

class ElementWiseUnaryNodeDispatcher_ONNX : public codegen::LibraryNodeDispatcher {
public:
    ElementWiseUnaryNodeDispatcher_ONNX(
        codegen::LanguageExtension& language_extension,
        const Function& function,
        const data_flow::DataFlowGraph& data_flow_graph,
        const math::tensor::ElementWiseUnaryNode& node
    );

    void dispatch_code(
        codegen::PrettyPrinter& stream,
        codegen::PrettyPrinter& globals_stream,
        codegen::CodeSnippetFactory& library_snippet_factory
    ) override;

protected:
    const math::tensor::ElementWiseUnaryNode& unary_node_;
};

class ElementWiseBinaryNodeDispatcher_ONNX : public codegen::LibraryNodeDispatcher {
public:
    ElementWiseBinaryNodeDispatcher_ONNX(
        codegen::LanguageExtension& language_extension,
        const Function& function,
        const data_flow::DataFlowGraph& data_flow_graph,
        const math::tensor::ElementWiseBinaryNode& node
    );

    void dispatch_code(
        codegen::PrettyPrinter& stream,
        codegen::PrettyPrinter& globals_stream,
        codegen::CodeSnippetFactory& library_snippet_factory
    ) override;

protected:
    const math::tensor::ElementWiseBinaryNode& binary_node_;
};

} // namespace tensor
} // namespace onnx
} // namespace sdfg
