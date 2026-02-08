#pragma once

#include "sdfg/codegen/dispatchers/block_dispatcher.h"
#include "sdfg/data_flow/library_nodes/math/blas/gemm_node.h"
#include "sdfg/targets/onnx/blas/utils.h"

namespace sdfg {
namespace onnx {
namespace blas {

class GEMMNodeDispatcher_ONNX : public codegen::LibraryNodeDispatcher {
public:
    GEMMNodeDispatcher_ONNX(
        codegen::LanguageExtension& language_extension,
        const Function& function,
        const data_flow::DataFlowGraph& data_flow_graph,
        const math::blas::GEMMNode& node
    );

    void dispatch_code(
        codegen::PrettyPrinter& stream,
        codegen::PrettyPrinter& globals_stream,
        codegen::CodeSnippetFactory& library_snippet_factory
    ) override;

protected:
    const math::blas::GEMMNode& gemm_node_;
};

} // namespace blas
} // namespace onnx
} // namespace sdfg
