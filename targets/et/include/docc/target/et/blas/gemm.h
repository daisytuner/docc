#pragma once

#include <sdfg/codegen/dispatchers/library_node_dispatcher_base.h>
#include <sdfg/data_flow/library_nodes/math/blas/gemm_node.h>

namespace docc::target::et::blas {

using namespace sdfg;

class GEMMNodeDispatcher_ETSOC_WithTransfers
    : public sdfg::codegen::LibraryNodeDispatcherBase<sdfg::math::blas::GEMMNode> {
public:
    GEMMNodeDispatcher_ETSOC_WithTransfers(
        codegen::LanguageExtension& language_extension,
        const Function& function,
        const data_flow::DataFlowGraph& data_flow_graph,
        const math::blas::GEMMNode& node
    );

    void dispatch(
        codegen::PrettyPrinter& stream,
        codegen::PrettyPrinter& globals_stream,
        codegen::CodeSnippetFactory& library_snippet_factory
    ) override;
};


} // namespace docc::target::et::blas
