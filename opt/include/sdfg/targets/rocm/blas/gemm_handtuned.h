#pragma once

#include "sdfg/codegen/dispatchers/block_dispatcher.h"
#include "sdfg/data_flow/library_nodes/math/blas/gemm_node.h"

namespace sdfg::rocm::blas {

class GEMMNodeDispatcher_ROCMHandTuned : public codegen::LibraryNodeDispatcher {
public:
    GEMMNodeDispatcher_ROCMHandTuned(
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
};

} // namespace sdfg::rocm::blas
