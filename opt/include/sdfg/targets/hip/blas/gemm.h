#pragma once

#include "sdfg/codegen/dispatchers/block_dispatcher.h"
#include "sdfg/data_flow/library_nodes/math/blas/gemm_node.h"

namespace sdfg::hip::blas {

void generate_kernel_gemm(
    codegen::PrettyPrinter& stream,
    codegen::LanguageExtension& language_extension,
    const math::blas::GEMMNode& gemm_node
);

class GEMMNodeDispatcher_HIPBLASWithTransfers : public codegen::LibraryNodeDispatcher {
public:
    GEMMNodeDispatcher_HIPBLASWithTransfers(
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

class GEMMNodeDispatcher_HIPBLASWithoutTransfers : public codegen::LibraryNodeDispatcher {
public:
    GEMMNodeDispatcher_HIPBLASWithoutTransfers(
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


} // namespace sdfg::hip::blas
