#pragma once

#include "sdfg/codegen/dispatchers/block_dispatcher.h"
#include "sdfg/data_flow/library_nodes/math/blas/gemm_node.h"

namespace sdfg::rocm::blas {

void generate_kernel_gemm(
    codegen::PrettyPrinter& stream,
    codegen::LanguageExtension& language_extension,
    const math::blas::GEMMNode& gemm_node
);

class GEMMNodeDispatcher_ROCMBLASWithTransfers : public codegen::LibraryNodeDispatcher {
public:
    GEMMNodeDispatcher_ROCMBLASWithTransfers(
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

    codegen::InstrumentationInfo instrumentation_info() const override;
};

class GEMMNodeDispatcher_ROCMBLASWithoutTransfers : public codegen::LibraryNodeDispatcher {
public:
    GEMMNodeDispatcher_ROCMBLASWithoutTransfers(
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

    codegen::InstrumentationInfo instrumentation_info() const override;
};


} // namespace sdfg::rocm::blas
