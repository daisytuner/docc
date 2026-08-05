#pragma once

#include "sdfg/codegen/dispatchers/block_dispatcher.h"
#include "sdfg/data_flow/library_nodes/math/blas/dot_node.h"

namespace sdfg::cuda::blas {


class DotNodeDispatcher_CUBLASWithTransfers : public codegen::LibraryNodeDispatcher {
public:
    DotNodeDispatcher_CUBLASWithTransfers(
        codegen::LanguageExtension& language_extension,
        const Function& function,
        const data_flow::DataFlowGraph& data_flow_graph,
        const math::blas::DotNode& node
    );

    void dispatch_code(
        codegen::PrettyPrinter& stream,
        codegen::PrettyPrinter& globals_stream,
        codegen::CodeSnippetFactory& library_snippet_factory
    ) override;

    codegen::InstrumentationInfo instrumentation_info() const override;
};

class DotNodeDispatcher_CUBLASWithoutTransfers : public codegen::LibraryNodeDispatcher {
public:
    DotNodeDispatcher_CUBLASWithoutTransfers(
        codegen::LanguageExtension& language_extension,
        const Function& function,
        const data_flow::DataFlowGraph& data_flow_graph,
        const math::blas::DotNode& node
    );

    void dispatch_code(
        codegen::PrettyPrinter& stream,
        codegen::PrettyPrinter& globals_stream,
        codegen::CodeSnippetFactory& library_snippet_factory
    ) override;

    codegen::InstrumentationInfo instrumentation_info() const override;
};

} // namespace sdfg::cuda::blas
