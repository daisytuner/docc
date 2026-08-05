#pragma once

#include "sdfg/codegen/dispatchers/block_dispatcher.h"
#include "sdfg/data_flow/library_nodes/math/blas/dot_node.h"

namespace sdfg::rocm::blas {


class DotNodeDispatcher_ROCMBLASWithTransfers : public codegen::LibraryNodeDispatcher {
public:
    DotNodeDispatcher_ROCMBLASWithTransfers(
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

class DotNodeDispatcher_ROCMBLASWithoutTransfers : public codegen::LibraryNodeDispatcher {
public:
    DotNodeDispatcher_ROCMBLASWithoutTransfers(
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

} // namespace sdfg::rocm::blas
