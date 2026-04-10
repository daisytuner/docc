#pragma once

#include "sdfg/codegen/dispatchers/block_dispatcher.h"
#include "sdfg/data_flow/library_nodes/stdlib/memset.h"

namespace sdfg::rocm::stdlib {

class MemsetNodeDispatcher_ROCMWithTransfers : public codegen::LibraryNodeDispatcher {
public:
    MemsetNodeDispatcher_ROCMWithTransfers(
        codegen::LanguageExtension& language_extension,
        const Function& function,
        const data_flow::DataFlowGraph& data_flow_graph,
        const sdfg::stdlib::MemsetNode& node
    );

    void dispatch_code(
        codegen::PrettyPrinter& stream,
        codegen::PrettyPrinter& globals_stream,
        codegen::CodeSnippetFactory& library_snippet_factory
    ) override;
};

class MemsetNodeDispatcher_ROCMWithoutTransfers : public codegen::LibraryNodeDispatcher {
public:
    MemsetNodeDispatcher_ROCMWithoutTransfers(
        codegen::LanguageExtension& language_extension,
        const Function& function,
        const data_flow::DataFlowGraph& data_flow_graph,
        const sdfg::stdlib::MemsetNode& node
    );

    void dispatch_code(
        codegen::PrettyPrinter& stream,
        codegen::PrettyPrinter& globals_stream,
        codegen::CodeSnippetFactory& library_snippet_factory
    ) override;
};

} // namespace sdfg::rocm::stdlib
