#pragma once

#include "sdfg/codegen/dispatchers/block_dispatcher.h"
#include "sdfg/data_flow/library_nodes/stdlib/memset.h"

namespace sdfg::cuda::stdlib {

class MemsetNodeDispatcher_CUDAWithTransfers : public codegen::LibraryNodeDispatcher {
public:
    MemsetNodeDispatcher_CUDAWithTransfers(
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

class MemsetNodeDispatcher_CUDAWithoutTransfers : public codegen::LibraryNodeDispatcher {
public:
    MemsetNodeDispatcher_CUDAWithoutTransfers(
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

} // namespace sdfg::cuda::stdlib
