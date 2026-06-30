#pragma once

#include "sdfg/codegen/dispatchers/block_dispatcher.h"
#include "sdfg/data_flow/library_nodes/stdlib/memcpy.h"

namespace sdfg::rocm::stdlib {

class MemcpyNodeDispatcher_ROCMWithTransfers : public codegen::LibraryNodeDispatcher {
public:
    MemcpyNodeDispatcher_ROCMWithTransfers(
        codegen::LanguageExtension& language_extension,
        const Function& function,
        const data_flow::DataFlowGraph& data_flow_graph,
        const sdfg::stdlib::MemcpyNode& node
    );

    void dispatch_code_with_edges(
        codegen::CodegenOutput& out,
        std::vector<codegen::DispatchInput>& inputs,
        std::vector<codegen::DispatchOutput>& outputs
    ) override;
};

class MemcpyNodeDispatcher_ROCMWithoutTransfers : public codegen::LibraryNodeDispatcher {
public:
    MemcpyNodeDispatcher_ROCMWithoutTransfers(
        codegen::LanguageExtension& language_extension,
        const Function& function,
        const data_flow::DataFlowGraph& data_flow_graph,
        const sdfg::stdlib::MemcpyNode& node
    );

    void dispatch_code_with_edges(
        codegen::CodegenOutput& out,
        std::vector<codegen::DispatchInput>& inputs,
        std::vector<codegen::DispatchOutput>& outputs
    ) override;
};

} // namespace sdfg::rocm::stdlib
