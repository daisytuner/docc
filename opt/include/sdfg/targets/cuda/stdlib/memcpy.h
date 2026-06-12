#pragma once

#include "sdfg/codegen/dispatchers/block_dispatcher.h"
#include "sdfg/data_flow/library_nodes/stdlib/memcpy.h"

namespace sdfg::cuda::stdlib {

class MemcpyNodeDispatcher_CUDAWithTransfers : public codegen::LibraryNodeDispatcher {
public:
    MemcpyNodeDispatcher_CUDAWithTransfers(
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

class MemcpyNodeDispatcher_CUDAWithoutTransfers : public codegen::LibraryNodeDispatcher {
public:
    MemcpyNodeDispatcher_CUDAWithoutTransfers(
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

} // namespace sdfg::cuda::stdlib
