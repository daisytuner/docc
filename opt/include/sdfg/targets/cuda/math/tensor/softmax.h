#pragma once

#include "sdfg/codegen/dispatchers/block_dispatcher.h"
#include "sdfg/data_flow/library_nodes/math/tensor/reduce_ops/softmax_node.h"

namespace sdfg::cuda::tensor {

class SoftmaxNodeDispatcher_CUDAWithTransfers : public codegen::LibraryNodeDispatcher {
public:
    SoftmaxNodeDispatcher_CUDAWithTransfers(
        codegen::LanguageExtension& language_extension,
        const Function& function,
        const data_flow::DataFlowGraph& data_flow_graph,
        const sdfg::math::tensor::SoftmaxNode& node
    );

    void dispatch_code_with_edges(
        codegen::CodegenOutput& out,
        std::vector<codegen::DispatchInput>& inputs,
        std::vector<codegen::DispatchOutput>& outputs
    ) override;
};

class SoftmaxNodeDispatcher_CUDAWithoutTransfers : public codegen::LibraryNodeDispatcher {
public:
    SoftmaxNodeDispatcher_CUDAWithoutTransfers(
        codegen::LanguageExtension& language_extension,
        const Function& function,
        const data_flow::DataFlowGraph& data_flow_graph,
        const sdfg::math::tensor::SoftmaxNode& node
    );

    void dispatch_code_with_edges(
        codegen::CodegenOutput& out,
        std::vector<codegen::DispatchInput>& inputs,
        std::vector<codegen::DispatchOutput>& outputs
    ) override;
};

} // namespace sdfg::cuda::tensor
