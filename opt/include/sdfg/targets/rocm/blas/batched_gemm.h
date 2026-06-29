#pragma once

#include "sdfg/codegen/dispatchers/block_dispatcher.h"
#include "sdfg/data_flow/library_nodes/math/blas/batched_gemm_node.h"

namespace sdfg::rocm::blas {

void generate_kernel_batched_gemm(
    codegen::PrettyPrinter& stream,
    codegen::LanguageExtension& language_extension,
    const math::blas::BatchedGEMMNode& node,
    const std::string& a_name,
    const std::string& b_name,
    const std::string& c_name,
    const std::string& alpha_name,
    const std::string& beta_name
);

class BatchedGEMMNodeDispatcher_ROCMBLASWithTransfers : public codegen::LibraryNodeDispatcher {
public:
    BatchedGEMMNodeDispatcher_ROCMBLASWithTransfers(
        codegen::LanguageExtension& language_extension,
        const Function& function,
        const data_flow::DataFlowGraph& data_flow_graph,
        const math::blas::BatchedGEMMNode& node
    );

    void dispatch_code_with_edges(
        codegen::CodegenOutput& out,
        std::vector<codegen::DispatchInput>& inputs,
        std::vector<codegen::DispatchOutput>& outputs
    ) override;
};

class BatchedGEMMNodeDispatcher_ROCMBLASWithoutTransfers : public codegen::LibraryNodeDispatcher {
public:
    BatchedGEMMNodeDispatcher_ROCMBLASWithoutTransfers(
        codegen::LanguageExtension& language_extension,
        const Function& function,
        const data_flow::DataFlowGraph& data_flow_graph,
        const math::blas::BatchedGEMMNode& node
    );

    void dispatch_code_with_edges(
        codegen::CodegenOutput& out,
        std::vector<codegen::DispatchInput>& inputs,
        std::vector<codegen::DispatchOutput>& outputs
    ) override;
};

} // namespace sdfg::rocm::blas
