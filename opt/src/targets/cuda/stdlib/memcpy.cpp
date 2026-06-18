#include "sdfg/targets/cuda/stdlib/memcpy.h"
#include "sdfg/targets/cuda/cuda.h"

namespace sdfg::cuda::stdlib {

MemcpyNodeDispatcher_CUDAWithTransfers::MemcpyNodeDispatcher_CUDAWithTransfers(
    codegen::LanguageExtension& language_extension,
    const Function& function,
    const data_flow::DataFlowGraph& data_flow_graph,
    const sdfg::stdlib::MemcpyNode& node
)
    : codegen::LibraryNodeDispatcher(language_extension, function, data_flow_graph, node) {}

void MemcpyNodeDispatcher_CUDAWithTransfers::dispatch_code_with_edges(
    codegen::CodegenOutput& out,
    std::vector<codegen::DispatchInput>& inputs,
    std::vector<codegen::DispatchOutput>& outputs
) {
    auto& node = static_cast<const sdfg::stdlib::MemcpyNode&>(node_);

    out.library_snippet_factory.add_global("#include <cuda.h>");

    out.stream << "cudaError_t err_cuda;" << std::endl;

    std::string num_expr = language_extension_.expression(node.count());

    out.stream << "void *d_ptr_in;" << std::endl;
    out.stream << "void *d_ptr_out;" << std::endl;

    out.stream << "err_cuda = cudaMalloc(&d_ptr_in, " << num_expr << ");" << std::endl;
    cuda_error_checking(out.stream, language_extension_, "err_cuda");
    out.stream << "err_cuda = cudaMalloc(&d_ptr_out, " << num_expr << ");" << std::endl;
    cuda_error_checking(out.stream, language_extension_, "err_cuda");

    out.stream << "err_cuda = cudaMemcpy(d_ptr_in, " << inputs.at(1).expr << ", " << num_expr
               << ", cudaMemcpyHostToDevice);" << std::endl;
    cuda_error_checking(out.stream, language_extension_, "err_cuda");

    out.stream << "err_cuda = cudaMemcpy(d_ptr_out, d_ptr_in, " << num_expr << ", cudaMemcpyDeviceToDevice);"
               << std::endl;
    cuda_error_checking(out.stream, language_extension_, "err_cuda");

    out.stream << "err_cuda = cudaMemcpy(" << inputs.at(0).expr << ", d_ptr_out, " << num_expr
               << ", cudaMemcpyDeviceToHost);" << std::endl;
    cuda_error_checking(out.stream, language_extension_, "err_cuda");

    out.stream << "err_cuda = cudaFree(d_ptr_in);" << std::endl;
    cuda_error_checking(out.stream, language_extension_, "err_cuda");
    out.stream << "err_cuda = cudaFree(d_ptr_out);" << std::endl;
    cuda_error_checking(out.stream, language_extension_, "err_cuda");
}

MemcpyNodeDispatcher_CUDAWithoutTransfers::MemcpyNodeDispatcher_CUDAWithoutTransfers(
    codegen::LanguageExtension& language_extension,
    const Function& function,
    const data_flow::DataFlowGraph& data_flow_graph,
    const sdfg::stdlib::MemcpyNode& node
)
    : codegen::LibraryNodeDispatcher(language_extension, function, data_flow_graph, node) {}

void MemcpyNodeDispatcher_CUDAWithoutTransfers::dispatch_code_with_edges(
    codegen::CodegenOutput& out,
    std::vector<codegen::DispatchInput>& inputs,
    std::vector<codegen::DispatchOutput>& outputs
) {
    auto& node = static_cast<const sdfg::stdlib::MemcpyNode&>(node_);

    out.library_snippet_factory.add_global("#include <cuda.h>");

    out.stream << "cudaError_t err_cuda;" << std::endl;
    out.stream << "err_cuda = cudaMemcpy(" << inputs.at(0).expr << ", " << inputs.at(1).expr << ", "
               << language_extension_.expression(node.count()) << ", cudaMemcpyDeviceToDevice);" << std::endl;
    cuda_error_checking(out.stream, language_extension_, "err_cuda");
}

} // namespace sdfg::cuda::stdlib
