#include "sdfg/targets/cuda/stdlib/memset.h"
#include "sdfg/targets/cuda/cuda.h"

namespace sdfg::cuda::stdlib {

MemsetNodeDispatcher_CUDAWithTransfers::MemsetNodeDispatcher_CUDAWithTransfers(
    codegen::LanguageExtension& language_extension,
    const Function& function,
    const data_flow::DataFlowGraph& data_flow_graph,
    const sdfg::stdlib::MemsetNode& node
)
    : codegen::LibraryNodeDispatcher(language_extension, function, data_flow_graph, node) {}

void MemsetNodeDispatcher_CUDAWithTransfers::dispatch_code(
    codegen::PrettyPrinter& stream,
    codegen::PrettyPrinter& globals_stream,
    codegen::CodeSnippetFactory& library_snippet_factory
) {
    auto& node = static_cast<const sdfg::stdlib::MemsetNode&>(node_);

    globals_stream << "#include <cuda.h>" << std::endl;

    stream << "cudaError_t err_cuda;" << std::endl;

    std::string num_expr = language_extension_.expression(node.num());

    stream << "void *d_ptr;" << std::endl;
    stream << "err_cuda = cudaMalloc(&d_ptr, " << num_expr << ");" << std::endl;
    cuda_error_checking(stream, language_extension_, "err_cuda");

    stream << "err_cuda = cudaMemset(d_ptr, " << language_extension_.expression(node.value()) << ", " << num_expr
           << ");" << std::endl;
    cuda_error_checking(stream, language_extension_, "err_cuda");

    stream << "err_cuda = cudaMemcpy(" << node.outputs().at(0) << ", d_ptr, " << num_expr
           << ", cudaMemcpyDeviceToHost);" << std::endl;
    cuda_error_checking(stream, language_extension_, "err_cuda");

    stream << "err_cuda = cudaFree(d_ptr);" << std::endl;
    cuda_error_checking(stream, language_extension_, "err_cuda");
}

MemsetNodeDispatcher_CUDAWithoutTransfers::MemsetNodeDispatcher_CUDAWithoutTransfers(
    codegen::LanguageExtension& language_extension,
    const Function& function,
    const data_flow::DataFlowGraph& data_flow_graph,
    const sdfg::stdlib::MemsetNode& node
)
    : codegen::LibraryNodeDispatcher(language_extension, function, data_flow_graph, node) {}

void MemsetNodeDispatcher_CUDAWithoutTransfers::dispatch_code(
    codegen::PrettyPrinter& stream,
    codegen::PrettyPrinter& globals_stream,
    codegen::CodeSnippetFactory& library_snippet_factory
) {
    auto& node = static_cast<const sdfg::stdlib::MemsetNode&>(node_);

    globals_stream << "#include <cuda.h>" << std::endl;

    stream << "cudaError_t err_cuda;" << std::endl;
    stream << "err_cuda = cudaMemset(" << node.outputs().at(0) << ", " << language_extension_.expression(node.value())
           << ", " << language_extension_.expression(node.num()) << ");" << std::endl;
    cuda_error_checking(stream, language_extension_, "err_cuda");
}

} // namespace sdfg::cuda::stdlib
