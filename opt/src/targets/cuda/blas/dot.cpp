#include "sdfg/targets/cuda/blas/dot.h"
#include <sdfg/data_flow/library_nodes/math/blas/dot_node.h>
#include "sdfg/codegen/dispatchers/block_dispatcher.h"
#include "sdfg/data_flow/library_nodes/math/blas/blas_node.h"
#include "sdfg/targets/cuda/blas/utils.h"
#include "sdfg/targets/cuda/cuda.h"

namespace sdfg::cuda::blas {

DotNodeDispatcher_CUBLASWithTransfers::DotNodeDispatcher_CUBLASWithTransfers(
    codegen::LanguageExtension& language_extension,
    const Function& function,
    const data_flow::DataFlowGraph& data_flow_graph,
    const math::blas::DotNode& node
)
    : codegen::LibraryNodeDispatcher(language_extension, function, data_flow_graph, node) {}

void DotNodeDispatcher_CUBLASWithTransfers::dispatch_code(
    codegen::PrettyPrinter& stream,
    codegen::PrettyPrinter& globals_stream,
    codegen::CodeSnippetFactory& library_snippet_factory
) {
    auto& dot_node = static_cast<const math::blas::DotNode&>(this->node_);

    globals_stream << "#include <cuda.h>" << std::endl;
    globals_stream << "#include <cublas_v2.h>" << std::endl;

    std::string type, type2;
    switch (dot_node.precision()) {
        case sdfg::math::blas::BLAS_Precision::s:
            type = "float";
            type2 = "S";
            break;
        case sdfg::math::blas::BLAS_Precision::d:
            type = "double";
            type2 = "D";
            break;
        default:
            throw std::runtime_error("Invalid precision for CUBLAS DOT node");
    }

    const std::string x_size =
        this->language_extension_.expression(
            symbolic::add(symbolic::mul(symbolic::sub(dot_node.n(), symbolic::one()), dot_node.incx()), symbolic::one())
        ) +
        " * sizeof(" + type + ")";
    const std::string y_size =
        this->language_extension_.expression(
            symbolic::add(symbolic::mul(symbolic::sub(dot_node.n(), symbolic::one()), dot_node.incy()), symbolic::one())
        ) +
        " * sizeof(" + type + ")";

    stream << "cudaError_t err_cuda;" << std::endl;
    stream << type << " *dx, *dy;" << std::endl;
    stream << "err_cuda = cudaMalloc((void**) &dx, " << x_size << ");" << std::endl;
    cuda_error_checking(stream, this->language_extension_, "err_cuda");
    stream << "err_cuda = cudaMalloc((void**) &dy, " << y_size << ");" << std::endl;
    cuda_error_checking(stream, this->language_extension_, "err_cuda");

    stream << "err_cuda = cudaMemcpy(dx, __x, " << x_size << ", cudaMemcpyHostToDevice);" << std::endl;
    cuda_error_checking(stream, this->language_extension_, "err_cuda");
    stream << "err_cuda = cudaMemcpy(dy, __y, " << y_size << ", cudaMemcpyHostToDevice);" << std::endl;
    cuda_error_checking(stream, this->language_extension_, "err_cuda");

    create_blas_handle(stream, this->language_extension_);
    stream << "cublasStatus_t err;" << std::endl;

    stream << "err = cublas" << type2 << "dot(handle, " << this->language_extension_.expression(dot_node.n())
           << ", dx, " << this->language_extension_.expression(dot_node.incx()) << ", dy, "
           << this->language_extension_.expression(dot_node.incy()) << ", &__out);" << std::endl;
    cublas_error_checking(stream, this->language_extension_, "err");
    check_cuda_kernel_launch_errors(stream, this->language_extension_, false);

    destroy_blas_handle(stream, this->language_extension_);

    stream << "err_cuda = cudaFree(dx);" << std::endl;
    cuda_error_checking(stream, this->language_extension_, "err_cuda");
    stream << "err_cuda = cudaFree(dy);" << std::endl;
    cuda_error_checking(stream, this->language_extension_, "err_cuda");
}

DotNodeDispatcher_CUBLASWithoutTransfers::DotNodeDispatcher_CUBLASWithoutTransfers(
    codegen::LanguageExtension& language_extension,
    const Function& function,
    const data_flow::DataFlowGraph& data_flow_graph,
    const math::blas::DotNode& node
)
    : codegen::LibraryNodeDispatcher(language_extension, function, data_flow_graph, node) {}

void DotNodeDispatcher_CUBLASWithoutTransfers::dispatch_code(
    codegen::PrettyPrinter& stream,
    codegen::PrettyPrinter& globals_stream,
    codegen::CodeSnippetFactory& library_snippet_factory
) {
    auto& dot_node = static_cast<const math::blas::DotNode&>(this->node_);
    globals_stream << "#include <cuda.h>" << std::endl;
    globals_stream << "#include <cublas_v2.h>" << std::endl;

    stream << "cudaError_t err_cuda;" << std::endl;
    stream << "cublasStatus_t err;" << std::endl;

    create_blas_handle(stream, this->language_extension_);

    stream << "err = cublas";
    switch (dot_node.precision()) {
        case sdfg::math::blas::BLAS_Precision::s:
            stream << "S";
            break;
        case sdfg::math::blas::BLAS_Precision::d:
            stream << "D";
            break;
        default:
            throw std::runtime_error("Invalid precision for CUBLAS DOT node");
    }
    stream << "dot(handle, " << this->language_extension_.expression(dot_node.n()) << ", __x, "
           << this->language_extension_.expression(dot_node.incx()) << ", __y, "
           << this->language_extension_.expression(dot_node.incy()) << ", &__out);" << std::endl;

    cublas_error_checking(stream, this->language_extension_, "err");
    check_cuda_kernel_launch_errors(stream, this->language_extension_, false);

    destroy_blas_handle(stream, this->language_extension_);
}

} // namespace sdfg::cuda::blas
