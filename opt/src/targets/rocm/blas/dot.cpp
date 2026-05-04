#include "sdfg/targets/rocm/blas/dot.h"
#include <sdfg/data_flow/library_nodes/math/blas/dot_node.h>
#include "sdfg/codegen/dispatchers/block_dispatcher.h"
#include "sdfg/data_flow/library_nodes/math/blas/blas_node.h"
#include "sdfg/targets/rocm/blas/utils.h"
#include "sdfg/targets/rocm/rocm.h"

namespace sdfg::rocm::blas {

DotNodeDispatcher_ROCMBLASWithTransfers::DotNodeDispatcher_ROCMBLASWithTransfers(
    codegen::LanguageExtension& language_extension,
    const Function& function,
    const data_flow::DataFlowGraph& data_flow_graph,
    const math::blas::DotNode& node
)
    : codegen::LibraryNodeDispatcher(language_extension, function, data_flow_graph, node) {}

void DotNodeDispatcher_ROCMBLASWithTransfers::dispatch_code(
    codegen::PrettyPrinter& stream,
    codegen::PrettyPrinter& globals_stream,
    codegen::CodeSnippetFactory& library_snippet_factory
) {
    auto& dot_node = static_cast<const math::blas::DotNode&>(this->node_);

    library_snippet_factory.add_global("#include <hip/hip_runtime.h>");
    library_snippet_factory.add_global("#include <hipblas/hipblas.h>");

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
            throw std::runtime_error("Invalid precision for ROCMBLAS DOT node");
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

    stream << "hipError_t err_hip;" << std::endl;
    stream << type << " *dx, *dy;" << std::endl;
    stream << "err_hip = hipMalloc((void**) &dx, " << x_size << ");" << std::endl;
    rocm_error_checking(stream, this->language_extension_, "err_hip");
    stream << "err_hip = hipMalloc((void**) &dy, " << y_size << ");" << std::endl;
    rocm_error_checking(stream, this->language_extension_, "err_hip");

    stream << "err_hip = hipMemcpy(dx, __x, " << x_size << ", hipMemcpyHostToDevice);" << std::endl;
    rocm_error_checking(stream, this->language_extension_, "err_hip");
    stream << "err_hip = hipMemcpy(dy, __y, " << y_size << ", hipMemcpyHostToDevice);" << std::endl;
    rocm_error_checking(stream, this->language_extension_, "err_hip");

    setup_blas_handle(library_snippet_factory, this->language_extension_);
    stream << "hipblasStatus_t err;" << std::endl;

    stream << "err = hipblas" << type2 << "dot(handle, " << this->language_extension_.expression(dot_node.n())
           << ", dx, " << this->language_extension_.expression(dot_node.incx()) << ", dy, "
           << this->language_extension_.expression(dot_node.incy()) << ", &__out);" << std::endl;
    rocmblas_error_checking(stream, this->language_extension_, "err");
    check_rocm_kernel_launch_errors(stream, this->language_extension_);

    stream << "err_hip = hipFree(dx);" << std::endl;
    rocm_error_checking(stream, this->language_extension_, "err_hip");
    stream << "err_hip = hipFree(dy);" << std::endl;
    rocm_error_checking(stream, this->language_extension_, "err_hip");
}

DotNodeDispatcher_ROCMBLASWithoutTransfers::DotNodeDispatcher_ROCMBLASWithoutTransfers(
    codegen::LanguageExtension& language_extension,
    const Function& function,
    const data_flow::DataFlowGraph& data_flow_graph,
    const math::blas::DotNode& node
)
    : codegen::LibraryNodeDispatcher(language_extension, function, data_flow_graph, node) {}

void DotNodeDispatcher_ROCMBLASWithoutTransfers::dispatch_code(
    codegen::PrettyPrinter& stream,
    codegen::PrettyPrinter& globals_stream,
    codegen::CodeSnippetFactory& library_snippet_factory
) {
    auto& dot_node = static_cast<const math::blas::DotNode&>(this->node_);
    library_snippet_factory.add_global("#include <hip/hip_runtime.h>");
    library_snippet_factory.add_global("#include <hipblas/hipblas.h>");

    setup_blas_handle(library_snippet_factory, this->language_extension_);

    stream << "hipError_t err_hip;" << std::endl;
    stream << "hipblasStatus_t err;" << std::endl;

    stream << "err = hipblas";
    switch (dot_node.precision()) {
        case sdfg::math::blas::BLAS_Precision::s:
            stream << "S";
            break;
        case sdfg::math::blas::BLAS_Precision::d:
            stream << "D";
            break;
        default:
            throw std::runtime_error("Invalid precision for ROCMBLAS DOT node");
    }
    stream << "dot(handle, " << this->language_extension_.expression(dot_node.n()) << ", __x, "
           << this->language_extension_.expression(dot_node.incx()) << ", __y, "
           << this->language_extension_.expression(dot_node.incy()) << ", &__out);" << std::endl;

    rocmblas_error_checking(stream, this->language_extension_, "err");
    check_rocm_kernel_launch_errors(stream, this->language_extension_);
}

} // namespace sdfg::rocm::blas
