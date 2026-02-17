#include "sdfg/targets/hip/blas/dot.h"
#include <sdfg/data_flow/library_nodes/math/blas/dot_node.h>
#include "sdfg/codegen/dispatchers/block_dispatcher.h"
#include "sdfg/data_flow/library_nodes/math/blas/blas_node.h"
#include "sdfg/targets/hip/blas/utils.h"
#include "sdfg/targets/hip/hip.h"

namespace sdfg::hip::blas {

DotNodeDispatcher_HIPBLASWithTransfers::DotNodeDispatcher_HIPBLASWithTransfers(
    codegen::LanguageExtension& language_extension,
    const Function& function,
    const data_flow::DataFlowGraph& data_flow_graph,
    const math::blas::DotNode& node
)
    : codegen::LibraryNodeDispatcher(language_extension, function, data_flow_graph, node) {}

void DotNodeDispatcher_HIPBLASWithTransfers::dispatch_code(
    codegen::PrettyPrinter& stream,
    codegen::PrettyPrinter& globals_stream,
    codegen::CodeSnippetFactory& library_snippet_factory
) {
    auto& dot_node = static_cast<const math::blas::DotNode&>(this->node_);

    globals_stream << "#include <hip/hip_runtime.h>" << std::endl;
    globals_stream << "#include <hipblas/hipblas.h>" << std::endl;

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
            throw std::runtime_error("Invalid precision for HIPBLAS DOT node");
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
    hip_error_checking(stream, this->language_extension_, "err_hip");
    stream << "err_hip = hipMalloc((void**) &dy, " << y_size << ");" << std::endl;
    hip_error_checking(stream, this->language_extension_, "err_hip");

    stream << "err_hip = hipMemcpy(dx, __x, " << x_size << ", hipMemcpyHostToDevice);" << std::endl;
    hip_error_checking(stream, this->language_extension_, "err_hip");
    stream << "err_hip = hipMemcpy(dy, __y, " << y_size << ", hipMemcpyHostToDevice);" << std::endl;
    hip_error_checking(stream, this->language_extension_, "err_hip");

    create_blas_handle(stream, this->language_extension_);
    stream << "hipblasStatus_t err;" << std::endl;

    stream << "err = hipblas" << type2 << "dot(handle, " << this->language_extension_.expression(dot_node.n())
           << ", dx, " << this->language_extension_.expression(dot_node.incx()) << ", dy, "
           << this->language_extension_.expression(dot_node.incy()) << ", &__out);" << std::endl;
    hipblas_error_checking(stream, this->language_extension_, "err");
    check_hip_kernel_launch_errors(stream, this->language_extension_);

    destroy_blas_handle(stream, this->language_extension_);

    stream << "err_hip = hipFree(dx);" << std::endl;
    hip_error_checking(stream, this->language_extension_, "err_hip");
    stream << "err_hip = hipFree(dy);" << std::endl;
    hip_error_checking(stream, this->language_extension_, "err_hip");
}

DotNodeDispatcher_HIPBLASWithoutTransfers::DotNodeDispatcher_HIPBLASWithoutTransfers(
    codegen::LanguageExtension& language_extension,
    const Function& function,
    const data_flow::DataFlowGraph& data_flow_graph,
    const math::blas::DotNode& node
)
    : codegen::LibraryNodeDispatcher(language_extension, function, data_flow_graph, node) {}

void DotNodeDispatcher_HIPBLASWithoutTransfers::dispatch_code(
    codegen::PrettyPrinter& stream,
    codegen::PrettyPrinter& globals_stream,
    codegen::CodeSnippetFactory& library_snippet_factory
) {
    auto& dot_node = static_cast<const math::blas::DotNode&>(this->node_);
    globals_stream << "#include <hip/hip_runtime.h>" << std::endl;
    globals_stream << "#include <hipblas/hipblas.h>" << std::endl;

    stream << "hipError_t err_hip;" << std::endl;
    stream << "hipblasStatus_t err;" << std::endl;

    create_blas_handle(stream, this->language_extension_);

    stream << "err = hipblas";
    switch (dot_node.precision()) {
        case sdfg::math::blas::BLAS_Precision::s:
            stream << "S";
            break;
        case sdfg::math::blas::BLAS_Precision::d:
            stream << "D";
            break;
        default:
            throw std::runtime_error("Invalid precision for HIPBLAS DOT node");
    }
    stream << "dot(handle, " << this->language_extension_.expression(dot_node.n()) << ", __x, "
           << this->language_extension_.expression(dot_node.incx()) << ", __y, "
           << this->language_extension_.expression(dot_node.incy()) << ", &__out);" << std::endl;

    hipblas_error_checking(stream, this->language_extension_, "err");
    check_hip_kernel_launch_errors(stream, this->language_extension_);

    destroy_blas_handle(stream, this->language_extension_);
}

} // namespace sdfg::hip::blas
