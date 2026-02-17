#include "sdfg/targets/hip/blas/gemm.h"
#include "sdfg/data_flow/library_nodes/math/blas/gemm_node.h"
#include "sdfg/targets/hip/blas/utils.h"
#include "sdfg/targets/hip/hip.h"

namespace sdfg::hip::blas {

GEMMNodeDispatcher_HIPBLASWithTransfers::GEMMNodeDispatcher_HIPBLASWithTransfers(
    codegen::LanguageExtension& language_extension,
    const Function& function,
    const data_flow::DataFlowGraph& data_flow_graph,
    const math::blas::GEMMNode& node
)
    : codegen::LibraryNodeDispatcher(language_extension, function, data_flow_graph, node) {}


void add_guard_clause(
    codegen::PrettyPrinter& stream,
    codegen::LanguageExtension& language_extension,
    const math::blas::GEMMNode& gemm_node
) {
    stream << "if (" << language_extension.expression(gemm_node.m()) << " != 0 && "
           << language_extension.expression(gemm_node.n()) << " != 0 && "
           << language_extension.expression(gemm_node.k()) << " != 0) {" << std::endl;
    stream.setIndent(stream.indent() + 4);
}

void remove_guard_clause(codegen::PrettyPrinter& stream) {
    stream.setIndent(stream.indent() - 4);
    stream << "}" << std::endl;
}

void GEMMNodeDispatcher_HIPBLASWithTransfers::dispatch_code(
    codegen::PrettyPrinter& stream,
    codegen::PrettyPrinter& globals_stream,
    codegen::CodeSnippetFactory& library_snippet_factory
) {
    auto& gemm_node = static_cast<const math::blas::GEMMNode&>(this->node_);

    globals_stream << "#include <hip/hip_runtime.h>" << std::endl;
    globals_stream << "#include <hipblas/hipblas.h>" << std::endl;

    std::string type, type2;
    switch (gemm_node.precision()) {
        case sdfg::math::blas::BLAS_Precision::s:
            type = "float";
            break;
        case sdfg::math::blas::BLAS_Precision::d:
            type = "double";
            break;
        default:
            throw std::runtime_error("Invalid precision for HIPBLAS GEMM node");
    }

    std::string size_A = "(" + this->language_extension_.expression(symbolic::mul(gemm_node.m(), gemm_node.k())) +
                         ") * sizeof(" + type + ")";

    std::string size_B = "(" + this->language_extension_.expression(symbolic::mul(gemm_node.k(), gemm_node.n())) +
                         ") * sizeof(" + type + ")";

    std::string size_C = "(" + this->language_extension_.expression(symbolic::mul(gemm_node.m(), gemm_node.n())) +
                         ") * sizeof(" + type + ")";

    add_guard_clause(stream, this->language_extension_, gemm_node);

    stream << "hipError_t err_hip;" << std::endl;

    stream << type << " *dA, *dB, *dC;" << std::endl;

    stream << "err_hip = hipMalloc((void**) &dA, " << size_A << ");" << std::endl;
    hip_error_checking(stream, this->language_extension_, "err_hip");
    stream << "err_hip = hipMalloc((void**) &dB, " << size_B << ");" << std::endl;
    hip_error_checking(stream, this->language_extension_, "err_hip");
    stream << "err_hip = hipMalloc((void**) &dC, " << size_C << ");" << std::endl;
    hip_error_checking(stream, this->language_extension_, "err_hip");

    stream << "err_hip = hipMemcpy(dA, __A, " << size_A << ", hipMemcpyHostToDevice);" << std::endl;
    hip_error_checking(stream, this->language_extension_, "err_hip");
    stream << "err_hip = hipMemcpy(dB, __B, " << size_B << ", hipMemcpyHostToDevice);" << std::endl;
    hip_error_checking(stream, this->language_extension_, "err_hip");
    stream << "err_hip = hipMemcpy(dC, __C, " << size_C << ", hipMemcpyHostToDevice);" << std::endl;
    hip_error_checking(stream, this->language_extension_, "err_hip");

    create_blas_handle(stream, this->language_extension_);

    generate_kernel_gemm(stream, this->language_extension_, gemm_node);

    stream << "err_hip = hipMemcpy(__C, dC, " << size_C << ", hipMemcpyDeviceToHost);" << std::endl;
    hip_error_checking(stream, this->language_extension_, "err_hip");

    stream << "err_hip = hipFree(dA);" << std::endl;
    hip_error_checking(stream, this->language_extension_, "err_hip");
    stream << "err_hip = hipFree(dB);" << std::endl;
    hip_error_checking(stream, this->language_extension_, "err_hip");
    stream << "err_hip = hipFree(dC);" << std::endl;
    hip_error_checking(stream, this->language_extension_, "err_hip");

    destroy_blas_handle(stream, this->language_extension_);

    remove_guard_clause(stream);
}

GEMMNodeDispatcher_HIPBLASWithoutTransfers::GEMMNodeDispatcher_HIPBLASWithoutTransfers(
    codegen::LanguageExtension& language_extension,
    const Function& function,
    const data_flow::DataFlowGraph& data_flow_graph,
    const math::blas::GEMMNode& node
)
    : codegen::LibraryNodeDispatcher(language_extension, function, data_flow_graph, node) {}

void GEMMNodeDispatcher_HIPBLASWithoutTransfers::dispatch_code(
    codegen::PrettyPrinter& stream,
    codegen::PrettyPrinter& globals_stream,
    codegen::CodeSnippetFactory& library_snippet_factory
) {
    auto& gemm_node = static_cast<const math::blas::GEMMNode&>(this->node_);

    globals_stream << "#include <hip/hip_runtime.h>" << std::endl;
    globals_stream << "#include <hipblas/hipblas.h>" << std::endl;

    add_guard_clause(stream, this->language_extension_, gemm_node);

    create_blas_handle(stream, this->language_extension_);

    generate_kernel_gemm(stream, this->language_extension_, gemm_node);

    destroy_blas_handle(stream, this->language_extension_);
    remove_guard_clause(stream);
}

void generate_kernel_gemm(
    codegen::PrettyPrinter& stream,
    codegen::LanguageExtension& language_extension,
    const math::blas::GEMMNode& gemm_node
) {
    std::string type;
    switch (gemm_node.precision()) {
        case sdfg::math::blas::BLAS_Precision::s:
            type = "S";
            break;
        case sdfg::math::blas::BLAS_Precision::d:
            type = "D";
            break;
        default:
            throw std::runtime_error("Invalid precision for HIPBLAS GEMM node");
    }

    auto first_dim = gemm_node.m();
    auto second_dim = gemm_node.n();
    auto first_mat = "A";
    auto second_mat = "B";
    auto ld_first = gemm_node.lda();
    auto ld_second = gemm_node.ldb();
    auto ldc = gemm_node.ldc();
    auto trans_first = gemm_node.trans_a();
    auto trans_second = gemm_node.trans_b();
    if (gemm_node.layout() == sdfg::math::blas::BLAS_Layout::RowMajor) {
        first_dim = gemm_node.n();
        second_dim = gemm_node.m();
        first_mat = "B";
        second_mat = "A";
        ldc = gemm_node.n();
        trans_first = gemm_node.trans_b();
        trans_second = gemm_node.trans_a();
        ld_first = (trans_first == sdfg::math::blas::BLAS_Transpose::No) ? gemm_node.n() : gemm_node.k();
        ld_second = (trans_second == sdfg::math::blas::BLAS_Transpose::No) ? gemm_node.k() : gemm_node.m();
    }

    std::string trans_first_str = (trans_first == sdfg::math::blas::BLAS_Transpose::No) ? "HIPBLAS_OP_N"
                                                                                        : "HIPBLAS_OP_T";
    std::string trans_second_str = (trans_second == sdfg::math::blas::BLAS_Transpose::No) ? "HIPBLAS_OP_N"
                                                                                          : "HIPBLAS_OP_T";


    stream << "hipblasStatus_t err;" << std::endl;
    stream << "err = hipblas" << type << "gemm(handle, " << trans_first_str << ", " << trans_second_str << ", "
           << language_extension.expression(first_dim) << ", " << language_extension.expression(second_dim) << ", "
           << language_extension.expression(gemm_node.k()) << ", "
           << "&__alpha, "
           << "d" << first_mat << ", " << language_extension.expression(ld_first) << ", "
           << "d" << second_mat << ", " << language_extension.expression(ld_second) << ", "
           << "&__beta, "
           << "dC, " << language_extension.expression(ldc) << ");" << std::endl;
    hipblas_error_checking(stream, language_extension, "err");
    check_hip_kernel_launch_errors(stream, language_extension);
}

} // namespace sdfg::hip::blas
