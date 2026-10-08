#include "sdfg/targets/rocm/blas/gemm.h"
#include "sdfg/data_flow/library_nodes/math/blas/gemm_node.h"
#include "sdfg/targets/rocm/blas/utils.h"
#include "sdfg/targets/rocm/rocm.h"

namespace sdfg::rocm::blas {

GEMMNodeDispatcher_ROCMBLASWithTransfers::GEMMNodeDispatcher_ROCMBLASWithTransfers(
    codegen::LanguageExtension& language_extension,
    const Function& function,
    const data_flow::DataFlowGraph& data_flow_graph,
    const math::blas::GEMMNode& node
)
    : codegen::LibraryNodeDispatcher(language_extension, function, data_flow_graph, node) {
}


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

void GEMMNodeDispatcher_ROCMBLASWithTransfers::dispatch_code(
    codegen::PrettyPrinter& stream,
    codegen::PrettyPrinter& globals_stream,
    codegen::CodeSnippetFactory& library_snippet_factory
) {
    auto& gemm_node = static_cast<const math::blas::GEMMNode&>(this->node_);

    library_snippet_factory.add_global("#include <hip/hip_runtime.h>");
    library_snippet_factory.add_global("#include <hipblas/hipblas.h>");

    std::string type, type2;
    switch (gemm_node.precision()) {
        case sdfg::math::blas::BLAS_Precision::h:
            type = "_Float16";
            break;
        case sdfg::math::blas::BLAS_Precision::s:
            type = "float";
            break;
        case sdfg::math::blas::BLAS_Precision::d:
            type = "double";
            break;
        default:
            throw std::runtime_error("Invalid precision for ROCMBLAS GEMM node");
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
    rocm_error_checking(stream, this->language_extension_, "err_hip");
    stream << "err_hip = hipMalloc((void**) &dB, " << size_B << ");" << std::endl;
    rocm_error_checking(stream, this->language_extension_, "err_hip");
    stream << "err_hip = hipMalloc((void**) &dC, " << size_C << ");" << std::endl;
    rocm_error_checking(stream, this->language_extension_, "err_hip");

    stream << "err_hip = hipMemcpy(dA, __A, " << size_A << ", hipMemcpyHostToDevice);" << std::endl;
    rocm_error_checking(stream, this->language_extension_, "err_hip");
    stream << "err_hip = hipMemcpy(dB, __B, " << size_B << ", hipMemcpyHostToDevice);" << std::endl;
    rocm_error_checking(stream, this->language_extension_, "err_hip");
    stream << "err_hip = hipMemcpy(dC, __C, " << size_C << ", hipMemcpyHostToDevice);" << std::endl;
    rocm_error_checking(stream, this->language_extension_, "err_hip");

    setup_blas_handle(library_snippet_factory, this->language_extension_);

    generate_kernel_gemm(stream, this->language_extension_, gemm_node);

    stream << "err_hip = hipMemcpy(__C, dC, " << size_C << ", hipMemcpyDeviceToHost);" << std::endl;
    rocm_error_checking(stream, this->language_extension_, "err_hip");

    stream << "err_hip = hipFree(dA);" << std::endl;
    rocm_error_checking(stream, this->language_extension_, "err_hip");
    stream << "err_hip = hipFree(dB);" << std::endl;
    rocm_error_checking(stream, this->language_extension_, "err_hip");
    stream << "err_hip = hipFree(dC);" << std::endl;
    rocm_error_checking(stream, this->language_extension_, "err_hip");

    remove_guard_clause(stream);
}

codegen::InstrumentationInfo GEMMNodeDispatcher_ROCMBLASWithTransfers::instrumentation_info() const {
    return {
        node_.element_id(),
        std::string(node_.element_type()) + ":::" + node_.code().value(),
        TargetType_ROCM,
        codegen::InstrumentationEventType::CUDA,
        analysis::LoopInfo{},
        {}
    };
}

GEMMNodeDispatcher_ROCMBLASWithoutTransfers::GEMMNodeDispatcher_ROCMBLASWithoutTransfers(
    codegen::LanguageExtension& language_extension,
    const Function& function,
    const data_flow::DataFlowGraph& data_flow_graph,
    const math::blas::GEMMNode& node
)
    : codegen::LibraryNodeDispatcher(language_extension, function, data_flow_graph, node) {
}

void GEMMNodeDispatcher_ROCMBLASWithoutTransfers::dispatch_code(
    codegen::PrettyPrinter& stream,
    codegen::PrettyPrinter& globals_stream,
    codegen::CodeSnippetFactory& library_snippet_factory
) {
    auto& gemm_node = static_cast<const math::blas::GEMMNode&>(this->node_);

    library_snippet_factory.add_global("#include <hip/hip_runtime.h>");
    library_snippet_factory.add_global("#include <hipblas/hipblas.h>");

    add_guard_clause(stream, this->language_extension_, gemm_node);

    setup_blas_handle(library_snippet_factory, this->language_extension_);

    generate_kernel_gemm(stream, this->language_extension_, gemm_node);

    remove_guard_clause(stream);
}

void generate_kernel_gemm(
    codegen::PrettyPrinter& stream,
    codegen::LanguageExtension& language_extension,
    const math::blas::GEMMNode& gemm_node
) {
    std::string type;
    switch (gemm_node.precision()) {
        case sdfg::math::blas::BLAS_Precision::h:
            type = "H";
            break;
        case sdfg::math::blas::BLAS_Precision::s:
            type = "S";
            break;
        case sdfg::math::blas::BLAS_Precision::d:
            type = "D";
            break;
        default:
            throw std::runtime_error("Invalid precision for ROCMBLAS GEMM node");
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

    std::string prefix = gemm_node.implementation_type() == rocm::ImplementationType_ROCMWithTransfers ? "d" : "__";

    // hipblasHgemm expects hipblasHalf operands. The SDFG types the operands as _Float16, which is
    // bit-identical to hipblasHalf but a distinct C++ type, so bind correctly-typed locals once.
    std::string alpha_arg = "&__alpha";
    std::string beta_arg = "&__beta";
    std::string a_arg = prefix + first_mat;
    std::string b_arg = prefix + second_mat;
    std::string c_arg = prefix + "C";
    if (gemm_node.precision() == sdfg::math::blas::BLAS_Precision::h) {
        stream << "const hipblasHalf* alpha_h = reinterpret_cast<const hipblasHalf*>(&__alpha);" << std::endl;
        stream << "const hipblasHalf* beta_h = reinterpret_cast<const hipblasHalf*>(&__beta);" << std::endl;
        stream << "const hipblasHalf* A_h = reinterpret_cast<const hipblasHalf*>(" << a_arg << ");" << std::endl;
        stream << "const hipblasHalf* B_h = reinterpret_cast<const hipblasHalf*>(" << b_arg << ");" << std::endl;
        stream << "hipblasHalf* C_h = reinterpret_cast<hipblasHalf*>(" << c_arg << ");" << std::endl;
        alpha_arg = "alpha_h";
        beta_arg = "beta_h";
        a_arg = "A_h";
        b_arg = "B_h";
        c_arg = "C_h";
    }

    stream << "hipblasStatus_t err;" << std::endl;
    stream << "err = hipblas" << type << "gemm(handle, " << trans_first_str << ", " << trans_second_str << ", "
           << language_extension.expression(first_dim) << ", " << language_extension.expression(second_dim) << ", "
           << language_extension.expression(gemm_node.k()) << ", " << alpha_arg << ", " << a_arg << ", "
           << language_extension.expression(ld_first) << ", " << b_arg << ", "
           << language_extension.expression(ld_second) << ", " << beta_arg << ", " << c_arg << ", "
           << language_extension.expression(ldc) << ");" << std::endl;
    rocmblas_error_checking(stream, language_extension, "err");
    check_rocm_kernel_launch_errors(stream, language_extension);
}

codegen::InstrumentationInfo GEMMNodeDispatcher_ROCMBLASWithoutTransfers::instrumentation_info() const {
    return {
        node_.element_id(),
        std::string(node_.element_type()) + ":::" + node_.code().value(),
        TargetType_ROCM,
        codegen::InstrumentationEventType::CUDA,
        analysis::LoopInfo{},
        {}
    };
}

} // namespace sdfg::rocm::blas
