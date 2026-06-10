#include "sdfg/targets/cuda/blas/batched_gemm.h"
#include "sdfg/data_flow/library_nodes/math/blas/batched_gemm_node.h"
#include "sdfg/targets/cuda/blas/utils.h"
#include "sdfg/targets/cuda/cuda.h"
#include "sdfg/types/scalar.h"

namespace sdfg::cuda::blas {

BatchedGEMMNodeDispatcher_CUBLASWithTransfers::BatchedGEMMNodeDispatcher_CUBLASWithTransfers(
    codegen::LanguageExtension& language_extension,
    const Function& function,
    const data_flow::DataFlowGraph& data_flow_graph,
    const math::blas::BatchedGEMMNode& node
)
    : codegen::LibraryNodeDispatcher(language_extension, function, data_flow_graph, node) {}

void BatchedGEMMNodeDispatcher_CUBLASWithTransfers::dispatch_code_with_edges(
    codegen::CodegenOutput& out,
    std::vector<codegen::DispatchInput>& inputs,
    std::vector<codegen::DispatchOutput>& outputs
) {
    auto& node = static_cast<const math::blas::BatchedGEMMNode&>(this->node_);

    out.library_snippet_factory.add_global("#include <cuda.h>");
    out.library_snippet_factory.add_global("#include <cublas_v2.h>");

    std::string type;
    switch (node.precision()) {
        case sdfg::math::blas::BLAS_Precision::s:
            type = "float";
            break;
        case sdfg::math::blas::BLAS_Precision::d:
            type = "double";
            break;
        default:
            throw std::runtime_error("Invalid precision for CUBLAS batched GEMM node");
    }

    auto batch_count_expr = this->language_extension_.expression(node.batch_count());

    std::string size_A = "(" +
                         this->language_extension_.expression(symbolic::mul(node.batch_count(), node.stride_a())) +
                         ") * sizeof(" + type + ")";

    std::string size_B = "(" +
                         this->language_extension_.expression(symbolic::mul(node.batch_count(), node.stride_b())) +
                         ") * sizeof(" + type + ")";

    std::string size_C = "(" +
                         this->language_extension_.expression(symbolic::mul(node.batch_count(), node.stride_c())) +
                         ") * sizeof(" + type + ")";

    auto& a_expr = inputs.at(math::blas::BatchedGEMMNode::A_INPUT_IDX).expr;
    auto& b_expr = inputs.at(math::blas::BatchedGEMMNode::B_INPUT_IDX).expr;
    auto& c_expr = inputs.at(math::blas::BatchedGEMMNode::C_INPUT_IDX).expr;
    auto& alpha_expr = inputs.at(math::blas::BatchedGEMMNode::ALPHA_INPUT_IDX).expr;
    auto& beta_expr = inputs.at(math::blas::BatchedGEMMNode::BETA_INPUT_IDX).expr;
    types::PrimitiveType prim_type = (node.precision() == sdfg::math::blas::BLAS_Precision::s)
                                         ? types::PrimitiveType::Float
                                         : types::PrimitiveType::Double;

    // Guard clause
    out.stream << "if (" << this->language_extension_.expression(node.m()) << " != 0 && "
               << this->language_extension_.expression(node.n()) << " != 0 && "
               << this->language_extension_.expression(node.k()) << " != 0 && " << batch_count_expr << " != 0) {"
               << std::endl;
    out.stream.setIndent(out.stream.indent() + 4);

    out.stream << "cudaError_t err_cuda;" << std::endl;

    out.stream << type << " *dA, *dB, *dC;" << std::endl;

    out.stream << "err_cuda = cudaMalloc((void**) &dA, " << size_A << ");" << std::endl;
    cuda_error_checking(out.stream, this->language_extension_, "err_cuda");
    out.stream << "err_cuda = cudaMalloc((void**) &dB, " << size_B << ");" << std::endl;
    cuda_error_checking(out.stream, this->language_extension_, "err_cuda");
    out.stream << "err_cuda = cudaMalloc((void**) &dC, " << size_C << ");" << std::endl;
    cuda_error_checking(out.stream, this->language_extension_, "err_cuda");

    out.stream << "err_cuda = cudaMemcpy(dA, " << a_expr << ", " << size_A << ", cudaMemcpyHostToDevice);" << std::endl;
    cuda_error_checking(out.stream, this->language_extension_, "err_cuda");
    out.stream << "err_cuda = cudaMemcpy(dB, " << b_expr << ", " << size_B << ", cudaMemcpyHostToDevice);" << std::endl;
    cuda_error_checking(out.stream, this->language_extension_, "err_cuda");
    out.stream << "err_cuda = cudaMemcpy(dC, " << c_expr << ", " << size_C << ", cudaMemcpyHostToDevice);" << std::endl;
    cuda_error_checking(out.stream, this->language_extension_, "err_cuda");

    setup_blas_handle(out.library_snippet_factory, this->language_extension_);

    out.stream << out.language_extension.primitive_type(prim_type) << " alpha_scalar = " << alpha_expr << ";"
               << std::endl;
    out.stream << out.language_extension.primitive_type(prim_type) << " beta_scalar = " << beta_expr << ";"
               << std::endl;

    generate_kernel_batched_gemm(
        out.stream, this->language_extension_, node, "dA", "dB", "dC", "&alpha_scalar", "&beta_scalar"
    );

    out.stream << "err_cuda = cudaMemcpy(" << c_expr << ", dC, " << size_C << ", cudaMemcpyDeviceToHost);" << std::endl;
    cuda_error_checking(out.stream, this->language_extension_, "err_cuda");

    out.stream << "err_cuda = cudaFree(dA);" << std::endl;
    cuda_error_checking(out.stream, this->language_extension_, "err_cuda");
    out.stream << "err_cuda = cudaFree(dB);" << std::endl;
    cuda_error_checking(out.stream, this->language_extension_, "err_cuda");
    out.stream << "err_cuda = cudaFree(dC);" << std::endl;
    cuda_error_checking(out.stream, this->language_extension_, "err_cuda");

    out.stream.setIndent(out.stream.indent() - 4);
    out.stream << "}" << std::endl;
}

BatchedGEMMNodeDispatcher_CUBLASWithoutTransfers::BatchedGEMMNodeDispatcher_CUBLASWithoutTransfers(
    codegen::LanguageExtension& language_extension,
    const Function& function,
    const data_flow::DataFlowGraph& data_flow_graph,
    const math::blas::BatchedGEMMNode& node
)
    : codegen::LibraryNodeDispatcher(language_extension, function, data_flow_graph, node) {}

void BatchedGEMMNodeDispatcher_CUBLASWithoutTransfers::dispatch_code_with_edges(
    codegen::CodegenOutput& out,
    std::vector<codegen::DispatchInput>& inputs,
    std::vector<codegen::DispatchOutput>& outputs
) {
    auto& node = static_cast<const math::blas::BatchedGEMMNode&>(this->node_);

    out.library_snippet_factory.add_global("#include <cuda.h>");
    out.library_snippet_factory.add_global("#include <cublas_v2.h>");

    auto batch_count_expr = this->language_extension_.expression(node.batch_count());

    auto& a_expr = inputs.at(math::blas::BatchedGEMMNode::A_INPUT_IDX).expr;
    auto& b_expr = inputs.at(math::blas::BatchedGEMMNode::B_INPUT_IDX).expr;
    auto& c_expr = inputs.at(math::blas::BatchedGEMMNode::C_INPUT_IDX).expr;
    auto& alpha_expr = inputs.at(math::blas::BatchedGEMMNode::ALPHA_INPUT_IDX).expr;
    auto& beta_expr = inputs.at(math::blas::BatchedGEMMNode::BETA_INPUT_IDX).expr;
    types::PrimitiveType prim_type = (node.precision() == sdfg::math::blas::BLAS_Precision::s)
                                         ? types::PrimitiveType::Float
                                         : types::PrimitiveType::Double;

    // Guard clause
    out.stream << "if (" << this->language_extension_.expression(node.m()) << " != 0 && "
               << this->language_extension_.expression(node.n()) << " != 0 && "
               << this->language_extension_.expression(node.k()) << " != 0 && " << batch_count_expr << " != 0) {"
               << std::endl;
    out.stream.setIndent(out.stream.indent() + 4);

    setup_blas_handle(out.library_snippet_factory, this->language_extension_);

    out.stream << out.language_extension.primitive_type(prim_type) << " alpha_scalar = " << alpha_expr << ";"
               << std::endl;
    out.stream << out.language_extension.primitive_type(prim_type) << " beta_scalar = " << beta_expr << ";"
               << std::endl;

    generate_kernel_batched_gemm(
        out.stream, this->language_extension_, node, a_expr, b_expr, c_expr, "&alpha_scalar", "&beta_scalar"
    );

    out.stream.setIndent(out.stream.indent() - 4);
    out.stream << "}" << std::endl;
}

void generate_kernel_batched_gemm(
    codegen::PrettyPrinter& stream,
    codegen::LanguageExtension& language_extension,
    const math::blas::BatchedGEMMNode& node,
    const std::string& a_name,
    const std::string& b_name,
    const std::string& c_name,
    const std::string& alpha_ptr,
    const std::string& beta_ptr
) {
    std::string type;
    switch (node.precision()) {
        case sdfg::math::blas::BLAS_Precision::s:
            type = "S";
            break;
        case sdfg::math::blas::BLAS_Precision::d:
            type = "D";
            break;
        default:
            throw std::runtime_error("Invalid precision for CUBLAS batched GEMM node");
    }

    // cuBLAS is column-major native, so for row-major we swap A and B
    auto first_dim = node.m();
    auto second_dim = node.n();
    auto first_mat = a_name;
    auto second_mat = b_name;
    auto ld_first = node.lda();
    auto ld_second = node.ldb();
    auto ldc = node.ldc();
    auto stride_first = node.stride_a();
    auto stride_second = node.stride_b();
    auto stride_c = node.stride_c();
    auto trans_first = node.trans_a();
    auto trans_second = node.trans_b();

    if (node.layout() == sdfg::math::blas::BLAS_Layout::RowMajor) {
        first_dim = node.n();
        second_dim = node.m();
        first_mat = b_name;
        second_mat = a_name;
        ldc = node.n();
        stride_first = node.stride_b();
        stride_second = node.stride_a();
        trans_first = node.trans_b();
        trans_second = node.trans_a();
        ld_first = (trans_first == sdfg::math::blas::BLAS_Transpose::No) ? node.n() : node.k();
        ld_second = (trans_second == sdfg::math::blas::BLAS_Transpose::No) ? node.k() : node.m();
    }

    std::string trans_first_str = (trans_first == sdfg::math::blas::BLAS_Transpose::No) ? "CUBLAS_OP_N" : "CUBLAS_OP_T";
    std::string trans_second_str = (trans_second == sdfg::math::blas::BLAS_Transpose::No) ? "CUBLAS_OP_N"
                                                                                          : "CUBLAS_OP_T";

    stream << "cublasStatus_t err;" << std::endl;
    stream << "err = cublas" << type << "gemmStridedBatched(handle, " << trans_first_str << ", " << trans_second_str
           << ", " << language_extension.expression(first_dim) << ", " << language_extension.expression(second_dim)
           << ", " << language_extension.expression(node.k()) << ", " << alpha_ptr << ", " << first_mat << ", "
           << language_extension.expression(ld_first) << ", " << language_extension.expression(stride_first) << ", "
           << second_mat << ", " << language_extension.expression(ld_second) << ", "
           << language_extension.expression(stride_second) << ", " << beta_ptr << ", " << c_name << ", "
           << language_extension.expression(ldc) << ", " << language_extension.expression(stride_c) << ", "
           << language_extension.expression(node.batch_count()) << ");" << std::endl;
    cublas_error_checking(stream, language_extension, "err");
    check_cuda_kernel_launch_errors(stream, language_extension, false);
}

} // namespace sdfg::cuda::blas
