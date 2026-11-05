#include "sdfg/targets/rocm/blas/batched_gemm.h"

#include "sdfg/data_flow/library_nodes/math/blas/batched_gemm_node.h"
#include "sdfg/data_flow/library_nodes/math/blas/blas_node.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/targets/rocm/blas/utils.h"
#include "sdfg/targets/rocm/rocm.h"
#include "sdfg/targets/rocm/rocm_data_offloading_node.h"

namespace sdfg::rocm::blas {

BatchedGEMMNodeDispatcher_ROCMBLASWithTransfers::BatchedGEMMNodeDispatcher_ROCMBLASWithTransfers(
    codegen::LanguageExtension& language_extension,
    const Function& function,
    const data_flow::DataFlowGraph& data_flow_graph,
    const math::blas::BatchedGEMMNode& node
)
    : codegen::LibraryNodeDispatcher(language_extension, function, data_flow_graph, node) {
}

static void add_guard_clause(
    codegen::PrettyPrinter& stream,
    codegen::LanguageExtension& language_extension,
    const math::blas::BatchedGEMMNode& node
) {
    stream << "if (" << language_extension.expression(node.m()) << " != 0 && "
           << language_extension.expression(node.n()) << " != 0 && " << language_extension.expression(node.k())
           << " != 0 && " << language_extension.expression(node.batch_count()) << " != 0) {" << std::endl;
    stream.changeIndent(+4);
}

static void remove_guard_clause(codegen::PrettyPrinter& stream) {
    stream.changeIndent(-4);
    stream << "}" << std::endl;
}

void BatchedGEMMNodeDispatcher_ROCMBLASWithTransfers::dispatch_code_with_edges(
    codegen::CodegenOutput& out,
    std::vector<codegen::DispatchInput>& inputs,
    std::vector<codegen::DispatchOutput>& outputs
) {
    auto& node = static_cast<const math::blas::BatchedGEMMNode&>(this->node_);

    out.library_snippet_factory.add_global("#include <hip/hip_runtime.h>");
    out.library_snippet_factory.add_global("#include <hipblas/hipblas.h>");

    const types::PrimitiveType prim_type = math::blas::BLAS_Precision_to_primitive_type(node.precision());
    types::Scalar base_type(prim_type);

    const symbolic::Expression size_A =
        symbolic::mul(symbolic::mul(node.batch_count(), node.stride_a()), symbolic::size_of_type(base_type));
    const symbolic::Expression size_B =
        symbolic::mul(symbolic::mul(node.batch_count(), node.stride_b()), symbolic::size_of_type(base_type));
    const symbolic::Expression size_C =
        symbolic::mul(symbolic::mul(node.batch_count(), node.stride_c()), symbolic::size_of_type(base_type));

    auto& a_expr = inputs.at(math::blas::BatchedGEMMNode::A_INPUT_IDX).expr;
    auto& b_expr = inputs.at(math::blas::BatchedGEMMNode::B_INPUT_IDX).expr;
    auto& c_expr = inputs.at(math::blas::BatchedGEMMNode::C_INPUT_IDX).expr;
    auto& alpha_expr = inputs.at(math::blas::BatchedGEMMNode::ALPHA_INPUT_IDX).expr;
    auto& beta_expr = inputs.at(math::blas::BatchedGEMMNode::BETA_INPUT_IDX).expr;

    add_guard_clause(out.stream, this->language_extension_, node);

    const std::string* async_alloc_md = this->function_.metadata_if_exists("rocm_async_alloc");
    const bool async_alloc = async_alloc_md != nullptr && *async_alloc_md == "1";

    out.stream << "hipError_t err_hip;" << std::endl;
    out.stream << this->language_extension_.primitive_type(base_type.primitive_type()) << " *dA, *dB, *dC;"
               << std::endl;

    dispatch_hip_malloc(out.stream, this->language_extension_, "dA", size_A, "err_hip", async_alloc);
    dispatch_hip_malloc(out.stream, this->language_extension_, "dB", size_B, "err_hip", async_alloc);
    dispatch_hip_malloc(out.stream, this->language_extension_, "dC", size_C, "err_hip", async_alloc);

    dispatch_hip_memcpy(
        out.stream, this->language_extension_, "dA", a_expr, size_A, "hipMemcpyHostToDevice", "err_hip", async_alloc
    );
    dispatch_hip_memcpy(
        out.stream, this->language_extension_, "dB", b_expr, size_B, "hipMemcpyHostToDevice", "err_hip", async_alloc
    );
    dispatch_hip_memcpy(
        out.stream, this->language_extension_, "dC", c_expr, size_C, "hipMemcpyHostToDevice", "err_hip", async_alloc
    );

    setup_blas_handle(out.library_snippet_factory, this->language_extension_, async_alloc);

    out.stream << out.language_extension.primitive_type(prim_type) << " alpha_scalar = " << alpha_expr << ";"
               << std::endl;
    out.stream << out.language_extension.primitive_type(prim_type) << " beta_scalar = " << beta_expr << ";"
               << std::endl;

    generate_kernel_batched_gemm(
        out.stream, this->language_extension_, node, "dA", "dB", "dC", "&alpha_scalar", "&beta_scalar"
    );

    dispatch_hip_memcpy(
        out.stream, this->language_extension_, c_expr, "dC", size_C, "hipMemcpyDeviceToHost", "err_hip", async_alloc
    );

    dispatch_hip_free(out.stream, this->language_extension_, "dA", "err_hip", async_alloc);
    dispatch_hip_free(out.stream, this->language_extension_, "dB", "err_hip", async_alloc);
    dispatch_hip_free(out.stream, this->language_extension_, "dC", "err_hip", async_alloc);

    remove_guard_clause(out.stream);
}

codegen::InstrumentationInfo BatchedGEMMNodeDispatcher_ROCMBLASWithTransfers::instrumentation_info() const {
    return {
        node_.element_id(),
        std::string(node_.element_type()) + ":::" + node_.code().value(),
        TargetType_ROCM,
        codegen::InstrumentationEventType::CUDA,
        analysis::LoopInfo{},
        {}
    };
}

BatchedGEMMNodeDispatcher_ROCMBLASWithoutTransfers::BatchedGEMMNodeDispatcher_ROCMBLASWithoutTransfers(
    codegen::LanguageExtension& language_extension,
    const Function& function,
    const data_flow::DataFlowGraph& data_flow_graph,
    const math::blas::BatchedGEMMNode& node
)
    : codegen::LibraryNodeDispatcher(language_extension, function, data_flow_graph, node) {
}

void BatchedGEMMNodeDispatcher_ROCMBLASWithoutTransfers::dispatch_code_with_edges(
    codegen::CodegenOutput& out,
    std::vector<codegen::DispatchInput>& inputs,
    std::vector<codegen::DispatchOutput>& outputs
) {
    auto& node = static_cast<const math::blas::BatchedGEMMNode&>(this->node_);

    out.library_snippet_factory.add_global("#include <hip/hip_runtime.h>");
    out.library_snippet_factory.add_global("#include <hipblas/hipblas.h>");

    auto batch_count_expr = this->language_extension_.expression(node.batch_count());

    auto& a_expr = inputs.at(math::blas::BatchedGEMMNode::A_INPUT_IDX).expr;
    auto& b_expr = inputs.at(math::blas::BatchedGEMMNode::B_INPUT_IDX).expr;
    auto& c_expr = inputs.at(math::blas::BatchedGEMMNode::C_INPUT_IDX).expr;
    auto& alpha_expr = inputs.at(math::blas::BatchedGEMMNode::ALPHA_INPUT_IDX).expr;
    auto& beta_expr = inputs.at(math::blas::BatchedGEMMNode::BETA_INPUT_IDX).expr;
    const types::PrimitiveType prim_type = math::blas::BLAS_Precision_to_primitive_type(node.precision());

    add_guard_clause(out.stream, this->language_extension_, node);

    const std::string* async_alloc_md = this->function_.metadata_if_exists("rocm_async_alloc");
    const bool async_alloc = async_alloc_md != nullptr && *async_alloc_md == "1";

    setup_blas_handle(out.library_snippet_factory, this->language_extension_, async_alloc);

    out.stream << out.language_extension.primitive_type(prim_type) << " alpha_scalar = " << alpha_expr << ";"
               << std::endl;
    out.stream << out.language_extension.primitive_type(prim_type) << " beta_scalar = " << beta_expr << ";"
               << std::endl;

    generate_kernel_batched_gemm(
        out.stream, this->language_extension_, node, a_expr, b_expr, c_expr, "&alpha_scalar", "&beta_scalar"
    );

    remove_guard_clause(out.stream);
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
    std::string precision_str(math::blas::BLAS_Precision_to_string(node.precision()));
    std::transform(precision_str.begin(), precision_str.end(), precision_str.begin(), ::toupper);

    // hipBLAS is column-major native, so for row-major we swap A and B
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

    std::string trans_first_str = (trans_first == sdfg::math::blas::BLAS_Transpose::No) ? "HIPBLAS_OP_N"
                                                                                        : "HIPBLAS_OP_T";
    std::string trans_second_str = (trans_second == sdfg::math::blas::BLAS_Transpose::No) ? "HIPBLAS_OP_N"
                                                                                          : "HIPBLAS_OP_T";

    stream << "hipblasStatus_t err;" << std::endl;
    stream << "err = hipblas" << precision_str << "gemmStridedBatched(handle, " << trans_first_str << ", "
           << trans_second_str << ", " << language_extension.expression(first_dim) << ", "
           << language_extension.expression(second_dim) << ", " << language_extension.expression(node.k()) << ", "
           << alpha_ptr << ", " << first_mat << ", " << language_extension.expression(ld_first) << ", "
           << language_extension.expression(stride_first) << ", " << second_mat << ", "
           << language_extension.expression(ld_second) << ", " << language_extension.expression(stride_second) << ", "
           << beta_ptr << ", " << c_name << ", " << language_extension.expression(ldc) << ", "
           << language_extension.expression(stride_c) << ", " << language_extension.expression(node.batch_count())
           << ");" << std::endl;
    rocmblas_error_checking(stream, language_extension, "err");
    check_rocm_kernel_launch_errors(stream, language_extension);
}

codegen::InstrumentationInfo BatchedGEMMNodeDispatcher_ROCMBLASWithoutTransfers::instrumentation_info() const {
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
