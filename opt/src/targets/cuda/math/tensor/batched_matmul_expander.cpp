#include "sdfg/targets/cuda/math/tensor/batched_matmul_expander.h"

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/library_nodes/math/blas/batched_gemm_node.h"
#include "sdfg/data_flow/library_nodes/math/blas/blas_node.h"
#include "sdfg/exceptions.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/targets/cuda/cuda.h"
#include "sdfg/types/pointer.h"
#include "sdfg/types/scalar.h"

namespace sdfg {
namespace offloading {

bool CudaBatchedMatMulExpander::
    expand(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    auto& dataflow = node_.get_parent();
    auto& block = static_cast<structured_control_flow::Block&>(*dataflow.get_parent());

    if (dataflow.in_degree(node_) != 3 || dataflow.out_degree(node_) != 0) {
        return false;
    }

    auto& layout_a = node_.layout_a();
    auto& layout_b = node_.layout_b();

    // Only handle batched matmuls (dims > 2)
    if (layout_a.dims() <= 2 && layout_b.dims() <= 2) {
        return false;
    }

    // Require both have the same number of batch dimensions (no broadcasting for batched GEMM)
    size_t batch_dims_a = layout_a.dims() - 2;
    size_t batch_dims_b = layout_b.dims() - 2;
    if (batch_dims_a != batch_dims_b) {
        return false;
    }

    // Verify batch dimensions match
    for (size_t i = 0; i < batch_dims_a; ++i) {
        if (!symbolic::eq(layout_a.get_dim(i), layout_b.get_dim(i))) {
            return false;
        }
    }

    // Require contiguous (no padding) layout for strided batched
    if (!layout_a.has_linear_accesses_no_padding() && !layout_a.has_transposed_strides_no_padding()) {
        return false;
    }
    if (!layout_b.has_linear_accesses_no_padding() && !layout_b.has_transposed_strides_no_padding()) {
        return false;
    }

    // Determine precision
    auto prim_type = node_.uniform_quantization(dataflow);
    if (!prim_type) {
        return false;
    }
    math::blas::BLAS_Precision precision;
    switch (prim_type.value()) {
        case types::PrimitiveType::Float:
            precision = math::blas::BLAS_Precision::s;
            break;
        case types::PrimitiveType::Double:
            precision = math::blas::BLAS_Precision::d;
            break;
        default:
            return false;
    }

    // Determine transpose flags
    math::blas::BLAS_Transpose trans_a, trans_b;
    if (layout_a.has_linear_accesses_no_padding()) {
        trans_a = math::blas::BLAS_Transpose::No;
    } else if (layout_a.has_transposed_strides_no_padding()) {
        trans_a = math::blas::BLAS_Transpose::Trans;
    } else {
        return false;
    }
    if (layout_b.has_linear_accesses_no_padding()) {
        trans_b = math::blas::BLAS_Transpose::No;
    } else if (layout_b.has_transposed_strides_no_padding()) {
        trans_b = math::blas::BLAS_Transpose::Trans;
    } else {
        return false;
    }

    auto& parent = static_cast<structured_control_flow::Sequence&>(*block.get_parent());
    int index = parent.index(block);

    // Get input edges
    auto iedges = dataflow.in_edges_by_connector(node_);
    auto* iedge_y = iedges.at(math::tensor::MatMulNode::Y_INPUT_IDX);
    auto* iedge_a = iedges.at(math::tensor::MatMulNode::A_INPUT_IDX);
    auto* iedge_b = iedges.at(math::tensor::MatMulNode::B_INPUT_IDX);

    auto& input_node_a = static_cast<data_flow::AccessNode&>(iedge_a->src());
    auto& input_node_b = static_cast<data_flow::AccessNode&>(iedge_b->src());
    auto& output_ptr = static_cast<data_flow::AccessNode&>(iedge_y->src());

    if (dataflow.in_degree(input_node_a) != 0 || dataflow.in_degree(input_node_b) != 0 ||
        dataflow.in_degree(output_ptr) != 0) {
        return false;
    }

    auto m = node_.m();
    auto n = node_.n();
    auto k = node_.k();

    // Compute batch count = product of all batch dimensions
    symbolic::Expression batch_count = symbolic::one();
    for (size_t i = 0; i < batch_dims_a; ++i) {
        batch_count = symbolic::mul(batch_count, layout_a.get_dim(i));
    }

    // Leading dimensions
    symbolic::Expression lda, ldb;
    if (trans_a == math::blas::BLAS_Transpose::No) {
        lda = layout_a.get_stride_innermost(1);
    } else {
        lda = layout_a.get_stride_innermost(0);
    }
    if (trans_b == math::blas::BLAS_Transpose::No) {
        ldb = layout_b.get_stride_innermost(1);
    } else {
        ldb = layout_b.get_stride_innermost(0);
    }
    auto ldc = n;

    // Strides between batches: stride of the outermost batch dim (last batch dim before matrix dims)
    // For contiguous layout, stride_a = m*k, stride_b = k*n, stride_c = m*n
    auto stride_a = symbolic::mul(m, k);
    auto stride_b = symbolic::mul(k, n);
    auto stride_c = symbolic::mul(m, n);

    auto scalar_type = types::Scalar(prim_type.value());

    // Add new graph
    auto& new_sequence = builder.add_sequence_before(parent, block, block.debug_info());
    auto& gemm_block = builder.add_block(new_sequence, block.debug_info());

    // Add batched GEMM library node
    auto& batched_gemm = builder.add_library_node<math::blas::BatchedGEMMNode>(
        gemm_block,
        node_.debug_info(),
        cuda::ImplementationType_CUDAWithTransfers,
        precision,
        math::blas::BLAS_Layout::RowMajor,
        trans_a,
        trans_b,
        batch_count,
        m,
        n,
        k,
        lda,
        ldb,
        ldc,
        stride_a,
        stride_b,
        stride_c
    );

    // Create access nodes
    auto& a_access = builder.add_access(gemm_block, input_node_a.data(), node_.debug_info());
    auto& b_access = builder.add_access(gemm_block, input_node_b.data(), node_.debug_info());
    auto& c_access = builder.add_access(gemm_block, output_ptr.data(), node_.debug_info());
    auto& alpha_const = builder.add_constant(gemm_block, "1.0", scalar_type, node_.debug_info());
    auto& beta_const = builder.add_constant(gemm_block, "0.0", scalar_type, node_.debug_info());

    // Connect edges
    builder.add_computational_memlet(
        gemm_block, a_access, batched_gemm, "__A", {}, types::Pointer(scalar_type), node_.debug_info()
    );
    builder.add_computational_memlet(
        gemm_block, b_access, batched_gemm, "__B", {}, types::Pointer(scalar_type), node_.debug_info()
    );
    builder.add_computational_memlet(
        gemm_block, c_access, batched_gemm, "__C", {}, types::Pointer(scalar_type), node_.debug_info()
    );
    builder
        .add_computational_memlet(gemm_block, alpha_const, batched_gemm, "__alpha", {}, scalar_type, node_.debug_info());
    builder
        .add_computational_memlet(gemm_block, beta_const, batched_gemm, "__beta", {}, scalar_type, node_.debug_info());

    // Clean up old block
    builder.clear_code_node_legacy(block, node_);
    builder.remove_child(parent, index + 1);

    return true;
}

} // namespace offloading
} // namespace sdfg
