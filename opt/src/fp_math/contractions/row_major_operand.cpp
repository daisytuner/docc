#include "sdfg/fp_math/contractions/row_major_operand.h"

#include <utility>
#include <vector>

#include "contraction_kernels.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/library_nodes/math/blas/gemm_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/matmul_node.h"
#include "sdfg/data_flow/library_nodes/stdlib/free.h"
#include "sdfg/data_flow/library_nodes/stdlib/malloc.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/types/pointer.h"
#include "sdfg/types/scalar.h"
#include "sdfg/types/tensor.h"
#include "sdfg/types/utils.h"

namespace sdfg {
namespace fp_math {

using math::blas::BLAS_Transpose;
using math::blas::GEMMNode;
using math::tensor::MatMulNode;
using math::tensor::TensorLayout;

bool b_stored_transposed(const data_flow::LibraryNode& node) {
    if (auto* gemm = dynamic_cast<const GEMMNode*>(&node)) {
        return gemm->trans_b() == BLAS_Transpose::Trans;
    }
    if (auto* matmul = dynamic_cast<const MatMulNode*>(&node)) {
        return TensorLayout::is_last_dims_col_or_row_major(matmul->layout_b().strides()) ==
               TensorLayout::LAYOUT_COL_MAJOR;
    }
    return false;
}

RowMajorB read_b_row_major(builder::StructuredSDFGBuilder& builder, data_flow::LibraryNode& node) {
    auto& block = static_cast<structured_control_flow::Block&>(*node.get_parent().get_parent());
    auto& parent = static_cast<structured_control_flow::Sequence&>(*block.get_parent());
    auto& dataflow = block.dataflow();
    auto debug_info = node.debug_info();
    auto gemm = dynamic_cast<GEMMNode*>(&node);
    auto matmul = dynamic_cast<MatMulNode*>(&node);
    std::string connector = gemm ? "__B" : "B";
    auto k = gemm ? gemm->k() : matmul->k();
    auto n = gemm ? gemm->n() : matmul->n();
    auto stored = gemm ? TensorLayout({k, n}, {symbolic::one(), gemm->ldb()}, symbolic::zero()) : matmul->layout_b();
    auto row_major = gemm
                         ? TensorLayout({k, n}, {n, symbolic::one()}, symbolic::zero())
                         : TensorLayout(stored.shape(), TensorLayout::linear_strides(stored.shape()), symbolic::zero());

    data_flow::Memlet* read_b = nullptr;
    for (auto& memlet : dataflow.in_edges(node)) {
        if (memlet.dst_conn() == connector) {
            read_b = &memlet;
        }
    }
    auto& source = static_cast<data_flow::AccessNode&>(read_b->src());
    types::Scalar element(detail::contraction_type(node));
    types::Pointer pointer(element);
    auto copy = builder.find_new_name(source.data() + "_t");
    builder.add_container(copy, pointer);

    auto& allocation = builder.add_block_before(parent, block, debug_info);
    auto bytes = symbolic::mul(row_major.total_elements(), types::get_type_size(element, false));
    auto& malloc = builder.add_library_node<stdlib::MallocNode>(allocation, debug_info, bytes);
    builder.add_computational_memlet(
        allocation, malloc, "_ret", builder.add_access(allocation, copy, debug_info), {}, pointer, debug_info
    );
    auto& copy_block = builder.add_block_before(parent, block, debug_info);
    auto& transpose = builder.add_library_node<math::tensor::TensorCopyNode>(copy_block, debug_info, stored, row_major);
    builder.add_computational_memlet(
        copy_block,
        builder.add_access(copy_block, source.data(), debug_info),
        transpose,
        "X",
        {},
        types::Tensor(element, stored),
        debug_info
    );
    builder.add_computational_memlet(
        copy_block,
        builder.add_access(copy_block, copy, debug_info),
        transpose,
        "Y",
        {},
        types::Tensor(element, row_major),
        debug_info
    );
    auto& release = builder.add_block_after(parent, block, debug_info);
    auto& free = builder.add_library_node<stdlib::FreeNode>(release, debug_info);
    builder.add_computational_memlet(
        release,
        builder.add_access(release, copy, debug_info),
        free,
        "_ptr",
        {},
        types::Pointer(types::Scalar(types::PrimitiveType::Void)),
        debug_info
    );

    data_flow::LibraryNode* contraction = &node;
    if (gemm) {
        contraction = &builder.add_library_node<GEMMNode>(
            block,
            debug_info,
            std::as_const(*gemm).implementation_type(),
            gemm->precision(),
            gemm->layout(),
            gemm->trans_a(),
            BLAS_Transpose::No,
            gemm->m(),
            n,
            k,
            gemm->lda(),
            n,
            gemm->ldc()
        );
    } else {
        matmul->layout_b() = row_major;
    }
    std::vector<data_flow::Memlet*> inputs, outputs;
    for (auto& memlet : dataflow.in_edges(node)) {
        inputs.push_back(&memlet);
    }
    for (auto& memlet : dataflow.out_edges(node)) {
        outputs.push_back(&memlet);
    }
    for (auto* memlet : inputs) {
        auto* src = &memlet->src();
        if (memlet == read_b) {
            src = &builder.add_access(block, copy, debug_info);
        } else if (contraction == &node) {
            continue;
        }
        builder.add_memlet(
            block,
            *src,
            memlet->src_conn(),
            *contraction,
            memlet->dst_conn(),
            memlet->subset(),
            memlet->base_type(),
            memlet->debug_info()
        );
        builder.remove_memlet(block, *memlet);
    }
    for (auto* memlet : contraction == &node ? std::vector<data_flow::Memlet*>{} : outputs) {
        builder.add_memlet(
            block,
            *contraction,
            memlet->src_conn(),
            memlet->dst(),
            memlet->dst_conn(),
            memlet->subset(),
            memlet->base_type(),
            memlet->debug_info()
        );
        builder.remove_memlet(block, *memlet);
    }
    if (dataflow.in_degree(source) == 0 && dataflow.out_degree(source) == 0) {
        builder.remove_node(block, source);
    }
    if (contraction != &node) {
        builder.remove_node(block, node);
    }
    return {static_cast<math::tensor::TensorCopyNode&>(transpose), *contraction};
}

} // namespace fp_math
} // namespace sdfg
