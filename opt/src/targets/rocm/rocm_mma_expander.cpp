#include "sdfg/targets/rocm/rocm_mma.h"

namespace sdfg::gpu::rocm {

bool RocmMmaExpander::matches_possible_mma_pattern(const math::tensor::MatMulNode& node) const {
    auto* mma_arch = arch_.mma_support();

    if (!mma_arch) {
        return false;
    }

    // basic sanity checks
    auto& dims_a = node.layout_a();
    auto& dims_b = node.layout_b();
    if (dims_a.dims() != 2 || dims_b.dims() != 2) {
        return false;
    }

    if (!symbolic::eq(dims_a.get_dim(1), dims_b.get_dim(0))) {
        // K dimension must match
        return false;
    }

    auto layout_a = dims_a.is_2d_col_or_row_major();
    auto layout_b = dims_b.is_2d_col_or_row_major();
    if (layout_a == math::tensor::TensorLayout::LAYOUT_OTHER || layout_b == math::tensor::TensorLayout::LAYOUT_OTHER ||
        node.layout_y().is_2d_col_or_row_major() == math::tensor::TensorLayout::LAYOUT_OTHER) {
        // only support row-major or col-major layouts
        return false;
    }

    auto& m = dims_a.get_dim(0);
    auto& n = dims_b.get_dim(1);
    auto& k = dims_a.get_dim(1);

    auto m_blocks = GpuMmaSupport::get_integer_block_count(m, mma_arch->mma_block_m);
    auto n_blocks = GpuMmaSupport::get_integer_block_count(n, mma_arch->mma_block_n);
    auto k_blocks = GpuMmaSupport::get_integer_block_count(k, mma_arch->mma_block_k);

    if (!m_blocks || !n_blocks || !k_blocks) {
        return false;
    }

    return mma_arch->valid_block_counts(mma_arch->mma_block_m, m_blocks, n_blocks, k_blocks);
}

passes::LibNodeExpander::ExpandOutcome RocmMmaExpander::handle_expand(
    LibNodeExpander::ExpandContext& context, structured_control_flow::Block& block, math::tensor::MatMulNode& node
) const {
    // should only ever arrive here, if we already checked node preconditions

    node.implementation_type() = ImplementationType_ROCM_MMA;

    return context.successfully_modified_node_only();
}


} // namespace sdfg::gpu::rocm
