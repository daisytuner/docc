#include "sdfg/targets/gpu/gpu_mma_expander.h"

#include "sdfg/symbolic/symbolic.h"
#include "sdfg/targets/gpu/gpu_offload_schedule_type.h"
#include "sdfg/types/type.h"
#include "sdfg/types/utils.h"

namespace sdfg::gpu {

const passes::LibNodeExpander* GpuMmaExpander::for_lib_node(const data_flow::LibraryNode& node) const {
    if (auto* ptr = CodeLibNodeExpander<math::tensor::MatMulNode>::for_lib_node(node)) {
        if (this->matches_possible_mma_pattern(static_cast<const math::tensor::MatMulNode&>(node))) {
            return this;
        }
    }
    return nullptr;
}

passes::LibNodeExpander::ExpandOutcome GpuMmaExpander::expand_mma(
    LibNodeExpander::AccessNodeExpand& standalone,
    const GpuArch& arch,
    GpuMmaTiling& mma_tiling,
    const math::tensor::TensorLayout& layout_a,
    const math::tensor::TensorLayout& layout_b,
    const math::tensor::TensorLayout& layout_y,
    types::PrimitiveType input_type,
    types::PrimitiveType output_type,
    const data_flow::ImplementationType& impl_type
) {
    auto& m_dim = layout_a.get_dim(0);

    auto& builder = standalone.builder();
    auto thread_x = symbolic::symbol(builder.find_new_name("wave_x"));
    auto threads_x_count = symbolic::integer(mma_tiling.macro_blocks_m * mma_tiling.threads_per_mma_block_m);
    builder.add_container(
        thread_x->get_name(), types::Scalar(types::get_primitive_type_to_hold_upper_bound(threads_x_count))
    );

    auto threads_y_count = symbolic::integer(mma_tiling.macro_blocks_n);
    auto& col_map = standalone.replace_with_structured_loop(
        AccessNodeExpand::LoopType::Map,
        thread_x,
        symbolic::Lt(thread_x, threads_x_count),
        symbolic::zero(),
        symbolic::add(thread_x, symbolic::integer(1)),
        ScheduleType_GPU_Offload::create(arch, TargetLevel::X_BLOCK, threads_x_count)
    );

    symbolic::Symbol col_inner;
    if (mma_tiling.macro_blocks_m > 1) {
        col_inner = symbolic::symbol(builder.find_new_name("wave_col"));
        builder
            .add_container(col_inner->get_name(), types::Scalar(types::get_primitive_type_to_hold_upper_bound(m_dim)));
        builder.add_assignments(
            col_map.root(),
            {{col_inner, symbolic::div(thread_x, symbolic::integer(mma_tiling.threads_per_mma_block_m))}}
        );
    }

    auto row_inner = symbolic::symbol(builder.find_new_name("wave_row"));
    builder.add_container(
        row_inner->get_name(), types::Scalar(types::get_primitive_type_to_hold_upper_bound(threads_y_count))
    );

    auto& row_map = builder.add_map(
        col_map.root(),
        row_inner,
        symbolic::Lt(row_inner, threads_y_count),
        symbolic::zero(),
        symbolic::add(row_inner, symbolic::integer(1)),
        ScheduleType_GPU_Offload::create(arch, TargetLevel::Y_BLOCK, symbolic::integer(mma_tiling.macro_blocks_n))
    );

    auto& inner_block = builder.add_block(row_map.root());
    auto& input_y = standalone.add_scalar_input_access(inner_block, 0);
    auto& input_a = standalone.add_scalar_input_access(inner_block, 1);
    auto& input_b = standalone.add_scalar_input_access(inner_block, 2);

    math::tensor::TensorLayout new_layout_a = layout_a;
    if (mma_tiling.macro_blocks_n > 1) {
        new_layout_a = add_to_offset(
            layout_a,
            symbolic::mul(row_inner, symbolic::mul(layout_a.get_stride(0), symbolic::integer(mma_tiling.mma_block_m)))
        );
    }
    math::tensor::TensorLayout new_layout_b = layout_b;
    if (mma_tiling.macro_blocks_m > 1) {
        new_layout_b = add_to_offset(
            layout_b,
            symbolic::mul(col_inner, symbolic::mul(layout_b.get_stride(1), symbolic::integer(mma_tiling.mma_block_n)))
        );
    }
    math::tensor::TensorLayout new_layout_y = layout_y;
    if (mma_tiling.macro_blocks_m > 1 || mma_tiling.macro_blocks_n > 1) {
        symbolic::Expression add = symbolic::zero();
        if (mma_tiling.macro_blocks_m > 1) {
            add = symbolic::add(
                add,
                symbolic::mul(row_inner, symbolic::mul(layout_y.get_stride(0), symbolic::integer(mma_tiling.mma_block_m)))
            );
        }
        if (mma_tiling.macro_blocks_n > 1) {
            add = symbolic::add(
                add,
                symbolic::mul(col_inner, symbolic::mul(layout_y.get_stride(1), symbolic::integer(mma_tiling.mma_block_n)))
            );
        }
        new_layout_y = add_to_offset(layout_y, add);
    }

    auto& inner_node = builder.add_library_node<
        math::tensor::MatMulNode>(inner_block, col_map.debug_info(), layout_a, layout_b, input_type, &layout_y, impl_type);

    types::Scalar input_scalar_type(input_type);
    types::Scalar output_scalar_type(output_type);
    types::Pointer ptr_type(input_scalar_type);

    builder.add_computational_memlet(
        inner_block,
        input_y,
        inner_node,
        inner_node.input(math::tensor::MatMulNode::Y_INPUT_IDX),
        {},
        types::Pointer(output_scalar_type)
    );
    builder.add_computational_memlet(
        inner_block,
        input_a,
        inner_node,
        inner_node.input(math::tensor::MatMulNode::A_INPUT_IDX),
        {},
        types::Pointer(input_scalar_type)
    );
    builder.add_computational_memlet(
        inner_block,
        input_b,
        inner_node,
        inner_node.input(math::tensor::MatMulNode::B_INPUT_IDX),
        {},
        types::Pointer(input_scalar_type)
    );

    return standalone.successfully_expanded();
}

passes::LibNodeExpander::ExpandOutcome GpuMmaExpander::handle_expand(
    LibNodeExpander::ExpandContext& context, structured_control_flow::Block& block, math::tensor::MatMulNode& node
) const {
    auto result_layout = node.layout_y();
    auto k_dim = node.layout_a().get_dim(1);

    auto mma_tiling = get_mma_tiling({result_layout.get_dim(0), result_layout.get_dim(1), k_dim});

    auto input_type = node.uniform_quantization(node.get_parent()).value();
    auto output_type = input_type;

    auto new_impl_type = arch_->mma_support()->get_matmul_impl_type(*arch_, mma_tiling);

    if (!new_impl_type.has_value()) {
        return context.unable();
    }

    auto standalone = context.replacement_requires_access_nodes({InputUse::Scalar, InputUse::Scalar, InputUse::Scalar});

    if (standalone) {
        return expand_mma(
            *standalone,
            *arch_,
            mma_tiling,
            node.layout_a(),
            node.layout_b(),
            result_layout,
            input_type,
            output_type,
            new_impl_type.value()
        );
    } else {
        return context.unable();
    }
}

math::tensor::TensorLayout GpuMmaExpander::
    add_to_offset(const math::tensor::TensorLayout& layout, const symbolic::Expression& offset_add) {
    auto new_offset = symbolic::add(layout.offset(), offset_add);
    return {layout.shape(), layout.strides(), new_offset};
}

bool GpuMmaExpander::matches_possible_mma_pattern(const math::tensor::MatMulNode& node) const {
    auto* mma_arch = arch_->mma_support();
    if (!mma_arch) {
        return false;
    }
    GpuMmaTiling dummy_tiling;
    if (!mma_arch->get_matmul_impl_type(*arch_, dummy_tiling).has_value()) {
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
        // only support row-major or col-major layout
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

    if (!mma_arch->valid_block_counts(mma_arch->mma_block_m, m_blocks, n_blocks, k_blocks)) {
        return false;
    }

    auto input_type = node.uniform_quantization(node.get_parent());
    auto output_type = input_type;

    if (!input_type || !output_type || input_type.value() == types::PrimitiveType::Void ||
        output_type.value() == types::PrimitiveType::Void) {
        return false;
    }

    return mma_arch->supported_types(input_type.value(), output_type.value());
}

GpuMmaTiling GpuMmaExpander::get_mma_tiling(const symbolic::MultiExpression& res_shape) const {
    auto* mma_arch = arch_->mma_support();
    if (!mma_arch) {
        throw std::runtime_error("No MMA architecture available for this GPU target.");
    }

    return mma_arch->get_mma_tiling(res_shape);
}

} // namespace sdfg::gpu
