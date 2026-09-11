#include "sdfg/targets/gpu/gpu_mma.h"

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

passes::LibNodeExpander::ExpandOutcome GpuMmaExpander::handle_expand(
    LibNodeExpander::ExpandContext& context, structured_control_flow::Block& block, math::tensor::MatMulNode& node
) const {
    auto result_layout = node.layout_y();
    auto k_dim = node.layout_a().get_dim(1);
    auto m_dim = node.layout_a().get_dim(0);
    auto n_dim = node.layout_b().get_dim(1);

    auto mma_tiling = get_mma_tiling({result_layout.get_dim(0), result_layout.get_dim(1), k_dim});

    auto standalone = context.replacement_requires_access_nodes({InputUse::Scalar, InputUse::Scalar, InputUse::Scalar});

    if (standalone) {
        auto& builder = standalone->builder();
        auto thread_x = symbolic::symbol(builder.find_new_name("wave_x"));
        auto threads_x_count = symbolic::integer(mma_tiling.macro_blocks_m * mma_tiling.threads_per_mma_block_m);
        builder.add_container(
            thread_x->get_name(), types::Scalar(types::get_primitive_type_to_hold_upper_bound(threads_x_count))
        );

        auto threads_y_count = symbolic::integer(mma_tiling.macro_blocks_n);
        auto& col_map = standalone->replace_with_structured_loop(
            AccessNodeExpand::LoopType::Map,
            thread_x,
            symbolic::Lt(thread_x, threads_x_count),
            symbolic::zero(),
            symbolic::add(thread_x, symbolic::integer(1)),
            get_schedule_type(TargetLevel::X_BLOCK, threads_x_count)
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
            get_schedule_type(TargetLevel::Y_BLOCK, symbolic::integer(mma_tiling.macro_blocks_n))
        );

        auto& inner_block = builder.add_block(row_map.root());
        auto& input_y = standalone->add_scalar_input_access(inner_block, 0);
        auto& input_a = standalone->add_scalar_input_access(inner_block, 1);
        auto& input_b = standalone->add_scalar_input_access(inner_block, 2);

        auto& inner_node = static_cast<math::tensor::MatMulNode&>(builder.copy_node(inner_block, node));
        if (mma_tiling.macro_blocks_n > 1) {
            inner_node.layout_a() = add_to_offset(
                node.layout_a(),
                symbolic::
                    mul(row_inner,
                        symbolic::mul(node.layout_a().get_stride(0), symbolic::integer(mma_tiling.mma_block_m)))
            );
        }
        if (mma_tiling.macro_blocks_m > 1) {
            inner_node.layout_b() = add_to_offset(
                node.layout_b(),
                symbolic::
                    mul(col_inner,
                        symbolic::mul(node.layout_b().get_stride(1), symbolic::integer(mma_tiling.mma_block_n)))
            );
        }
        if (mma_tiling.macro_blocks_m > 1 || mma_tiling.macro_blocks_n > 1) {
            symbolic::Expression add = symbolic::zero();
            if (mma_tiling.macro_blocks_m > 1) {
                add = symbolic::
                    add(add,
                        symbolic::
                            mul(row_inner,
                                symbolic::mul(node.layout_y().get_stride(0), symbolic::integer(mma_tiling.mma_block_m)))
                    );
            }
            if (mma_tiling.macro_blocks_n > 1) {
                add = symbolic::
                    add(add,
                        symbolic::
                            mul(col_inner,
                                symbolic::mul(node.layout_y().get_stride(1), symbolic::integer(mma_tiling.mma_block_n)))
                    );
            }
            inner_node.layout_y() = add_to_offset(node.layout_y(), add);
        }

        set_implementation_type_mma(inner_node, mma_tiling);

        types::Scalar scalar_type(inner_node.fixed_quantization());
        types::Pointer ptr_type(scalar_type);

        builder.add_computational_memlet(
            inner_block, input_y, inner_node, inner_node.input(math::tensor::MatMulNode::Y_INPUT_IDX), {}, ptr_type
        );
        builder.add_computational_memlet(
            inner_block, input_a, inner_node, inner_node.input(math::tensor::MatMulNode::A_INPUT_IDX), {}, ptr_type
        );
        builder.add_computational_memlet(
            inner_block, input_b, inner_node, inner_node.input(math::tensor::MatMulNode::B_INPUT_IDX), {}, ptr_type
        );

        return standalone->successfully_expanded();
    } else {
        return context.unable();
    }
}

math::tensor::TensorLayout GpuMmaExpander::
    add_to_offset(const math::tensor::TensorLayout& layout, const symbolic::Expression& offset_add) const {
    auto new_offset = symbolic::add(layout.offset(), offset_add);
    return {layout.shape(), layout.strides(), new_offset};
}

} // namespace sdfg::gpu
