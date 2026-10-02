#include "sdfg/targets/gpu/gpu_mma_expander.h"

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/targets/gpu/gpu_mma_fill_node.h"
#include "sdfg/targets/gpu/gpu_mma_fragment_eltwise_add_node.h"
#include "sdfg/targets/gpu/gpu_mma_fragment_load_node.h"
#include "sdfg/targets/gpu/gpu_mma_fragment_store_node.h"
#include "sdfg/targets/gpu/gpu_mma_matmul_node.h"
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

void GpuMmaExpander::create_fragment_load(
    passes::LibNodeExpander::AccessNodeExpand& standalone,
    types::PrimitiveType input_type,
    const data_flow::ImplementationType& impl_type,
    const DebugInfo& org_debug_info,
    builder::StructuredSDFGBuilder& builder,
    std::string frag_name,
    const MmaBlockSize& mma_block_size,
    MmaFragmentType frag_type,
    const GpuMmaFromMemoryLayout& from_memory,
    unsigned input_index,
    Block& block
) {
    auto& load_node = builder.add_library_node<
        GpuMmaFragmentLoadNode>(block, org_debug_info, mma_block_size, frag_type, from_memory, input_type, impl_type);
    auto& frag_ptr = builder.add_access(block, frag_name);
    auto& mem_ptr = standalone.add_scalar_input_access(block, input_index);
    builder.add_computational_memlet(
        block,
        frag_ptr,
        load_node,
        load_node.input(GpuMmaFragmentLoadNode::FRAG_INPUT_IDX),
        {},
        types::Pointer(types::Scalar(input_type))
    );
    builder.add_computational_memlet(
        block,
        mem_ptr,
        load_node,
        load_node.input(GpuMmaFragmentLoadNode::PTR_INPUT_IDX),
        {},
        types::Pointer(types::Scalar(input_type))
    );
}

void GpuMmaExpander::create_fragment_store(
    passes::LibNodeExpander::AccessNodeExpand& standalone,
    types::PrimitiveType output_type,
    const data_flow::ImplementationType& impl_type,
    const DebugInfo& org_debug_info,
    builder::StructuredSDFGBuilder& builder,
    std::string frag_name,
    const MmaBlockSize& mma_block_size,
    MmaFragmentType frag_type,
    const GpuMmaFromMemoryLayout& to_memory,
    unsigned input_index,
    Block& block
) {
    auto& store_node = builder.add_library_node<
        GpuMmaFragmentStoreNode>(block, org_debug_info, mma_block_size, frag_type, to_memory, output_type, impl_type);
    auto& frag_ptr = builder.add_access(block, frag_name);
    auto& mem_ptr = standalone.add_scalar_input_access(block, input_index);
    builder.add_computational_memlet(
        block,
        frag_ptr,
        store_node,
        store_node.input(GpuMmaFragmentStoreNode::FRAG_INPUT_IDX),
        {},
        types::Pointer(types::Scalar(output_type))
    );
    builder.add_computational_memlet(
        block,
        mem_ptr,
        store_node,
        store_node.input(GpuMmaFragmentStoreNode::PTR_INPUT_IDX),
        {},
        types::Pointer(types::Scalar(output_type))
    );
}

passes::LibNodeExpander::ExpandOutcome GpuMmaExpander::expand_mma_standalone(
    LibNodeExpander::AccessNodeExpand& standalone,
    const GpuArch& arch,
    GpuMmaTiling& mma_tiling,
    const math::tensor::TensorLayout& layout_a,
    const math::tensor::TensorLayout& layout_b,
    const math::tensor::TensorLayout& layout_y,
    types::PrimitiveType input_type,
    types::PrimitiveType acc_type,
    types::PrimitiveType output_type,
    const data_flow::ImplementationType& impl_type,
    bool include_c_add,
    const DebugInfo& org_debug_info
) {
    auto* mma_arch = arch.mma_support();
    auto mma_impl_type = mma_arch->get_mma_impl_type();

    auto& m_dim = layout_a.get_dim(0);
    auto& k_dim = layout_b.get_dim(1);

    auto& builder = standalone.builder();
    auto thread_x = symbolic::symbol(builder.find_new_name("wave_x"));
    auto threads_x_count = symbolic::integer(mma_tiling.macro_blocks_m * mma_tiling.threads_per_mma_block_m);
    builder.add_container(
        thread_x->get_name(), types::Scalar(types::get_primitive_type_to_hold_upper_bound(threads_x_count))
    );

    auto acc_frag_name = builder.find_new_name("mma_acc");
    types::Pointer acc_frag_type{types::Scalar(acc_type)};
    mma_arch->set_mma_fragment_storage_type(
        acc_frag_type.storage_type(),
        mma_tiling.mma_block_size,
        MmaFragmentType::C,
        MmaFragmentLayout::MMA_LAYOUT_UNSPECIFIED
    );
    builder.add_container(acc_frag_name, acc_frag_type);

    types::Pointer a_frag_type{types::Scalar(input_type)};
    auto a_frag_name = builder.find_new_name("mma_a");
    auto a_col_major = layout_a.is_2d_col_or_row_major() == math::tensor::TensorLayout::LAYOUT_COL_MAJOR;
    mma_arch->set_mma_fragment_storage_type(
        a_frag_type.storage_type(),
        mma_tiling.mma_block_size,
        MmaFragmentType::A,
        a_col_major ? MmaFragmentLayout::MMA_LAYOUT_COL_MAJOR : MmaFragmentLayout::MMA_LAYOUT_ROW_MAJOR
    );
    builder.add_container(a_frag_name, a_frag_type);

    types::Pointer b_frag_type{types::Scalar(input_type)};
    auto b_frag_name = builder.find_new_name("mma_b");
    auto b_col_major = layout_b.is_2d_col_or_row_major() == math::tensor::TensorLayout::LAYOUT_COL_MAJOR;
    mma_arch->set_mma_fragment_storage_type(
        b_frag_type.storage_type(),
        mma_tiling.mma_block_size,
        MmaFragmentType::B,
        b_col_major ? MmaFragmentLayout::MMA_LAYOUT_COL_MAJOR : MmaFragmentLayout::MMA_LAYOUT_ROW_MAJOR
    );
    builder.add_container(b_frag_name, b_frag_type);
    auto y_col_major = layout_y.is_2d_col_or_row_major() == math::tensor::TensorLayout::LAYOUT_COL_MAJOR;

    auto threads_y_count = symbolic::integer(mma_tiling.macro_blocks_n);
    auto& col_map = standalone.replace_with_structured_loop(
        AccessNodeExpand::LoopType::Map,
        thread_x,
        symbolic::Lt(thread_x, threads_x_count),
        symbolic::zero(),
        symbolic::add(thread_x, symbolic::integer(1)),
        ScheduleType_GPU_Offload::create(arch, TargetLevel::X_BLOCK, threads_x_count)
    );

    symbolic::Expression brow_in_tile;
    if (mma_tiling.macro_blocks_m > 1) {
        auto wave_row_name = builder.find_new_name("wave_row");
        auto brow_in_tile_sym = symbolic::symbol(wave_row_name);
        brow_in_tile = brow_in_tile_sym;
        builder.add_container(wave_row_name, types::Scalar(types::get_primitive_type_to_hold_upper_bound(m_dim)));
        builder.add_assignments(
            col_map.root(),
            {{brow_in_tile_sym, symbolic::div(thread_x, symbolic::integer(mma_tiling.threads_per_mma_block_m))}}
        );
    } else {
        brow_in_tile = symbolic::zero();
    }

    auto bcol_in_tile = symbolic::symbol(builder.find_new_name("wave_col"));
    builder.add_container(
        bcol_in_tile->get_name(), types::Scalar(types::get_primitive_type_to_hold_upper_bound(threads_y_count))
    );

    auto& row_map = builder.add_map(
        col_map.root(),
        bcol_in_tile,
        symbolic::Lt(bcol_in_tile, threads_y_count),
        symbolic::zero(),
        symbolic::add(bcol_in_tile, symbolic::integer(1)),
        ScheduleType_GPU_Offload::create(arch, TargetLevel::Y_BLOCK, symbolic::integer(mma_tiling.macro_blocks_m))
    );

    auto lda = layout_a.get_stride(a_col_major ? 1 : 0);
    auto a_offset = SymEngine::add({
        layout_a.offset(),
        SymEngine::mul({brow_in_tile, symbolic::integer(mma_tiling.mma_block_size.m), layout_a.get_stride(0)}),
        SymEngine::mul({bcol_in_tile, symbolic::integer(mma_tiling.mma_block_size.n), layout_a.get_stride(1)}),
    });
    auto ldb = layout_b.get_stride(b_col_major ? 1 : 0);
    auto b_offset = SymEngine::add(
        {layout_b.offset(),
         SymEngine::mul({brow_in_tile, symbolic::integer(mma_tiling.mma_block_size.n), layout_b.get_stride(0)}),
         SymEngine::mul({bcol_in_tile, symbolic::integer(mma_tiling.mma_block_size.k), layout_b.get_stride(1)})}
    );
    auto ldc = layout_y.get_stride(y_col_major ? 1 : 0);
    auto y_offset = SymEngine::add(
        {layout_y.offset(),
         SymEngine::mul({bcol_in_tile, symbolic::integer(mma_tiling.mma_block_size.m), layout_y.get_stride(0)}),
         SymEngine::mul({brow_in_tile, symbolic::integer(mma_tiling.mma_block_size.n), layout_y.get_stride(1)})}
    );

    auto& per_wavefront_block = builder.add_block(row_map.root());
    {
        auto& fill_node = builder.add_library_node<
            GpuMmaFillNode>(per_wavefront_block, org_debug_info, mma_tiling.mma_block_size, acc_type, acc_frag_name);
        auto& acc_frag_ptr = builder.add_access(per_wavefront_block, acc_frag_name);
        builder.add_computational_memlet(
            per_wavefront_block, acc_frag_ptr, fill_node, fill_node.input(0), {}, types::Pointer(types::Scalar(acc_type))
        );
    }

    auto k_tile = symbolic::symbol(builder.find_new_name("tile_k"));
    builder.add_container(k_tile->get_name(), types::Scalar(types::get_primitive_type_to_hold_upper_bound(k_dim)));
    auto& k_sweep = builder.add_for(
        row_map.root(),
        k_tile,
        symbolic::Lt(k_tile, k_dim),
        symbolic::zero(),
        symbolic::add(k_tile, symbolic::integer(mma_tiling.mma_block_size.k))
    );

    auto& load_block = builder.add_block(k_sweep.root());
    create_fragment_load(
        standalone,
        input_type,
        impl_type,
        org_debug_info,
        builder,
        a_frag_name,
        mma_tiling.mma_block_size,
        MmaFragmentType::A,
        {.offset = a_offset, .ldstride = lda, .layout = MMA_LAYOUT_UNSPECIFIED},
        1,
        load_block
    );
    create_fragment_load(
        standalone,
        input_type,
        impl_type,
        org_debug_info,
        builder,
        b_frag_name,
        mma_tiling.mma_block_size,
        MmaFragmentType::B,
        {.offset = b_offset, .ldstride = ldb, .layout = MMA_LAYOUT_UNSPECIFIED},
        2,
        load_block
    );

    auto& inner_block = builder.add_block(k_sweep.root());

    create_fragment_mma(
        standalone,
        arch,
        mma_tiling,
        input_type,
        a_frag_name,
        b_frag_name,
        acc_type,
        acc_frag_name,
        impl_type,
        org_debug_info,
        builder,
        inner_block
    );

    auto y_layout = y_col_major ? MMA_LAYOUT_COL_MAJOR : MMA_LAYOUT_ROW_MAJOR;
    std::string store_frag_name;
    if (include_c_add) {
        auto c_frag_name = builder.find_new_name("mma_c");
        types::Pointer c_type{types::Scalar(output_type)};
        mma_arch->set_mma_fragment_storage_type(
            c_type.storage_type(),
            mma_tiling.mma_block_size,
            MmaFragmentType::C,
            MmaFragmentLayout::MMA_LAYOUT_UNSPECIFIED
        );
        builder.add_container(c_frag_name, c_type);
        auto& load_c_block = builder.add_block(row_map.root());
        create_fragment_load(
            standalone,
            output_type,
            impl_type,
            org_debug_info,
            builder,
            c_frag_name,
            mma_tiling.mma_block_size,
            MmaFragmentType::C,
            {.offset = y_offset, .ldstride = ldc, .layout = y_layout},
            0,
            load_c_block
        );

        store_frag_name = builder.find_new_name("mma_d");
        types::Pointer d_type{types::Scalar(output_type)};
        mma_arch->set_mma_fragment_storage_type(
            d_type.storage_type(),
            mma_tiling.mma_block_size,
            MmaFragmentType::C,
            MmaFragmentLayout::MMA_LAYOUT_UNSPECIFIED
        );
        builder.add_container(store_frag_name, d_type);
        auto& eltwise_add_block = builder.add_block(row_map.root());
        create_eltwise_add_block(
            standalone,
            mma_tiling.mma_block_size,
            acc_type,
            output_type,
            impl_type,
            org_debug_info,
            builder,
            c_frag_name,
            acc_frag_name,
            store_frag_name,
            eltwise_add_block
        );
    } else {
        assert(output_type == acc_type && "Output and Accumulate type must be identical if we do not add C");
        store_frag_name = acc_frag_name;
    }

    auto& store_block = builder.add_block(row_map.root());
    create_fragment_store(
        standalone,
        output_type,
        impl_type,
        org_debug_info,
        builder,
        store_frag_name,
        mma_tiling.mma_block_size,
        MmaFragmentType::C,
        {.offset = y_offset, .ldstride = ldc, .layout = y_layout},
        0,
        store_block
    );

    return standalone.successfully_expanded();
}

void GpuMmaExpander::create_fragment_mma(
    LibNodeExpander::AccessNodeExpand& standalone,
    const GpuArch& arch,
    GpuMmaTiling& mma_tiling,
    types::PrimitiveType input_type,
    const std::string& frag_a,
    const std::string& frag_b,
    types::PrimitiveType acc_type,
    const std::string& frag_acc,
    const data_flow::ImplementationType& impl_type,
    const DebugInfo& org_debug_info,
    builder::StructuredSDFGBuilder& builder,
    structured_control_flow::Block& block
) {
    auto& inner_node = builder.add_library_node<
        GpuMmaMatmulNode>(block, org_debug_info, mma_tiling.mma_block_size, input_type, acc_type, impl_type);

    types::Scalar input_scalar_type(input_type);
    types::Scalar acc_scalar_type(acc_type);
    types::Pointer ptr_type(input_scalar_type);

    auto& acc_access = builder.add_access(block, frag_acc);
    builder.add_computational_memlet(
        block,
        acc_access,
        inner_node,
        inner_node.input(math::tensor::MatMulNode::Y_INPUT_IDX),
        {},
        types::Pointer(acc_scalar_type)
    );
    auto& a_access = builder.add_access(block, frag_a);
    builder.add_computational_memlet(
        block,
        a_access,
        inner_node,
        inner_node.input(math::tensor::MatMulNode::A_INPUT_IDX),
        {},
        types::Pointer(input_scalar_type)
    );
    auto& b_access = builder.add_access(block, frag_b);
    builder.add_computational_memlet(
        block,
        b_access,
        inner_node,
        inner_node.input(math::tensor::MatMulNode::B_INPUT_IDX),
        {},
        types::Pointer(input_scalar_type)
    );
}

passes::LibNodeExpander::ExpandOutcome GpuMmaExpander::handle_expand(
    LibNodeExpander::ExpandContext& context, structured_control_flow::Block& block, math::tensor::MatMulNode& node
) const {
    auto result_layout = node.layout_y();
    auto k_dim = node.layout_a().get_dim(1);

    auto mma_tiling = get_mma_tiling({result_layout.get_dim(0), result_layout.get_dim(1), k_dim});

    auto input_type = node.uniform_quantization(node.get_parent()).value();
    auto output_type = input_type;

    auto new_impl_type = arch_->mma_support()->get_mma_impl_type();

    if (!new_impl_type.has_value()) {
        return context.unable();
    }

    auto standalone = context.replacement_requires_access_nodes({InputUse::Scalar, InputUse::Scalar, InputUse::Scalar});

    if (standalone) {
        return expand_mma_standalone(
            *standalone,
            *arch_,
            mma_tiling,
            node.layout_a(),
            node.layout_b(),
            result_layout,
            input_type,
            output_type,
            output_type,
            new_impl_type.value(),
            true,
            node.debug_info()
        );
    } else {
        return context.unable();
    }
}

void GpuMmaExpander::create_eltwise_add_block(
    AccessNodeExpand& standalone,
    const MmaBlockSize& mma_block_size,
    types::PrimitiveType acc_type,
    types::PrimitiveType output_type,
    const data_flow::ImplementationType& impl_type,
    const DebugInfo& debug_info,
    builder::StructuredSDFGBuilder& builder,
    const std::string& c_frag_name,
    const std::string& acc_frag_name,
    const std::string& store_frag_name,
    Block& block
) {
    auto& eltwise_add_node = builder.add_library_node<
        GpuMmaFragmentEltwiseAddNode>(block, debug_info, mma_block_size, output_type, acc_type, impl_type);

    auto& c_access = builder.add_access(block, c_frag_name);
    builder.add_computational_memlet(
        block,
        c_access,
        eltwise_add_node,
        eltwise_add_node.input(GpuMmaFragmentEltwiseAddNode::FRAG_C_INPUT_IDX),
        {},
        types::Pointer(types::Scalar(output_type))
    );

    auto& acc_access = builder.add_access(block, acc_frag_name);
    builder.add_computational_memlet(
        block,
        acc_access,
        eltwise_add_node,
        eltwise_add_node.input(GpuMmaFragmentEltwiseAddNode::FRAG_ACC_INPUT_IDX),
        {},
        types::Pointer(types::Scalar(acc_type))
    );

    auto& store_access = builder.add_access(block, store_frag_name);
    builder.add_computational_memlet(
        block,
        store_access,
        eltwise_add_node,
        eltwise_add_node.input(GpuMmaFragmentEltwiseAddNode::FRAG_D_INPUT_IDX),
        {},
        types::Pointer(types::Scalar(output_type))
    );
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

    auto m_blocks = GpuMmaSupport::get_integer_block_count(m, mma_arch->mma_block_size.m);
    auto n_blocks = GpuMmaSupport::get_integer_block_count(n, mma_arch->mma_block_size.n);
    auto k_blocks = GpuMmaSupport::get_integer_block_count(k, mma_arch->mma_block_size.k);

    if (!m_blocks || !n_blocks || !k_blocks) {
        return false;
    }

    if (!mma_arch->valid_block_counts(mma_arch->mma_block_size.m, m_blocks, n_blocks, k_blocks)) {
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
