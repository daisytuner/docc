#include "sdfg/targets/gpu/gpu_mma_expander.h"

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/reduce.h"
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
    const char* operand_label,
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
    auto& mem_memlet = builder.add_computational_memlet(
        block,
        mem_ptr,
        load_node,
        load_node.input(GpuMmaFragmentLoadNode::PTR_INPUT_IDX),
        {},
        types::Pointer(types::Scalar(input_type))
    );
    if (operand_label != nullptr) {
        mem_memlet.add_metadata(math::tensor::MatMulNode::OPERAND_METADATA_KEY, operand_label);
    }
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
    const char* operand_label,
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
    auto& mem_memlet = builder.add_computational_memlet(
        block,
        mem_ptr,
        store_node,
        store_node.input(GpuMmaFragmentStoreNode::PTR_INPUT_IDX),
        {},
        types::Pointer(types::Scalar(output_type))
    );
    if (operand_label != nullptr) {
        mem_memlet.add_metadata(math::tensor::MatMulNode::OPERAND_METADATA_KEY, operand_label);
    }
}

passes::LibNodeExpander::ExpandOutcome GpuMmaExpander::expand_mma_standalone(
    LibNodeExpander::AccessNodeExpand& standalone,
    const GpuArch& arch,
    GpuMmaTiling& mma_tiling,
    const math::tensor::TensorLayout& layout_a,
    const math::tensor::TensorLayout& layout_b,
    const math::tensor::TensorLayout& layout_y,
    types::PrimitiveType input_type,
    types::PrimitiveType output_type,
    bool include_c_add,
    const DebugInfo& org_debug_info,
    const std::array<int, 3>& args_order, // {y, a, b}
    bool ragged,
    int k_block_multiple
) {
    auto* mma_arch = arch.mma_support();

    auto& m_dim = layout_a.get_dim(0);
    auto& k_dim = layout_b.get_dim(0);

    auto impl_type = mma_arch->get_mma_impl_type(mma_tiling);

    auto& builder = standalone.builder();
    auto wave_row = symbolic::symbol(builder.find_new_name("wave_row"));
    auto waves_m = symbolic::integer(mma_tiling.get_mma_blocks_m_per_wavefront());
    builder.add_container(wave_row->get_name(), types::Scalar(types::get_primitive_type_to_hold_upper_bound(waves_m)));

    std::vector<std::string> acc_frag_names; // row-major over the wave's wt_m x wt_n blocks
    for (int f = 0; f < mma_tiling.wave_tiling.total_blocks(); ++f) {
        auto acc_frag_name = builder.find_new_name("mma_acc");
        types::Pointer acc_frag_type{types::Scalar(mma_tiling.acc_type)};
        mma_arch->set_mma_fragment_storage_type(
            acc_frag_type.storage_type(), mma_tiling, MmaFragmentType::C, MmaFragmentLayout::MMA_LAYOUT_UNSPECIFIED
        );
        builder.add_container(acc_frag_name, acc_frag_type);
        acc_frag_names.push_back(acc_frag_name);
    }

    auto a_col_major = layout_a.is_2d_col_or_row_major() == math::tensor::TensorLayout::LAYOUT_COL_MAJOR;
    MmaFragmentLayout a_frag_layout = a_col_major ? MmaFragmentLayout::MMA_LAYOUT_COL_MAJOR
                                                  : MmaFragmentLayout::MMA_LAYOUT_ROW_MAJOR;
    std::vector<std::string> a_frag_names;
    for (int i = 0; i < mma_tiling.wave_tiling.blocks_m; ++i) {
        types::Pointer a_frag_type{types::Scalar(input_type)};
        auto a_frag_name = builder.find_new_name("mma_a");
        mma_arch
            ->set_mma_fragment_storage_type(a_frag_type.storage_type(), mma_tiling, MmaFragmentType::A, a_frag_layout);
        builder.add_container(a_frag_name, a_frag_type);
        a_frag_names.push_back(a_frag_name);
    }

    auto b_col_major = layout_b.is_2d_col_or_row_major() == math::tensor::TensorLayout::LAYOUT_COL_MAJOR;
    MmaFragmentLayout b_frag_layout = b_col_major ? MmaFragmentLayout::MMA_LAYOUT_COL_MAJOR
                                                  : MmaFragmentLayout::MMA_LAYOUT_ROW_MAJOR;
    std::vector<std::string> b_frag_names;
    for (int j = 0; j < mma_tiling.wave_tiling.blocks_n; ++j) {
        types::Pointer b_frag_type{types::Scalar(input_type)};
        auto b_frag_name = builder.find_new_name("mma_b");
        mma_arch
            ->set_mma_fragment_storage_type(b_frag_type.storage_type(), mma_tiling, MmaFragmentType::B, b_frag_layout);
        builder.add_container(b_frag_name, b_frag_type);
        b_frag_names.push_back(b_frag_name);
    }
    auto y_col_major = layout_y.is_2d_col_or_row_major() == math::tensor::TensorLayout::LAYOUT_COL_MAJOR;

    auto waves_n = symbolic::integer(mma_tiling.get_mma_blocks_n_per_wavefront());
    // Each wave_row iteration is one whole wave: the MMA nodes below are wave-collective.
    auto wave_row_sched = ScheduleType_GPU_Offload::create(arch, TargetLevel::X_BLOCK, waves_m);
    ScheduleType_GPU_Offload::lanes(wave_row_sched, symbolic::integer(mma_tiling.threads_per_mma_wavefront));
    auto& col_map = standalone.replace_with_structured_loop(
        AccessNodeExpand::LoopType::Map,
        wave_row,
        symbolic::Lt(wave_row, waves_m),
        symbolic::zero(),
        symbolic::add(wave_row, symbolic::integer(1)),
        wave_row_sched
    );

    symbolic::Expression brow_in_tile = wave_row;

    auto bcol_in_tile = symbolic::symbol(builder.find_new_name("wave_col"));
    builder
        .add_container(bcol_in_tile->get_name(), types::Scalar(types::get_primitive_type_to_hold_upper_bound(waves_n)));

    auto& row_map = builder.add_map(
        col_map.root(),
        bcol_in_tile,
        symbolic::Lt(bcol_in_tile, waves_n),
        symbolic::zero(),
        symbolic::add(bcol_in_tile, symbolic::integer(1)),
        ScheduleType_GPU_Offload::create(arch, TargetLevel::Y_BLOCK, waves_n)
    );

    // K-block loop (unit stride, so LoopTiling can strip-mine it into staged K-panels).
    auto k_tile = symbolic::symbol(builder.find_new_name("tile_k"));
    int k_blocks = 0;
    if (ragged) {
        auto k_pad = symbolic::simplify(symbolic::expand(symbolic::overapproximate(k_dim)));
        k_blocks = GpuMmaSupport::get_ceil_block_count(k_pad, mma_tiling.mma_block_size.k);
        if (k_blocks > 0 && k_block_multiple > 1) {
            k_blocks = (k_blocks + k_block_multiple - 1) / k_block_multiple * k_block_multiple;
        }
    } else {
        k_blocks = GpuMmaSupport::get_integer_block_count(k_dim, mma_tiling.mma_block_size.k);
    }
    if (!k_blocks) {
        throw std::runtime_error("MMA expansion: K is not a multiple of the MMA block size");
    }
    auto k_count = symbolic::integer(k_blocks);
    builder.add_container(k_tile->get_name(), types::Scalar(types::get_primitive_type_to_hold_upper_bound(k_count)));
    auto k_elem = symbolic::mul(k_tile, symbolic::integer(mma_tiling.mma_block_size.k));

    auto lda = layout_a.get_stride(a_col_major ? 1 : 0);
    auto block_m = symbolic::integer(mma_tiling.mma_block_size.m);
    auto block_n = symbolic::integer(mma_tiling.mma_block_size.n);
    // M block index of fragment i of this wave, N block index of fragment j.
    auto brow = [&](int i) {
        return symbolic::
            add(symbolic::mul(brow_in_tile, symbolic::integer(mma_tiling.wave_tiling.blocks_m)), symbolic::integer(i));
    };
    auto bcol = [&](int j) {
        return symbolic::
            add(symbolic::mul(bcol_in_tile, symbolic::integer(mma_tiling.wave_tiling.blocks_n)), symbolic::integer(j));
    };
    auto a_offset = [&](int i) {
        return SymEngine::add({
            layout_a.offset(),
            SymEngine::mul({brow(i), block_m, layout_a.get_stride(0)}),
            SymEngine::mul({k_elem, layout_a.get_stride(1)}),
        });
    };
    auto ldb = layout_b.get_stride(b_col_major ? 1 : 0);
    auto b_offset = [&](int j) {
        return SymEngine::add(
            {layout_b.offset(),
             SymEngine::mul({k_elem, layout_b.get_stride(0)}),
             SymEngine::mul({bcol(j), block_n, layout_b.get_stride(1)})}
        );
    };
    auto ldc = layout_y.get_stride(y_col_major ? 1 : 0);
    auto y_offset = [&](int i, int j) {
        return SymEngine::add(
            {layout_y.offset(),
             SymEngine::mul({brow(i), block_m, layout_y.get_stride(0)}),
             SymEngine::mul({bcol(j), block_n, layout_y.get_stride(1)})}
        );
    };

    // Ragged extents: a fragment keeps the in-bounds rows/cols counted from its origin (null = never clipped).
    // A `min` argument of the exact extent at or above the covered tile size never clips a fragment of the tile,
    // e.g. min(N, i0 + 64) - i0 over a 64-tile clips like N - i0.
    auto clip = [&](const symbolic::Expression& exact, int covered, const symbolic::Expression& origin) {
        auto e = symbolic::simplify(exact);
        if (!ragged || symbolic::eq(e, symbolic::integer(covered))) {
            return symbolic::Expression(SymEngine::null);
        }
        e = symbolic::expand(e);
        if (SymEngine::is_a<SymEngine::Add>(*e)) {
            symbolic::Expression minterm = SymEngine::null;
            symbolic::Expression rest = symbolic::zero();
            for (const auto& arg : e->get_args()) {
                if (minterm.is_null() && SymEngine::is_a<SymEngine::Min>(*arg)) {
                    minterm = arg;
                } else {
                    rest = symbolic::add(rest, arg);
                }
            }
            if (!minterm.is_null()) {
                SymEngine::vec_basic kept;
                for (const auto& a : minterm->get_args()) {
                    auto v = symbolic::simplify(symbolic::expand(symbolic::add(a, rest)));
                    if (SymEngine::is_a<SymEngine::Integer>(*v) &&
                        SymEngine::rcp_static_cast<const SymEngine::Integer>(v)->as_int() >= covered) {
                        continue;
                    }
                    kept.push_back(v);
                }
                if (kept.empty()) {
                    return symbolic::Expression(SymEngine::null);
                }
                e = kept.size() == 1 ? kept.front() : SymEngine::min(kept);
            }
        }
        return symbolic::simplify(symbolic::sub(e, origin));
    };
    auto& n_dim = layout_b.get_dim(1);
    const int m_covered = mma_tiling.macro_tiling.blocks_m * mma_tiling.mma_block_size.m;
    const int n_covered = mma_tiling.macro_tiling.blocks_n * mma_tiling.mma_block_size.n;
    const int k_covered = k_blocks * mma_tiling.mma_block_size.k;
    auto m_clip = [&](int i) {
        return clip(m_dim, m_covered, symbolic::mul(brow(i), block_m));
    };
    auto n_clip = [&](int j) {
        return clip(n_dim, n_covered, symbolic::mul(bcol(j), block_n));
    };
    auto k_clip = clip(k_dim, k_covered, k_elem);

    auto& per_wavefront_block = builder.add_block(row_map.root());
    for (const auto& acc_frag_name : acc_frag_names) {
        auto& fill_node = builder.add_library_node<GpuMmaFillNode>(
            per_wavefront_block, org_debug_info, mma_tiling.mma_block_size, mma_tiling.acc_type, impl_type
        );
        auto& acc_frag_ptr = builder.add_access(per_wavefront_block, acc_frag_name);
        builder.add_computational_memlet(
            per_wavefront_block,
            acc_frag_ptr,
            fill_node,
            fill_node.input(0),
            {},
            types::Pointer(types::Scalar(mma_tiling.acc_type))
        );
    }

    std::vector<structured_control_flow::ReductionInfo> k_reductions;
    for (const auto& acc_frag_name : acc_frag_names) {
        k_reductions.push_back({structured_control_flow::ReductionOperation::Add, acc_frag_name});
    }
    auto& k_sweep = builder.add_reduce(
        row_map.root(),
        k_tile,
        symbolic::Lt(k_tile, k_count),
        symbolic::zero(),
        symbolic::add(k_tile, symbolic::integer(1)),
        k_reductions,
        structured_control_flow::ScheduleType_Sequential::create()
    );

    auto& load_block = builder.add_block(k_sweep.root());
    for (int i = 0; i < mma_tiling.wave_tiling.blocks_m; ++i) {
        create_fragment_load(
            standalone,
            input_type,
            impl_type,
            org_debug_info,
            builder,
            a_frag_names[i],
            mma_tiling.mma_block_size,
            MmaFragmentType::A,
            {.offset = a_offset(i), .ldstride = lda, .layout = a_frag_layout, .rows = m_clip(i), .cols = k_clip},
            args_order.at(1),
            "A",
            load_block
        );
    }
    for (int j = 0; j < mma_tiling.wave_tiling.blocks_n; ++j) {
        create_fragment_load(
            standalone,
            input_type,
            impl_type,
            org_debug_info,
            builder,
            b_frag_names[j],
            mma_tiling.mma_block_size,
            MmaFragmentType::B,
            {.offset = b_offset(j), .ldstride = ldb, .layout = b_frag_layout, .rows = k_clip, .cols = n_clip(j)},
            args_order.at(2),
            "B",
            load_block
        );
    }

    auto& inner_block = builder.add_block(k_sweep.root());
    for (int i = 0; i < mma_tiling.wave_tiling.blocks_m; ++i) {
        for (int j = 0; j < mma_tiling.wave_tiling.blocks_n; ++j) {
            create_fragment_mma(
                standalone,
                arch,
                mma_tiling,
                input_type,
                a_frag_names[i],
                b_frag_names[j],
                acc_frag_names[i * mma_tiling.wave_tiling.blocks_n + j],
                impl_type,
                org_debug_info,
                builder,
                inner_block
            );
        }
    }

    auto y_layout = y_col_major ? MMA_LAYOUT_COL_MAJOR : MMA_LAYOUT_ROW_MAJOR;
    std::string c_frag_name;
    std::string d_frag_name;
    // Converting the accumulator into a narrower output goes through the elementwise add, against C or, if Y is
    // overwritten, against zeros.
    const bool convert_output = include_c_add || output_type != mma_tiling.acc_type;
    if (convert_output) {
        c_frag_name = builder.find_new_name("mma_c");
        types::Pointer c_type{types::Scalar(output_type)};
        mma_arch->set_mma_fragment_storage_type(
            c_type.storage_type(), mma_tiling, MmaFragmentType::C, MmaFragmentLayout::MMA_LAYOUT_UNSPECIFIED
        );
        builder.add_container(c_frag_name, c_type);
        d_frag_name = builder.find_new_name("mma_d");
        types::Pointer d_type{types::Scalar(output_type)};
        mma_arch->set_mma_fragment_storage_type(
            d_type.storage_type(), mma_tiling, MmaFragmentType::C, MmaFragmentLayout::MMA_LAYOUT_UNSPECIFIED
        );
        builder.add_container(d_frag_name, d_type);
    }

    for (int i = 0; i < mma_tiling.wave_tiling.blocks_m; ++i) {
        for (int j = 0; j < mma_tiling.wave_tiling.blocks_n; ++j) {
            const auto& acc_frag_name = acc_frag_names[i * mma_tiling.wave_tiling.blocks_n + j];
            GpuMmaFromMemoryLayout y_mem{
                .offset = y_offset(i, j), .ldstride = ldc, .layout = y_layout, .rows = m_clip(i), .cols = n_clip(j)
            };
            std::string store_frag_name = acc_frag_name;
            if (convert_output) {
                auto& load_c_block = builder.add_block(row_map.root());
                if (include_c_add) {
                    create_fragment_load(
                        standalone,
                        output_type,
                        impl_type,
                        org_debug_info,
                        builder,
                        c_frag_name,
                        mma_tiling.mma_block_size,
                        MmaFragmentType::C,
                        y_mem,
                        args_order.at(0),
                        "Y",
                        load_c_block
                    );
                } else {
                    auto& fill_node = builder.add_library_node<
                        GpuMmaFillNode>(load_c_block, org_debug_info, mma_tiling.mma_block_size, output_type, impl_type);
                    builder.add_computational_memlet(
                        load_c_block,
                        builder.add_access(load_c_block, c_frag_name),
                        fill_node,
                        fill_node.input(0),
                        {},
                        types::Pointer(types::Scalar(output_type))
                    );
                }
                auto& eltwise_add_block = builder.add_block(row_map.root());
                create_eltwise_add_block(
                    standalone,
                    mma_tiling.mma_block_size,
                    mma_tiling.acc_type,
                    output_type,
                    impl_type,
                    org_debug_info,
                    builder,
                    c_frag_name,
                    acc_frag_name,
                    d_frag_name,
                    eltwise_add_block
                );
                store_frag_name = d_frag_name;
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
                y_mem,
                args_order.at(0),
                "Y",
                store_block
            );
        }
    }

    return standalone.successfully_expanded();
}

void GpuMmaExpander::create_fragment_mma(
    LibNodeExpander::AccessNodeExpand& standalone,
    const GpuArch& arch,
    GpuMmaTiling& mma_tiling,
    types::PrimitiveType input_type,
    const std::string& frag_a,
    const std::string& frag_b,
    const std::string& frag_acc,
    const data_flow::ImplementationType& impl_type,
    const DebugInfo& org_debug_info,
    builder::StructuredSDFGBuilder& builder,
    structured_control_flow::Block& block
) {
    auto& inner_node = builder.add_library_node<
        GpuMmaMatmulNode>(block, org_debug_info, mma_tiling.mma_block_size, input_type, mma_tiling.acc_type, impl_type);

    types::Scalar input_scalar_type(input_type);
    types::Scalar acc_scalar_type(mma_tiling.acc_type);
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

    auto input_type = node.input_type();
    auto output_type = node.output_type();

    auto mma_arch = arch_->mma_support();
    auto acc_type = mma_arch->get_accumulator_type(input_type, output_type, node.types().accumulator);

    auto mma_tiling = mma_arch->get_mma_tiling(
        {result_layout.get_dim(0), result_layout.get_dim(1), k_dim},
        input_type,
        acc_type,
        block_size_hint_,
        wave_tile_hint_
    );

    auto standalone = context.replacement_requires_access_nodes({InputUse::Scalar, InputUse::Scalar, InputUse::Scalar});

    if (standalone) {
        return expand_mma_standalone(
            *standalone,
            *arch_,
            mma_tiling.value(),
            node.layout_a(),
            node.layout_b(),
            result_layout,
            input_type,
            output_type,
            false, // MatMul overwrites Y
            node.debug_info(),
            {0, 1, 2} // {y, a, b}: MatMulNode access nodes are already ordered Y, A, B
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

    auto input_type = node.input_type();
    auto output_type = node.output_type();

    if (input_type == types::PrimitiveType::Void || output_type == types::PrimitiveType::Void) {
        return false;
    }

    if (!mma_arch->supported_types(input_type, output_type)) {
        return false;
    }

    auto acc_type = mma_arch->get_accumulator_type(input_type, output_type, node.types().accumulator);
    if (node.types().accumulator != types::PrimitiveType::Void && acc_type != node.types().accumulator) {
        return false;
    }

    auto tiling = mma_arch->get_mma_tiling({m, n, k}, input_type, acc_type, block_size_hint_, wave_tile_hint_);

    return tiling.has_value();
}

} // namespace sdfg::gpu
