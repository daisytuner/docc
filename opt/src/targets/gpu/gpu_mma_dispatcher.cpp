#include "sdfg/targets/gpu/gpu_mma_dispatcher.h"

#include "sdfg/symbolic/symbolic.h"
#include "sdfg/targets/gpu/gpu_mma_fill_node.h"
#include "sdfg/targets/gpu/gpu_mma_fragment_eltwise_add_node.h"
#include "sdfg/targets/gpu/gpu_mma_fragment_load_node.h"
#include "sdfg/targets/gpu/gpu_mma_fragment_store_node.h"
#include "sdfg/targets/gpu/gpu_mma_matmul_node.h"
#include "sdfg/types/scalar.h"
#include "sdfg/types/type.h"


namespace sdfg::gpu {

using namespace sdfg::math::tensor;

void GpuMmaMatmulDispatcher::emit_eltwise_compute(
    codegen::CodegenOutput& out,
    const std::string& main_frag,
    const std::vector<std::string>& additional_frags,
    const std::function<void(
        codegen::CodegenOutput&,
        const std::string& main_frag_elem,
        const std::string& idx,
        const std::vector<std::string>& other_frag_elems
    )>& compute
) const {
    out.stream << "for (int _ei = 0; _ei < " << main_frag << ".num_elements; ++_ei) {" << std::endl;
    out.stream.changeIndent(+4);
    std::vector<std::string> args;
    args.reserve(additional_frags.size());
    for (auto& frag : additional_frags) {
        args.push_back(frag + ".x[_ei]");
    }
    compute(out, main_frag + ".x[_ei]", "_ei", args);
    out.stream.changeIndent(-4);
    out.stream << "}" << std::endl;
}

void GpuMmaMatmulDispatcher::dispatch_code_with_edges(
    codegen::CodegenOutput& out,
    std::vector<codegen::DispatchInput>& inputs,
    std::vector<codegen::DispatchOutput>& outputs
) {
    auto& node = static_cast<const math::tensor::MatMulNode&>(this->node_);

    auto uniform_type = node.uniform_quantization(node.get_parent()).value();

    auto result_layout = node.layout_y();
    if (result_layout.dims() > 2) {
        throw InvalidSDFGException(
            "MatMulNode #" + std::to_string(node.element_id()) +
            " has more than 2 dimensions in the result layout. ROCm MMA only supports 2D matrices."
        );
    }
    auto result_col_major = is_col_major(result_layout);
    auto result_line_size = result_col_major ? result_layout.get_stride(1) : result_layout.get_stride(0);

    auto& layout_org_a = node.layout_a();
    bool layout_org_a_col_major = is_col_major(layout_org_a);
    auto k_dim = layout_org_a.get_dim(1);
    auto layout_a_line_size = layout_org_a_col_major ? layout_org_a.get_stride(1) : layout_org_a.get_stride(0);

    auto tiling = get_mma_tiling({result_layout.get_dim(0), result_layout.get_dim(1), k_dim});

    auto wave_tile_m = tiling.wave_tile_blocks_m * tiling.mma_block_size.m;
    auto wave_tile_n = tiling.wave_tile_blocks_n * tiling.mma_block_size.n;
    auto wave_tile_k = tiling.mma_block_size.k;

    std::array<int, 3> wave_tile_dims = {wave_tile_m, wave_tile_n, wave_tile_k};

    auto& layout_org_b = node.layout_b();
    auto layout_org_b_col_major = is_col_major(layout_org_b);
    auto layout_b_line_size = layout_org_b_col_major ? layout_org_b.get_stride(1) : layout_org_b.get_stride(0);

    types::Scalar offset_type(types::PrimitiveType::UInt32);

    emit_needed_declarations(out);

    auto matA_glbl_offset = "matA_glbl_offset";
    auto matB_glbl_offset = "matB_glbl_offset";

    out.stream << language_extension_.declaration(matA_glbl_offset, offset_type) << " = "
               << language_extension_.expression(get_start_offset(layout_org_a)) << ";" << std::endl;
    out.stream << language_extension_.declaration(matB_glbl_offset, offset_type) << " = "
               << language_extension_.expression(get_start_offset(layout_org_b)) << ";" << std::endl;


    emit_block_frag_declaration(
        out,
        "fragA",
        MmaFragmentType::A,
        wave_tile_dims,
        layout_org_a_col_major ? MmaFragmentLayout::MMA_LAYOUT_COL_MAJOR : MmaFragmentLayout::MMA_LAYOUT_ROW_MAJOR,
        uniform_type
    );
    emit_block_frag_declaration(
        out,
        "fragB",
        MmaFragmentType::B,
        wave_tile_dims,
        layout_org_b_col_major ? MmaFragmentLayout::MMA_LAYOUT_COL_MAJOR : MmaFragmentLayout::MMA_LAYOUT_ROW_MAJOR,
        uniform_type
    );

    emit_block_frag_declaration(
        out, "fragAcc", MmaFragmentType::C, wave_tile_dims, MmaFragmentLayout::MMA_LAYOUT_UNSPECIFIED, uniform_type
    );

    emit_frag_zero_init(out, "fragAcc", uniform_type);

    auto sym_k = symbolic::symbol("k");

    out.stream << "for (int k = 0; k < " << language_extension_.expression(k_dim)
               << "; k += " << tiling.mma_block_size.k << ") {" << std::endl;
    out.stream.changeIndent(+4);

    emit_load_macro(
        out,
        "fragA",
        inputs.at(math::tensor::MatMulNode::A_INPUT_IDX).expr,
        symbolic::symbol(matA_glbl_offset),
        layout_a_line_size
    );
    emit_load_macro(
        out,
        "fragB",
        inputs.at(math::tensor::MatMulNode::B_INPUT_IDX).expr,
        symbolic::symbol(matB_glbl_offset),
        layout_b_line_size
    );

    emit_mma_compute(out, "fragAcc", "fragA", "fragB", "fragAcc");

    auto mma_k = symbolic::integer(tiling.mma_block_size.k);

    out.stream << matA_glbl_offset
               << " += " << language_extension_.expression(symbolic::mul(mma_k, layout_org_a.get_stride(1))) << ";"
               << std::endl;
    out.stream << matB_glbl_offset
               << " += " << language_extension_.expression(symbolic::mul(mma_k, layout_org_b.get_stride(0))) << ";"
               << std::endl;

    out.stream.changeIndent(-4);
    out.stream << "}" << std::endl;

    emit_block_frag_declaration(
        out, "fragC", MmaFragmentType::C, wave_tile_dims, MmaFragmentLayout::MMA_LAYOUT_UNSPECIFIED, uniform_type
    );
    emit_load_macro(
        out,
        "fragC",
        inputs.at(math::tensor::MatMulNode::Y_INPUT_IDX).expr,
        get_start_offset(result_layout),
        result_line_size,
        result_col_major ? MmaFragmentLayout::MMA_LAYOUT_COL_MAJOR : MmaFragmentLayout::MMA_LAYOUT_ROW_MAJOR
    );

    emit_eltwise_compute(out, "fragAcc", {"fragC"}, [](auto& out, auto& main_elem, auto& idx, auto& args) {
        out.stream << main_elem << " += " << args.at(0) << ";" << std::endl;
    });

    emit_store_macro(
        out,
        "fragAcc",
        inputs.at(0).expr,
        get_start_offset(result_layout),
        result_line_size,
        result_col_major ? MmaFragmentLayout::MMA_LAYOUT_COL_MAJOR : MmaFragmentLayout::MMA_LAYOUT_ROW_MAJOR
    );
}

bool GpuMmaMatmulDispatcher::is_col_major(const math::tensor::TensorLayout& layout) {
    auto l = layout.is_2d_col_or_row_major();
    if (l == math::tensor::TensorLayout::LAYOUT_COL_MAJOR) {
        return true;
    } else if (l == math::tensor::TensorLayout::LAYOUT_ROW_MAJOR) {
        return false;
    } else {
        throw std::invalid_argument("only supports col or row-major");
    }
}

symbolic::Expression GpuMmaMatmulDispatcher::get_start_offset(const math::tensor::TensorLayout& layout) const {
    return layout.offset();
}

void GpuMmaMatmulDispatcher::dispatch_single_mma(codegen::CodegenOutput& out, std::vector<codegen::DispatchInput>& inputs)
    const {
    emit_needed_declarations(out);

    // All three operands are fragments (declared by their containers); this is only the mma_sync.
    const std::string& acc = inputs.at(GpuMmaMatmulNode::Y_INPUT_IDX).expr;
    const std::string& frag_a = inputs.at(GpuMmaMatmulNode::A_INPUT_IDX).expr;
    const std::string& frag_b = inputs.at(GpuMmaMatmulNode::B_INPUT_IDX).expr;

    emit_mma_compute(out, acc, frag_a, frag_b, acc);
}

void GpuMmaMatmulDispatcher::
    dispatch_fragment_load(codegen::CodegenOutput& out, std::vector<codegen::DispatchInput>& inputs) const {
    auto& node = static_cast<const GpuMmaFragmentLoadNode&>(node_);
    auto& layout = node.layout();

    emit_needed_declarations(out);

    const std::string& frag = inputs.at(GpuMmaFragmentLoadNode::FRAG_INPUT_IDX).expr;
    const std::string& base = inputs.at(GpuMmaFragmentLoadNode::PTR_INPUT_IDX).expr;

    emit_load_macro(out, frag, base, layout.offset, layout.ldstride, layout.layout);
}

void GpuMmaMatmulDispatcher::
    dispatch_fragment_store(codegen::CodegenOutput& out, std::vector<codegen::DispatchInput>& inputs) const {
    auto& node = static_cast<const GpuMmaFragmentStoreNode&>(node_);
    auto& layout = node.layout();

    emit_needed_declarations(out);

    const std::string& frag = inputs.at(GpuMmaFragmentStoreNode::FRAG_INPUT_IDX).expr;
    const std::string& base = inputs.at(GpuMmaFragmentStoreNode::PTR_INPUT_IDX).expr;

    emit_store_macro(out, frag, base, layout.offset, layout.ldstride, layout.layout);
}

void GpuMmaMatmulDispatcher::dispatch_eltwise_add(codegen::CodegenOutput& out, std::vector<codegen::DispatchInput>& inputs)
    const {
    auto& node = static_cast<const GpuMmaFragmentEltwiseAddNode&>(node_);

    auto otype = out.language_extension.primitive_type(node.output_type());
    auto atype = out.language_extension.primitive_type(node.acc_type());

    emit_needed_declarations(out);

    const std::string& frag_d = inputs.at(GpuMmaFragmentEltwiseAddNode::FRAG_D_INPUT_IDX).expr;
    const std::string& frag_acc = inputs.at(GpuMmaFragmentEltwiseAddNode::FRAG_ACC_INPUT_IDX).expr;
    const std::string& frag_c = inputs.at(GpuMmaFragmentEltwiseAddNode::FRAG_C_INPUT_IDX).expr;

    emit_eltwise_compute(
        out,
        frag_d,
        {frag_acc, frag_c},
        [&otype, &atype](auto& out, auto& main_elem, auto& idx, auto& args) {
            out.stream << main_elem << " = static_cast<" << otype << ">(" << args.at(0) << " + static_cast<" << atype
                       << ">(" << args.at(1) << "));" << std::endl;
        }
    );
}

void GpuMmaMatmulDispatcher::
    dispatch_accumulator_fill(codegen::CodegenOutput& out, std::vector<codegen::DispatchInput>& inputs) const {
    auto& node = static_cast<const GpuMmaFillNode&>(node_);
    auto fill_type = node.fill_type();

    emit_needed_declarations(out);

    const std::string& acc = inputs.at(GpuMmaFillNode::Y_INPUT_IDX).expr;

    emit_frag_zero_init(out, acc, fill_type);
}

} // namespace sdfg::gpu
