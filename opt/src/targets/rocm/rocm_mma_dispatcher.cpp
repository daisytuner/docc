#include "sdfg/targets/rocm/rocm_mma_dispatcher.h"

#include "sdfg/targets/rocm/rocm_arch.h"


namespace sdfg::gpu::rocm {

const GpuMmaSupport* RocmMmaMatmulDispatcher::get_mma_arch() const { return ROCM_ARCH_GFX1201.mma_support(); }

void RocmMmaMatmulDispatcher::emit_block_frag_declaration(
    codegen::CodegenOutput& out,
    const std::string& name,
    FragmentType type,
    std::array<int, 3> dims,
    math::tensor::TensorLayout::TensorLayoutType layout,
    types::PrimitiveType scalar_type,
    std::optional<std::pair<int, int>> coop_dims
) const {
    out.stream << "rocwmma::fragment<";
    switch (type) {
        case FragmentType::A:
            out.stream << "rocwmma::matrix_a, ";
            break;
        case FragmentType::B:
            out.stream << "rocwmma::matrix_b, ";
            break;
        case FragmentType::C:
            out.stream << "rocwmma::accumulator, ";
            break;
        default:
            throw std::invalid_argument("invalid fragment type");
    }
    out.stream << dims[0] << ", ";
    out.stream << dims[1] << ", ";
    out.stream << dims[2] << ", ";
    switch (scalar_type) {
        case types::PrimitiveType::BFloat:
            out.stream << "rocwmma::bfloat16_t";
            break;
        case types::PrimitiveType::Half:
            out.stream << "rocwmma::float16_t";
            break;
        case types::PrimitiveType::Float:
            out.stream << "rocwmma::float32_t";
            break;
        default:
            throw std::invalid_argument(
                "invalid scalar type: " + std::string(types::primitive_type_to_string(scalar_type)) + " on mma #" +
                std::to_string(node_.element_id())
            );
    }

    if (layout == math::tensor::TensorLayout::LAYOUT_COL_MAJOR) {
        out.stream << ", rocwmma::col_major";
    } else if (layout == math::tensor::TensorLayout::LAYOUT_ROW_MAJOR) {
        out.stream << ", rocwmma::row_major";
    }

    if (coop_dims) {
        out.stream << ", rocwmma::fragment_scheduler::coop_row_major_2d<" << coop_dims->first << ", "
                   << coop_dims->second << ">";
    }

    out.stream << "> " << name << ";" << std::endl;
}

GpuMmaTiling RocmMmaMatmulDispatcher::get_mma_tiling(const symbolic::MultiExpression& res_shape) const {
    auto* mma_arch = get_mma_arch();
    if (!mma_arch) {
        throw std::runtime_error("No MMA architecture available for this GPU target.");
    }

    return mma_arch->get_mma_tiling(res_shape);
}

void RocmMmaMatmulDispatcher::emit_load_macro(
    codegen::CodegenOutput& out,
    const std::string& name,
    const std::string& base_addr,
    const symbolic::Expression& offset,
    const symbolic::Expression& line_size,
    math::tensor::TensorLayout::TensorLayoutType layout
) const {
    out.stream << "rocwmma::load_matrix_sync(" << name << ", " << base_addr;
    if (!offset.is_null()) {
        out.stream << " + " << language_extension_.expression(offset);
    }
    out.stream << ", " << language_extension_.expression(line_size);
    if (layout == math::tensor::TensorLayout::LAYOUT_COL_MAJOR) {
        out.stream << ", rocwmma::mem_col_major";
    } else if (layout == math::tensor::TensorLayout::LAYOUT_ROW_MAJOR) {
        out.stream << ", rocwmma::mem_row_major";
    }
    out.stream << ");" << std::endl;
}

void RocmMmaMatmulDispatcher::emit_frag_zero_init(
    codegen::CodegenOutput& out, const std::string& frag_name, types::PrimitiveType scalar_type
) const {
    out.stream << "rocwmma::fill_fragment(" << frag_name << ", static_cast<"
               << language_extension_.primitive_type(scalar_type) << ">(0.0));" << std::endl;
}

void RocmMmaMatmulDispatcher::emit_mma_compute(
    codegen::CodegenOutput& out,
    const std::string& frag_d_out,
    const std::string& frag_a,
    const std::string& frag_b,
    const std::string& frag_c_in
) const {
    out.stream << "rocwmma::mma_sync(" << frag_d_out << ", " << frag_a << ", " << frag_b << ", " << frag_c_in << ");"
               << std::endl;
}

void RocmMmaMatmulDispatcher::emit_needed_declarations(codegen::CodegenOutput& out) const {
    GpuMmaMatmulDispatcher::emit_needed_declarations(out);

    out.library_snippet_factory.require_dependency(RocmWmmaLibDependency::instance());
}

void RocmMmaMatmulDispatcher::emit_store_macro(
    const codegen::CodegenOutput& out,
    const std::string& frag_name,
    const std::string& base_addr,
    const symbolic::Expression& offset,
    const symbolic::Expression& line_size,
    math::tensor::TensorLayout::TensorLayoutType layout
) const {
    out.stream << "rocwmma::store_matrix_sync(" << base_addr;
    if (!offset.is_null()) {
        out.stream << " + " << language_extension_.expression(offset);
    }
    out.stream << ", " << frag_name;
    out.stream << ", " << language_extension_.expression(line_size);
    if (layout == math::tensor::TensorLayout::LAYOUT_COL_MAJOR) {
        out.stream << ", rocwmma::mem_col_major";
    } else if (layout == math::tensor::TensorLayout::LAYOUT_ROW_MAJOR) {
        out.stream << ", rocwmma::mem_row_major";
    }
    out.stream << ");" << std::endl;
}

} // namespace sdfg::gpu::rocm
