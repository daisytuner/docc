#include "sdfg/targets/rocm/rocm_mma_dispatcher.h"

#include "sdfg/targets/rocm/codegen/rocm_language_extension.h"
#include "sdfg/targets/rocm/rocm_arch.h"


namespace sdfg::gpu::rocm {

const GpuMmaSupport* RocmMmaBaseDispatcher::get_mma_arch_from_impl_type_hack() const {
    return ROCM_ARCH_GFX1201.mma_support();
}

const GpuArch* RocmMmaBaseDispatcher::get_gpu_arch_from_context(codegen::CodegenOutput& out) const {
    auto* rocm_lang = dynamic_cast<sdfg::rocm::ROCMLanguageExtension*>(&out.language_extension);
    if (rocm_lang) {
        return rocm_lang->gpu_arch();
    } else {
        return nullptr;
    }
}

void RocmMmaBaseDispatcher::emit_block_frag_declaration(
    codegen::CodegenOutput& out,
    const std::string& name,
    MmaFragmentType type,
    std::array<int, 3> dims,
    MmaFragmentLayout layout,
    types::PrimitiveType scalar_type,
    std::optional<std::pair<int, int>> coop_dims
) const {
    std::stringstream ss;
    RocmMmaSupport::emit_block_frag_type(ss, type, dims, layout, scalar_type, coop_dims);
    out.stream << ss.str() << " " << name << ";" << std::endl;
}

GpuMmaTiling RocmMmaBaseDispatcher::get_mma_tiling(const symbolic::MultiExpression& res_shape) const {
    auto* mma_arch = get_mma_arch_from_impl_type_hack();
    if (!mma_arch) {
        throw std::runtime_error("No MMA architecture available for this GPU target.");
    }

    return mma_arch->get_mma_tiling(res_shape);
}

void RocmMmaBaseDispatcher::emit_load_macro(
    codegen::CodegenOutput& out,
    const std::string& name,
    const std::string& base_addr,
    const symbolic::Expression& offset,
    const symbolic::Expression& line_size,
    MmaFragmentLayout layout
) const {
    out.stream << "rocwmma::load_matrix_sync(" << name << ", " << base_addr;
    if (!offset.is_null()) {
        out.stream << " + " << language_extension_.expression(offset);
    }
    out.stream << ", " << language_extension_.expression(line_size);
    if (layout == MmaFragmentLayout::MMA_LAYOUT_COL_MAJOR) {
        out.stream << ", rocwmma::mem_col_major";
    } else if (layout == MmaFragmentLayout::MMA_LAYOUT_ROW_MAJOR) {
        out.stream << ", rocwmma::mem_row_major";
    }
    out.stream << ");" << std::endl;
}

void RocmMmaBaseDispatcher::emit_frag_zero_init(
    codegen::CodegenOutput& out, const std::string& frag_name, types::PrimitiveType scalar_type
) const {
    out.stream << "rocwmma::fill_fragment(" << frag_name << ", static_cast<"
               << language_extension_.primitive_type(scalar_type) << ">(0.0));" << std::endl;
}

void RocmMmaBaseDispatcher::emit_mma_compute(
    codegen::CodegenOutput& out,
    const std::string& frag_d_out,
    const std::string& frag_a,
    const std::string& frag_b,
    const std::string& frag_c_in
) const {
    out.stream << "rocwmma::mma_sync(" << frag_d_out << ", " << frag_a << ", " << frag_b << ", " << frag_c_in << ");"
               << std::endl;
}

void RocmMmaBaseDispatcher::emit_needed_declarations(codegen::CodegenOutput& out) const {
    GpuMmaMatmulDispatcher::emit_needed_declarations(out);

    out.library_snippet_factory.require_dependency(RocmWmmaLibDependency::instance());
}

void RocmMmaBaseDispatcher::emit_store_macro(
    const codegen::CodegenOutput& out,
    const std::string& frag_name,
    const std::string& base_addr,
    const symbolic::Expression& offset,
    const symbolic::Expression& line_size,
    MmaFragmentLayout layout
) const {
    out.stream << "rocwmma::store_matrix_sync(" << base_addr;
    if (!offset.is_null()) {
        out.stream << " + " << language_extension_.expression(offset);
    }
    out.stream << ", " << frag_name;
    out.stream << ", " << language_extension_.expression(line_size);
    if (layout == MmaFragmentLayout::MMA_LAYOUT_COL_MAJOR) {
        out.stream << ", rocwmma::mem_col_major";
    } else if (layout == MmaFragmentLayout::MMA_LAYOUT_ROW_MAJOR) {
        out.stream << ", rocwmma::mem_row_major";
    }
    out.stream << ");" << std::endl;
}

} // namespace sdfg::gpu::rocm
