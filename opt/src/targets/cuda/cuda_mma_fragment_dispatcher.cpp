#include "sdfg/targets/cuda/cuda_mma_fragment_dispatcher.h"


namespace sdfg::gpu::cuda {

void CudaMmaFragmentStrategy::emit_needed_declarations(codegen::CodegenOutput& out) const {
    out.library_snippet_factory.require_dependency(CudaWmmaLibDependency::instance());
}

void CudaMmaFragmentStrategy::
    emit_fill(codegen::CodegenOutput& out, const std::string& acc_frag, types::PrimitiveType fill_type) const {
    out.stream << "nvcuda::wmma::fill_fragment(" << acc_frag << ", static_cast<"
               << out.language_extension.primitive_type(fill_type) << ">(0.0));" << std::endl;
}

void CudaMmaFragmentStrategy::emit_mma(
    codegen::CodegenOutput& out,
    const std::string& frag_d,
    const std::string& frag_a,
    const std::string& frag_b,
    const std::string& frag_c
) const {
    out.stream << "nvcuda::wmma::mma_sync(" << frag_d << ", " << frag_a << ", " << frag_b << ", " << frag_c << ");"
               << std::endl;
}

void CudaMmaFragmentStrategy::emit_fragment_load(
    codegen::CodegenOutput& out,
    const std::string& frag,
    const std::string& base,
    const GpuMmaFromMemoryLayout& layout,
    MmaFragmentType /*frag_type*/,
    types::PrimitiveType /*element_type*/,
    MmaFragmentLayout fragment_layout
) const {
    out.stream << "nvcuda::wmma::load_matrix_sync(" << frag << ", " << base;
    if (!layout.offset.is_null()) {
        out.stream << " + " << out.language_extension.expression(layout.offset);
    }
    out.stream << ", " << out.language_extension.expression(layout.ldstride);
    if (fragment_layout == MMA_LAYOUT_UNSPECIFIED || fragment_layout != layout.layout) {
        if (layout.layout == MmaFragmentLayout::MMA_LAYOUT_COL_MAJOR) {
            out.stream << ", nvcuda::wmma::mem_col_major";
        } else if (layout.layout == MmaFragmentLayout::MMA_LAYOUT_ROW_MAJOR) {
            out.stream << ", nvcuda::wmma::mem_row_major";
        }
    }
    out.stream << ");" << std::endl;
}

void CudaMmaFragmentStrategy::emit_fragment_store(
    codegen::CodegenOutput& out,
    const std::string& frag,
    const std::string& base,
    const GpuMmaFromMemoryLayout& layout,
    MmaFragmentType /*frag_type*/,
    types::PrimitiveType /*element_type*/
) const {
    out.stream << "nvcuda::wmma::store_matrix_sync(" << base;
    if (!layout.offset.is_null()) {
        out.stream << " + " << out.language_extension.expression(layout.offset);
    }
    out.stream << ", " << frag;
    out.stream << ", " << out.language_extension.expression(layout.ldstride);
    if (layout.layout == MmaFragmentLayout::MMA_LAYOUT_COL_MAJOR) {
        out.stream << ", nvcuda::wmma::mem_col_major";
    } else if (layout.layout == MmaFragmentLayout::MMA_LAYOUT_ROW_MAJOR) {
        out.stream << ", nvcuda::wmma::mem_row_major";
    }
    out.stream << ");" << std::endl;
}

void CudaMmaFragmentStrategy::emit_eltwise(
    codegen::CodegenOutput& out,
    const std::string& main_frag,
    const std::vector<std::string>& additional_frags,
    const EltwiseCompute& compute
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

} // namespace sdfg::gpu::cuda
