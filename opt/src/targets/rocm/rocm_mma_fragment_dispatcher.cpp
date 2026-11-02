#include "sdfg/targets/rocm/rocm_mma_fragment_dispatcher.h"

#include "sdfg/targets/rocm/rocm.h"

namespace sdfg::gpu::rocm {

std::optional<std::pair<std::string, std::string>> RocmMmaFragmentStrategy::fragment_element_coords(
    MmaFragmentType frag_type, const MmaBlockSize& block_size, const std::string& idx, std::string& condition
) const {
    const int wavefront = ::sdfg::rocm::rocm_wavefront_size();
    if (frag_type != MmaFragmentType::C || block_size.m != 16 || block_size.n != 16 || wavefront != 64) {
        return std::nullopt;
    }
    condition.clear();
    const std::string lane = "(threadIdx.x % " + std::to_string(wavefront) + ")";
    return std::make_pair("(4 * (" + lane + " / 16) + " + idx + ")", std::string("(threadIdx.x % 16)"));
}

void RocmMmaFragmentStrategy::emit_needed_declarations(codegen::CodegenOutput& out) const {
    out.library_snippet_factory.require_dependency(RocmWmmaLibDependency::instance());
}

void RocmMmaFragmentStrategy::
    emit_fill(codegen::CodegenOutput& out, const std::string& acc_frag, types::PrimitiveType fill_type) const {
    out.stream << "rocwmma::fill_fragment(" << acc_frag << ", static_cast<"
               << out.language_extension.primitive_type(fill_type) << ">(0.0));" << std::endl;
}

void RocmMmaFragmentStrategy::emit_mma(
    codegen::CodegenOutput& out,
    const std::string& frag_d,
    const std::string& frag_a,
    const std::string& frag_b,
    const std::string& frag_c
) const {
    out.stream << "rocwmma::mma_sync(" << frag_d << ", " << frag_a << ", " << frag_b << ", " << frag_c << ");"
               << std::endl;
}

void RocmMmaFragmentStrategy::emit_fragment_load(
    codegen::CodegenOutput& out,
    const std::string& frag,
    const std::string& base,
    const GpuMmaFromMemoryLayout& layout,
    MmaFragmentType /*frag_type*/,
    types::PrimitiveType /*element_type*/,
    MmaFragmentLayout fragment_layout
) const {
    out.stream << "rocwmma::load_matrix_sync(" << frag << ", " << base;
    if (!layout.offset.is_null()) {
        out.stream << " + " << out.language_extension.expression(layout.offset);
    }
    out.stream << ", " << out.language_extension.expression(layout.ldstride);
    if (fragment_layout == MMA_LAYOUT_UNSPECIFIED || fragment_layout != layout.layout) {
        if (layout.layout == MmaFragmentLayout::MMA_LAYOUT_COL_MAJOR) {
            out.stream << ", rocwmma::mem_col_major";
        } else if (layout.layout == MmaFragmentLayout::MMA_LAYOUT_ROW_MAJOR) {
            out.stream << ", rocwmma::mem_row_major";
        }
    }
    out.stream << ");" << std::endl;
}

void RocmMmaFragmentStrategy::emit_fragment_store(
    codegen::CodegenOutput& out,
    const std::string& frag,
    const std::string& base,
    const GpuMmaFromMemoryLayout& layout,
    MmaFragmentType /*frag_type*/,
    types::PrimitiveType /*element_type*/
) const {
    out.stream << "rocwmma::store_matrix_sync(" << base;
    if (!layout.offset.is_null()) {
        out.stream << " + " << out.language_extension.expression(layout.offset);
    }
    out.stream << ", " << frag;
    out.stream << ", " << out.language_extension.expression(layout.ldstride);
    if (layout.layout == MmaFragmentLayout::MMA_LAYOUT_COL_MAJOR) {
        out.stream << ", rocwmma::mem_col_major";
    } else if (layout.layout == MmaFragmentLayout::MMA_LAYOUT_ROW_MAJOR) {
        out.stream << ", rocwmma::mem_row_major";
    }
    out.stream << ");" << std::endl;
}

void RocmMmaFragmentStrategy::emit_eltwise(
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

} // namespace sdfg::gpu::rocm
