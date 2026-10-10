#include "sdfg/targets/rocm/rocm_mfma_dispatcher.h"

#include "sdfg/targets/gpu/gpu_mma_fragment_load_node.h"
#include "sdfg/targets/gpu/gpu_mma_fragment_store_node.h"
#include "sdfg/types/pointer.h"
#include "sdfg/types/scalar.h"

namespace sdfg::gpu::rocm {

namespace {

constexpr const char* kLane = "(threadIdx.x % 64)";

/// True if the affine @p expr is a multiple of @p n for every symbol value (constant and all
/// symbol coefficients divisible). Non-affine terms are rejected.
bool affine_multiple_of(const symbolic::Expression& expr, long long n) {
    auto e = symbolic::expand(expr);
    symbolic::ExpressionMapping zeros;
    for (const auto& s : symbolic::atoms(e)) {
        zeros[s] = symbolic::zero();
    }
    auto constant = symbolic::subs(e, zeros);
    auto divisible = [n](const symbolic::Expression& c) {
        return SymEngine::is_a<SymEngine::Integer>(*c) &&
               SymEngine::rcp_static_cast<const SymEngine::Integer>(c)->as_int() % n == 0;
    };
    if (!divisible(constant)) {
        return false;
    }
    for (const auto& s : symbolic::atoms(e)) {
        auto at_one = zeros;
        at_one[s] = symbolic::one();
        auto coeff = symbolic::expand(symbolic::sub(symbolic::subs(e, at_one), constant));
        auto at_two = zeros;
        at_two[s] = symbolic::integer(2);
        auto twice = symbolic::expand(symbolic::sub(symbolic::subs(e, at_two), constant));
        if (!divisible(coeff) || !symbolic::eq(twice, symbolic::mul(symbolic::integer(2), coeff))) {
            return false;
        }
    }
    return true;
}

} // namespace

void RocmMfmaBaseDispatcher::emit_frag_zero_init(
    codegen::CodegenOutput& out, const std::string& frag_name, types::PrimitiveType scalar_type
) const {
    out.stream << frag_name << " = static_cast<" << language_extension_.primitive_type(scalar_type) << ">(0.0);"
               << std::endl;
}

void RocmMfmaBaseDispatcher::emit_mma_compute(
    codegen::CodegenOutput& out,
    const std::string& frag_d_out,
    const std::string& frag_a,
    const std::string& frag_b,
    const std::string& frag_c_in
) const {
    out.stream << frag_d_out << " = __builtin_amdgcn_mfma_f32_32x32x8f16(" << frag_a << ", " << frag_b << ", "
               << frag_c_in << ", 0, 0, 0);" << std::endl;
}

void RocmMfmaBaseDispatcher::emit_eltwise_compute(
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
    out.stream << "for (int _ei = 0; _ei < static_cast<int>(sizeof(" << main_frag << ") / sizeof(" << main_frag
               << "[0])); ++_ei) {" << std::endl;
    out.stream.changeIndent(+4);
    std::vector<std::string> args;
    for (auto& frag : additional_frags) {
        args.push_back(frag + "[_ei]");
    }
    compute(out, main_frag + "[_ei]", "_ei", args);
    out.stream.changeIndent(-4);
    out.stream << "}" << std::endl;
}

void RocmMfmaBaseDispatcher::emit_needed_declarations(codegen::CodegenOutput& out) const {
    GpuMmaMatmulDispatcher::emit_needed_declarations(out);
}

void RocmMfmaBaseDispatcher::emit_fragment_transfer(
    codegen::CodegenOutput& out,
    bool load,
    MmaFragmentType type,
    const GpuMmaFromMemoryLayout& layout,
    types::PrimitiveType element_type,
    const std::string& frag,
    const std::string& base
) const {
    if (type != MmaFragmentType::C && !load) {
        throw std::runtime_error("MFMA: only accumulator fragments can be stored");
    }
    const bool row_major = layout.layout == MmaFragmentLayout::MMA_LAYOUT_ROW_MAJOR;
    const std::string ld = "(" + language_extension_.expression(layout.ldstride) + ")";
    const std::string off = layout.offset.is_null() ? std::string("0")
                                                    : "(" + language_extension_.expression(layout.offset) + ")";
    const std::string ptr = language_extension_.type_cast(base, types::Pointer(types::Scalar(element_type)));
    const std::string lane = kLane;
    const std::string lo = "(" + lane + " % 32)";
    const std::string kq = "(4 * (" + lane + " / 32))";

    std::string index; // element _e of this lane's fragment
    std::string first; // the lane's first element when its four k values are adjacent
    switch (type) {
        case MmaFragmentType::A: // (row lo, k kq+_e)
            index = row_major ? lo + " * " + ld + " + " + kq + " + _e" : lo + " + (" + kq + " + _e) * " + ld;
            first = row_major ? lo + " * " + ld + " + " + kq : "";
            break;
        case MmaFragmentType::B: // (k kq+_e, col lo)
            index = row_major ? "(" + kq + " + _e) * " + ld + " + " + lo : kq + " + _e + " + lo + " * " + ld;
            first = row_major ? "" : kq + " + " + lo + " * " + ld;
            break;
        default: { // C: (row 8*(_e/4) + kq + _e%4, col lo)
            const std::string row = "(8 * (_e / 4) + " + kq + " + _e % 4)";
            index = row_major ? row + " * " + ld + " + " + lo : row + " + " + lo + " * " + ld;
            break;
        }
    }
    // One 8-byte access when the lane's four k values are adjacent and 4-element aligned.
    const bool contiguous = !first.empty() && affine_multiple_of(layout.ldstride, 4) &&
                            (layout.offset.is_null() || affine_multiple_of(layout.offset, 4));
    if (contiguous) {
        out.stream << frag << " = *reinterpret_cast<const __typeof__(" << frag << ")*>(&(" << ptr << ")[" << off
                   << " + " << first << "]);" << std::endl;
        return;
    }
    out.stream << "for (int _e = 0; _e < static_cast<int>(sizeof(" << frag << ") / sizeof(" << frag << "[0])); ++_e) {"
               << std::endl;
    out.stream.changeIndent(+4);
    const std::string mem = "(" + ptr + ")[" + off + " + " + index + "]";
    if (load) {
        out.stream << frag << "[_e] = " << mem << ";" << std::endl;
    } else {
        out.stream << mem << " = " << frag << "[_e];" << std::endl;
    }
    out.stream.changeIndent(-4);
    out.stream << "}" << std::endl;
}

void RocmMfmaFragmentLoadDispatcher::dispatch_code_with_edges(
    codegen::CodegenOutput& out,
    std::vector<codegen::DispatchInput>& inputs,
    std::vector<codegen::DispatchOutput>& outputs
) {
    auto& node = static_cast<const GpuMmaFragmentLoadNode&>(node_);
    emit_needed_declarations(out);
    emit_fragment_transfer(
        out,
        true,
        node.fragment_type(),
        node.layout(),
        node.element_type(),
        inputs.at(GpuMmaFragmentLoadNode::FRAG_INPUT_IDX).expr,
        inputs.at(GpuMmaFragmentLoadNode::PTR_INPUT_IDX).expr
    );
}

void RocmMfmaFragmentStoreDispatcher::dispatch_code_with_edges(
    codegen::CodegenOutput& out,
    std::vector<codegen::DispatchInput>& inputs,
    std::vector<codegen::DispatchOutput>& outputs
) {
    auto& node = static_cast<const GpuMmaFragmentStoreNode&>(node_);
    emit_needed_declarations(out);
    emit_fragment_transfer(
        out,
        false,
        node.fragment_type(),
        node.layout(),
        node.element_type(),
        inputs.at(GpuMmaFragmentStoreNode::FRAG_INPUT_IDX).expr,
        inputs.at(GpuMmaFragmentStoreNode::PTR_INPUT_IDX).expr
    );
}

} // namespace sdfg::gpu::rocm
