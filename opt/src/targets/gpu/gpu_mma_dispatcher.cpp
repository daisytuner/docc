#include "sdfg/targets/gpu/gpu_mma_dispatcher.h"

#include "sdfg/data_flow/access_node.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/targets/gpu/gpu_mma_fill_node.h"
#include "sdfg/targets/gpu/gpu_mma_fragment_eltwise_add_node.h"
#include "sdfg/targets/gpu/gpu_mma_fragment_load_node.h"
#include "sdfg/targets/gpu/gpu_mma_fragment_store_node.h"
#include "sdfg/targets/gpu/gpu_mma_matmul_node.h"
#include "sdfg/types/pointer.h"
#include "sdfg/types/scalar.h"

namespace sdfg::gpu {

void emit_clipped_fragment_transfer(
    codegen::CodegenOutput& out,
    const GpuMmaDispatcherStrategy& strategy,
    bool load,
    const std::string& frag,
    const std::string& base,
    const GpuMmaFromMemoryLayout& layout,
    const MmaBlockSize& block_size,
    MmaFragmentType frag_type,
    types::PrimitiveType element_type,
    MmaFragmentLayout fragment_layout
) {
    auto& lang = out.language_extension;
    auto shape = block_size.get_shape(frag_type);
    const auto R = SymEngine::rcp_static_cast<const SymEngine::Integer>(shape[0])->as_int();
    const auto C = SymEngine::rcp_static_cast<const SymEngine::Integer>(shape[1])->as_int();
    const bool row_major = layout.layout == MmaFragmentLayout::MMA_LAYOUT_ROW_MAJOR;

    GpuMmaFromMemoryLayout unclipped = layout;
    unclipped.rows = SymEngine::null;
    unclipped.cols = SymEngine::null;

    symbolic::Condition full = SymEngine::boolTrue;
    if (!layout.rows.is_null()) {
        full = symbolic::And(full, symbolic::Ge(layout.rows, shape[0]));
    }
    if (!layout.cols.is_null()) {
        full = symbolic::And(full, symbolic::Ge(layout.cols, shape[1]));
    }

    out.stream << "if (" << lang.expression(full) << ") {" << std::endl;
    out.stream.changeIndent(+4);
    if (load) {
        strategy.emit_fragment_load(out, frag, base, unclipped, frag_type, element_type, fragment_layout);
    } else {
        strategy.emit_fragment_store(out, frag, base, unclipped, frag_type, element_type);
    }
    out.stream.changeIndent(-4);
    out.stream << "} else {" << std::endl;
    out.stream.changeIndent(+4);

    out.stream << "using __docc_elem_t = std::remove_cv_t<std::remove_reference_t<decltype("
               << strategy.fragment_element(frag, "0") << ")>>;" << std::endl;

    const std::string ld = "(" + lang.expression(layout.ldstride) + ")";
    const std::string off = layout.offset.is_null() ? "0" : "(" + lang.expression(layout.offset) + ")";
    const std::string ptr = lang.type_cast(base, types::Pointer(types::Scalar(element_type)));
    std::string in_bounds = "true";
    if (!layout.rows.is_null()) {
        in_bounds += " && __docc_r < (" + lang.expression(layout.rows) + ")";
    }
    if (!layout.cols.is_null()) {
        in_bounds += " && __docc_c < (" + lang.expression(layout.cols) + ")";
    }
    const std::string mem = "(" + ptr + ")[" + off + " + " +
                            (row_major ? "__docc_r * " + ld + " + __docc_c" : "__docc_r + __docc_c * " + ld) + "]";
    const std::string elem = strategy.fragment_element(frag, "__docc_e");

    // Iota table over the block in the fragment's memory layout: loading it through the same fragment load yields
    // each lane element's linear block coordinate, independent of the fragment's register layout.
    auto emit_iota = [&]() {
        const std::string n = std::to_string(R * C);
        out.stream << "struct __docc_iota_t { __docc_elem_t v[" << n << "]; };" << std::endl;
        out.stream << "static constexpr __docc_iota_t __docc_iota = []() constexpr { __docc_iota_t t{}; for (int i "
                      "= 0; i < "
                   << n << "; ++i) { t.v[i] = static_cast<__docc_elem_t>(i); } return t; }();" << std::endl;
        out.stream << "__typeof__(" << frag << ") __docc_idx;" << std::endl;
        GpuMmaFromMemoryLayout iota_layout{
            .offset = symbolic::zero(), .ldstride = symbolic::integer(row_major ? C : R), .layout = layout.layout
        };
        strategy.emit_fragment_load(
            out,
            "__docc_idx",
            "const_cast<__docc_elem_t*>(__docc_iota.v)",
            iota_layout,
            frag_type,
            element_type,
            fragment_layout
        );
    };
    auto emit_coords_from_iota = [&]() {
        out.stream << "const int __docc_v = static_cast<int>(static_cast<float>("
                   << strategy.fragment_element("__docc_idx", "__docc_e") << "));" << std::endl;
        if (row_major) {
            out.stream << "const int __docc_r = __docc_v / " << C << ", __docc_c = __docc_v % " << C << ";"
                       << std::endl;
        } else {
            out.stream << "const int __docc_c = __docc_v / " << R << ", __docc_r = __docc_v % " << R << ";"
                       << std::endl;
        }
    };

    std::string condition;
    auto coords = strategy.fragment_element_coords(frag_type, block_size, "__docc_e", condition);
    if (coords && !condition.empty()) {
        out.stream << "#if !(" << condition << ")" << std::endl;
        emit_iota();
        out.stream << "#endif" << std::endl;
    } else if (!coords) {
        emit_iota();
    }

    out.stream << "for (int __docc_e = 0; __docc_e < " << strategy.fragment_element_count(frag) << "; ++__docc_e) {"
               << std::endl;
    out.stream.changeIndent(+4);
    if (coords) {
        if (!condition.empty()) {
            out.stream << "#if " << condition << std::endl;
        }
        out.stream << "const int __docc_r = " << coords->first << ", __docc_c = " << coords->second << ";" << std::endl;
        if (!condition.empty()) {
            out.stream << "#else" << std::endl;
            emit_coords_from_iota();
            out.stream << "#endif" << std::endl;
        }
    } else {
        emit_coords_from_iota();
    }
    if (load) {
        out.stream << elem << " = (" << in_bounds << ") ? static_cast<__docc_elem_t>(" << mem
                   << ") : static_cast<__docc_elem_t>(0);" << std::endl;
    } else {
        out.stream << "if (" << in_bounds << ") { " << mem << " = " << elem << "; }" << std::endl;
    }
    out.stream.changeIndent(-4);
    out.stream << "}" << std::endl;
    out.stream.changeIndent(-4);
    out.stream << "}" << std::endl;
}
void GpuMmaMatmulDispatcher::dispatch_code_with_edges(
    codegen::CodegenOutput& out,
    std::vector<codegen::DispatchInput>& inputs,
    std::vector<codegen::DispatchOutput>& /*outputs*/
) {
    strategy_->emit_needed_declarations(out);

    // All three operands are fragments (declared by their containers); this is only the MMA step.
    const std::string& acc = inputs.at(GpuMmaMatmulNode::Y_INPUT_IDX).expr;
    const std::string& frag_a = inputs.at(GpuMmaMatmulNode::A_INPUT_IDX).expr;
    const std::string& frag_b = inputs.at(GpuMmaMatmulNode::B_INPUT_IDX).expr;

    strategy_->emit_mma(out, acc, frag_a, frag_b, acc);
}

void GpuMmaFillDispatcher::dispatch_code_with_edges(
    codegen::CodegenOutput& out,
    std::vector<codegen::DispatchInput>& inputs,
    std::vector<codegen::DispatchOutput>& /*outputs*/
) {
    auto& node = static_cast<const GpuMmaFillNode&>(node_);

    strategy_->emit_needed_declarations(out);

    const std::string& acc = inputs.at(GpuMmaFillNode::Y_INPUT_IDX).expr;
    strategy_->emit_fill(out, acc, node.fill_type());
}

static MmaFragmentLayout get_fragment_memory_layout(const std::string& fragment_container, const Function& function) {
    auto frag_storage = function.type(fragment_container).storage_type();
    auto& frag_args = frag_storage.args();
    if (frag_args.size() > 4) {
        auto layout_arg = frag_args.at(4);
        auto layout_int = SymEngine::rcp_dynamic_cast<const SymEngine::Integer>(layout_arg);
        if (!layout_int.is_null()) {
            return static_cast<MmaFragmentLayout>(layout_int->as_int());
        }
    }
    return MmaFragmentLayout::MMA_LAYOUT_UNSPECIFIED;
}

void GpuMmaFragmentLoadDispatcher::dispatch_code_with_edges(
    codegen::CodegenOutput& out,
    std::vector<codegen::DispatchInput>& inputs,
    std::vector<codegen::DispatchOutput>& /*outputs*/
) {
    auto& node = static_cast<const GpuMmaFragmentLoadNode&>(node_);

    strategy_->emit_needed_declarations(out);

    const std::string& frag = inputs.at(GpuMmaFragmentLoadNode::FRAG_INPUT_IDX).expr;
    const std::string& base = inputs.at(GpuMmaFragmentLoadNode::PTR_INPUT_IDX).expr;

    // A/B fragments with a matching declared layout take the direct overload; the runtime-layout one goes through a
    // temporary that defeats packing of the operand registers.

    auto& frag_node =
        static_cast<const data_flow::AccessNode&>(inputs.at(GpuMmaFragmentLoadNode::FRAG_INPUT_IDX).edge.src());
    auto frag_layout = get_fragment_memory_layout(frag_node.data(), function_);

    if (node.layout().clipped()) {
        emit_clipped_fragment_transfer(
            out,
            *strategy_,
            true,
            frag,
            base,
            node.layout(),
            node.block_size(),
            node.fragment_type(),
            node.element_type(),
            frag_layout
        );
        return;
    }
    strategy_
        ->emit_fragment_load(out, frag, base, node.layout(), node.fragment_type(), node.element_type(), frag_layout);
}

void GpuMmaFragmentStoreDispatcher::dispatch_code_with_edges(
    codegen::CodegenOutput& out,
    std::vector<codegen::DispatchInput>& inputs,
    std::vector<codegen::DispatchOutput>& /*outputs*/
) {
    auto& node = static_cast<const GpuMmaFragmentStoreNode&>(node_);

    strategy_->emit_needed_declarations(out);

    const std::string& frag = inputs.at(GpuMmaFragmentStoreNode::FRAG_INPUT_IDX).expr;
    const std::string& base = inputs.at(GpuMmaFragmentStoreNode::PTR_INPUT_IDX).expr;

    if (node.layout().clipped()) {
        auto& frag_node =
            static_cast<const data_flow::AccessNode&>(inputs.at(GpuMmaFragmentStoreNode::FRAG_INPUT_IDX).edge.src());
        emit_clipped_fragment_transfer(
            out,
            *strategy_,
            false,
            frag,
            base,
            node.layout(),
            node.block_size(),
            node.fragment_type(),
            node.element_type(),
            get_fragment_memory_layout(frag_node.data(), function_)
        );
        return;
    }
    strategy_->emit_fragment_store(out, frag, base, node.layout(), node.fragment_type(), node.element_type());
}

void GpuMmaEltwiseAddDispatcher::dispatch_code_with_edges(
    codegen::CodegenOutput& out,
    std::vector<codegen::DispatchInput>& inputs,
    std::vector<codegen::DispatchOutput>& /*outputs*/
) {
    auto& node = static_cast<const GpuMmaFragmentEltwiseAddNode&>(node_);

    auto otype = out.language_extension.primitive_type(node.output_type());
    auto atype = out.language_extension.primitive_type(node.acc_type());

    strategy_->emit_needed_declarations(out);

    const std::string& frag_d = inputs.at(GpuMmaFragmentEltwiseAddNode::FRAG_D_INPUT_IDX).expr;
    const std::string& frag_acc = inputs.at(GpuMmaFragmentEltwiseAddNode::FRAG_ACC_INPUT_IDX).expr;
    const std::string& frag_c = inputs.at(GpuMmaFragmentEltwiseAddNode::FRAG_C_INPUT_IDX).expr;

    strategy_->emit_eltwise(
        out, frag_d, {frag_acc, frag_c}, [&otype, &atype](auto& out, auto& main_elem, auto& idx, auto& args) {
            out.stream << main_elem << " = static_cast<" << otype << ">(" << args.at(0) << " + static_cast<" << atype
                       << ">(" << args.at(1) << "));" << std::endl;
        }
    );
}

} // namespace sdfg::gpu
