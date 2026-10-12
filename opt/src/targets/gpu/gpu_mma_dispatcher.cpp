#include "sdfg/targets/gpu/gpu_mma_dispatcher.h"

#include "sdfg/targets/gpu/gpu_mma_fill_node.h"
#include "sdfg/targets/gpu/gpu_mma_fragment_eltwise_add_node.h"
#include "sdfg/targets/gpu/gpu_mma_fragment_load_node.h"
#include "sdfg/targets/gpu/gpu_mma_fragment_store_node.h"
#include "sdfg/targets/gpu/gpu_mma_matmul_node.h"

namespace sdfg::gpu {

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

void GpuMmaFragmentLoadDispatcher::dispatch_code_with_edges(
    codegen::CodegenOutput& out,
    std::vector<codegen::DispatchInput>& inputs,
    std::vector<codegen::DispatchOutput>& /*outputs*/
) {
    auto& node = static_cast<const GpuMmaFragmentLoadNode&>(node_);

    strategy_->emit_needed_declarations(out);

    const std::string& frag = inputs.at(GpuMmaFragmentLoadNode::FRAG_INPUT_IDX).expr;
    const std::string& base = inputs.at(GpuMmaFragmentLoadNode::PTR_INPUT_IDX).expr;

    strategy_->emit_fragment_load(out, frag, base, node.layout(), node.fragment_type(), node.element_type());
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
