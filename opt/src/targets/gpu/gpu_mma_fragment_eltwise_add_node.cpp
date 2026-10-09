#include "sdfg/targets/gpu/gpu_mma_fragment_eltwise_add_node.h"

#include <nlohmann/json.hpp>

#include "sdfg/builder/structured_sdfg_builder.h"

#include <sstream>

#include "sdfg/data_flow/data_flow_graph.h"
#include "sdfg/exceptions.h"
#include "sdfg/types/type.h"

namespace sdfg::gpu {

GpuMmaFragmentEltwiseAddNode::GpuMmaFragmentEltwiseAddNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const MmaBlockSize& block_size,
    types::PrimitiveType output_type,
    types::PrimitiveType acc_type,
    const data_flow::ImplementationType& impl_type
)
    : LibraryNode(
          element_id,
          debug_info,
          vertex,
          parent,
          LibraryNodeType_GpuMmaFragmentEltwiseAdd,
          {},
          {"fragD", "fragAcc", "fragC"},
          false,
          impl_type
      ),
      block_size_(block_size), output_type_(output_type), acc_type_(acc_type) {
}

void GpuMmaFragmentEltwiseAddNode::validate(const Function& function) const {
    auto& graph = this->get_parent();
    if (graph.in_degree(*this) != 3) {
        throw InvalidSDFGException("MmaFragmentLoadNode: Expected exactly 2 inputs (frag, ptr)");
    }
    if (graph.out_degree(*this) != 0) {
        throw InvalidSDFGException("MmaFragmentLoadNode: Expected no outputs");
    }
}

symbolic::SymbolSet GpuMmaFragmentEltwiseAddNode::symbols() const {
    symbolic::SymbolSet syms;
    return syms;
}

void GpuMmaFragmentEltwiseAddNode::
    replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) {
}

void GpuMmaFragmentEltwiseAddNode::replace(const symbolic::ExpressionMapping& replacements) {
}

std::unique_ptr<data_flow::DataFlowNode> GpuMmaFragmentEltwiseAddNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::unique_ptr<data_flow::DataFlowNode>(new GpuMmaFragmentEltwiseAddNode(
        element_id, debug_info(), vertex, parent, block_size_, output_type_, acc_type_, implementation_type_
    ));
}

std::string GpuMmaFragmentEltwiseAddNode::toStr() const {
    std::stringstream ss;
    auto otype = types::primitive_type_to_string(output_type_);
    auto atype = types::primitive_type_to_string(acc_type_);
    ss << "MmaFragmentEltwiseAdd(" << block_size_.dim_str(MmaFragmentType::C) << ", D(" << otype << " = Acc(" << atype
       << ") + C(" << otype << ")";
    ss << ", " << implementation_type_.value();
    ss << ")";
    return ss.str();
}

data_flow::PointerAccessType GpuMmaFragmentEltwiseAddNode::pointer_access_type(int input_idx) const {
    auto frag_elems = SymEngine::mul(block_size_.get_shape(MmaFragmentType::C));
    if (input_idx == FRAG_D_INPUT_IDX) {
        return data_flow::PointerAccessMeta::create_full_write_only(frag_elems, true);
    } else if (input_idx == FRAG_C_INPUT_IDX || input_idx == FRAG_ACC_INPUT_IDX) {
        return data_flow::PointerAccessMeta::create_read_only(frag_elems, true);
    }
    return LibraryNode::pointer_access_type(input_idx);
}

nlohmann::json GpuMmaFragmentEltwiseAddNodeSerializer::serialize(const data_flow::LibraryNode& library_node) {
    const auto& node = static_cast<const GpuMmaFragmentEltwiseAddNode&>(library_node);
    nlohmann::json j;
    j["code"] = node.code().value();
    serialize_mma_block_size(j["block_size"], node.block_size());
    j["output_type"] = node.output_type();
    j["acc_type"] = node.acc_type();
    return j;
}

data_flow::LibraryNode& GpuMmaFragmentEltwiseAddNodeSerializer::deserialize(
    const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
) {
    serializer::JSONSerializer serializer;
    DebugInfo debug_info = serializer.json_to_debug_info(j["debug_info"]);
    auto block_size = deserialize_mma_block_size(j.at("block_size"));
    auto output_type = j.at("output_type").get<types::PrimitiveType>();
    auto acc_type = j.at("acc_type").get<types::PrimitiveType>();
    return builder.add_library_node<GpuMmaFragmentEltwiseAddNode>(
        parent, debug_info, block_size, output_type, acc_type, data_flow::ImplementationType_NONE
    );
}

} // namespace sdfg::gpu
