#include "sdfg/targets/gpu/gpu_mma_matmul_node.h"

#include <nlohmann/json.hpp>

#include "sdfg/builder/structured_sdfg_builder.h"

#include <sstream>

#include "sdfg/data_flow/data_flow_node.h"
#include "sdfg/exceptions.h"
#include "sdfg/types/type.h"

namespace sdfg::gpu {

GpuMmaMatmulNode::GpuMmaMatmulNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const MmaBlockSize& mma_block_size,
    types::PrimitiveType input_type,
    types::PrimitiveType acc_type,
    const data_flow::ImplementationType& impl_type
)
    : LibraryNode(
          element_id, debug_info, vertex, parent, LibraryNodeType_GpuMmaMatmul, {}, {"Y", "A", "B"}, false, impl_type
      ),
      mma_block_size_(mma_block_size), input_type_(input_type), acc_type_(acc_type) {
    if (mma_block_size_.m == 0 || mma_block_size_.n == 0 || mma_block_size_.k == 0) {
        throw std::invalid_argument("GpuMmaMatmulNode: MMA block size must be non-zero");
    }
}

void GpuMmaMatmulNode::validate(const Function& function) const {
    auto& graph = this->get_parent();
    if (graph.in_degree(*this) != 3) {
        throw InvalidSDFGException("GpuMmaMatmulNode: Expected exactly 3 inputs (Y, A, B)");
    }
    if (graph.out_degree(*this) != 0) {
        throw InvalidSDFGException("GpuMmaMatmulNode: Expected no outputs");
    }
}

symbolic::SymbolSet GpuMmaMatmulNode::symbols() const {
    symbolic::SymbolSet syms;
    return syms;
}

void GpuMmaMatmulNode::replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) {
}

void GpuMmaMatmulNode::replace(const symbolic::ExpressionMapping& replacements) {
}

std::unique_ptr<data_flow::DataFlowNode> GpuMmaMatmulNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::unique_ptr<GpuMmaMatmulNode>(new GpuMmaMatmulNode(
        element_id, debug_info(), vertex, parent, mma_block_size_, input_type_, acc_type_, implementation_type_
    ));
}

std::string GpuMmaMatmulNode::toStr() const {
    std::stringstream ss;
    ss << "GpuMma(" << mma_block_size_;
    ss << ": " << implementation_type_.value();
    ss << ")";
    return ss.str();
}

symbolic::Expression GpuMmaMatmulNode::flop() const {
    auto res_elems = symbolic::mul(this->m(), this->n());
    // one multiply and one add per contraction element
    return symbolic::mul(symbolic::integer(2), symbolic::mul(res_elems, this->k()));
}

data_flow::PointerAccessType GpuMmaMatmulNode::pointer_access_type(int input_idx) const {
    if (input_idx == Y_INPUT_IDX) {
        return data_flow::PointerAccessMeta::create_generic(
            data_flow::ConvexAccessPattern::create(symbolic::integer(mma_block_size_.m * mma_block_size_.n), true),
            data_flow::ConvexAccessPattern::create(symbolic::integer(mma_block_size_.m * mma_block_size_.n), true),
            true
        );
    } else if (input_idx == A_INPUT_IDX) {
        return data_flow::PointerAccessMeta::
            create_read_only(symbolic::integer(mma_block_size_.m * mma_block_size_.k), true);
    } else if (input_idx == B_INPUT_IDX) {
        return data_flow::PointerAccessMeta::
            create_read_only(symbolic::integer(mma_block_size_.k * mma_block_size_.n), true);
    }
    throw std::invalid_argument("GpuMmaNode: Invalid input index for pointer access type");
}

nlohmann::json GpuMmaMatmulNodeSerializer::serialize(const data_flow::LibraryNode& library_node) {
    const auto& node = static_cast<const GpuMmaMatmulNode&>(library_node);
    nlohmann::json j;
    j["code"] = node.code().value();
    serialize_mma_block_size(j["block_size"], node.mma_block_size());
    j["input_type"] = node.input_type();
    j["acc_type"] = node.acc_type();
    return j;
}

data_flow::LibraryNode& GpuMmaMatmulNodeSerializer::deserialize(
    const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
) {
    serializer::JSONSerializer serializer;
    DebugInfo debug_info = serializer.json_to_debug_info(j["debug_info"]);
    auto block_size = deserialize_mma_block_size(j.at("block_size"));
    auto input_type = j.at("input_type").get<types::PrimitiveType>();
    auto acc_type = j.at("acc_type").get<types::PrimitiveType>();
    return builder.add_library_node<
        GpuMmaMatmulNode>(parent, debug_info, block_size, input_type, acc_type, data_flow::ImplementationType_NONE);
}

} // namespace sdfg::gpu
