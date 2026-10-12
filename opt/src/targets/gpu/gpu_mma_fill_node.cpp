#include "sdfg/targets/gpu/gpu_mma_fill_node.h"

#include <nlohmann/json.hpp>

#include "sdfg/builder/structured_sdfg_builder.h"

#include <sstream>

#include "sdfg/exceptions.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/type.h"

namespace sdfg::gpu {

GpuMmaFillNode::GpuMmaFillNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const MmaBlockSize& mma_block_size,
    types::PrimitiveType fill_type,
    const data_flow::ImplementationType& impl_type
)
    : LibraryNode(element_id, debug_info, vertex, parent, LibraryNodeType_GpuMmaFill, {}, {"Y"}, false, impl_type),
      mma_block_size_(mma_block_size), fill_type_(fill_type) {
    if (mma_block_size_.m == 0 || mma_block_size_.n == 0 || mma_block_size_.k == 0) {
        throw std::invalid_argument("GpuMmaFillNode: MMA block size must be non-zero");
    }
}

void GpuMmaFillNode::validate(const Function& function) const {
    auto& graph = this->get_parent();
    if (graph.in_degree(*this) != 1) {
        throw InvalidSDFGException("GpuMmaFillNode: Expected exactly 1 input (Y)");
    }
    if (graph.out_degree(*this) != 0) {
        throw InvalidSDFGException("GpuMmaFillNode: Expected no outputs");
    }
}

symbolic::SymbolSet GpuMmaFillNode::symbols() const {
    symbolic::SymbolSet syms;
    return syms;
}

void GpuMmaFillNode::replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) {
}

void GpuMmaFillNode::replace(const symbolic::ExpressionMapping& replacements) {
}

std::unique_ptr<data_flow::DataFlowNode> GpuMmaFillNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::unique_ptr<data_flow::DataFlowNode>(
        new GpuMmaFillNode(element_id, debug_info(), vertex, parent, mma_block_size_, fill_type_, implementation_type_)
    );
}

std::string GpuMmaFillNode::toStr() const {
    std::stringstream ss;
    ss << "GpuMmaFill(" << mma_block_size_;
    ss << ", impl: " << implementation_type_.value();
    ss << ")";
    return ss.str();
}

data_flow::PointerAccessType GpuMmaFillNode::pointer_access_type(int input_idx) const {
    if (input_idx == Y_INPUT_IDX) {
        return data_flow::PointerAccessMeta::
            create_full_write_only(SymEngine::mul(mma_block_size_.get_shape(MmaFragmentType::C)), true);
    }
    throw std::invalid_argument("GpuMmaFillNode: Invalid input index for pointer access type");
}

nlohmann::json GpuMmaFillNodeSerializer::serialize(const data_flow::LibraryNode& library_node) {
    const auto& node = static_cast<const GpuMmaFillNode&>(library_node);
    nlohmann::json j;
    j["code"] = node.code().value();
    serialize_mma_block_size(j["block_size"], node.mma_block_size());
    j["fill_type"] = node.fill_type();
    return j;
}

data_flow::LibraryNode& GpuMmaFillNodeSerializer::deserialize(
    const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
) {
    serializer::JSONSerializer serializer;
    DebugInfo debug_info = serializer.json_to_debug_info(j["debug_info"]);
    auto block_size = deserialize_mma_block_size(j.at("block_size"));
    auto fill_type = j.at("fill_type").get<types::PrimitiveType>();
    return builder
        .add_library_node<GpuMmaFillNode>(parent, debug_info, block_size, fill_type, data_flow::ImplementationType_NONE);
}

} // namespace sdfg::gpu
