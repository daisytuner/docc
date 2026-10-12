#include "sdfg/targets/gpu/gpu_mma_fragment_store_node.h"

#include <nlohmann/json.hpp>

#include "sdfg/builder/structured_sdfg_builder.h"

#include <sstream>

#include "sdfg/data_flow/data_flow_graph.h"
#include "sdfg/exceptions.h"
#include "sdfg/symbolic/utils.h"

namespace sdfg::gpu {

GpuMmaFragmentStoreNode::GpuMmaFragmentStoreNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const MmaBlockSize& block_size,
    MmaFragmentType fragment_type,
    const GpuMmaFromMemoryLayout& layout,
    types::PrimitiveType element_type,
    const data_flow::ImplementationType& impl_type
)
    : LibraryNode(
          element_id,
          debug_info,
          vertex,
          parent,
          LibraryNodeType_GpuMmaFragmentStore,
          {},
          {"frag", "ptr"},
          false,
          impl_type
      ),
      block_size_(block_size), fragment_type_(fragment_type), layout_(layout), element_type_(element_type) {
}

void GpuMmaFragmentStoreNode::validate(const Function& function) const {
    auto& graph = this->get_parent();
    if (graph.in_degree(*this) != 2) {
        throw InvalidSDFGException("MmaFragmentStoreNode: Expected exactly 2 inputs (frag, ptr)");
    }
    if (graph.out_degree(*this) != 0) {
        throw InvalidSDFGException("MmaFragmentStoreNode: Expected no outputs");
    }
}

symbolic::SymbolSet GpuMmaFragmentStoreNode::symbols() const {
    symbolic::SymbolSet syms;
    layout_.collect_symbols(syms);
    return syms;
}

void GpuMmaFragmentStoreNode::replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) {
    layout_.replace(old_expression, new_expression);
}

void GpuMmaFragmentStoreNode::replace(const symbolic::ExpressionMapping& replacements) {
    layout_.replace(replacements);
}

std::unique_ptr<data_flow::DataFlowNode> GpuMmaFragmentStoreNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::unique_ptr<data_flow::DataFlowNode>(new GpuMmaFragmentStoreNode(
        element_id, debug_info(), vertex, parent, block_size_, fragment_type_, layout_, element_type_, implementation_type_
    ));
}

std::string GpuMmaFragmentStoreNode::toStr() const {
    std::stringstream ss;
    ss << "MmaFragmentStore(" << block_size_.dim_str(fragment_type_) << ", " << layout_;
    ss << ", " << types::primitive_type_to_string(element_type_);
    ss << ", " << implementation_type_.value();
    ss << ")";
    return ss.str();
}

data_flow::PointerAccessType GpuMmaFragmentStoreNode::pointer_access_type(int input_idx) const {
    if (input_idx == FRAG_INPUT_IDX) {
        return data_flow::PointerAccessMeta::create_read_only(SymEngine::mul(block_size_.get_shape(fragment_type_)), true);
    } else if (input_idx == PTR_INPUT_IDX) {
        auto t_layout = layout_.to_tensor_layout(block_size_, fragment_type_);
        data_flow::MemoryAccessPatternType pattern;
        if (t_layout) {
            pattern = data_flow::ConvexAccessPattern::create(symbolic::__nullptr__(), false);
        } else {
            pattern = data_flow::TensorLayoutPattern::create(t_layout.value(), true);
        }
        return data_flow::PointerAccessMeta::
            create_generic(data_flow::NoAccessPattern::instance(), std::move(pattern), true);
    }
    return LibraryNode::pointer_access_type(input_idx);
}

bool GpuMmaFragmentStoreNode::can_relocalize_operand(int input_idx, const math::tensor::TensorLayout& packed) const {
    if (input_idx == PTR_INPUT_IDX) {
        if (!symbolic::vectors_of_expressions_match(packed.shape(), block_size_.get_shape(fragment_type_))) {
            return false;
        }
        if (auto new_layout = GpuMmaFromMemoryLayout::from_tensor_layout(packed)) {
            return true;
        }
    }
    return false;
}

bool GpuMmaFragmentStoreNode::relocalize_operand(int input_idx, const math::tensor::TensorLayout& packed) {
    if (input_idx == PTR_INPUT_IDX) {
        if (!symbolic::vectors_of_expressions_match(packed.shape(), block_size_.get_shape(fragment_type_))) {
            return false;
        }
        if (auto new_layout = GpuMmaFromMemoryLayout::from_tensor_layout(packed)) {
            layout_ = new_layout.value();
            return true;
        }
    }
    return false;
}

nlohmann::json GpuMmaFragmentStoreNodeSerializer::serialize(const data_flow::LibraryNode& library_node) {
    const auto& node = static_cast<const GpuMmaFragmentStoreNode&>(library_node);
    nlohmann::json j;
    j["code"] = node.code().value();
    serialize_mma_block_size(j["block_size"], node.block_size());
    j["fragment_type"] = static_cast<int>(node.fragment_type());
    serialize_mma_from_memory_layout(j["layout"], node.layout());
    j["element_type"] = node.element_type();
    return j;
}

data_flow::LibraryNode& GpuMmaFragmentStoreNodeSerializer::deserialize(
    const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
) {
    serializer::JSONSerializer serializer;
    DebugInfo debug_info = serializer.json_to_debug_info(j["debug_info"]);
    auto block_size = deserialize_mma_block_size(j.at("block_size"));
    auto fragment_type = static_cast<MmaFragmentType>(j.at("fragment_type").get<int>());
    auto layout = deserialize_mma_from_memory_layout(j.at("layout"));
    auto element_type = j.at("element_type").get<types::PrimitiveType>();
    return builder.add_library_node<GpuMmaFragmentStoreNode>(
        parent, debug_info, block_size, fragment_type, layout, element_type, data_flow::ImplementationType_NONE
    );
}

} // namespace sdfg::gpu
