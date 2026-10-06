#include "sdfg/targets/gpu/gpu_mma_fragment_load_node.h"

#include <nlohmann/json.hpp>

#include "sdfg/builder/structured_sdfg_builder.h"

#include <sstream>

#include "sdfg/data_flow/data_flow_graph.h"
#include "sdfg/exceptions.h"
#include "sdfg/symbolic/utils.h"
#include "sdfg/types/type.h"

namespace sdfg::gpu {

GpuMmaFragmentLoadNode::GpuMmaFragmentLoadNode(
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
          element_id, debug_info, vertex, parent, LibraryNodeType_GpuMmaFragmentLoad, {}, {"frag", "ptr"}, false, impl_type
      ),
      block_size_(block_size), fragment_type_(fragment_type), layout_(layout), element_type_(element_type) {
    if (layout_.layout == MmaFragmentLayout::MMA_LAYOUT_UNSPECIFIED) {
        throw std::invalid_argument("GpuMmaFragmentLoadNode: MMA from-memory layout must be specified");
    }
}

void GpuMmaFragmentLoadNode::validate(const Function& function) const {
    auto& graph = this->get_parent();
    if (graph.in_degree(*this) != 2) {
        throw InvalidSDFGException("MmaFragmentLoadNode: Expected exactly 2 inputs (frag, ptr)");
    }
    if (graph.out_degree(*this) != 0) {
        throw InvalidSDFGException("MmaFragmentLoadNode: Expected no outputs");
    }
}

symbolic::SymbolSet GpuMmaFragmentLoadNode::symbols() const {
    symbolic::SymbolSet syms;
    layout_.collect_symbols(syms);
    return syms;
}

void GpuMmaFragmentLoadNode::replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) {
    layout_.replace(old_expression, new_expression);
}

void GpuMmaFragmentLoadNode::replace(const symbolic::ExpressionMapping& replacements) {
    layout_.replace(replacements);
}

std::unique_ptr<data_flow::DataFlowNode> GpuMmaFragmentLoadNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::unique_ptr<data_flow::DataFlowNode>(new GpuMmaFragmentLoadNode(
        element_id, debug_info(), vertex, parent, block_size_, fragment_type_, layout_, element_type_, implementation_type_
    ));
}

data_flow::PointerAccessType GpuMmaFragmentLoadNode::pointer_access_type(int input_idx) const {
    if (input_idx == FRAG_INPUT_IDX) {
        return data_flow::PointerAccessMeta::
            create_full_write_only(SymEngine::mul(block_size_.get_shape(fragment_type_)), true);
    } else if (input_idx == PTR_INPUT_IDX) {
        auto t_layout = layout_.to_tensor_layout(block_size_, fragment_type_);
        return data_flow::PointerAccessMeta::create_read_only(symbolic::__nullptr__(), true, t_layout);
    }
    return LibraryNode::pointer_access_type(input_idx);
}

bool GpuMmaFragmentLoadNode::
    relocalize_operand_internal(int input_idx, const math::tensor::TensorLayout& packed, bool check_only) {
    if (input_idx == PTR_INPUT_IDX) {
        auto type = packed.is_2d_col_or_row_major();
        if (type == math::tensor::TensorLayout::LAYOUT_ROW_MAJOR) {
            if (!check_only) {
                layout_ = GpuMmaFromMemoryLayout{
                    .offset = packed.offset(), .ldstride = packed.get_stride(1), .layout = MMA_LAYOUT_ROW_MAJOR
                };
            }
            return true;
        } else if (type == math::tensor::TensorLayout::LAYOUT_COL_MAJOR) {
            if (!check_only) {
                layout_ = GpuMmaFromMemoryLayout{
                    .offset = packed.offset(), .ldstride = packed.get_stride(0), .layout = MMA_LAYOUT_ROW_MAJOR
                };
            }
            return true;
        }
    }
    return false;
}

bool GpuMmaFragmentLoadNode::can_relocalize_operand(int input_idx, const math::tensor::TensorLayout& packed) const {
    if (input_idx == PTR_INPUT_IDX) {
        if (!symbolic::vectors_of_expressions_match(packed.shape(), block_size_.get_shape(fragment_type_))) {
            return false;
        }
        if (GpuMmaFromMemoryLayout::from_tensor_layout(packed)) {
            return true;
        }
    }
    return false;
}

bool GpuMmaFragmentLoadNode::relocalize_operand(int input_idx, const math::tensor::TensorLayout& packed) {
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

std::string GpuMmaFragmentLoadNode::toStr() const {
    std::stringstream ss;
    ss << "MmaFragmentLoad(" << block_size_.dim_str(fragment_type_) << ", " << layout_;
    ss << ", " << types::primitive_type_to_string(element_type_);
    ss << ", " << implementation_type_.value();
    ss << ")";
    return ss.str();
}

nlohmann::json GpuMmaFragmentLoadNodeSerializer::serialize(const data_flow::LibraryNode& library_node) {
    const auto& node = static_cast<const GpuMmaFragmentLoadNode&>(library_node);
    nlohmann::json j;
    j["code"] = node.code().value();
    serialize_mma_block_size(j["block_size"], node.block_size());
    j["fragment_type"] = static_cast<int>(node.fragment_type());
    serialize_mma_from_memory_layout(j["layout"], node.layout());
    j["element_type"] = node.element_type();
    return j;
}

data_flow::LibraryNode& GpuMmaFragmentLoadNodeSerializer::deserialize(
    const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
) {
    serializer::JSONSerializer serializer;
    DebugInfo debug_info = serializer.json_to_debug_info(j["debug_info"]);
    auto block_size = deserialize_mma_block_size(j.at("block_size"));
    auto fragment_type = static_cast<MmaFragmentType>(j.at("fragment_type").get<int>());
    auto layout = deserialize_mma_from_memory_layout(j.at("layout"));
    auto element_type = j.at("element_type").get<types::PrimitiveType>();
    return builder.add_library_node<GpuMmaFragmentLoadNode>(
        parent, debug_info, block_size, fragment_type, layout, element_type, data_flow::ImplementationType_NONE
    );
}

} // namespace sdfg::gpu
