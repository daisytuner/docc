#include "sdfg/data_flow/library_nodes/stdlib/free.h"

namespace sdfg {
namespace stdlib {

FreeNode::FreeNode(
    size_t element_id, const DebugInfo& debug_info, const graph::Vertex vertex, data_flow::DataFlowGraph& parent
)
    : StdlibNode(
          element_id,
          debug_info,
          vertex,
          parent,
          LibraryNodeType_Free,
          {},
          {"_ptr"},
          true,
          data_flow::ImplementationType_NONE
      ) {}


void FreeNode::validate(const Function& function) const { LibraryNode::validate(function); }

symbolic::SymbolSet FreeNode::symbols() const { return {}; }

std::unique_ptr<data_flow::DataFlowNode> FreeNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::make_unique<FreeNode>(element_id, debug_info_, vertex, parent);
}

void FreeNode::replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) {
    // Do nothing
    return;
}

data_flow::PointerAccessType FreeNode::pointer_access_type(int input_idx) const {
    if (input_idx == 0) {
        return data_flow::PointerAccessMeta::create_invalidate();
    } else {
        return LibraryNode::pointer_access_type(input_idx);
    }
}

nlohmann::json FreeNodeSerializer::serialize(const data_flow::LibraryNode& library_node) {
    const FreeNode& node = static_cast<const FreeNode&>(library_node);

    nlohmann::json j;
    j["code"] = node.code().value();

    return j;
}

data_flow::LibraryNode& FreeNodeSerializer::deserialize(
    const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
) {
    assert(j.contains("code"));
    assert(j.contains("debug_info"));

    auto code = j["code"].get<std::string>();
    if (code != LibraryNodeType_Free.value()) {
        throw InvalidSDFGException("Invalid library node code");
    }

    sdfg::serializer::JSONSerializer serializer;
    DebugInfo debug_info = serializer.json_to_debug_info(j["debug_info"]);

    return builder.add_library_node<FreeNode>(parent, debug_info);
}

FreeNodeDispatcher::FreeNodeDispatcher(
    codegen::LanguageExtension& language_extension,
    const Function& function,
    const data_flow::DataFlowGraph& data_flow_graph,
    const FreeNode& node
)
    : codegen::LibraryNodeDispatcher(language_extension, function, data_flow_graph, node) {}

void FreeNodeDispatcher::dispatch_code_with_edges(
    codegen::CodegenOutput& out,
    std::vector<codegen::DispatchInput>& inputs,
    std::vector<codegen::DispatchOutput>& outputs
) {
    out.stream << out.language_extension.external_prefix() << "free(" << inputs.at(0).expr << ");" << std::endl;
}

FreeNode& add_free_node(
    builder::StructuredSDFGBuilder& builder,
    Block& block,
    const std::string& ptr,
    const types::IType& ptr_type,
    DebugInfo debug_info
) {
    auto& dst_ptr_access = builder.add_access(block, ptr);
    auto& libnode = builder.add_library_node<stdlib::FreeNode>(block, debug_info);
    builder.add_computational_memlet(block, dst_ptr_access, libnode, "_ptr", {}, ptr_type);

    return static_cast<FreeNode&>(libnode);
}

std::tuple<Block&, FreeNode&> add_free_block(
    builder::StructuredSDFGBuilder& builder,
    Sequence& parent,
    const std::string& ptr,
    const types::IType& ptr_type,
    DebugInfo debug_info
) {
    auto& block = builder.add_block(parent);

    auto& libnode = add_free_node(builder, block, ptr, ptr_type, debug_info);

    return {block, libnode};
}

} // namespace stdlib
} // namespace sdfg
