#include "sdfg/data_flow/library_nodes/stdlib/memmove.h"

namespace sdfg {
namespace stdlib {

MemmoveNode::MemmoveNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const symbolic::Expression count
)
    : StdlibNode(
          element_id,
          debug_info,
          vertex,
          parent,
          LibraryNodeType_Memmove,
          {},
          {"_dst", "_src"},
          true,
          data_flow::ImplementationType_NONE
      ),
      count_(count) {}

const symbolic::Expression MemmoveNode::count() const { return count_; }

void MemmoveNode::validate(const Function& function) const { LibraryNode::validate(function); }

symbolic::SymbolSet MemmoveNode::symbols() const {
    auto count_symbols = symbolic::atoms(this->count_);
    return count_symbols;
}

std::unique_ptr<data_flow::DataFlowNode> MemmoveNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::make_unique<MemmoveNode>(element_id, debug_info_, vertex, parent, count_);
}

void MemmoveNode::replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) {
    this->count_ = symbolic::subs(this->count_, old_expression, new_expression);
}

nlohmann::json MemmoveNodeSerializer::serialize(const data_flow::LibraryNode& library_node) {
    const MemmoveNode& node = static_cast<const MemmoveNode&>(library_node);

    nlohmann::json j;
    j["code"] = node.code().value();

    sdfg::serializer::JSONSerializer serializer;
    j["count"] = serializer.expression(node.count());

    return j;
}

data_flow::LibraryNode& MemmoveNodeSerializer::deserialize(
    const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
) {
    assert(j.contains("code"));
    assert(j.contains("debug_info"));
    assert(j.contains("count"));

    auto code = j["code"].get<std::string>();
    if (code != LibraryNodeType_Memmove.value()) {
        throw InvalidSDFGException("Invalid library node code");
    }

    // Extract debug info using JSONSerializer
    sdfg::serializer::JSONSerializer serializer;
    DebugInfo debug_info = serializer.json_to_debug_info(j["debug_info"]);

    // Extract properties
    auto count = symbolic::parse(j.at("count"));

    return builder.add_library_node<MemmoveNode>(parent, debug_info, count);
}

MemmoveNodeDispatcher::MemmoveNodeDispatcher(
    codegen::LanguageExtension& language_extension,
    const Function& function,
    const data_flow::DataFlowGraph& data_flow_graph,
    const MemmoveNode& node
)
    : codegen::LibraryNodeDispatcher(language_extension, function, data_flow_graph, node) {}

void MemmoveNodeDispatcher::dispatch_code_with_edges(
    codegen::CodegenOutput& out,
    std::vector<codegen::DispatchInput>& inputs,
    std::vector<codegen::DispatchOutput>& outputs
) {
    auto& node = static_cast<const MemmoveNode&>(node_);

    out.stream << language_extension_.external_prefix() << "memmove(" << inputs.at(0).expr << ", " << inputs.at(1).expr
               << ", " << language_extension_.expression(node.count()) << ")" << ";";
    out.stream << std::endl;
}

MemmoveNode& add_memmove_node(
    builder::StructuredSDFGBuilder& builder,
    Block& block,
    const std::string& src_ptr,
    const std::string& dst_ptr,
    const symbolic::Expression& count,
    const types::IType& ptr_type,
    DebugInfo debug_info
) {
    auto& src_ptr_access = builder.add_access(block, src_ptr);
    auto& dst_ptr_access = builder.add_access(block, dst_ptr);
    auto& libnode = builder.add_library_node<stdlib::MemmoveNode>(block, debug_info, count);
    builder.add_computational_memlet(block, src_ptr_access, libnode, "_src", {}, ptr_type);
    builder.add_computational_memlet(block, dst_ptr_access, libnode, "_dst", {}, ptr_type);

    return static_cast<MemmoveNode&>(libnode);
}

std::tuple<Block&, MemmoveNode&> add_memmove_block(
    builder::StructuredSDFGBuilder& builder,
    Sequence& parent,
    const std::string& src_ptr,
    const std::string& dst_ptr,
    const symbolic::Expression& count,
    const types::IType& ptr_type,
    DebugInfo debug_info
) {
    auto& block = builder.add_block(parent);

    auto& libnode = add_memmove_node(builder, block, src_ptr, dst_ptr, count, ptr_type, debug_info);

    return {block, libnode};
}

} // namespace stdlib
} // namespace sdfg
