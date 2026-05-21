#include "sdfg/data_flow/library_nodes/stdlib/memcpy.h"

namespace sdfg {
namespace stdlib {

MemcpyNode::MemcpyNode(
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
          LibraryNodeType_Memcpy,
          {},
          {"_dst", "_src"},
          true,
          data_flow::ImplementationType_NONE
      ),
      count_(count) {}

const symbolic::Expression MemcpyNode::count() const { return count_; }

void MemcpyNode::validate(const Function& function) const { LibraryNode::validate(function); }

symbolic::SymbolSet MemcpyNode::symbols() const {
    auto count_symbols = symbolic::atoms(this->count_);
    return count_symbols;
}

std::unique_ptr<data_flow::DataFlowNode> MemcpyNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::make_unique<MemcpyNode>(element_id, debug_info_, vertex, parent, count_);
}

data_flow::PointerAccessType MemcpyNode::pointer_access_type(int input_idx) const {
    if (input_idx == 0) {
        return data_flow::PointerAccessMeta::create_full_write_only(count_, true);
    } else if (input_idx == 1) {
        return data_flow::PointerAccessMeta::create_read_only(count_, true);
    } else {
        return StdlibNode::pointer_access_type(input_idx);
    }
}

std::string MemcpyNode::toStr() const { return StdlibNode::toStr() + "(n: " + count_->__str__() + ")"; }

void MemcpyNode::replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) {
    this->count_ = symbolic::subs(this->count_, old_expression, new_expression);
}

nlohmann::json MemcpyNodeSerializer::serialize(const data_flow::LibraryNode& library_node) {
    const MemcpyNode& node = static_cast<const MemcpyNode&>(library_node);

    nlohmann::json j;
    j["code"] = node.code().value();

    sdfg::serializer::JSONSerializer serializer;
    j["count"] = serializer.expression(node.count());

    return j;
}

data_flow::LibraryNode& MemcpyNodeSerializer::deserialize(
    const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
) {
    assert(j.contains("code"));
    assert(j.contains("debug_info"));
    assert(j.contains("count"));

    auto code = j["code"].get<std::string>();
    if (code != LibraryNodeType_Memcpy.value()) {
        throw InvalidSDFGException("Invalid library node code");
    }

    // Extract debug info using JSONSerializer
    sdfg::serializer::JSONSerializer serializer;
    DebugInfo debug_info = serializer.json_to_debug_info(j["debug_info"]);

    // Extract properties
    auto count = symbolic::parse(j.at("count"));

    return builder.add_library_node<MemcpyNode>(parent, debug_info, count);
}

MemcpyNodeDispatcher::MemcpyNodeDispatcher(
    codegen::LanguageExtension& language_extension,
    const Function& function,
    const data_flow::DataFlowGraph& data_flow_graph,
    const MemcpyNode& node
)
    : codegen::LibraryNodeDispatcher(language_extension, function, data_flow_graph, node) {}

void MemcpyNodeDispatcher::dispatch_code_with_edges(
    codegen::CodegenOutput& out,
    std::vector<codegen::DispatchInput>& inputs,
    std::vector<codegen::DispatchOutput>& outputs
) {
    auto& node = static_cast<const MemcpyNode&>(node_);

    out.stream << language_extension_.external_prefix() << "memcpy(" << inputs.at(0).expr << ", " << inputs.at(1).expr
               << ", " << language_extension_.expression(node.count()) << ")" << ";";
    out.stream << std::endl;
}

} // namespace stdlib
} // namespace sdfg
