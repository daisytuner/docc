#include "sdfg/data_flow/library_nodes/stdlib/memset.h"

namespace sdfg {
namespace stdlib {

MemsetNode::MemsetNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const symbolic::Expression value,
    const symbolic::Expression num
)
    : StdlibNode(
          element_id,
          debug_info,
          vertex,
          parent,
          LibraryNodeType_Memset,
          {},
          {"_ptr"},
          true,
          data_flow::ImplementationType_NONE
      ),
      num_(num), value_(value) {}

const symbolic::Expression MemsetNode::value() const { return value_; }

const symbolic::Expression MemsetNode::num() const { return num_; }

void MemsetNode::validate(const Function& function) const { LibraryNode::validate(function); }

symbolic::SymbolSet MemsetNode::symbols() const {
    auto value_symbols = symbolic::atoms(this->value_);
    auto num_symbols = symbolic::atoms(this->num_);
    num_symbols.insert(value_symbols.begin(), value_symbols.end());
    return num_symbols;
}

std::unique_ptr<data_flow::DataFlowNode> MemsetNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::make_unique<MemsetNode>(element_id, debug_info_, vertex, parent, value_, num_);
}

data_flow::PointerAccessType MemsetNode::pointer_access_type(int input_idx) const {
    if (input_idx == 0) {
        return data_flow::PointerAccessMeta::create_full_write_only(num_, true);
    } else {
        return StdlibNode::pointer_access_type(input_idx);
    }
}

std::string MemsetNode::toStr() const {
    return StdlibNode::toStr() + "(n: " + num_->__str__() + ", v: " + value_->__str__() + ")";
}

void MemsetNode::replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) {
    this->value_ = symbolic::subs(this->value_, old_expression, new_expression);
    this->num_ = symbolic::subs(this->num_, old_expression, new_expression);
}

nlohmann::json MemsetNodeSerializer::serialize(const data_flow::LibraryNode& library_node) {
    const MemsetNode& node = static_cast<const MemsetNode&>(library_node);

    nlohmann::json j;
    j["code"] = node.code().value();

    sdfg::serializer::JSONSerializer serializer;
    j["value"] = serializer.expression(node.value());
    j["num"] = serializer.expression(node.num());

    return j;
}

data_flow::LibraryNode& MemsetNodeSerializer::deserialize(
    const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
) {
    assert(j.contains("code"));
    assert(j.contains("debug_info"));
    assert(j.contains("value"));
    assert(j.contains("num"));

    auto code = j["code"].get<std::string>();
    if (code != LibraryNodeType_Memset.value()) {
        throw InvalidSDFGException("Invalid library node code");
    }

    // Extract debug info using JSONSerializer
    sdfg::serializer::JSONSerializer serializer;
    DebugInfo debug_info = serializer.json_to_debug_info(j["debug_info"]);

    // Extract properties
    auto value = symbolic::parse(j.at("value"));
    auto num = symbolic::parse(j.at("num"));

    return builder.add_library_node<MemsetNode>(parent, debug_info, value, num);
}

MemsetNodeDispatcher::MemsetNodeDispatcher(
    codegen::LanguageExtension& language_extension,
    const Function& function,
    const data_flow::DataFlowGraph& data_flow_graph,
    const MemsetNode& node
)
    : codegen::LibraryNodeDispatcher(language_extension, function, data_flow_graph, node) {}

void MemsetNodeDispatcher::dispatch_code_with_edges(
    codegen::CodegenOutput& out,
    std::vector<codegen::DispatchInput>& inputs,
    std::vector<codegen::DispatchOutput>& outputs
) {
    auto& node = static_cast<const MemsetNode&>(node_);

    out.stream << language_extension_.external_prefix() << "memset(" << inputs.at(0).expr << ", "
               << language_extension_.expression(node.value()) << ", " << language_extension_.expression(node.num())
               << ")"
               << ";";
    out.stream << std::endl;
}

MemsetNode& add_memset_node(
    builder::StructuredSDFGBuilder& builder,
    Block& block,
    const std::string& dst_ptr,
    const symbolic::Expression& value,
    const symbolic::Expression& num,
    const types::IType& ptr_type,
    DebugInfo debug_info
) {
    auto& dst_ptr_access = builder.add_access(block, dst_ptr);
    auto& libnode = builder.add_library_node<stdlib::MemsetNode>(block, debug_info, value, num);
    builder.add_computational_memlet(block, dst_ptr_access, libnode, "_ptr", {}, ptr_type);

    return static_cast<MemsetNode&>(libnode);
}

std::tuple<Block&, MemsetNode&> add_memset_block(
    builder::StructuredSDFGBuilder& builder,
    Sequence& parent,
    const std::string& dst_ptr,
    const symbolic::Expression& value,
    const symbolic::Expression& num,
    const types::IType& ptr_type,
    DebugInfo debug_info
) {
    auto& block = builder.add_block(parent);

    auto& libnode = add_memset_node(builder, block, dst_ptr, value, num, ptr_type, debug_info);

    return {block, libnode};
}

} // namespace stdlib
} // namespace sdfg
