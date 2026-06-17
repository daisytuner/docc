#include "sdfg/data_flow/library_nodes/invoke_node.h"

namespace sdfg {
namespace data_flow {

InvokeNode::InvokeNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const std::string& callee_name,
    const std::vector<std::string>& outputs,
    const std::vector<std::string>& inputs
)
    : LibraryNode(
          element_id,
          debug_info,
          vertex,
          parent,
          LibraryNodeType_Invoke,
          outputs,
          inputs,
          true,
          data_flow::ImplementationType_NONE
      ),
      callee_name_(callee_name) {
    this->outputs_.push_back("_unwind"); // Add unwind output
}

const std::string& InvokeNode::callee_name() const { return this->callee_name_; }

bool InvokeNode::is_void(const Function& sdfg) const { return outputs_.size() == 1 || outputs_.at(0) != "_ret"; }

bool InvokeNode::is_indirect_call(const Function& sdfg) const {
    auto& type = sdfg.type(this->callee_name_);
    return dynamic_cast<const types::Pointer*>(&type) != nullptr;
}

void InvokeNode::validate(const Function& function) const {
    LibraryNode::validate(function);

    if (!function.exists(this->callee_name_)) {
        throw InvalidSDFGException("InvokeNode: Function '" + this->callee_name_ + "' does not exist.");
    }
    auto& type = function.type(this->callee_name_);
    if (!dynamic_cast<const types::Function*>(&type) && !dynamic_cast<const types::Pointer*>(&type)) {
        throw InvalidSDFGException("InvokeNode: '" + this->callee_name_ + "' is not a function or pointer.");
    }

    if (auto func_type = dynamic_cast<const types::Function*>(&type)) {
        if (!function.is_external(this->callee_name_)) {
            throw InvalidSDFGException("InvokeNode: Function '" + this->callee_name_ + "' must be declared.");
        }
        if (!func_type->is_var_arg() && inputs_.size() != func_type->num_params()) {
            throw InvalidSDFGException(
                "InvokeNode: Number of inputs does not match number of function parameters. Expected " +
                std::to_string(func_type->num_params()) + ", got " + std::to_string(inputs_.size())
            );
        }
        if (!this->is_void(function) && outputs_.size() < 1) {
            throw InvalidSDFGException(
                "CallNode: Non-void function must have at least one output to store the return value."
            );
        }
    }
}

symbolic::SymbolSet InvokeNode::symbols() const { return {symbolic::symbol(this->callee_name_)}; }

std::unique_ptr<data_flow::DataFlowNode> InvokeNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::make_unique<InvokeNode>(element_id, debug_info_, vertex, parent, callee_name_, outputs_, inputs_);
}

void InvokeNode::replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) {}

void InvokeNode::replace(const symbolic::ExpressionMapping& replacements) {}

nlohmann::json InvokeNodeSerializer::serialize(const data_flow::LibraryNode& library_node) {
    const InvokeNode& node = static_cast<const InvokeNode&>(library_node);

    nlohmann::json j;
    j["code"] = node.code().value();
    j["callee_name"] = node.callee_name();
    j["outputs"] = node.outputs();
    j["inputs"] = node.inputs();

    return j;
}

data_flow::LibraryNode& InvokeNodeSerializer::deserialize(
    const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
) {
    assert(j.contains("code"));
    assert(j.contains("callee_name"));
    assert(j.contains("outputs"));
    assert(j.contains("inputs"));
    assert(j.contains("debug_info"));

    auto code = j["code"].get<std::string>();
    if (code != LibraryNodeType_Invoke.value()) {
        throw InvalidSDFGException("Invalid library node code");
    }

    sdfg::serializer::JSONSerializer serializer;
    DebugInfo debug_info = serializer.json_to_debug_info(j["debug_info"]);

    std::string callee_name = j["callee_name"].get<std::string>();
    auto outputs = j["outputs"].get<std::vector<std::string>>();
    auto inputs = j["inputs"].get<std::vector<std::string>>();

    return builder.add_library_node<InvokeNode>(parent, debug_info, callee_name, outputs, inputs);
}

InvokeNodeDispatcher::InvokeNodeDispatcher(
    codegen::LanguageExtension& language_extension,
    const Function& function,
    const data_flow::DataFlowGraph& data_flow_graph,
    const InvokeNode& node
)
    : codegen::LibraryNodeDispatcher(language_extension, function, data_flow_graph, node) {}

void InvokeNodeDispatcher::dispatch_code(
    codegen::PrettyPrinter& stream,
    codegen::PrettyPrinter& globals_stream,
    codegen::CodeSnippetFactory& library_snippet_factory
) {
    auto& node = static_cast<const InvokeNode&>(node_);

    stream << "try {" << std::endl;
    stream.setIndent(stream.indent() + 4);

    if (!node.is_void(function_)) {
        stream << node.outputs().at(0) << " = ";
    }
    if (node.is_indirect_call(function_)) {
        auto& graph = node.get_parent();

        // Collect return memlet
        const data_flow::Memlet* ret_memlet = nullptr;
        for (auto& oedge : graph.out_edges(node)) {
            if (oedge.src_conn() == "_ret") {
                ret_memlet = &oedge;
                break;
            }
        }

        // Collect input memlets
        std::unordered_map<std::string, const data_flow::Memlet*> input_memlets;
        for (auto& iedge : graph.in_edges(node)) {
            input_memlets[iedge.dst_conn()] = &iedge;
        }

        // Cast callee to function pointer type
        std::string func_ptr_type;

        // Return type
        if (ret_memlet) {
            auto ret_type = ret_memlet->result_type(function_);
            func_ptr_type = language_extension_.declaration("", *ret_type) + " (*)";
        } else {
            func_ptr_type = "void (*)";
        }

        // Parameters
        func_ptr_type += "(";
        for (size_t i = 0; i < node.inputs().size(); i++) {
            auto memlet_in = input_memlets.find(node.inputs().at(i));
            assert(memlet_in != input_memlets.end());
            auto in_type = memlet_in->second->result_type(function_);
            func_ptr_type += language_extension_.declaration("", *in_type);
            if (i < node.inputs().size() - 1) {
                func_ptr_type += ", ";
            }
        }
        func_ptr_type += ")";

        if (this->language_extension_.language() == "C") {
            stream << "((" << func_ptr_type << ") " << node.callee_name() << ")" << "(";
        } else if (this->language_extension_.language() == "C++") {
            stream << "reinterpret_cast<" << func_ptr_type << ">(" << node.callee_name() << ")" << "(";
        }
    } else {
        stream << this->language_extension_.external_prefix() << node.callee_name() << "(";
    }
    for (size_t i = 0; i < node.inputs().size(); ++i) {
        stream << node.inputs().at(i);
        if (i < node.inputs().size() - 1) {
            stream << ", ";
        }
    }
    stream << ")" << ";";
    stream << std::endl;

    stream << node_.outputs().at(node.outputs().size() - 1) << " = false;" << std::endl;
    stream.setIndent(stream.indent() - 4);
    stream << "} catch (...) {" << std::endl;
    stream.setIndent(stream.indent() + 4);
    stream << node_.outputs().at(node.outputs().size() - 1) << " = true;" << std::endl;
    stream.setIndent(stream.indent() - 4);
    stream << "}" << std::endl;
}

} // namespace data_flow
} // namespace sdfg
