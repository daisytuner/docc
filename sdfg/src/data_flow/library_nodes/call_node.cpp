#include "sdfg/data_flow/library_nodes/call_node.h"

namespace sdfg {
namespace data_flow {

CallNode::CallNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const std::string& callee_name,
    const std::vector<std::string>& outputs,
    const std::vector<std::string>& inputs,
    std::vector<PointerAccessType> ptr_access_meta
)
    : LibraryNode(
          element_id,
          debug_info,
          vertex,
          parent,
          LibraryNodeType_Call,
          outputs,
          inputs,
          true,
          data_flow::ImplementationType_NONE
      ),
      callee_name_(callee_name), ptr_access_meta_(std::move(ptr_access_meta)) {}

const std::string& CallNode::callee_name() const { return this->callee_name_; }

bool CallNode::is_void(const Function& sdfg) const { return outputs_.empty() || outputs_.at(0) != "_ret"; }

bool CallNode::is_indirect_call(const Function& sdfg) const {
    auto& type = sdfg.type(this->callee_name_);
    return dynamic_cast<const types::Pointer*>(&type) != nullptr;
}

void CallNode::validate(const Function& function) const {
    LibraryNode::validate(function);

    if (!function.exists(this->callee_name_)) {
        throw InvalidSDFGException("CallNode: Function '" + this->callee_name_ + "' does not exist.");
    }
    auto& type = function.type(this->callee_name_);
    if (!dynamic_cast<const types::Function*>(&type) && !dynamic_cast<const types::Pointer*>(&type)) {
        throw InvalidSDFGException("CallNode: '" + this->callee_name_ + "' is not a function or pointer.");
    }

    if (auto func_type = dynamic_cast<const types::Function*>(&type)) {
        if (!function.is_external(this->callee_name_)) {
            throw InvalidSDFGException("CallNode: Function '" + this->callee_name_ + "' must be declared.");
        }
        if (!func_type->is_var_arg() && inputs_.size() != func_type->num_params()) {
            throw InvalidSDFGException(
                "CallNode: Number of inputs does not match number of function parameters. Expected " +
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

symbolic::SymbolSet CallNode::symbols() const { return {symbolic::symbol(this->callee_name_)}; }

std::unique_ptr<data_flow::DataFlowNode> CallNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    std::vector<PointerAccessType> ptr_access_meta_clone;
    ptr_access_meta_clone.reserve(ptr_access_meta_.size());
    for (auto& ptr_access_meta : ptr_access_meta_) {
        if (ptr_access_meta) {
            ptr_access_meta_clone.push_back(ptr_access_meta->clone());
        } else {
            ptr_access_meta_clone.push_back(nullptr);
        }
    }
    return std::make_unique<CallNode>(
        element_id, debug_info_, vertex, parent, callee_name_, outputs_, inputs_, std::move(ptr_access_meta_clone)
    );
}

PointerAccessType CallNode::pointer_access_type(int input_idx) const {
    if (ptr_access_meta_.size() > input_idx) {
        return ptr_access_meta_.at(0)->ref();
    } else {
        return LibraryNode::pointer_access_type(input_idx);
    }
}

const std::vector<PointerAccessType>& CallNode::pointer_access_meta() const { return ptr_access_meta_; }

std::string CallNode::toStr() const { return LibraryNode::toStr() + "('" + callee_name_ + "')"; }

void CallNode::replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) {
    for (auto& meta : ptr_access_meta_) {
        if (meta) {
            meta->replace(old_expression, new_expression);
        }
    }
}

void CallNode::replace(const symbolic::ExpressionMapping& replacements) {
    for (auto& meta : ptr_access_meta_) {
        if (meta) {
            meta->replace(replacements);
        }
    }
}

nlohmann::json CallNodeSerializer::serialize(const data_flow::LibraryNode& library_node) {
    const CallNode& node = static_cast<const CallNode&>(library_node);

    nlohmann::json j;
    j["code"] = node.code().value();
    j["callee_name"] = node.callee_name();
    j["outputs"] = node.outputs();
    j["inputs"] = node.inputs();
    j["ptr_access_meta"] = PointerAccessMetaSerializer::serialize(node.pointer_access_meta());

    return j;
}

data_flow::LibraryNode& CallNodeSerializer::deserialize(
    const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
) {
    assert(j.contains("code"));
    assert(j.contains("callee_name"));
    assert(j.contains("outputs"));
    assert(j.contains("inputs"));
    assert(j.contains("debug_info"));

    auto code = j["code"].get<std::string>();
    if (code != LibraryNodeType_Call.value()) {
        throw InvalidSDFGException("Invalid library node code");
    }

    sdfg::serializer::JSONSerializer serializer;
    DebugInfo debug_info = serializer.json_to_debug_info(j["debug_info"]);

    std::string callee_name = j["callee_name"].get<std::string>();
    auto outputs = j["outputs"].get<std::vector<std::string>>();
    auto inputs = j["inputs"].get<std::vector<std::string>>();
    auto ptr_access_meta = PointerAccessMetaSerializer::deserialize_list(j.find("ptr_access_meta"), j);

    return builder
        .add_library_node<CallNode>(parent, debug_info, callee_name, outputs, inputs, std::move(ptr_access_meta));
}

CallNodeDispatcher::CallNodeDispatcher(
    codegen::LanguageExtension& language_extension,
    const Function& function,
    const data_flow::DataFlowGraph& data_flow_graph,
    const CallNode& node
)
    : codegen::LibraryNodeDispatcher(language_extension, function, data_flow_graph, node) {}

void CallNodeDispatcher::dispatch_code_with_edges(
    codegen::CodegenOutput& out,
    std::vector<codegen::DispatchInput>& inputs,
    std::vector<codegen::DispatchOutput>& outputs
) {
    auto& node = static_cast<const CallNode&>(node_);

    codegen::DispatchOutput* output = nullptr;
    if (!node.is_void(function_)) {
        output = &outputs.at(0);
        pre_allocate_output(out, *output, node.output(0));
        out.stream << *output->local_name << " = ";
    }
    if (node.is_indirect_call(function_)) {
        // Cast callee to function pointer type
        std::string func_ptr_type;

        // Return type
        if (output) {
            func_ptr_type = language_extension_.declaration("", *output->out_type) + " (*)";
        } else {
            func_ptr_type = "void (*)";
        }

        // Parameters
        func_ptr_type += "(";
        for (size_t i = 0; i < inputs.size(); i++) {
            auto& input = inputs.at(i);

            auto in_type = input.edge.result_type(function_);
            func_ptr_type += language_extension_.declaration("", *in_type);
            if (i < node.inputs().size() - 1) {
                func_ptr_type += ", ";
            }
        }
        func_ptr_type += ")";

        if (this->language_extension_.language() == "C") {
            out.stream << "((" << func_ptr_type << ") " << node.callee_name() << ")" << "(";
        } else if (this->language_extension_.language() == "C++") {
            out.stream << "reinterpret_cast<" << func_ptr_type << ">(" << node.callee_name() << ")" << "(";
        }
    } else {
        out.stream << this->language_extension_.external_prefix() << node.callee_name() << "(";
    }
    for (size_t i = 0; i < inputs.size(); ++i) {
        out.stream << inputs.at(i).expr;
        if (i < node.inputs().size() - 1) {
            out.stream << ", ";
        }
    }
    out.stream << ")" << ";";
    out.stream << std::endl;
}

} // namespace data_flow
} // namespace sdfg
