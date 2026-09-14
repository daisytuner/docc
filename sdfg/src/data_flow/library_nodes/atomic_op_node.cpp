#include "sdfg/data_flow/library_nodes/atomic_op_node.h"

#include "sdfg/types/array.h"
#include "sdfg/types/pointer.h"

namespace sdfg {
namespace data_flow {

AtomicScalarOpNode::AtomicScalarOpNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    DataFlowGraph& parent,
    types::PrimitiveType data_type,
    AtomicOpType atomic_op,
    const AtomicScalarOpImpl* impl
)
    : LibraryNode(
          element_id,
          debug_info,
          vertex,
          parent,
          LibraryNodeType_AtomicScalarOp,
          {"_prev"},
          {"_dst", "_src"},
          true, // atomic store to (global) memory is an observable side effect
          impl->implementation_type()
      ),
      data_type_(data_type), atomic_op_(atomic_op), impl_(impl) {
    verify_impl_exits();
};

void AtomicScalarOpNode::verify_impl_exits() const {
    if (!impl_->supports(data_type_, atomic_op_)) {
        throw InvalidSDFGException(
            "AtomicScalarOpNode: Unsupported implementation type " + std::string(implementation_type_.value())
        );
    }
}

const AtomicScalarOpImpl* AtomicScalarOpNode::get_implementation(const std::string& type) {
    const AtomicScalarOpImpl* impl;
    if (type == AtomicScalarOpCPUImpl::TYPE_NAME) {
        return AtomicScalarOpCPUImpl::instance();
    } else if (type == AtomicScalarOpCudaImpl::TYPE_NAME) {
        return AtomicScalarOpCudaImpl::instance();
    } else if (type == AtomicScalarOpRocmImpl::TYPE_NAME) {
        return AtomicScalarOpRocmImpl::instance();
    } else {
        throw std::runtime_error("Invalid implementation type");
    }
}

void AtomicScalarOpNode::switch_implementation(const AtomicScalarOpImpl& new_impl) {
    if (!new_impl.supports(data_type_, atomic_op_)) {
        throw InvalidSDFGException(
            "AtomicScalarOpNode: Implementation '" + std::string(new_impl.type_name()) + "' does not support " +
            std::string(atomic_op_type_to_string(atomic_op_)) + " on " +
            std::string(types::primitive_type_to_string(data_type_))
        );
    }
    impl_ = &new_impl;
    set_implementation_type(new_impl.implementation_type());
}

void AtomicScalarOpNode::validate(const Function& function) const {
    LibraryNode::validate(function);

    verify_impl_matches();
    verify_impl_exits();
}

symbolic::SymbolSet AtomicScalarOpNode::symbols() const { return {}; };

std::unique_ptr<DataFlowNode> AtomicScalarOpNode::
    clone(size_t element_id, const graph::Vertex vertex, DataFlowGraph& parent) const {
    return std::unique_ptr<AtomicScalarOpNode>(
        new AtomicScalarOpNode(element_id, this->debug_info_, vertex, parent, data_type_, atomic_op_, this->impl_)
    );
};

void AtomicScalarOpNode::replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) {
    // The addressing lives on the in memlets, not the node.
};

void AtomicScalarOpNode::replace(const symbolic::ExpressionMapping& replacements) {
    // The addressing lives on the in memlets, not the node.
}

void AtomicScalarOpNode::verify_impl_matches() const {
    if (impl_->type_name() != implementation_type_.value()) {
        throw InvalidSDFGException("AtomicScalarOpNode: Implementation type mismatch");
    }
}

//
// AtomicScalarOpCPUImpl
//

const AtomicScalarOpCPUImpl* AtomicScalarOpCPUImpl::instance() {
    static const AtomicScalarOpCPUImpl instance;
    return &instance;
}

bool AtomicScalarOpCPUImpl::supports(types::PrimitiveType data_type, AtomicOpType op) const {
    switch (data_type) {
        case types::PrimitiveType::Int8:
        case types::PrimitiveType::Int16:
        case types::PrimitiveType::Int32:
        case types::PrimitiveType::Int64:
        case types::PrimitiveType::Int128:
        case types::PrimitiveType::UInt8:
        case types::PrimitiveType::UInt16:
        case types::PrimitiveType::UInt32:
        case types::PrimitiveType::UInt64:
        case types::PrimitiveType::Float:
        case types::PrimitiveType::Double:
        case types::PrimitiveType::Half:
        case types::PrimitiveType::BFloat:
            return true;
        default:
            return false;
    }
}

//
// AtomicScalarOpCudaImpl
//

const AtomicScalarOpCudaImpl* AtomicScalarOpCudaImpl::instance() {
    static const AtomicScalarOpCudaImpl instance;
    return &instance;
}

bool AtomicScalarOpCudaImpl::supports(types::PrimitiveType data_type, AtomicOpType op) const {
    switch (op) {
        case AtomicOpType::Add:
            switch (data_type) {
                case types::PrimitiveType::Int32:
                case types::PrimitiveType::UInt32:
                case types::PrimitiveType::Int64:
                case types::PrimitiveType::UInt64:
                case types::PrimitiveType::Float:
                case types::PrimitiveType::Double:
                case types::PrimitiveType::Half:
                case types::PrimitiveType::BFloat:
                    return true;
                default:
                    return false;
            }
        case AtomicOpType::Subtract:
            switch (data_type) {
                case types::PrimitiveType::UInt32:
                case types::PrimitiveType::Int32:
                    return true;
                default:
                    return false;
            }
        case AtomicOpType::Min:
        case AtomicOpType::Max:
            switch (data_type) {
                case types::PrimitiveType::Int32:
                case types::PrimitiveType::UInt32:
                case types::PrimitiveType::Int64:
                case types::PrimitiveType::UInt64:
                    return true;
                default:
                    return false;
            }
        default:
            return false;
    }
}

//
// AtomicScalarOpRocmImpl
//

const AtomicScalarOpRocmImpl* AtomicScalarOpRocmImpl::instance() {
    static const AtomicScalarOpRocmImpl instance;
    return &instance;
}

bool AtomicScalarOpRocmImpl::supports(types::PrimitiveType data_type, AtomicOpType op) const {
    switch (op) {
        case AtomicOpType::Add:
            switch (data_type) {
                case types::PrimitiveType::Int32:
                case types::PrimitiveType::UInt32:
                case types::PrimitiveType::Int64:
                case types::PrimitiveType::UInt64:
                case types::PrimitiveType::Float:
                case types::PrimitiveType::Double:
                case types::PrimitiveType::Half:
                case types::PrimitiveType::BFloat:
                    return true;
                default:
                    return false;
            }
        case AtomicOpType::Subtract:
            switch (data_type) {
                case types::PrimitiveType::Int32:
                case types::PrimitiveType::UInt32:
                    return true;
                default:
                    return false;
            }
        case AtomicOpType::Min:
        case AtomicOpType::Max:
            switch (data_type) {
                case types::PrimitiveType::Int32:
                case types::PrimitiveType::UInt32:
                case types::PrimitiveType::Int64:
                case types::PrimitiveType::UInt64:
                case types::PrimitiveType::Float:
                case types::PrimitiveType::Double:
                    return true;
                default:
                    return false;
            }
        default:
            return false;
    }
}

nlohmann::json AtomicScalarOpNodeSerializer::serialize(const sdfg::data_flow::LibraryNode& library_node) {
    if (library_node.code() != data_flow::LibraryNodeType_AtomicScalarOp) {
        throw std::runtime_error("Invalid library node code");
    }
    auto& node = static_cast<const AtomicScalarOpNode&>(library_node);
    nlohmann::json j;
    j["code"] = std::string(library_node.code().value());
    j["primitive_type"] = types::primitive_type_to_string(node.data_type());
    j["atomic_op"] = atomic_op_type_to_string(node.atomic_op());

    return j;
}

data_flow::LibraryNode& AtomicScalarOpNodeSerializer::deserialize(
    const nlohmann::json& j, sdfg::builder::StructuredSDFGBuilder& builder, sdfg::structured_control_flow::Block& parent
) {
    auto code = j["code"].get<std::string>();
    if (code != data_flow::LibraryNodeType_AtomicScalarOp.value()) {
        throw std::runtime_error("Invalid library node code");
    }

    auto data_type = types::primitive_type_from_string(j.at("primitive_type").get<std::string>());
    auto atomic_op = atomic_op_type_from_string(j.at("atomic_op").get<std::string>());
    auto type = j.at("implementation_type").get<std::string>();
    auto* impl = AtomicScalarOpNode::get_implementation(type);
    return builder.add_library_node<data_flow::AtomicScalarOpNode>(parent, DebugInfo(), data_type, atomic_op, impl);
};

AtomicScalarOpNodeDispatcher::AtomicScalarOpNodeDispatcher(
    codegen::LanguageExtension& language_extension,
    const Function& function,
    const data_flow::DataFlowGraph& data_flow_graph,
    const AtomicScalarOpNode& node
)
    : codegen::LibraryNodeDispatcher(language_extension, function, data_flow_graph, node) {}

void AtomicScalarOpCPUNodeDispatcher::dispatch_code_with_edges(
    codegen::CodegenOutput& out,
    std::vector<codegen::DispatchInput>& inputs,
    std::vector<codegen::DispatchOutput>& outputs
) {
    auto& node = static_cast<const AtomicScalarOpNode&>(node_);
    node.verify_impl_matches();
    node.verify_impl_exits();

    std::string ptr_name = "ptr";
    std::string value_name = "value";
    out.stream << language_extension_.declaration(ptr_name, types::Pointer(types::Scalar(node.data_type()))) << " = "
               << inputs.at(0).expr << ";" << std::endl;
    out.stream << language_extension_.declaration(value_name, types::Scalar(node.data_type())) << " = "
               << inputs.at(1).expr << ";" << std::endl;

    const std::string* prev_name = nullptr;
    if (!outputs.empty()) {
        pre_allocate_output(out, outputs.at(0), node.output(0));
        prev_name = outputs.at(0).local_name;
    }

    auto omp_simple_out = [&](const std::string& op) {
        out.stream << "#pragma omp atomic" << std::endl;
        if (prev_name) {
            out.stream << "{" << std::endl;
            out.stream.changeIndent(+4);
            out.stream << outputs.at(0).local_name << "*(" << ptr_name << ")" << std::endl;
        }
        out.stream << "*(" << ptr_name << ") " << op << value_name << ";" << std::endl;
        if (prev_name) {
            out.stream.changeIndent(-4);
            out.stream << "}" << std::endl;
        }
    };

    auto omp_compr_out = [&](const std::string& op) {
        if (!prev_name) {
            prev_name = &node.output(0);
            out.stream << language_extension_.declaration(*prev_name, types::Scalar(node.data_type())) << ";"
                       << std::endl;
        }
        out.stream << "#pragma omp atomic" << std::endl;
        out.stream << "{" << std::endl;
        out.stream.changeIndent(+4);
        out.stream << *prev_name << " = *(" << ptr_name << ")" << std::endl;
        out.stream << "*(" << ptr_name << ") " << "= *(" << *prev_name << " " << op << " " << value_name << ")? "
                   << *prev_name << " : " << value_name << ";" << std::endl;
        out.stream.changeIndent(-4);
        out.stream << "}" << std::endl;
    };

    switch (node.atomic_op()) {
        case AtomicOpType::Add:
            omp_simple_out("+=");
            break;
        case AtomicOpType::Subtract:
            omp_simple_out("-=");
            break;
        case AtomicOpType::Min:
            omp_compr_out("<");
            break;
        case AtomicOpType::Max:
            omp_compr_out(">");
            break;
        default:
            throw std::invalid_argument("Invalid AtomicOpType type on #" + std::to_string(node.element_id()));
    }
}

void AtomicScalarOpGPUNodeDispatcher::dispatch_code_with_edges(
    codegen::CodegenOutput& out,
    std::vector<codegen::DispatchInput>& inputs,
    std::vector<codegen::DispatchOutput>& outputs
) {
    const auto& node = static_cast<const AtomicScalarOpNode&>(node_);
    node.verify_impl_matches();
    node.verify_impl_exits();
    // Input order follows the connector order {"_dst", "_src"}.
    std::string dst = inputs.at(0).expr;
    std::string src = inputs.at(1).expr;

    if (node.data_type() == types::PrimitiveType::Half) {
        // Route through float: __half(_Float16) is an ambiguous functional cast
        // (_Float16 converts to both float and double), so name the float ctor.
        src = "__half(static_cast<float>(" + src + "))";
        dst = "reinterpret_cast<__half*>(" + dst + ")";
    }

    switch (node.atomic_op()) {
        case AtomicOpType::Add:
            out.stream << "atomicAdd(" << dst << ", " << src << ");" << std::endl;
            break;
        case AtomicOpType::Subtract:
            out.stream << "atomicSub(" << dst << ", " << src << ");" << std::endl;
            break;
        case AtomicOpType::Min:
            out.stream << "atomicMin(" << dst << ", " << src << ");" << std::endl;
            break;
        case AtomicOpType::Max:
            out.stream << "atomicMax(" << dst << ", " << src << ");" << std::endl;
            break;
        default:
            throw std::invalid_argument("Invalid AtomicOpType type on #" + std::to_string(node.element_id()));
    }
}

} // namespace data_flow
} // namespace sdfg
