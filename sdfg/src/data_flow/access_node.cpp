#include "sdfg/data_flow/access_node.h"

#include <unordered_set>
#include "sdfg/data_flow/data_flow_graph.h"
#include "sdfg/function.h"

namespace sdfg {
namespace data_flow {

AccessNode::AccessNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    DataFlowGraph& parent,
    const std::string& data
)
    : DataFlowNode(element_id, debug_info, vertex, parent), data_(data) {

      };

void AccessNode::validate(const Function& function) const {
    if (!function.exists(this->data_)) {
        throw InvalidSDFGException("Access node " + this->data_ + " uses non-existent variable");
    }

    auto& graph = this->get_parent();

    if (graph.out_degree(*this) > 1) {
        MemletType type = (*graph.out_edges(*this).begin()).type();
        for (auto& oedge : graph.out_edges(*this)) {
            if (oedge.type() != type) {
                throw InvalidSDFGException("Access node " + this->data() + " used with multiple memlet types");
            }
        }
    }

    if (graph.in_degree(*this) > 1) {
        MemletType type = (*graph.in_edges(*this).begin()).type();
        for (auto& iedge : graph.in_edges(*this)) {
            if (iedge.type() != type) {
                throw InvalidSDFGException("Access node " + this->data() + " used with multiple memlet types");
            }
        }
    }
}

const std::string& AccessNode::data() const { return this->data_; };

void AccessNode::data(const std::string data) { this->data_ = data; }

bool AccessNode::side_effect() const { return get_parent().in_degree(*this) > 0; }

std::unique_ptr<DataFlowNode> AccessNode::clone(size_t element_id, const graph::Vertex vertex, DataFlowGraph& parent)
    const {
    return std::unique_ptr<AccessNode>(new AccessNode(element_id, this->debug_info_, vertex, parent, this->data_));
};

void AccessNode::replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) {
    if (SymEngine::is_a<SymEngine::Symbol>(*old_expression) && SymEngine::is_a<SymEngine::Symbol>(*new_expression)) {
        auto old_symbol = SymEngine::rcp_static_cast<const SymEngine::Symbol>(old_expression);
        if (this->data_ == old_symbol->get_name()) {
            auto new_symbol = SymEngine::rcp_static_cast<const SymEngine::Symbol>(new_expression);
            this->data_ = new_symbol->get_name();
        }
    }
}

/**
 * Every read can be removed
 */
EdgeRemoveOption AccessNode::can_remove_out_edge(const data_flow::DataFlowGraph& graph, const Memlet* memlet) const {
    return EdgeRemoveOption::Trivially;
}

namespace {
bool is_special_constant(const std::string& data) {
    static const std::unordered_set<std::string> constants = {
        "true",
        "false",
        "INT8_MIN",
        "INT8_MAX",
        "UINT8_MAX",
        "INT16_MIN",
        "INT16_MAX",
        "UINT16_MAX",
        "INT32_MIN",
        "INT32_MAX",
        "UINT32_MAX",
        "INT64_MIN",
        "INT64_MAX",
        "UINT64_MAX",
    };
    return constants.find(data) != constants.end();
}
} // namespace

ConstantNode::ConstantNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    DataFlowGraph& parent,
    const std::string& data,
    const types::IType& type
)
    : AccessNode(element_id, debug_info, vertex, parent, data), type_(type.clone()) {};

void ConstantNode::validate(const Function& function) const {
    if (function.exists(this->data_)) {
        throw InvalidSDFGException("ConstantNode " + this->data_ + " uses variable");
    }

    auto& graph = this->get_parent();
    if (graph.in_degree(*this) > 0) {
        throw InvalidSDFGException("ConstantNode " + this->data_ + " has incoming edges");
    }

    switch (this->type_->type_id()) {
        case types::TypeID::Scalar: {
            auto& scalar_type = static_cast<const types::Scalar&>(*this->type_);
            switch (scalar_type.primitive_type()) {
                case types::PrimitiveType::Bool:
                case types::PrimitiveType::Int8:
                case types::PrimitiveType::Int16:
                case types::PrimitiveType::Int32:
                case types::PrimitiveType::Int64:
                case types::PrimitiveType::UInt8:
                case types::PrimitiveType::UInt16:
                case types::PrimitiveType::UInt32:
                case types::PrimitiveType::UInt64: {
                    auto data = this->data();
                    if (is_special_constant(data)) {
                        break;
                    }

                    try {
                        helpers::parse_number_signed(data);
                    } catch (const std::exception& e) {
                        throw InvalidSDFGException("ConstantNode " + data + " has non-integer scalar type");
                    }
                    break;
                }
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }
}

const types::IType& ConstantNode::type() const { return *this->type_; };

std::unique_ptr<DataFlowNode> ConstantNode::clone(size_t element_id, const graph::Vertex vertex, DataFlowGraph& parent)
    const {
    return std::unique_ptr<
        ConstantNode>(new ConstantNode(element_id, this->debug_info_, vertex, parent, this->data(), *this->type_));
};

} // namespace data_flow
} // namespace sdfg
