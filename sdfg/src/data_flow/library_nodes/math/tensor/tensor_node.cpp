#include "sdfg/data_flow/library_nodes/math/tensor/tensor_node.h"

#include "sdfg/types/tensor.h"

namespace sdfg {
namespace math {
namespace tensor {

TensorNode::TensorNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const data_flow::LibraryNodeCode& code,
    const std::vector<std::string>& outputs,
    const std::vector<std::string>& inputs,
    data_flow::ImplementationType impl_type
)
    : MathNode(element_id, debug_info, vertex, parent, code, outputs, inputs, impl_type) {}

void TensorNode::validate(const Function& function) const {
    MathNode::validate(function);

    auto& graph = this->get_parent();

    // Check that all input memlets are tensor of scalar
    for (auto& iedge : graph.in_edges(*this)) {
        if (iedge.base_type().type_id() != types::TypeID::Tensor) {
            throw InvalidSDFGException(
                "TensorNode: Input memlet must be of tensor type. Found type: " + iedge.base_type().print()
            );
        }
    }

    // Check that all output memlets are tensor of scalar
    for (auto& oedge : graph.out_edges(*this)) {
        if (oedge.base_type().type_id() != types::TypeID::Tensor) {
            throw InvalidSDFGException(
                "TensorNode: Output memlet must be of tensor type. Found type: " + oedge.base_type().print()
            );
        }
    }

    // Validate that all memlets have the same primitive type
    types::PrimitiveType prim_type = primitive_type(graph);

    // Check if this operation supports integer types
    if (!supports_integer_types() && types::is_integer(prim_type)) {
        throw InvalidSDFGException(
            "TensorNode: This operation does not support integer types. Found type: " +
            std::string(types::primitive_type_to_string(prim_type))
        );
    }
}

types::PrimitiveType TensorNode::primitive_type(const data_flow::DataFlowGraph& graph) const {
    types::PrimitiveType result_type = types::PrimitiveType::Void;
    bool first = true;

    // Check all input edges
    for (auto& iedge : graph.in_edges(*this)) {
        types::PrimitiveType edge_type;
        auto& tensor_type = static_cast<const types::Tensor&>(iedge.base_type());
        edge_type = tensor_type.primitive_type();

        if (first) {
            result_type = edge_type;
            first = false;
        } else if (result_type != edge_type) {
            throw InvalidSDFGException(
                "TensorNode: All input memlets must have the same primitive type. Found " +
                std::string(types::primitive_type_to_string(result_type)) + " and " +
                std::string(types::primitive_type_to_string(edge_type))
            );
        }
    }

    // Check all output edges
    for (auto& oedge : graph.out_edges(*this)) {
        types::PrimitiveType edge_type;
        auto& tensor_type = static_cast<const types::Tensor&>(oedge.base_type());
        edge_type = tensor_type.primitive_type();

        if (first) {
            result_type = edge_type;
            first = false;
        } else if (result_type != edge_type) {
            throw InvalidSDFGException(
                "TensorNode: All output memlets must have the same primitive type. Found " +
                std::string(types::primitive_type_to_string(result_type)) + " and " +
                std::string(types::primitive_type_to_string(edge_type))
            );
        }
    }

    if (first) {
        throw InvalidSDFGException("TensorNode: No edges found to determine primitive type");
    }

    return result_type;
}

data_flow::TaskletCode TensorNode::get_integer_minmax_tasklet(types::PrimitiveType prim_type, bool is_max) {
    bool is_signed = types::is_signed(prim_type);
    if (is_max) {
        return is_signed ? data_flow::TaskletCode::int_smax : data_flow::TaskletCode::int_umax;
    } else {
        return is_signed ? data_flow::TaskletCode::int_smin : data_flow::TaskletCode::int_umin;
    }
}

} // namespace tensor
} // namespace math
} // namespace sdfg
