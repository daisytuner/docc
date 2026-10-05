#include "sdfg/targets/gpu/math/tensor/tensor_operands.h"

#include "sdfg/data_flow/library_nodes/math/tensor/embedding_node.h"
#include "sdfg/exceptions.h"

namespace sdfg::gpu::tensor {

symbolic::Expression TensorOperand::num_bytes() const {
    return symbolic::mul(num_elements, symbolic::integer(types::bit_width(element_type) / 8));
}

static types::PrimitiveType connector_type(
    const data_flow::LibraryNode& node, const data_flow::DataFlowGraph& graph, const std::string& connector
) {
    auto* edge = graph.in_edge_for_connector(node, connector);
    if (edge == nullptr) {
        throw InvalidSDFGException(node.code().value() + ": missing input connector " + connector);
    }
    return edge->base_type().primitive_type();
}

static symbolic::Expression product(const std::vector<symbolic::Expression>& dims) {
    symbolic::Expression result = symbolic::one();
    for (auto& dim : dims) {
        result = symbolic::mul(result, dim);
    }
    return result;
}

static std::vector<TensorOperand>
embedding_operands(const math::tensor::EmbeddingNode& node, const data_flow::DataFlowGraph& graph) {
    auto weight_type = connector_type(node, graph, "W");
    auto num_indices = product(node.index_shape());
    return {
        {"Y", weight_type, symbolic::mul(num_indices, node.weight_shape().at(1)), false, true},
        {"W", weight_type, product(node.weight_shape()), true, false},
        {"I", connector_type(node, graph, "I"), num_indices, true, false},
    };
}

std::optional<std::vector<TensorOperand>>
tensor_operands(const data_flow::LibraryNode& node, const data_flow::DataFlowGraph& graph) {
    if (auto* embedding = dynamic_cast<const math::tensor::EmbeddingNode*>(&node)) {
        return embedding_operands(*embedding, graph);
    }
    return std::nullopt;
}

const TensorOperand& find_operand(const std::vector<TensorOperand>& operands, const std::string& connector) {
    for (auto& operand : operands) {
        if (operand.connector == connector) {
            return operand;
        }
    }
    throw InvalidSDFGException("GPU tensor operand " + connector + " not found");
}

} // namespace sdfg::gpu::tensor
