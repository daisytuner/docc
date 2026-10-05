#include "sdfg/targets/gpu/math/tensor/tensor_operands.h"

#include <algorithm>

#include "sdfg/data_flow/library_nodes/math/tensor/embedding_node.h"
#include "sdfg/data_flow/pointer_metadata.h"
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

static TensorOperand make_operand(
    const data_flow::LibraryNode& node,
    const std::string& connector,
    types::PrimitiveType element_type,
    const symbolic::Expression& num_elements
) {
    auto& inputs = node.inputs();
    auto position = std::find(inputs.begin(), inputs.end(), connector);
    if (position == inputs.end()) {
        throw InvalidSDFGException(node.code().value() + ": unknown input connector " + connector);
    }

    TensorOperand operand{connector, element_type, num_elements, true, true};
    auto access = node.pointer_access_type(static_cast<int>(std::distance(inputs.begin(), position)));
    if (!access) {
        return operand;
    }
    if (access->invalidated_after()) {
        throw InvalidSDFGException(node.code().value() + ": operand " + connector + " is invalidated by the node");
    }
    // Only a full overwrite makes the incoming host contents irrelevant.
    bool full_write = dynamic_cast<const data_flow::PointerFullWriteOnly*>(access.get()) != nullptr;
    operand.copy_to_device = access->may_contain_reads() || (access->may_contain_writes() && !full_write);
    operand.copy_to_host = access->may_contain_writes();
    return operand;
}

static std::vector<TensorOperand>
embedding_operands(const math::tensor::EmbeddingNode& node, const data_flow::DataFlowGraph& graph) {
    auto weight_type = connector_type(node, graph, "W");
    auto num_indices = product(node.index_shape());
    return {
        make_operand(node, "Y", weight_type, symbolic::mul(num_indices, node.weight_shape().at(1))),
        make_operand(node, "W", weight_type, product(node.weight_shape())),
        make_operand(node, "I", connector_type(node, graph, "I"), num_indices),
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

std::vector<TensorBuffer> tensor_buffers(
    const std::vector<TensorOperand>& operands, const std::unordered_map<std::string, std::string>& host_of_connector
) {
    std::vector<TensorBuffer> buffers;
    for (auto& operand : operands) {
        auto& host = host_of_connector.at(operand.connector);
        auto buffer = std::find_if(buffers.begin(), buffers.end(), [&](const TensorBuffer& candidate) {
            return candidate.host == host;
        });
        if (buffer == buffers.end()) {
            buffers.push_back({host, {}, operand.num_bytes(), false, false});
            buffer = std::prev(buffers.end());
        } else {
            buffer->num_bytes = symbolic::max(buffer->num_bytes, operand.num_bytes());
        }
        buffer->connectors.push_back(operand.connector);
        buffer->copy_to_device = buffer->copy_to_device || operand.copy_to_device;
        buffer->copy_to_host = buffer->copy_to_host || operand.copy_to_host;
    }
    return buffers;
}

} // namespace sdfg::gpu::tensor
