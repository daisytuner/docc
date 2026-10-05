#pragma once

#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "sdfg/data_flow/data_flow_graph.h"
#include "sdfg/data_flow/library_node.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/type.h"

namespace sdfg::gpu::tensor {

/// One pointer operand (input connector) of a GPU-dispatched tensor library node.
struct TensorOperand {
    std::string connector;
    types::PrimitiveType element_type;
    symbolic::Expression num_elements;
    /// The node reads the data or only partially overwrites it, so the device copy starts from the host contents.
    bool copy_to_device;
    /// The node may write the data, so the host receives the device contents afterwards.
    bool copy_to_host;

    symbolic::Expression num_bytes() const;
};

/// One device buffer per host pointer, shared by all operands aliasing it.
struct TensorBuffer {
    std::string host;
    std::vector<std::string> connectors;
    symbolic::Expression num_bytes;
    bool copy_to_device;
    bool copy_to_host;
};

/**
 * @brief Device-memory footprint of a tensor library node with a GPU dispatcher.
 *
 * Shared by the GPU tensor dispatchers (transfers inside the dispatcher) and the
 * transfer extraction (explicit offloading blocks). Transfer directions follow the
 * node's own `pointer_access_type`. Adding a layer means adding its operands here
 * plus a dispatcher.
 *
 * @return std::nullopt for library nodes without a GPU tensor dispatcher
 */
std::optional<std::vector<TensorOperand>>
tensor_operands(const data_flow::LibraryNode& node, const data_flow::DataFlowGraph& graph);

const TensorOperand& find_operand(const std::vector<TensorOperand>& operands, const std::string& connector);

/// Group operands by the host pointer they are bound to (`host_of_connector`), in operand order.
std::vector<TensorBuffer> tensor_buffers(
    const std::vector<TensorOperand>& operands, const std::unordered_map<std::string, std::string>& host_of_connector
);

} // namespace sdfg::gpu::tensor
