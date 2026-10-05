#pragma once

#include <optional>
#include <string>
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
    /// Contents are consumed and must be on the device before the node runs.
    bool read;
    /// Contents are produced and must be copied back to the host afterwards.
    bool written;

    symbolic::Expression num_bytes() const;
};

/**
 * @brief Device-memory footprint of a tensor library node with a GPU dispatcher.
 *
 * Shared by the GPU tensor dispatchers (transfers inside the dispatcher) and the
 * transfer extraction (explicit offloading blocks). Adding a layer means adding
 * its operands here plus a dispatcher.
 *
 * @return std::nullopt for library nodes without a GPU tensor dispatcher
 */
std::optional<std::vector<TensorOperand>>
tensor_operands(const data_flow::LibraryNode& node, const data_flow::DataFlowGraph& graph);

const TensorOperand& find_operand(const std::vector<TensorOperand>& operands, const std::string& connector);

} // namespace sdfg::gpu::tensor
