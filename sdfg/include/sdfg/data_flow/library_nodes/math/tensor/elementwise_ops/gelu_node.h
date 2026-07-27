#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include <nlohmann/json_fwd.hpp>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/data_flow_graph.h"
#include "sdfg/data_flow/data_flow_node.h"
#include "sdfg/data_flow/library_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/elementwise_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/tensor_node.h"
#include "sdfg/element.h"
#include "sdfg/graph/graph.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/type.h"

namespace sdfg {
namespace math {
namespace tensor {

inline data_flow::LibraryNodeCode LibraryNodeType_GELU("ml::GELU");

/** @brief Applies the Gaussion Error Linear Units function.
 *
 * If tanh_approx is set to true, the tanh approximation is used.
 * Expands to:
 * Precise: GELU(x) = 0.5 * x * (1 + erf(x * sqrt(1 / 2)))
 * Tanh approximation: GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))
 */
class GELUNode : public ElementWiseDataflowTensorNode {
private:
    bool tanh_approx_;

public:
    GELUNode(
        size_t element_id,
        const DebugInfo& debug_info,
        const graph::Vertex vertex,
        data_flow::DataFlowGraph& parent,
        const std::vector<symbolic::Expression>& shape,
        bool tanh_approx = false,
        QuantizationType quantization = QUANTIZATION_MATCH_INPUTS,
        const data_flow::ImplementationType& impl_type = data_flow::ImplementationType_NONE
    );

    bool tanh_approx() const;

    ElementOutput expand_operation_dataflow_precise(
        builder::StructuredSDFGBuilder& builder,
        structured_control_flow::Block& block,
        std::vector<ElementInput>& needed_inputs,
        types::PrimitiveType expected_type
    );

    ElementOutput expand_operation_dataflow_tanh_approx(
        builder::StructuredSDFGBuilder& builder,
        structured_control_flow::Block& block,
        std::vector<ElementInput>& needed_inputs,
        types::PrimitiveType expected_type
    );

    ElementOutput expand_operation_dataflow(
        builder::StructuredSDFGBuilder& builder,
        structured_control_flow::Block& block,
        std::vector<ElementInput>& needed_inputs,
        types::PrimitiveType expected_type
    ) override;

    bool supports_integer_types() const override { return false; }

    std::unique_ptr<data_flow::DataFlowNode>
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const override;

    std::string toStr() const override;
};

class GELUNodeSerializer : public BaseElementWiseDataflowTensorNodeSerializer {
public:
    nlohmann::json serialize(const data_flow::LibraryNode& library_node) override;

    data_flow::LibraryNode& deserialize(
        const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
    ) override;
};

} // namespace tensor
} // namespace math
} // namespace sdfg
