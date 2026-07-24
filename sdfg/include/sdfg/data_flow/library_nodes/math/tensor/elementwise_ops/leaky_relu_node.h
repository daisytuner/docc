#pragma once

#include "sdfg/data_flow/library_nodes/math/tensor/elementwise_node.h"

#include "sdfg/codegen/dispatchers/block_dispatcher.h"
#include "sdfg/serializer/json_serializer.h"

namespace sdfg {
namespace math {
namespace tensor {

inline data_flow::LibraryNodeCode LibraryNodeType_LeakyReLU("ml::LeakyReLU");

class LeakyReLUNode : public ElementWiseDataflowTensorNode {
public:
    LeakyReLUNode(
        size_t element_id,
        const DebugInfo& debug_info,
        const graph::Vertex vertex,
        data_flow::DataFlowGraph& parent,
        const std::vector<symbolic::Expression>& shape,
        QuantizationType quantization = QUANTIZATION_MATCH_INPUTS,
        const data_flow::ImplementationType& impl_type = data_flow::ImplementationType_NONE
    );

    int tensor_input_count() const override { return 2; }

    ElementOutput expand_operation_dataflow(
        builder::StructuredSDFGBuilder& builder,
        Block& block,
        std::vector<ElementInput>& needed_inputs,
        types::PrimitiveType expected_type
    ) override;

    bool supports_integer_types() const override { return false; }

    std::unique_ptr<data_flow::DataFlowNode>
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const override;
};

typedef SimpleElementWiseDataflowTensorNodeSerializer<LeakyReLUNode> LeakyReLUNodeSerializer;

} // namespace tensor
} // namespace math
} // namespace sdfg
