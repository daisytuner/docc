#pragma once

#include "sdfg/data_flow/library_nodes/math/tensor/elementwise_node.h"

#include "sdfg/codegen/dispatchers/block_dispatcher.h"
#include "sdfg/serializer/json_serializer.h"
#include "sdfg/types/type.h"

namespace sdfg {
namespace math {
namespace tensor {

inline data_flow::LibraryNodeCode LibraryNodeType_Cast("ml::Cast");

class CastNode : public ElementWiseDataflowTensorNode {
public:
    CastNode(
        size_t element_id,
        const DebugInfo& debug_info,
        const graph::Vertex vertex,
        data_flow::DataFlowGraph& parent,
        const std::vector<symbolic::Expression>& shape,
        types::PrimitiveType target_type, // we reuse the fixed_quantization field for this
        const data_flow::ImplementationType& impl_type = data_flow::ImplementationType_NONE
    );

    ElementOutput expand_operation_dataflow(
        builder::StructuredSDFGBuilder& builder,
        Block& block,
        std::vector<ElementInput>& needed_inputs,
        types::PrimitiveType expected_type
    ) override;

    bool supports_integer_types() const override { return true; }

    types::PrimitiveType target_type() const { return fixed_quantization_; }

    void validate(const Function& function) const override;

    std::unique_ptr<data_flow::DataFlowNode>
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const override;
};

typedef SimpleElementWiseDataflowTensorNodeSerializer<CastNode> CastNodeSerializer;

} // namespace tensor
} // namespace math
} // namespace sdfg
