#pragma once

#include "sdfg/data_flow/library_nodes/math/tensor/elementwise_node.h"

#include "sdfg/codegen/dispatchers/block_dispatcher.h"
#include "sdfg/serializer/json_serializer.h"

namespace sdfg {
namespace math {
namespace tensor {

inline data_flow::LibraryNodeCode LibraryNodeType_HardSigmoid("ml::HardSigmoid");

class HardSigmoidNode : public ElementWiseUnaryNode {
public:
    HardSigmoidNode(
        size_t element_id,
        const DebugInfo& debug_info,
        const graph::Vertex vertex,
        data_flow::DataFlowGraph& parent,
        const std::vector<symbolic::Expression>& shape
    );

    bool expand_operation(
        builder::StructuredSDFGBuilder& builder,
        analysis::AnalysisManager& analysis_manager,
        structured_control_flow::Sequence& body,
        const std::string& input_name,
        const std::string& output_name,
        const types::Tensor& input_type,
        const types::Tensor& output_type,
        const data_flow::Subset& subset
    ) override;

    bool supports_integer_types() const override { return false; }

    std::unique_ptr<data_flow::DataFlowNode>
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const override;
};

typedef ElementWiseUnaryNodeSerializer<HardSigmoidNode> HardSigmoidNodeSerializer;

} // namespace tensor
} // namespace math
} // namespace sdfg
