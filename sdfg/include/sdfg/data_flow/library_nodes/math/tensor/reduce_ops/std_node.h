#pragma once

#include "sdfg/data_flow/library_nodes/math/tensor/reduce_node.h"

namespace sdfg {
namespace math {
namespace tensor {

inline data_flow::LibraryNodeCode LibraryNodeType_Std("ml::Std");

class StdNode : public ReduceNode {
public:
    StdNode(
        size_t element_id,
        const DebugInfo& debug_info,
        const graph::Vertex vertex,
        data_flow::DataFlowGraph& parent,
        const std::vector<symbolic::Expression>& shape,
        const std::vector<int64_t>& axes,
        bool keepdims
    );

    bool expand_reduction(
        builder::StructuredSDFGBuilder& builder,
        analysis::AnalysisManager& analysis_manager,
        structured_control_flow::Sequence& body,
        const std::string& input_name,
        const std::string& output_name,
        const types::Tensor& input_type,
        const types::Tensor& output_type,
        const data_flow::Subset& input_subset,
        const data_flow::Subset& output_subset
    ) override;

    std::string identity(types::PrimitiveType primitive_type) const override;

    bool supports_integer_types() const override { return false; }

    std::unique_ptr<data_flow::DataFlowNode>
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const override;

protected:
    bool expand_inner(
        builder::StructuredSDFGBuilder& builder,
        analysis::AnalysisManager& analysis_manager,
        structured_control_flow::Block& block,
        data_flow::DataFlowGraph& dataflow,
        structured_control_flow::Sequence& parent,
        Transition& transition,
        const data_flow::Memlet* iedge_input,
        const data_flow::Memlet* iedge_result,
        const data_flow::AccessNode* input_node,
        const data_flow::AccessNode* output_node,
        const std::vector<symbolic::Expression>& output_shape,
        const std::vector<int64_t>& sorted_axes
    ) override;
};

typedef ReduceNodeSerializer<StdNode> StdNodeSerializer;

} // namespace tensor
} // namespace math
} // namespace sdfg
