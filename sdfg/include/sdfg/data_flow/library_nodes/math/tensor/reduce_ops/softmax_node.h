#pragma once

#include "sdfg/data_flow/library_nodes/math/tensor/reduce_node.h"

namespace sdfg {
namespace math {
namespace tensor {

inline data_flow::LibraryNodeCode LibraryNodeType_Softmax("ml::Softmax");

class SoftmaxNode : public ReduceNode {
public:
    SoftmaxNode(
        size_t element_id,
        const DebugInfo& debug_info,
        const graph::Vertex vertex,
        data_flow::DataFlowGraph& parent,
        const std::vector<symbolic::Expression>& shape,
        const std::vector<int64_t>& axes,
        bool keepdims = false
    );

    void validate(const Function& function) const override;

    bool expand(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

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
    ) override {
        return false;
    }

    std::string identity(types::PrimitiveType primitive_type) const override { return ""; }

    bool supports_integer_types() const override { return false; }

    std::unique_ptr<data_flow::DataFlowNode>
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const override;
};

typedef ReduceNodeSerializer<SoftmaxNode> SoftmaxNodeSerializer;

} // namespace tensor
} // namespace math
} // namespace sdfg
