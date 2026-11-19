#pragma once

#include "sdfg/data_flow/library_nodes/math/tensor/reduce_node.h"

namespace sdfg {
namespace math {
namespace tensor {

inline data_flow::LibraryNodeCode LibraryNodeType_ArgSort("ml::ArgSort");

/**
 * @brief The indices that sort X along one axis (np.argsort with kind="stable"): equal elements keep their order,
 * and NaNs sort last.
 *
 * Y holds Int64 indices in the shape of X.
 */
class ArgSortNode : public ReduceNode {
public:
    ArgSortNode(
        size_t element_id,
        const DebugInfo& debug_info,
        const graph::Vertex vertex,
        data_flow::DataFlowGraph& parent,
        const std::vector<symbolic::Expression>& shape,
        const std::vector<int64_t>& axes,
        bool keepdims = false
    );

    void validate(const Function& function) const override;

    /// The shape of X: the axis is sorted, not reduced.
    std::vector<symbolic::Expression> output_shape() const override {
        return shape_;
    }

    passes::LibNodeExpander::ExpandOutcome expand(passes::LibNodeExpander::ExpandContext& context, Block& block) override;

    bool expand_reduction(
        passes::LibNodeExpander::AccessNodeExpand& expansion,
        builder::StructuredSDFGBuilder& builder,
        structured_control_flow::Sequence& body,
        const types::Tensor& input_type,
        const types::Tensor& output_type,
        const data_flow::Subset& input_subset,
        const data_flow::Subset& output_subset
    ) override {
        return false;
    }

    std::string identity(types::PrimitiveType primitive_type) const override {
        return "";
    }

    bool supports_integer_types() const override {
        return true;
    }

    std::unique_ptr<data_flow::DataFlowNode>
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const override;
};

typedef ReduceNodeSerializer<ArgSortNode> ArgSortNodeSerializer;

} // namespace tensor
} // namespace math
} // namespace sdfg
