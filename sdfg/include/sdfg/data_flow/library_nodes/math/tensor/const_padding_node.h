#pragma once

#include <cstddef>
#include <memory>
#include <nlohmann/json_fwd.hpp>
#include <string>

#include "sdfg/data_flow/data_flow_graph.h"
#include "sdfg/data_flow/data_flow_node.h"
#include "sdfg/data_flow/library_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/tensor_layout.h"
#include "sdfg/data_flow/library_nodes/math/tensor/tensor_node.h"
#include "sdfg/data_flow/pointer_metadata.h"
#include "sdfg/element.h"
#include "sdfg/function.h"
#include "sdfg/graph/graph.h"
#include "sdfg/passes/expansion/lib_node_expander.h"
#include "sdfg/serializer/json_serializer.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {
namespace math {
namespace tensor {

inline data_flow::LibraryNodeCode LibraryNodeType_ConstPadding("ml::ConstPadding");

/** @brief Applies padding by a constant value.
 * Y is the output tensor and X is the input tensor. Pads is the padding mask and Val is the constant value for padding.
 * The shape dimensions of Y and X must be identical. The padding mask must always be even, as it contains the lower and
 * upper padding size per dimension. It can, however, contains less dimensions than Y and X. In such a case, the padding
 * dimensions are the innermost dimension of the tensors.
 * Example:
 * X: [n, h, w]
 * Pads: [a, b, c, d]
 * Then, we can derive Y: [n, a + h + b, c + w + d]
 * In this example, the last two dimensions are padded. The lower pads are a and c and the higher pads are b and d. The
 * first dimension is left untouched.
 */
class ConstPaddingNode : public TensorNode {
private:
    symbolic::MultiExpression pads_;
    TensorLayout y_layout_;
    TensorLayout x_layout_;

public:
    ConstPaddingNode(
        size_t element_id,
        const DebugInfo& debug_info,
        const graph::Vertex vertex,
        data_flow::DataFlowGraph& parent,
        const symbolic::MultiExpression& pads,
        const TensorLayout& y_layout,
        const TensorLayout& x_layout,
        const data_flow::ImplementationType& impl_type = data_flow::ImplementationType_NONE
    );

    static int constexpr Y_INPUT_IDX = 0;
    static int constexpr X_INPUT_IDX = 1;
    static int constexpr VAL_INPUT_IDX = 2;

    const symbolic::MultiExpression& pads() const;

    const symbolic::Expression& get_lower_pad(int index) const;
    const symbolic::Expression& get_upper_pad(int index) const;

    const TensorLayout& y_layout() const;
    const TensorLayout& x_layout() const;

    virtual void validate(const Function& function) const override;

    virtual bool supports_integer_types() const override;

    virtual passes::LibNodeExpander::ExpandOutcome
    expand(passes::LibNodeExpander::ExpandContext& context, structured_control_flow::Block& block) override;

    /**
     * @brief Convert node to string representation
     * @return String representation of the node
     */
    virtual std::string toStr() const override;

    /**
     * @brief Get all symbols used in this node
     * @return Set of symbolic expressions used by this node
     */
    virtual symbolic::SymbolSet symbols() const override;

    /**
     * @brief Calculate floating point operations for this node
     * @return Symbolic expression for FLOP count
     */
    virtual symbolic::Expression flop() const override;

    /**
     * Describes what a pointer is used for
     * @param input_idx index of input that is a pointer.
     * @return Invalid if not asked about a pointer input
     */
    virtual data_flow::PointerAccessType pointer_access_type(int input_idx) const override;

    /**
     * @brief Clone this node for graph transformations
     * @param element_id New element identifier for the clone
     * @param vertex New graph vertex for the clone
     * @param parent Parent graph for the clone
     * @return Unique pointer to the cloned node
     */
    virtual std::unique_ptr<data_flow::DataFlowNode>
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const override;

    virtual void replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) override;

    virtual void replace(const symbolic::ExpressionMapping& replacements) override;
};

class ConstPaddingNodeSerializer : public serializer::LibraryNodeSerializer {
public:
    nlohmann::json serialize(const data_flow::LibraryNode& library_node) override;

    data_flow::LibraryNode& deserialize(
        const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
    ) override;
};

} // namespace tensor
} // namespace math
} // namespace sdfg
