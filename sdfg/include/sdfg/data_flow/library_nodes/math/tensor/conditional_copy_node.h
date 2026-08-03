/**
 * @file conditional_copy_node.h
 * @brief Conditional tensor copy node that copies values from one tensor or another depending on a tensor mask
 */
#pragma once

#include <cstddef>
#include <memory>
#include <nlohmann/json_fwd.hpp>
#include <string>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/data_flow_graph.h"
#include "sdfg/data_flow/data_flow_node.h"
#include "sdfg/data_flow/library_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/tensor_layout.h"
#include "sdfg/data_flow/library_nodes/math/tensor/tensor_node.h"
#include "sdfg/element.h"
#include "sdfg/function.h"
#include "sdfg/graph/graph.h"
#include "sdfg/passes/expansion/lib_node_expander.h"
#include "sdfg/serializer/json_serializer.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {
namespace math {
namespace tensor {

inline data_flow::LibraryNodeCode LibraryNodeType_ConditionalTensorCopy("ml::ConditionalCopy");

/** @brief Conditional tensor copy node that copies values from one tensor or another depending on a tensor mask.
 *
 * Exactly four inputs are required: Mask, X1, X2, and Y. All tensor types must have the same shapes. The tensor of Mask
 * must have a boolean element type. All the other tensors must have the same element type. The node is expanded into a
 * loop nest over the shape dimensions where an elementwise copy of X1 into Y is performed if the boolean value of Mask
 * is true, otherwise a copy of X2 into Y is performed.
 */
class ConditionalTensorCopyNode : public TensorNode {
private:
    TensorLayout layout_mask_;
    TensorLayout layout_x1_;
    TensorLayout layout_x2_;
    TensorLayout layout_y_;

    void validate_equal_shapes(const TensorLayout& layout1, const TensorLayout& layout2) const;

public:
    ConditionalTensorCopyNode(
        size_t element_id,
        const DebugInfo& debug_info,
        const graph::Vertex vertex,
        data_flow::DataFlowGraph& parent,
        const TensorLayout& layout_mask,
        const TensorLayout& layout_x1,
        const TensorLayout& layout_x2,
        const TensorLayout& layout_y,
        const data_flow::ImplementationType& impl_type = data_flow::ImplementationType_NONE
    );

    static auto constexpr MASK_INPUT_IDX = 0;
    static auto constexpr X1_INPUT_IDX = 1;
    static auto constexpr X2_INPUT_IDX = 2;
    static auto constexpr Y_INPUT_IDX = 3;

    const TensorLayout& layout_mask() const;
    const TensorLayout& layout_x1() const;
    const TensorLayout& layout_x2() const;
    const TensorLayout& layout_y() const;

    void validate(const Function& function) const override;

    virtual bool supports_integer_types() const override;

    virtual passes::LibNodeExpander::ExpandOutcome
    expand(passes::LibNodeExpander::ExpandContext& context, structured_control_flow::Block& block) override;

    virtual std::string toStr() const override;

    virtual symbolic::SymbolSet symbols() const override;

    virtual symbolic::Expression flop() const override;

    virtual std::unique_ptr<data_flow::DataFlowNode>
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const override;

    virtual void replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) override;

    void replace(const symbolic::ExpressionMapping& replacements) override;
};

class ConditionalTensorCopyNodeSerializer : public serializer::LibraryNodeSerializer {
public:
    nlohmann::json serialize(const data_flow::LibraryNode& library_node) override;

    data_flow::LibraryNode& deserialize(
        const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
    ) override;
};

} // namespace tensor
} // namespace math
} // namespace sdfg
