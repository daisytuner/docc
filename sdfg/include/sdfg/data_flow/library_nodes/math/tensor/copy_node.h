/**
 * @file copy_node.h
 * @brief Tensor copy node that copies values from one buffer to another according to their tensor access
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

inline data_flow::LibraryNodeCode LibraryNodeType_TensorCopy("ml::Copy");

/** @brief Tensor copy node that copies values form one container to another according to their tensor access.
 *
 * The only critertion for this copy node to work is that both tensor types must have the same number of elements.
 * There are four expansion modes:
 * - Identity mode: The shape of both tensors are the same. The tensor can just be copied straight forwardly.
 * - Permutation mode: The shapes of the tensors have the same length and the same elements but their ordering is
 * different. The tensor is copied by adapting the index variable access to the input tensor.
 * - Squeeze mode: The shape of the tensors contain the same elements in the same order but one of the tensors has
 * additional ones in it. The tensor is copied by filling the index variable access of the unsqueezed tensor with zeros.
 * - Reshape mode: None of the above modes. The tensors are copied by shifting one index variable over the whole size
 * with divisions and modulos.
 */
class TensorCopyNode : public TensorNode {
private:
    TensorLayout layout_x_;
    TensorLayout layout_y_;

    void expand_identity_mode(
        passes::LibNodeExpander::AccessNodeExpand& standalone,
        builder::StructuredSDFGBuilder& builder,
        structured_control_flow::Sequence& sequence
    );

    void expand_permutation_mode(
        passes::LibNodeExpander::AccessNodeExpand& standalone,
        builder::StructuredSDFGBuilder& builder,
        structured_control_flow::Sequence& sequence
    );

    void expand_squeeze_mode(
        passes::LibNodeExpander::AccessNodeExpand& standalone,
        builder::StructuredSDFGBuilder& builder,
        structured_control_flow::Sequence& sequence
    );

    void expand_reshape_mode(
        passes::LibNodeExpander::AccessNodeExpand& standalone,
        builder::StructuredSDFGBuilder& builder,
        structured_control_flow::Sequence& sequence
    );

public:
    TensorCopyNode(
        size_t element_id,
        const DebugInfo& debug_info,
        const graph::Vertex vertex,
        data_flow::DataFlowGraph& parent,
        const TensorLayout& layout_x,
        const TensorLayout& layout_y,
        const data_flow::ImplementationType& impl_type = data_flow::ImplementationType_NONE
    );

    static auto constexpr X_INPUT_IDX = 0;
    static auto constexpr Y_INPUT_IDX = 1;

    const TensorLayout& layout_x() const;
    const TensorLayout& layout_y() const;

    /** @brief The tensor copy node is in identity mode.
     *
     * The shape of both tensors are the same. The tensor can just be copied straight forwardly.
     * Example shapes:
     * - 2x3 -> 2x3
     *
     * @return True iff the tensor copy node is in identity mode
     */
    bool is_identity_mode() const;

    /** @brief The tensor copy node is in permutation mode.
     *
     * The shapes of the tensors have the same length and the same elements but their ordering is
     * different. The tensor is copied by adapting the index variable access to the input tensor.
     * Example shapes:
     * - 2x3 -> 3x2
     * - 1x2x3 -> 3x2x1
     *
     * @return True iff the tensor copy node is in permutation mode
     */
    bool is_permutation_mode() const;

    /** @brief The tensor copy node is in squeeze mode.
     *
     * The shape of the tensors contain the same elements in the same order but one of the tensors has
     * additional ones in it. The tensor is copied by filling the index variable access of the unsqueezed tensor with
     * zeros.
     * Example shapes:
     * - 2x3 -> 2x1x3
     * - 2x3 -> 1x2x1x3x1
     * - 2x1x3 -> 2x3
     * - 1x2x1x3x1 -> 2x3
     *
     * @return True iff the tensor copy node is in squeeze mode
     */
    bool is_squeeze_mode() const;

    /** @brief The tensor copy node is in reshape mode.
     *
     * The tensor copy node is neither in identity, permutation, nor squeeze mode. The tensors are copied by shifting
     * one index variable over the whole size with divisions and modulos.
     * Example shapes:
     * - 2x3 -> 6x1
     * - 2x3 -> 1x6
     * - 2x3 -> 6
     * - 6x1 -> 2x3
     * - 1x6 -> 2x3
     * - 6 -> 2x3
     * - 48 -> 2x4x3x2
     * - 2x4x3x2 -> 48
     *
     * @return True iff the tensor copy node is in reshape mode
     */
    bool is_reshape_mode() const;

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

class TensorCopyNodeSerializer : public serializer::LibraryNodeSerializer {
public:
    nlohmann::json serialize(const data_flow::LibraryNode& library_node) override;

    data_flow::LibraryNode& deserialize(
        const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
    ) override;
};

} // namespace tensor
} // namespace math
} // namespace sdfg
