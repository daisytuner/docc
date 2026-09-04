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

inline data_flow::LibraryNodeCode LibraryNodeType_Index("ml::Index");

/** @brief Tensor advanced-indexing node.
 *
 * This node mirrors the advanced indexing from NumPy/PyTorch.
 *
 * Suppose you have an input tensor (X) of shape [4, 4, 4, 4, 4] and you want to perform indexing on this tensor with
 * this access pattern: [:, (0, 2), (1,), :, :].
 * Then, this library node would require you to set indices to [1, 2] because these are the index positions you want to
 * perform indexing on (the rest are ignored with ":"). This will create the connectors "I1" and "I2" where the index
 * tensors (0, 2) and (1,) must be connected.
 * To derive the output shape, first all index tensors must be broadcasted to a common shape. This is done with the
 * method common_indices_shape(). In this case, the index tensor shapes are [2] and [1] and have a common shape of [2].
 * Since the index positions are contiguous, the common shape is placed at the location of the index posisitions to
 * derive the output shape. In this example: [4, 2, 4, 4].
 * In the case that the index positions are non-contiguous, the common shape is placed at the front of the output shape.
 * For example, if the access pattern is [:, (0, 2), :, (1,), :], then the indices are [1, 3] and non-contiguous. The
 * common shape is still [2], but the output shape now is [2, 4, 4, 4].
 */
class IndexNode : public TensorNode {
private:
    std::vector<long long> indices_;
    TensorLayout y_layout_;
    TensorLayout x_layout_;
    std::vector<TensorLayout> index_layouts_;

public:
    IndexNode(
        size_t element_id,
        const DebugInfo& debug_info,
        const graph::Vertex vertex,
        data_flow::DataFlowGraph& parent,
        const std::vector<long long>& indices,
        const TensorLayout& y_layout,
        const TensorLayout& x_layout,
        const std::vector<TensorLayout>& index_layouts,
        const data_flow::ImplementationType& impl_type = data_flow::ImplementationType_NONE
    );

    static int constexpr Y_INPUT_IDX = 0;
    static int constexpr X_INPUT_IDX = 1;
    static int constexpr INDEX_INPUT_OFFSET = 2;

    long long num_indices() const;
    const std::vector<long long>& indices() const;
    bool contiguous_indices() const;

    const TensorLayout& y_layout() const;
    const TensorLayout& x_layout() const;
    const std::vector<TensorLayout>& index_layouts() const;

    symbolic::MultiExpression common_indices_shape() const;

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

class IndexNodeSerializer : public serializer::LibraryNodeSerializer {
public:
    nlohmann::json serialize(const data_flow::LibraryNode& library_node) override;

    data_flow::LibraryNode& deserialize(
        const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
    ) override;
};

} // namespace tensor
} // namespace math
} // namespace sdfg
