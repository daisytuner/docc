#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <string>

#include <nlohmann/json_fwd.hpp>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/data_flow_graph.h"
#include "sdfg/data_flow/data_flow_node.h"
#include "sdfg/data_flow/library_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/tensor_layout.h"
#include "sdfg/data_flow/library_nodes/math/tensor/tensor_node.h"
#include "sdfg/data_flow/pointer_metadata.h"
#include "sdfg/graph/graph.h"
#include "sdfg/serializer/json_serializer.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg::math::tensor {

inline data_flow::LibraryNodeCode LibraryNodeType_LayerNorm("ml::LayerNorm");

/**
 * Applies layer normalization over the trailing `noramlized_shape` dimensions of the input.
 * The outputs (the inputs which are written to) are y, mean, and the standard deviation.
 * This inputs are x and epsilon.
 * Optionally, a affine scale can be added (gamma).
 * If an affine scale is added, a bias can be additionally added (beta).
 */
class LayerNormNode : public TensorNode {
    symbolic::MultiExpression normalized_shape_; ///< The dimensions to normalize over
    bool elementwise_affine_; ///< True iff learnable per-element affine parameters is present
    bool bias_; ///< True iff additive bias is present (only meaningful when elementwise_affine_)
    TensorLayout y_layout_;
    TensorLayout mean_layout_;
    TensorLayout rstd_layout_;
    TensorLayout x_layout_;
    std::optional<TensorLayout> gamma_layout_;
    std::optional<TensorLayout> beta_layout_;
    QuantizationType fixed_quantization_; ///< Fixed quantization

    void validate_equal_shapes(
        const std::string& msg, const symbolic::MultiExpression& shape1, const symbolic::MultiExpression& shape2
    ) const;

public:
    /**
     * Constructs a layer normalization node without learnable per-element affine parameter and without additive bias.
     */
    LayerNormNode(
        size_t element_id,
        const DebugInfo& debug_info,
        graph::Vertex vertex,
        data_flow::DataFlowGraph& parent,
        const symbolic::MultiExpression& normalized_shape,
        const TensorLayout& y_layout,
        const TensorLayout& mean_layout,
        const TensorLayout& rstd_layout,
        const TensorLayout& x_layout,
        QuantizationType quantization = QUANTIZATION_MATCH_INPUTS,
        data_flow::ImplementationType impl_type = data_flow::ImplementationType_NONE
    );

    /**
     * Constructs a layer normalization node with learnable per-element affine parameter, but without additive bias.
     */
    LayerNormNode(
        size_t element_id,
        const DebugInfo& debug_info,
        graph::Vertex vertex,
        data_flow::DataFlowGraph& parent,
        const symbolic::MultiExpression& normalized_shape,
        const TensorLayout& y_layout,
        const TensorLayout& mean_layout,
        const TensorLayout& rstd_layout,
        const TensorLayout& x_layout,
        const TensorLayout& gamma_layout,
        QuantizationType quantization = QUANTIZATION_MATCH_INPUTS,
        data_flow::ImplementationType impl_type = data_flow::ImplementationType_NONE
    );

    /**
     * Constructs a layer normalization node with learnable per-element affine parameter and with additive bias.
     */
    LayerNormNode(
        size_t element_id,
        const DebugInfo& debug_info,
        graph::Vertex vertex,
        data_flow::DataFlowGraph& parent,
        const symbolic::MultiExpression& normalized_shape,
        const TensorLayout& y_layout,
        const TensorLayout& mean_layout,
        const TensorLayout& rstd_layout,
        const TensorLayout& x_layout,
        const TensorLayout& gamma_layout,
        const TensorLayout& beta_layout,
        QuantizationType quantization = QUANTIZATION_MATCH_INPUTS,
        data_flow::ImplementationType impl_type = data_flow::ImplementationType_NONE
    );

    static auto constexpr Y_INPUT_IDX = 0;
    static auto constexpr MEAN_INPUT_IDX = 1;
    static auto constexpr RSTD_INPUT_IDX = 2;
    static auto constexpr X_INPUT_IDX = 3;
    static auto constexpr EPS_INPUT_IDX = 4;
    static auto constexpr GAMMA_INPUT_IDX = 5;
    static auto constexpr BETA_INPUT_IDX = 6;

    /** @brief The dimensions to normalize over
     */
    const symbolic::MultiExpression& normalized_shape() const;

    /** @brief the dimensions to not normalize over
     */
    symbolic::MultiExpression non_normalized_shape() const;

    /** @brief True iff learnable per-element affine parameters is present
     */
    bool elementwise_affine() const;

    /** @brief True iff additive bias is present (only meaningful when elementwise_affine)
     */
    bool bias() const;

    const TensorLayout& y_layout() const;
    const TensorLayout& mean_layout() const;
    const TensorLayout& rstd_layout() const;
    const TensorLayout& x_layout() const;
    const std::optional<TensorLayout>& gamma_layout() const;
    const std::optional<TensorLayout>& beta_layout() const;

    QuantizationType quantization() const;

    void set_quantization(const QuantizationType quant);

    void validate(const Function& function) const override;

    bool supports_integer_types() const override;

    virtual passes::LibNodeExpander::ExpandOutcome
    expand(passes::LibNodeExpander::ExpandContext& context, structured_control_flow::Block& block) override;

    virtual std::string toStr() const override;

    virtual symbolic::SymbolSet symbols() const override;

    virtual symbolic::Expression flop() const override;

    virtual data_flow::PointerAccessType pointer_access_type(int input_idx) const override;

    virtual std::unique_ptr<data_flow::DataFlowNode>
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const override;

    virtual void replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) override;

    virtual void replace(const symbolic::ExpressionMapping& replacements) override;
};

class LayerNormNodeSerializer : public serializer::LibraryNodeSerializer {
    nlohmann::json serialize(const data_flow::LibraryNode& library_node) override;

    data_flow::LibraryNode& deserialize(
        const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
    ) override;
};

} // namespace sdfg::math::tensor
