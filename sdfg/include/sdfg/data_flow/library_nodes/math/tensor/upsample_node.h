/**
 * @file upsample_node.h
 * @brief Bilinear 2D upsampling node compatible with aten::upsample_bilinear2d.vec
 *
 * This file defines the UpsampleBilinear2DNode class which implements a 2D
 * bilinear resize following the PyTorch ``aten::upsample_bilinear2d.vec``
 * operator semantics:
 *
 *   upsample_bilinear2d.vec(Tensor input, SymInt[]? output_size,
 *                           bool align_corners, float[]? scale_factors) -> Tensor
 *
 * ## Input/Output Requirements
 * - Input connector "X": Input tensor [N, C, H_in, W_in]
 * - Input connector "Y": Output tensor [N, C, H_out, W_out] (written in place)
 *
 * ## Source-coordinate mapping
 *
 * For each output index ``o`` along a spatial dimension the corresponding
 * (fractional) source coordinate is computed exactly as PyTorch does:
 * - align_corners:  src = o * (In - 1) / (Out - 1)      (0 if Out == 1)
 * - otherwise:      src = max(0, (o + 0.5) * rscale - 0.5)
 *   where ``rscale`` is ``1 / scale_factor`` when explicit scale factors were
 *   provided, and ``In / Out`` otherwise.
 *
 * The integer neighbours are ``i0 = floor(src)`` and ``i1 = min(i0 + 1, In - 1)``
 * with interpolation weight ``lambda = src - i0``.
 */

#pragma once

#include "sdfg/data_flow/library_nodes/math/tensor/tensor_node.h"

#include "sdfg/serializer/json_serializer.h"

namespace sdfg {
namespace math {
namespace tensor {

inline data_flow::LibraryNodeCode LibraryNodeType_UpsampleBilinear2D("ml::UpsampleBilinear2D");

/**
 * @class UpsampleBilinear2DNode
 * @brief 2D bilinear upsampling operation
 *
 * The node is expanded into a nested map over [N, C, H_out, W_out]. Inside the
 * loop nest the fractional source coordinates and the four contributing input
 * pixels are computed and combined using the standard separable bilinear
 * interpolation formula.
 */
class UpsampleBilinear2DNode : public TensorNode {
protected:
    std::vector<symbolic::Expression> input_shape_; ///< Input shape [N, C, H_in, W_in]
    std::vector<symbolic::Expression> output_shape_; ///< Output shape [N, C, H_out, W_out]
    bool align_corners_;
    std::vector<double> scale_factors_; ///< Optional [scale_h, scale_w]; empty if not provided

public:
    static auto constexpr Y_OUTPUT_IDX = 0;
    static auto constexpr X_INPUT_IDX = 1;

    UpsampleBilinear2DNode(
        size_t element_id,
        const DebugInfo& debug_info,
        const graph::Vertex vertex,
        data_flow::DataFlowGraph& parent,
        const std::vector<symbolic::Expression>& input_shape,
        const std::vector<symbolic::Expression>& output_shape,
        bool align_corners,
        const std::vector<double>& scale_factors,
        const data_flow::ImplementationType& impl_type = data_flow::ImplementationType_NONE
    );

    const std::vector<symbolic::Expression>& input_shape() const { return input_shape_; }
    const std::vector<symbolic::Expression>& output_shape() const { return output_shape_; }
    bool align_corners() const { return align_corners_; }
    const std::vector<double>& scale_factors() const { return scale_factors_; }

    void validate(const Function& function) const override;

    symbolic::SymbolSet symbols() const override;

    void replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) override;

    void replace(const symbolic::ExpressionMapping& replacements) override;

    bool supports_integer_types() const override { return false; }

    passes::LibNodeExpander::ExpandOutcome
    expand(passes::LibNodeExpander::ExpandContext& context, structured_control_flow::Block& block) override;

    std::unique_ptr<data_flow::DataFlowNode>
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const override;

    std::string toStr() const override;

    symbolic::Expression flop() const override;

    data_flow::PointerAccessType pointer_access_type(int input_idx) const override;
};

/**
 * @class UpsampleBilinear2DNodeSerializer
 * @brief Serializer for UpsampleBilinear2DNode
 */
class UpsampleBilinear2DNodeSerializer : public serializer::LibraryNodeSerializer {
public:
    nlohmann::json serialize(const data_flow::LibraryNode& library_node) override;

    data_flow::LibraryNode& deserialize(
        const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
    ) override;
};

} // namespace tensor
} // namespace math
} // namespace sdfg
