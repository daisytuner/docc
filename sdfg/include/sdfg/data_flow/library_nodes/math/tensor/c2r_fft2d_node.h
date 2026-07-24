/**
 * @file c2r_fft2d_node.h
 * @brief Inverse complex-to-real 2D FFT library node (hand-tuned GPU path).
 *
 * Represents a batched inverse 2D FFT from the non-redundant (Hermitian) half
 * spectrum back to a real signal, scaled by 1/(fftH*fftW):
 *
 *   Y[m, :, :] = IFFT2( X[m, :, 0:halfW] )   (m over `matrices`, halfW = fftW/2+1)
 *
 * X is complex (`CFloat`/`CDouble`) [matrices, fftH, halfW]; Y is real
 * [matrices, fftH, fftW]. Like @ref FFTConvNode, this node has no CPU expansion; it
 * is realised only by the CUDA/ROCm hand-tuned dispatchers that emit mixed-radix
 * Stockham kernels (column FFT + C2R rows). The companion forward is @ref R2CFFT2DNode.
 */

#pragma once

#include <vector>

#include "sdfg/data_flow/library_nodes/math/math_node.h"

#include "sdfg/serializer/json_serializer.h"

namespace sdfg {
namespace math {
namespace tensor {

inline data_flow::LibraryNodeCode LibraryNodeType_C2RFFT2D("ml::C2RFFT2D");

/**
 * @class C2RFFT2DNode
 * @brief Batched inverse 2D complex-to-real FFT (hand-tuned FFT kernels).
 */
class C2RFFT2DNode : public MathNode {
private:
    std::vector<symbolic::Expression> shape_; ///< Real output shape [matrices, fftH, fftW].
    types::PrimitiveType precision_; ///< Float or Double (real component type).

public:
    static constexpr int Y_INPUT_IDX = 0; ///< Real output (pointer input).
    static constexpr int X_INPUT_IDX = 1; ///< Complex half-spectrum input.

    C2RFFT2DNode(
        size_t element_id,
        const DebugInfo& debug_info,
        const graph::Vertex vertex,
        data_flow::DataFlowGraph& parent,
        const data_flow::ImplementationType& implementation_type,
        const std::vector<symbolic::Expression>& shape,
        types::PrimitiveType precision
    );

    const std::vector<symbolic::Expression>& shape() const;
    types::PrimitiveType real_primitive() const;
    types::PrimitiveType complex_primitive() const;

    void validate(const Function& function) const override;

    passes::LibNodeExpander::ExpandOutcome
    expand(passes::LibNodeExpander::ExpandContext& context, structured_control_flow::Block& block) override;

    symbolic::SymbolSet symbols() const override;

    void replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) override;

    void replace(const symbolic::ExpressionMapping& replacements) override;

    std::unique_ptr<data_flow::DataFlowNode>
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const override;

    std::string toStr() const override;

    data_flow::PointerAccessType pointer_access_type(int input_idx) const override;
};

class C2RFFT2DNodeSerializer : public serializer::LibraryNodeSerializer {
public:
    nlohmann::json serialize(const data_flow::LibraryNode& library_node) override;

    data_flow::LibraryNode& deserialize(
        const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
    ) override;
};

} // namespace tensor
} // namespace math
} // namespace sdfg
