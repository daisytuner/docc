/**
 * @file r2c_fft2d_node.h
 * @brief Forward real-to-complex 2D FFT library node (hand-tuned GPU path).
 *
 * Represents a batched forward 2D FFT of a real, already-padded input into the
 * non-redundant (Hermitian) half spectrum:
 *
 *   Y[m, :, 0:halfW] = FFT2( X[m, :, :] )   (m over `matrices`, halfW = fftW/2+1)
 *
 * X is real [matrices, fftH, fftW]; Y is complex (`CFloat`/`CDouble`)
 * [matrices, fftH, halfW]. Like @ref FFTConvNode, this node has no CPU expansion; it
 * is realised only by the CUDA/ROCm hand-tuned dispatchers that emit mixed-radix
 * Stockham kernels (R2C rows + column FFT). The companion inverse is @ref C2RFFT2DNode.
 */

#pragma once

#include <vector>

#include "sdfg/data_flow/library_nodes/math/math_node.h"

#include "sdfg/serializer/json_serializer.h"

namespace sdfg {
namespace math {
namespace tensor {

inline data_flow::LibraryNodeCode LibraryNodeType_R2CFFT2D("ml::R2CFFT2D");

/**
 * @class R2CFFT2DNode
 * @brief Batched forward 2D real-to-complex FFT (hand-tuned FFT kernels).
 */
class R2CFFT2DNode : public MathNode {
private:
    std::vector<symbolic::Expression> shape_; ///< Real input shape [matrices, fftH, fftW].
    types::PrimitiveType precision_; ///< Float or Double (real component type).

public:
    static constexpr int Y_INPUT_IDX = 0; ///< Complex half-spectrum output (pointer input).
    static constexpr int X_INPUT_IDX = 1; ///< Real padded input.

    R2CFFT2DNode(
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

class R2CFFT2DNodeSerializer : public serializer::LibraryNodeSerializer {
public:
    nlohmann::json serialize(const data_flow::LibraryNode& library_node) override;

    data_flow::LibraryNode& deserialize(
        const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
    ) override;
};

} // namespace tensor
} // namespace math
} // namespace sdfg
