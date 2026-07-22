/**
 * @file fft_node.h
 * @brief Fast Fourier Transform library nodes (forward R2C and inverse C2R)
 *
 * This file defines the FFTNode (forward, real-to-complex) and IFFTNode
 * (inverse, complex-to-real) library nodes. They model a batched multi
 * dimensional discrete Fourier transform following the cuFFT / hipFFT
 * conventions, in particular the Hermitian-symmetric layout of the R2C/C2R
 * transforms where the last transformed dimension is reduced to `n/2 + 1`
 * complex elements.
 *
 * ## Data layout
 *
 * A transform operates on `batch` independent signals, each of spatial extent
 * `shape = [n_0, n_1, ..., n_{d-1}]` (row-major, contiguous).
 *
 * - Real buffer element count:    `real_count = batch * prod(n_i)`
 * - Complex (Hermitian) element count:
 *       `complex_extent = batch * prod_{i<d-1}(n_i) * (n_{d-1}/2 + 1)`
 *
 * For FFTNode: input `__X` is real (`Float`/`Double`), output `__Y` is complex
 * (`CFloat`/`CDouble`) with `complex_extent` elements.
 * For IFFTNode: input `__X` is complex (`complex_extent`), output `__Y` is real
 * (`real_count`). C2R is unnormalized (matching cuFFT): callers must scale by
 * `1 / prod(n_i)` afterwards.
 *
 * @see math::tensor::ConvNode for the convolution that is rewritten into
 *      FFT -> complexMul -> IFFT.
 */

#pragma once

#include <vector>

#include "sdfg/data_flow/library_nodes/math/math_node.h"

#include "sdfg/serializer/json_serializer.h"

namespace sdfg {
namespace math {
namespace tensor {

inline data_flow::LibraryNodeCode LibraryNodeType_FFT("ml::FFT");
inline data_flow::LibraryNodeCode LibraryNodeType_IFFT("ml::IFFT");

/**
 * @enum FFTDirection
 * @brief Direction of the transform.
 */
enum class FFTDirection {
    Forward, ///< Real-to-complex (R2C) forward transform.
    Inverse, ///< Complex-to-real (C2R) inverse transform.
};

/**
 * @class FFTNodeBase
 * @brief Shared base for the forward and inverse FFT library nodes.
 *
 * Stores the batched transform geometry (spatial `shape`, `batch` count and the
 * real-side floating point `precision`) and exposes the Hermitian layout helpers
 * (`complex_last_dim()`, `complex_extent()`, `real_extent()`) that must be used
 * consistently by dispatchers, data-transfer extraction and expanders.
 */
class FFTNodeBase : public MathNode {
protected:
    std::vector<symbolic::Expression> shape_; ///< Spatial extents n_0..n_{d-1} (last dim is the Hermitian dim).
    symbolic::Expression batch_; ///< Number of independent transforms.
    types::PrimitiveType precision_; ///< Real-side precision: Float or Double.

public:
    static constexpr int Y_INPUT_IDX = 0; ///< Output pointer `__Y`.
    static constexpr int X_INPUT_IDX = 1; ///< Input pointer `__X`.

    FFTNodeBase(
        size_t element_id,
        const DebugInfo& debug_info,
        const graph::Vertex vertex,
        data_flow::DataFlowGraph& parent,
        const data_flow::LibraryNodeCode& code,
        const data_flow::ImplementationType& implementation_type,
        const std::vector<symbolic::Expression>& shape,
        symbolic::Expression batch,
        types::PrimitiveType precision
    );

    /// @brief Direction of the transform (forward or inverse).
    virtual FFTDirection direction() const = 0;

    /// @brief Spatial extents `[n_0, ..., n_{d-1}]`.
    const std::vector<symbolic::Expression>& shape() const;

    /// @brief Number of independent transforms.
    symbolic::Expression batch() const;

    /// @brief Transform rank (number of spatial dimensions).
    size_t rank() const;

    /// @brief Real-side primitive type (Float or Double).
    types::PrimitiveType real_primitive() const;

    /// @brief Complex-side primitive type (CFloat or CDouble), derived from precision.
    types::PrimitiveType complex_primitive() const;

    /// @brief Hermitian last dimension: `n_{d-1}/2 + 1`.
    symbolic::Expression complex_last_dim() const;

    /// @brief Total real element count: `batch * prod(n_i)`.
    symbolic::Expression real_extent() const;

    /// @brief Total complex element count: `batch * prod_{i<d-1}(n_i) * (n_{d-1}/2 + 1)`.
    symbolic::Expression complex_extent() const;

    void validate(const Function& function) const override;

    symbolic::SymbolSet symbols() const override;

    void replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) override;

    void replace(const symbolic::ExpressionMapping& replacements) override;

    data_flow::PointerAccessType pointer_access_type(int input_idx) const override;
};

/**
 * @class FFTNode
 * @brief Forward real-to-complex (R2C) batched FFT.
 *
 * Input connector `__X` (real, `real_extent` elements) -> output connector
 * `__Y` (complex, `complex_extent` elements).
 */
class FFTNode : public FFTNodeBase {
public:
    FFTNode(
        size_t element_id,
        const DebugInfo& debug_info,
        const graph::Vertex vertex,
        data_flow::DataFlowGraph& parent,
        const data_flow::ImplementationType& implementation_type,
        const std::vector<symbolic::Expression>& shape,
        symbolic::Expression batch,
        types::PrimitiveType precision
    );

    FFTDirection direction() const override { return FFTDirection::Forward; }

    passes::LibNodeExpander::ExpandOutcome
    expand(passes::LibNodeExpander::ExpandContext& context, structured_control_flow::Block& block) override;

    std::unique_ptr<data_flow::DataFlowNode>
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const override;

    std::string toStr() const override;
};

/**
 * @class IFFTNode
 * @brief Inverse complex-to-real (C2R) batched FFT (unnormalized).
 *
 * Input connector `__X` (complex, `complex_extent` elements) -> output connector
 * `__Y` (real, `real_extent` elements). The result is not divided by
 * `prod(n_i)` (matching cuFFT / hipFFT semantics).
 */
class IFFTNode : public FFTNodeBase {
public:
    IFFTNode(
        size_t element_id,
        const DebugInfo& debug_info,
        const graph::Vertex vertex,
        data_flow::DataFlowGraph& parent,
        const data_flow::ImplementationType& implementation_type,
        const std::vector<symbolic::Expression>& shape,
        symbolic::Expression batch,
        types::PrimitiveType precision
    );

    FFTDirection direction() const override { return FFTDirection::Inverse; }

    passes::LibNodeExpander::ExpandOutcome
    expand(passes::LibNodeExpander::ExpandContext& context, structured_control_flow::Block& block) override;

    std::unique_ptr<data_flow::DataFlowNode>
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const override;

    std::string toStr() const override;
};

class FFTNodeSerializer : public serializer::LibraryNodeSerializer {
public:
    nlohmann::json serialize(const data_flow::LibraryNode& library_node) override;

    data_flow::LibraryNode& deserialize(
        const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
    ) override;
};

class IFFTNodeSerializer : public serializer::LibraryNodeSerializer {
public:
    nlohmann::json serialize(const data_flow::LibraryNode& library_node) override;

    data_flow::LibraryNode& deserialize(
        const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
    ) override;
};

} // namespace tensor
} // namespace math
} // namespace sdfg
