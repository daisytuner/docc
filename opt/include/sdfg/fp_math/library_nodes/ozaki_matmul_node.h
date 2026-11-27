#pragma once

#include <string>
#include <vector>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/tensor_layout.h"
#include "sdfg/passes/expansion/lib_node_expander.h"
#include "sdfg/serializer/json_serializer.h"

namespace sdfg {
namespace fp_math {

inline data_flow::LibraryNodeCode LibraryNodeType_OzakiMatMul("fp::OzakiMatMul");

/**
 * @brief Y = A @ B with B given as its Ozaki slices, prepared once for many products, e.g. a model's weights.
 *
 * Computes what OzakiContraction with direct rescaling computes for a MatMul of the float m x k A and a k x n B whose
 * slices and column scales were prepared before (see build_slicing_direct_prepared): `s` row-major k x n Int8 slices
 * `B0 ... B{s-1}`, or column-major if B is stored transposed, and the n scales `SB`, the absolute maxima of B's
 * columns in A's type. A, Y follow their layouts. Dimensions of Y before its last two are a batch: A has them too,
 * and the slices and scales hold one matrix after the other, as a KV cache keeps its keys per head.
 *
 * Inputs, in order: Y (written), A, SB, B0 ... B{s-1}.
 */
class OzakiMatMulNode : public data_flow::LibraryNode {
    math::tensor::TensorLayout layout_a_;
    math::tensor::TensorLayout layout_y_;
    types::PrimitiveType element_;
    int slices_;
    bool b_transposed_;

public:
    static constexpr size_t Y_INPUT_IDX = 0;
    static constexpr size_t A_INPUT_IDX = 1;
    static constexpr size_t SCALES_INPUT_IDX = 2;
    static constexpr size_t FIRST_SLICE_INPUT_IDX = 3;

    OzakiMatMulNode(
        size_t element_id,
        const DebugInfo& debug_info,
        const graph::Vertex vertex,
        data_flow::DataFlowGraph& parent,
        const math::tensor::TensorLayout& layout_a,
        const math::tensor::TensorLayout& layout_y,
        types::PrimitiveType element,
        int slices,
        bool b_transposed = false,
        const data_flow::ImplementationType& impl_type = data_flow::ImplementationType_NONE
    );

    const math::tensor::TensorLayout& layout_a() const {
        return layout_a_;
    }
    const math::tensor::TensorLayout& layout_y() const {
        return layout_y_;
    }
    types::PrimitiveType element() const {
        return element_;
    }
    int slices() const {
        return slices_;
    }
    /// Whether each slice holds B transposed, its n rows of k elements, as a cache holds a row per token.
    bool b_transposed() const {
        return b_transposed_;
    }
    /// The extents of Y's batch dimensions, those before its last two; none for one matrix.
    symbolic::MultiExpression batch() const;

    symbolic::Expression m() const;
    symbolic::Expression n() const;
    symbolic::Expression k() const;

    void validate(const Function& function) const override;

    symbolic::SymbolSet symbols() const override;

    void replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) override;
    void replace(const symbolic::ExpressionMapping& replacements) override;

    std::unique_ptr<data_flow::DataFlowNode>
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const override;

    std::string toStr() const override;

    data_flow::PointerAccessType pointer_access_type(int input_idx) const override;
};

inline data_flow::LibraryNodeCode LibraryNodeType_OzakiSlice("fp::OzakiSlice");

/**
 * @brief The Ozaki slices of the rows of X, as OzakiMatMul's B takes them transposed, e.g. a token's keys as a cache
 * stores them: the scales `S`, each row's absolute maximum in X's type, and `s` row-major Int8 slices `D0 ... D{s-1}`
 * of X's shape, as slicing with direct rescaling slices A's rows. Dimensions of X before its last two are a batch.
 *
 * Inputs, in order: X, S (written), D0 ... D{s-1} (written).
 */
class OzakiSliceNode : public data_flow::LibraryNode {
    math::tensor::TensorLayout layout_x_;
    types::PrimitiveType element_;
    int slices_;

public:
    static constexpr size_t X_INPUT_IDX = 0;
    static constexpr size_t SCALES_INPUT_IDX = 1;
    static constexpr size_t FIRST_SLICE_INPUT_IDX = 2;

    OzakiSliceNode(
        size_t element_id,
        const DebugInfo& debug_info,
        const graph::Vertex vertex,
        data_flow::DataFlowGraph& parent,
        const math::tensor::TensorLayout& layout_x,
        types::PrimitiveType element,
        int slices,
        const data_flow::ImplementationType& impl_type = data_flow::ImplementationType_NONE
    );

    const math::tensor::TensorLayout& layout_x() const {
        return layout_x_;
    }
    types::PrimitiveType element() const {
        return element_;
    }
    int slices() const {
        return slices_;
    }
    symbolic::Expression rows() const;
    symbolic::Expression columns() const;
    symbolic::MultiExpression batch() const;

    void validate(const Function& function) const override;
    symbolic::SymbolSet symbols() const override;
    void replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) override;
    void replace(const symbolic::ExpressionMapping& replacements) override;
    std::unique_ptr<data_flow::DataFlowNode>
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const override;
    std::string toStr() const override;
    data_flow::PointerAccessType pointer_access_type(int input_idx) const override;
};

class OzakiSliceNodeSerializer : public serializer::LibraryNodeSerializer {
public:
    nlohmann::json serialize(const data_flow::LibraryNode& library_node) override;
    data_flow::LibraryNode& deserialize(
        const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
    ) override;
};

class OzakiSliceExpander : public passes::CodeLibNodeExpander<OzakiSliceNode> {
    double min_normal_;

public:
    /// @param min_normal the least scale of a row (see OzakiContraction)
    explicit OzakiSliceExpander(double min_normal = 0)
        : CodeLibNodeExpander(LibraryNodeType_OzakiSlice), min_normal_(min_normal) {
    }

    using TypedLibNodeExpander<OzakiSliceNode>::handle_expand;

    passes::LibNodeExpander::ExpandOutcome handle_expand(
        passes::LibNodeExpander::ExpandContext& context, structured_control_flow::Block& block, OzakiSliceNode& node
    ) const override;
};

class OzakiMatMulNodeSerializer : public serializer::LibraryNodeSerializer {
public:
    nlohmann::json serialize(const data_flow::LibraryNode& library_node) override;
    data_flow::LibraryNode& deserialize(
        const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
    ) override;
};

class OzakiMatMulExpander : public passes::CodeLibNodeExpander<OzakiMatMulNode> {
    int max_bits_;
    double min_normal_;

public:
    /// @param max_bits the bits the target computes results in (see OzakiContraction); with them, A takes the slices
    /// its own precision needs, which may differ from B's, else as many as B
    /// @param min_normal the least scale of a row of A (see OzakiContraction)
    explicit OzakiMatMulExpander(int max_bits = 0, double min_normal = 0)
        : CodeLibNodeExpander(LibraryNodeType_OzakiMatMul), max_bits_(max_bits), min_normal_(min_normal) {
    }

    using TypedLibNodeExpander<OzakiMatMulNode>::handle_expand;

    passes::LibNodeExpander::ExpandOutcome handle_expand(
        passes::LibNodeExpander::ExpandContext& context, structured_control_flow::Block& block, OzakiMatMulNode& node
    ) const override;
};

/// Expands every OzakiMatMul node of the SDFG into the slicing of A, the Int8 MatMuls and the rescaling, and every
/// OzakiSlice node into the slicing of its rows; whether any.
/// @param max_bits, min_normal see OzakiMatMulExpander
bool expand_ozaki_matmuls(
    builder::StructuredSDFGBuilder& builder,
    analysis::AnalysisManager& analysis_manager,
    int max_bits = 0,
    double min_normal = 0
);

} // namespace fp_math
} // namespace sdfg
