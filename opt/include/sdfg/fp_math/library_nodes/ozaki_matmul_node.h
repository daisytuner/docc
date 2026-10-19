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
 * `B0 ... B{s-1}` and the n scales `SB`, the absolute maxima of B's columns in A's type. A, Y follow their layouts.
 *
 * Inputs, in order: Y (written), A, SB, B0 ... B{s-1}.
 */
class OzakiMatMulNode : public data_flow::LibraryNode {
    math::tensor::TensorLayout layout_a_;
    math::tensor::TensorLayout layout_y_;
    types::PrimitiveType element_;
    int slices_;

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

class OzakiMatMulNodeSerializer : public serializer::LibraryNodeSerializer {
public:
    nlohmann::json serialize(const data_flow::LibraryNode& library_node) override;
    data_flow::LibraryNode& deserialize(
        const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
    ) override;
};

class OzakiMatMulExpander : public passes::CodeLibNodeExpander<OzakiMatMulNode> {
public:
    OzakiMatMulExpander() : CodeLibNodeExpander(LibraryNodeType_OzakiMatMul) {
    }

    using TypedLibNodeExpander<OzakiMatMulNode>::handle_expand;

    passes::LibNodeExpander::ExpandOutcome handle_expand(
        passes::LibNodeExpander::ExpandContext& context, structured_control_flow::Block& block, OzakiMatMulNode& node
    ) const override;
};

/// Expands every OzakiMatMul node of the SDFG into the slicing of A, the Int8 MatMuls and the rescaling; whether any.
bool expand_ozaki_matmuls(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager);

} // namespace fp_math
} // namespace sdfg
