#pragma once

#include <sdfg/data_flow/library_node.h>
#include <sdfg/data_flow/library_nodes/math/tensor/tensor_layout.h>

namespace sdfg {
namespace axelera {

inline data_flow::LibraryNodeCode LibraryNodeType_AxeleraMatMul("axelera.MatMul");
inline data_flow::LibraryNodeCode LibraryNodeType_AxeleraImcLoad("axelera.ImcLoad");

/**
 * `Y = A @ B` on the MVM of an AI core: A (M x K) and B (K x N) are int8, Y (M x N) is float32 and the products
 * accumulate exactly in int32. Every operand is a bare pointer addressed through its layout (shape outermost first),
 * so the tile transformations can repoint it at a staged buffer.
 */
class AxeleraMatMulNode : public data_flow::LibraryNode {
    math::tensor::TensorLayout layout_a_;
    math::tensor::TensorLayout layout_b_;
    math::tensor::TensorLayout layout_y_;

public:
    static constexpr int Y_INPUT_IDX = 0;
    static constexpr int A_INPUT_IDX = 1;
    static constexpr int B_INPUT_IDX = 2;

    AxeleraMatMulNode(
        size_t element_id,
        const DebugInfo& debug_info,
        const graph::Vertex vertex,
        data_flow::DataFlowGraph& parent,
        const math::tensor::TensorLayout& layout_a,
        const math::tensor::TensorLayout& layout_b,
        const math::tensor::TensorLayout& layout_y
    );

    const math::tensor::TensorLayout& layout_a() const {
        return layout_a_;
    }
    const math::tensor::TensorLayout& layout_b() const {
        return layout_b_;
    }
    const math::tensor::TensorLayout& layout_y() const {
        return layout_y_;
    }

    void validate(const Function& function) const override;
    symbolic::SymbolSet symbols() const override;
    void replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) override;
    void replace(const symbolic::ExpressionMapping& replacements) override;
    std::unique_ptr<data_flow::DataFlowNode>
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const override;
    std::string toStr() const override;

    data_flow::PointerAccessType pointer_access_type(int input_idx) const override;
    bool relocalize_operand(int input_idx, const math::tensor::TensorLayout& packed) override;
    bool can_relocalize_operand(int input_idx, const math::tensor::TensorLayout& packed) const override;
};

/// Makes the K x N int8 matrix at `_src` (in L1, addressed through @ref layout) resident in the IMC buffer `_imc`.
class AxeleraImcLoadNode : public data_flow::LibraryNode {
    math::tensor::TensorLayout layout_;

public:
    static constexpr int IMC_INPUT_IDX = 0;
    static constexpr int SRC_INPUT_IDX = 1;

    AxeleraImcLoadNode(
        size_t element_id,
        const DebugInfo& debug_info,
        const graph::Vertex vertex,
        data_flow::DataFlowGraph& parent,
        const math::tensor::TensorLayout& layout
    );

    const math::tensor::TensorLayout& layout() const {
        return layout_;
    }

    void validate(const Function& function) const override;
    symbolic::SymbolSet symbols() const override;
    void replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) override;
    void replace(const symbolic::ExpressionMapping& replacements) override;
    std::unique_ptr<data_flow::DataFlowNode>
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const override;
    std::string toStr() const override;

    data_flow::PointerAccessType pointer_access_type(int input_idx) const override;
};

} // namespace axelera
} // namespace sdfg
