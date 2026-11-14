#include "sdfg/passes/dataflow/matmul_batch_flattening.h"

#include <utility>

#include "sdfg/data_flow/library_nodes/math/tensor/matmul_node.h"
#include "sdfg/types/tensor.h"

namespace sdfg::passes {

namespace {

bool contiguous(const math::tensor::TensorLayout& layout) {
    symbolic::Expression stride = symbolic::one();
    for (int i = layout.dims() - 1; i >= 0; --i) {
        if (!symbolic::eq(layout.get_dim(i), symbolic::one()) && !symbolic::eq(layout.get_stride(i), stride)) {
            return false;
        }
        stride = symbolic::mul(stride, layout.get_dim(i));
    }
    return true;
}

} // namespace

bool MatMulBatchFlattening::accept(structured_control_flow::Block& block) {
    using math::tensor::MatMulNode;
    using math::tensor::TensorLayout;
    auto& dataflow = block.dataflow();
    bool applied = false;
    for (auto* library : dataflow.library_nodes()) {
        auto* node = dynamic_cast<MatMulNode*>(library);
        if (!node || node->code() != math::tensor::LibraryNodeType_MatMul ||
            std::as_const(*node).implementation_type() != data_flow::ImplementationType_NONE) {
            continue;
        }
        const auto& a = node->layout_a();
        const auto& b = node->layout_b();
        const auto& y = node->layout_y();
        if (a.dims() <= 2 || b.dims() != 2 || y.dims() != a.dims() || !contiguous(a) || !contiguous(y)) {
            continue;
        }
        bool compatible = true;
        symbolic::Expression rows = symbolic::one();
        for (int i = 0; i < a.dims() - 1; ++i) {
            compatible &= symbolic::eq(a.get_dim(i), y.get_dim(i));
            rows = symbolic::mul(rows, a.get_dim(i));
        }
        for (auto& edge : dataflow.in_edges(*node)) {
            // Tensor-coordinate subsets would also need remapping to the flattened rank.
            if (edge.base_type().type_id() == types::TypeID::Tensor && !edge.subset().empty()) {
                compatible = false;
            }
        }
        if (!compatible) {
            continue;
        }

        TensorLayout flat_a({rows, node->k()}, {}, a.offset());
        TensorLayout flat_y({rows, node->n()}, {}, y.offset());
        node->relocalize_operand(MatMulNode::A_INPUT_IDX, flat_a);
        node->relocalize_operand(MatMulNode::Y_INPUT_IDX, flat_y);
        for (auto& edge : dataflow.in_edges(*node)) {
            if (edge.dst_conn() == "B") {
                continue;
            }
            const auto& layout = edge.dst_conn() == "A" ? flat_a : flat_y;
            const auto& type = edge.base_type();
            if (auto* tensor = dynamic_cast<const types::Tensor*>(&type)) {
                edge.set_base_type(
                    types::Tensor(type.storage_type(), type.alignment(), type.initializer(), tensor->element_type(), layout)
                );
            }
        }
        applied = true;
    }
    if (applied) {
        analysis_manager_.invalidate_all();
    }
    return applied;
}

} // namespace sdfg::passes
