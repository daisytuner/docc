#include "sdfg/passes/dataflow/view_copy_elimination.h"

#include <functional>
#include <optional>

#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/copy_node.h"
#include "sdfg/data_flow/library_nodes/stdlib/free.h"
#include "sdfg/data_flow/library_nodes/stdlib/malloc.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/if_else.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/structured_control_flow/while.h"
#include "sdfg/types/pointer.h"

namespace sdfg {
namespace passes {

namespace {

/// Whether any block under @p node accesses @p container.
bool accesses(structured_control_flow::ControlFlowNode& node, const std::string& container) {
    if (auto* block = dynamic_cast<structured_control_flow::Block*>(&node)) {
        for (auto* access : block->dataflow().data_nodes()) {
            if (access->data() == container) {
                return true;
            }
        }
        return false;
    }
    if (auto* sequence = dynamic_cast<structured_control_flow::Sequence*>(&node)) {
        for (size_t i = 0; i < sequence->size(); ++i) {
            if (accesses(sequence->at(i), container)) {
                return true;
            }
        }
        return false;
    }
    if (auto* loop = dynamic_cast<structured_control_flow::StructuredLoop*>(&node)) {
        return accesses(loop->root(), container);
    }
    if (auto* loop = dynamic_cast<structured_control_flow::While*>(&node)) {
        return accesses(loop->root(), container);
    }
    if (auto* branches = dynamic_cast<structured_control_flow::IfElse*>(&node)) {
        for (size_t i = 0; i < branches->size(); ++i) {
            if (accesses(branches->at(i).first, container)) {
                return true;
            }
        }
        return false;
    }
    return true;
}

/// The library node of type @p Node of a block that holds it and accesses only @p container, or null.
template<typename Node>
Node* only(structured_control_flow::ControlFlowNode& node, const std::string& container) {
    auto* block = dynamic_cast<structured_control_flow::Block*>(&node);
    if (!block) {
        return nullptr;
    }
    Node* found = nullptr;
    for (auto& dataflow_node : block->dataflow().nodes()) {
        if (auto* access = dynamic_cast<data_flow::AccessNode*>(&dataflow_node)) {
            if (access->data() != container) {
                return nullptr;
            }
        } else if (auto* match = dynamic_cast<Node*>(&dataflow_node); match && !found) {
            found = match;
        } else {
            return nullptr;
        }
    }
    return found;
}

/// Whether @p layout is the row-major order of its shape from element 0.
bool contiguous(const math::tensor::TensorLayout& layout) {
    if (!symbolic::eq(layout.offset(), symbolic::zero())) {
        return false;
    }
    symbolic::Expression stride = symbolic::one();
    for (int i = layout.dims() - 1; i >= 0; --i) {
        if (!symbolic::eq(layout.get_dim(i), symbolic::one()) && !symbolic::eq(layout.get_stride(i), stride)) {
            return false;
        }
        stride = symbolic::mul(stride, layout.get_dim(i));
    }
    return true;
}

struct Copy {
    std::string source;
    std::string target;
};

/// The transient a block copies whole into a result, contiguous on both sides.
std::optional<Copy> view_copy(const Function& function, structured_control_flow::ControlFlowNode& node) {
    auto* block = dynamic_cast<structured_control_flow::Block*>(&node);
    if (!block) {
        return std::nullopt;
    }
    auto& graph = block->dataflow();
    math::tensor::TensorCopyNode* copy = nullptr;
    for (auto& dataflow_node : graph.nodes()) {
        if (auto* library_node = dynamic_cast<math::tensor::TensorCopyNode*>(&dataflow_node); library_node && !copy) {
            copy = library_node;
        } else if (!dynamic_cast<data_flow::AccessNode*>(&dataflow_node) ||
                   dynamic_cast<data_flow::ConstantNode*>(&dataflow_node)) {
            return std::nullopt;
        }
    }
    if (!copy || !contiguous(copy->layout_x()) || !contiguous(copy->layout_y()) ||
        !symbolic::eq(copy->layout_x().total_elements(), copy->layout_y().total_elements())) {
        return std::nullopt;
    }
    // The copy reads X and writes through the pointer Y, both inputs.
    Copy result;
    for (auto& memlet : graph.in_edges(*copy)) {
        auto& data = static_cast<const data_flow::AccessNode&>(memlet.src()).data();
        if (memlet.dst_conn() == copy->inputs().at(math::tensor::TensorCopyNode::X_INPUT_IDX)) {
            result.source = data;
        } else if (memlet.dst_conn() == copy->inputs().at(math::tensor::TensorCopyNode::Y_INPUT_IDX)) {
            result.target = data;
        }
    }
    if (result.source.empty() || result.target.empty() || result.source == result.target ||
        !function.is_transient(result.source) ||
        !(function.is_argument(result.target) || function.is_external(result.target))) {
        return std::nullopt;
    }
    auto* source = dynamic_cast<const types::Pointer*>(&function.type(result.source));
    auto* target = dynamic_cast<const types::Pointer*>(&function.type(result.target));
    if (!source || !target || !source->has_pointee_type() || !target->has_pointee_type() ||
        source->pointee_type().primitive_type() != target->pointee_type().primitive_type()) {
        return std::nullopt;
    }
    return result;
}

} // namespace

std::string ViewCopyElimination::name() {
    return "ViewCopyElimination";
}

bool ViewCopyElimination::run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();
    bool changed = false;
    size_t position = 0;
    while (position < root.size()) {
        auto copy = view_copy(sdfg, root.at(position));
        std::optional<size_t> allocation, free;
        bool eliminable = copy.has_value();
        for (size_t i = 0; i < root.size() && eliminable; ++i) {
            auto& child = root.at(i);
            if (i == position) {
                continue;
            }
            if (i < position && accesses(child, copy->target)) {
                eliminable = false;
            } else if (i < position && only<stdlib::MallocNode>(child, copy->source)) {
                eliminable = !allocation;
                allocation = i;
            } else if (i > position && accesses(child, copy->source)) {
                eliminable = !free && only<stdlib::FreeNode>(child, copy->source);
                free = i;
            }
        }
        if (!eliminable || !allocation) {
            ++position;
            continue;
        }
        for (size_t i = 0; i < position; ++i) {
            if (i != *allocation) {
                root.at(i).replace(symbolic::symbol(copy->source), symbolic::symbol(copy->target));
            }
        }
        if (free) {
            builder.remove_child(root, *free);
        }
        builder.remove_child(root, position);
        builder.remove_child(root, *allocation);
        builder.remove_container(copy->source);
        changed = true;
        position = 0;
    }
    if (changed) {
        analysis_manager.invalidate_all();
    }
    return changed;
}

} // namespace passes
} // namespace sdfg
