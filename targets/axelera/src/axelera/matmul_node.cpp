#include "docc/target/axelera/matmul_node.h"

#include <sdfg/data_flow/data_flow_graph.h>
#include <sdfg/exceptions.h>

namespace sdfg {
namespace axelera {

namespace {

void expect_operand(
    const data_flow::LibraryNode& node,
    const std::string& connector,
    types::PrimitiveType expected,
    const std::string& what
) {
    auto* edge = node.get_parent().in_edge_for_connector(node, connector);
    if (!edge) {
        throw InvalidSDFGException(what + ": missing operand " + connector);
    }
    auto actual = edge->base_type().primitive_type();
    if (actual != expected) {
        throw InvalidSDFGException(
            what + ": operand " + connector + " must be " + types::primitive_type_to_string(expected) + ", got " +
            types::primitive_type_to_string(actual)
        );
    }
}

} // namespace

AxeleraMatMulNode::AxeleraMatMulNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const math::tensor::TensorLayout& layout_a,
    const math::tensor::TensorLayout& layout_b,
    const math::tensor::TensorLayout& layout_y
)
    : LibraryNode(
          element_id,
          debug_info,
          vertex,
          parent,
          LibraryNodeType_AxeleraMatMul,
          {},
          {"Y", "A", "B"},
          false,
          data_flow::ImplementationType_NONE
      ),
      layout_a_(layout_a), layout_b_(layout_b), layout_y_(layout_y) {
    if (layout_a.dims() != 2 || layout_b.dims() != 2 || layout_y.dims() != 2) {
        throw std::invalid_argument("AxeleraMatMulNode: operands must be 2-dimensional");
    }
}

void AxeleraMatMulNode::validate(const Function& function) const {
    expect_operand(*this, "A", types::PrimitiveType::Int8, "AxeleraMatMulNode");
    expect_operand(*this, "B", types::PrimitiveType::Int8, "AxeleraMatMulNode");
    expect_operand(*this, "Y", types::PrimitiveType::Float, "AxeleraMatMulNode");
    if (!symbolic::eq(layout_a_.get_dim(1), layout_b_.get_dim(0)) ||
        !symbolic::eq(layout_a_.get_dim(0), layout_y_.get_dim(0)) ||
        !symbolic::eq(layout_b_.get_dim(1), layout_y_.get_dim(1))) {
        throw InvalidSDFGException("AxeleraMatMulNode: operand shapes do not form a matrix product");
    }
}

symbolic::SymbolSet AxeleraMatMulNode::symbols() const {
    symbolic::SymbolSet symbols;
    layout_a_.collect_symbols(symbols);
    layout_b_.collect_symbols(symbols);
    layout_y_.collect_symbols(symbols);
    return symbols;
}

void AxeleraMatMulNode::replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) {
    layout_a_.replace_symbols(old_expression, new_expression);
    layout_b_.replace_symbols(old_expression, new_expression);
    layout_y_.replace_symbols(old_expression, new_expression);
}

void AxeleraMatMulNode::replace(const symbolic::ExpressionMapping& replacements) {
    layout_a_.replace_symbols(replacements);
    layout_b_.replace_symbols(replacements);
    layout_y_.replace_symbols(replacements);
}

std::unique_ptr<data_flow::DataFlowNode> AxeleraMatMulNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::make_unique<AxeleraMatMulNode>(element_id, debug_info(), vertex, parent, layout_a_, layout_b_, layout_y_);
}

std::string AxeleraMatMulNode::toStr() const {
    return "AxeleraMatMul(A: " + layout_a_.toStr() + ", B: " + layout_b_.toStr() + ", Y: " + layout_y_.toStr() + ")";
}

data_flow::PointerAccessType AxeleraMatMulNode::pointer_access_type(int input_idx) const {
    switch (input_idx) {
        case Y_INPUT_IDX:
            return data_flow::PointerAccessMeta::create_full_write_only(layout_y_.total_elements(), true, layout_y_);
        case A_INPUT_IDX:
            return data_flow::PointerAccessMeta::create_read_only(layout_a_.total_elements(), true, layout_a_);
        case B_INPUT_IDX:
            return data_flow::PointerAccessMeta::create_read_only(layout_b_.total_elements(), true, layout_b_);
        default:
            return nullptr;
    }
}

bool AxeleraMatMulNode::can_relocalize_operand(int input_idx, const math::tensor::TensorLayout& packed) const {
    return input_idx >= Y_INPUT_IDX && input_idx <= B_INPUT_IDX && packed.dims() == 2;
}

bool AxeleraMatMulNode::relocalize_operand(int input_idx, const math::tensor::TensorLayout& packed) {
    if (!can_relocalize_operand(input_idx, packed)) {
        return false;
    }
    (input_idx == Y_INPUT_IDX ? layout_y_ : input_idx == A_INPUT_IDX ? layout_a_ : layout_b_) = packed;
    return true;
}

AxeleraImcLoadNode::AxeleraImcLoadNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const math::tensor::TensorLayout& layout
)
    : LibraryNode(
          element_id,
          debug_info,
          vertex,
          parent,
          LibraryNodeType_AxeleraImcLoad,
          {},
          {"_imc", "_src"},
          false,
          data_flow::ImplementationType_NONE
      ),
      layout_(layout) {
    if (layout.dims() != 2) {
        throw std::invalid_argument("AxeleraImcLoadNode: the matrix must be 2-dimensional");
    }
}

void AxeleraImcLoadNode::validate(const Function& function) const {
    expect_operand(*this, "_imc", types::PrimitiveType::Int8, "AxeleraImcLoadNode");
    expect_operand(*this, "_src", types::PrimitiveType::Int8, "AxeleraImcLoadNode");
}

symbolic::SymbolSet AxeleraImcLoadNode::symbols() const {
    symbolic::SymbolSet symbols;
    layout_.collect_symbols(symbols);
    return symbols;
}

void AxeleraImcLoadNode::replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) {
    layout_.replace_symbols(old_expression, new_expression);
}

void AxeleraImcLoadNode::replace(const symbolic::ExpressionMapping& replacements) {
    layout_.replace_symbols(replacements);
}

std::unique_ptr<data_flow::DataFlowNode> AxeleraImcLoadNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::make_unique<AxeleraImcLoadNode>(element_id, debug_info(), vertex, parent, layout_);
}

std::string AxeleraImcLoadNode::toStr() const {
    return "AxeleraImcLoad(" + layout_.toStr() + ")";
}

data_flow::PointerAccessType AxeleraImcLoadNode::pointer_access_type(int input_idx) const {
    switch (input_idx) {
        case IMC_INPUT_IDX:
            return data_flow::PointerAccessMeta::create_full_write_only(layout_.total_elements(), true);
        case SRC_INPUT_IDX:
            return data_flow::PointerAccessMeta::create_read_only(layout_.total_elements(), true, layout_);
        default:
            return nullptr;
    }
}

} // namespace axelera
} // namespace sdfg
