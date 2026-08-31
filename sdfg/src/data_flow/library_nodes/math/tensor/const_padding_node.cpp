#include "sdfg/data_flow/library_nodes/math/tensor/const_padding_node.h"

#include <cstddef>
#include <memory>
#include <sstream>
#include <string>

#include <nlohmann/json_fwd.hpp>

#include "sdfg/data_flow/data_flow_graph.h"
#include "sdfg/data_flow/data_flow_node.h"
#include "sdfg/data_flow/library_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/tensor_layout.h"
#include "sdfg/data_flow/library_nodes/math/tensor/tensor_node.h"
#include "sdfg/data_flow/memlet.h"
#include "sdfg/data_flow/pointer_metadata.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/element.h"
#include "sdfg/exceptions.h"
#include "sdfg/function.h"
#include "sdfg/graph/graph.h"
#include "sdfg/passes/expansion/lib_node_expander.h"
#include "sdfg/serializer/json_serializer.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/scalar.h"
#include "sdfg/types/tensor.h"
#include "sdfg/types/type.h"
#include "symengine/symengine_rcp.h"

namespace sdfg {
namespace math {
namespace tensor {

ConstPaddingNode::ConstPaddingNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const symbolic::MultiExpression& pads,
    const TensorLayout& y_layout,
    const TensorLayout& x_layout,
    const data_flow::ImplementationType& impl_type
)
    : TensorNode(element_id, debug_info, vertex, parent, LibraryNodeType_ConstPadding, {}, {"_y", "_x", "_val"}, impl_type),
      pads_(pads), y_layout_(y_layout), x_layout_(x_layout) {}

const symbolic::MultiExpression& ConstPaddingNode::pads() const { return this->pads_; }

const symbolic::Expression& ConstPaddingNode::get_lower_pad(int index) const { return this->pads_.at(2 * index); }

const symbolic::Expression& ConstPaddingNode::get_upper_pad(int index) const { return this->pads_.at(2 * index + 1); }

const TensorLayout& ConstPaddingNode::y_layout() const { return this->y_layout_; }

const TensorLayout& ConstPaddingNode::x_layout() const { return this->x_layout_; }

void ConstPaddingNode::validate(const Function& function) const {
    auto& graph = this->get_parent();
    TensorNode::validate(function);

    // Check presence of in and out edges
    if (graph.out_degree(*this) != 0) {
        throw InvalidSDFGException(
            "ConstPaddingNode: Expected no outputs but got: " + std::to_string(graph.out_degree(*this))
        );
    }
    if (graph.in_degree(*this) != 3) {
        throw InvalidSDFGException("ConstPaddingNode: Expected 3 input but got: " + std::to_string(graph.in_degree(*this)));
    }
    const data_flow::Memlet* y_edge = graph.in_edge_for_connector(*this, "_y");
    if (!y_edge) {
        throw InvalidSDFGException("ConstPaddingNode: No memlet connected at connector: _y");
    }
    const data_flow::Memlet* x_edge = graph.in_edge_for_connector(*this, "_x");
    if (!x_edge) {
        throw InvalidSDFGException("ConstPaddingNode: No memlet connected at connector: _x");
    }
    const data_flow::Memlet* val_edge = graph.in_edge_for_connector(*this, "_val");
    if (!val_edge) {
        throw InvalidSDFGException("ConstPaddingNode: No memlet connected at connector: _val");
    }

    // Check that the input edges have tensors as base types
    if (y_edge->base_type().type_id() != types::TypeID::Tensor) {
        throw InvalidSDFGException(
            "ConstPaddingNode: Expected tensor type at connector '_y' but got: " + y_edge->base_type().print()
        );
    }
    if (x_edge->base_type().type_id() != types::TypeID::Tensor) {
        throw InvalidSDFGException(
            "ConstPaddingNode: Expected tensor type at connector '_x' but got: " + x_edge->base_type().print()
        );
    }
    const auto& y_tensor = static_cast<const types::Tensor&>(y_edge->base_type());
    const auto& x_tensor = static_cast<const types::Tensor&>(x_edge->base_type());

    // Check that the tensor layouts match with the tensor types on the edges
    if (y_tensor.layout() != this->y_layout_) {
        throw InvalidSDFGException(
            "ConstPaddingNode: Provided tensor layout does not match the memlet tensor type for connector '_y': " +
            y_tensor.layout().toStr() + " != " + this->y_layout_.toStr()
        );
    }
    if (x_tensor.layout() != this->x_layout_) {
        throw InvalidSDFGException(
            "ConstPaddingNode: Provided tensor layout does not match the memlet tensor type for connector '_x': " +
            x_tensor.layout().toStr() + " != " + this->x_layout_.toStr()
        );
    }

    // Check shapes
    if (this->x_layout_.dims() != this->y_layout_.dims()) {
        throw InvalidSDFGException(
            "ConstPaddingNode: Layout dimensions do not match: " + std::to_string(this->x_layout_.dims()) +
            " != " + std::to_string(this->y_layout_.dims())
        );
    }
    if (this->pads_.size() % 2 != 0) {
        throw InvalidSDFGException(
            "ConstPaddingNode: Pads are not divisable by two: " + std::to_string(this->pads_.size())
        );
    }
    int num_pads = this->pads_.size() / 2;

    // Check that pads applied to x.shape -> y.shape
    symbolic::MultiExpression padded(this->y_layout_.dims(), SymEngine::null);
    int offset = this->y_layout_.dims() - num_pads;
    for (int i = 0; i < offset; i++) {
        padded[i] = this->x_layout_.get_dim(i);
    }
    for (int i = 0; i < num_pads; i++) {
        padded[offset + i] = SymEngine::add(
            {this->x_layout_.get_dim(offset + i),
             this->get_lower_pad(num_pads - i - 1),
             this->get_upper_pad(num_pads - i - 1)}
        );
    }
    TensorLayout dummy(padded, this->y_layout_.strides(), this->y_layout_.offset());
    if (dummy != this->y_layout_) {
        throw InvalidSDFGException(
            "ConstPaddingNode: Pads applied to input shape mismatches output shape: " + dummy.toStr() +
            " != " + this->y_layout_.toStr()
        );
    }
}

bool ConstPaddingNode::supports_integer_types() const { return true; }

using Dir = passes::LibNodeExpander::InputUse;

passes::LibNodeExpander::ExpandOutcome ConstPaddingNode::
    expand(passes::LibNodeExpander::ExpandContext& context, structured_control_flow::Block& block) {
    auto standalone = context.replacement_requires_access_nodes({Dir::IndirectWrite, Dir::IndirectRead, Dir::Scalar});
    if (!standalone) {
        return context.unable();
    }

    auto& new_sequence = standalone->replace_with_sequence();
    auto& builder = standalone->builder();

    auto& graph = this->get_parent();
    const auto* y_edge = graph.in_edge_for_connector(*this, "_y");
    const auto* x_edge = graph.in_edge_for_connector(*this, "_x");
    const auto* val_edge = graph.in_edge_for_connector(*this, "_val");
    auto prim_type = this->primitive_type(graph);
    types::Scalar base_type(prim_type);
    types::Scalar indvar_type(types::PrimitiveType::Int64);
    int num_pads = this->pads_.size() / 2;
    int pad_offset = this->y_layout_.dims() - num_pads;

    // Add map nest over y shape
    structured_control_flow::Sequence* current_seq = &new_sequence;
    int dims = this->y_layout_.dims();
    data_flow::Subset x_subset, y_subset;
    x_subset.reserve(dims);
    y_subset.reserve(dims);
    symbolic::Condition copy_condition = symbolic::__true__();
    for (int i = 0; i < dims; i++) {
        auto indvar_container = builder.find_new_name("_i");
        builder.add_container(indvar_container, indvar_type);
        auto indvar = symbolic::symbol(indvar_container);
        const auto& bound = this->y_layout_.get_dim(i);

        y_subset.push_back(indvar);
        if (i >= pad_offset) {
            auto lower = this->get_lower_pad(dims - i - 1);
            if (!symbolic::eq(lower, symbolic::zero())) {
                copy_condition = symbolic::And(copy_condition, symbolic::Ge(indvar, lower));
            }
            x_subset.push_back(symbolic::sub(indvar, lower));
            auto upper = this->get_upper_pad(dims - i - 1);
            if (!symbolic::eq(upper, symbolic::zero())) {
                copy_condition = symbolic::And(copy_condition, symbolic::Lt(indvar, symbolic::sub(bound, upper)));
            }
        } else {
            x_subset.push_back(indvar);
        }

        auto& map = builder.add_map(
            *current_seq,
            indvar,
            symbolic::Lt(indvar, bound),
            symbolic::zero(),
            symbolic::add(indvar, symbolic::one()),
            structured_control_flow::ScheduleType_Sequential::create(),
            this->debug_info_
        );
        current_seq = &map.root();
    }

    // Create branching
    auto& branch = builder.add_if_else(*current_seq, this->debug_info_);
    auto& copy_case = builder.add_case(branch, symbolic::Eq(copy_condition, symbolic::__true__()), this->debug_info_);
    auto& fill_case = builder.add_case(branch, symbolic::Eq(copy_condition, symbolic::__false__()), this->debug_info_);

    // Create copy case
    {
        auto& block = builder.add_block(copy_case, this->debug_info_);
        auto& x_access = standalone->add_indirect_read_access(block, X_INPUT_IDX);
        auto& y_access = standalone->add_indirect_write_access(block, Y_INPUT_IDX);
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"}, this->debug_info_);
        builder.add_computational_memlet(
            block, x_access, tasklet, "_in", x_subset, x_edge->base_type(), x_edge->debug_info()
        );
        builder.add_computational_memlet(
            block, tasklet, "_out", y_access, y_subset, y_edge->base_type(), y_edge->debug_info()
        );
    }

    // Create fill case
    {
        auto& block = builder.add_block(fill_case, this->debug_info_);
        auto& val_access = standalone->add_scalar_input_access(block, VAL_INPUT_IDX);
        auto& y_access = standalone->add_indirect_write_access(block, Y_INPUT_IDX);
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"}, this->debug_info_);
        builder.add_computational_memlet(
            block, val_access, tasklet, "_in", {}, val_edge->base_type(), val_edge->debug_info()
        );
        builder.add_computational_memlet(
            block, tasklet, "_out", y_access, y_subset, y_edge->base_type(), y_edge->debug_info()
        );
    }

    return standalone->successfully_expanded();
}

std::string ConstPaddingNode::toStr() const {
    std::stringstream stream;
    stream << "ConstPadding(pads: [";
    for (long long i = 0; i < this->pads_.size(); i++) {
        if (i > 0) {
            stream << ",";
        }
        stream << this->pads_.at(i)->__str__();
    }
    stream << "], y_layout: " << this->y_layout_ << ", x_layout: " << this->x_layout_ << ")";
    return stream.str();
}

symbolic::SymbolSet ConstPaddingNode::symbols() const {
    symbolic::SymbolSet syms;
    for (const auto& pad : this->pads_) {
        for (auto sym : symbolic::atoms(pad)) {
            syms.insert(sym);
        }
    }
    this->y_layout_.collect_symbols(syms);
    this->x_layout_.collect_symbols(syms);
    return syms;
}

symbolic::Expression ConstPaddingNode::flop() const { return symbolic::zero(); }

data_flow::PointerAccessType ConstPaddingNode::pointer_access_type(int input_idx) const {
    switch (input_idx) {
        case Y_INPUT_IDX:
            return data_flow::PointerAccessMeta::create_full_write_only(this->y_layout_.total_elements(), true);
        case X_INPUT_IDX:
            return data_flow::PointerAccessMeta::create_read_only(this->x_layout_.total_elements(), true);
        default:
            return nullptr;
    }
}

std::unique_ptr<data_flow::DataFlowNode> ConstPaddingNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::make_unique<ConstPaddingNode>(
        element_id,
        this->debug_info_,
        vertex,
        parent,
        this->pads_,
        this->y_layout_,
        this->x_layout_,
        this->implementation_type_
    );
}

void ConstPaddingNode::replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) {
    for (long long i = 0; i < this->pads_.size(); i++) {
        this->pads_[i] = symbolic::subs(this->pads_[i], old_expression, new_expression);
    }
    this->y_layout_.replace_symbols(old_expression, new_expression);
    this->x_layout_.replace_symbols(old_expression, new_expression);
}

void ConstPaddingNode::replace(const symbolic::ExpressionMapping& replacements) {
    for (long long i = 0; i < this->pads_.size(); i++) {
        this->pads_[i] = symbolic::subs(this->pads_[i], replacements);
    }
    this->y_layout_.replace_symbols(replacements);
    this->x_layout_.replace_symbols(replacements);
}

nlohmann::json ConstPaddingNodeSerializer::serialize(const data_flow::LibraryNode& library_node) {
    const auto& const_paddding_node = static_cast<const ConstPaddingNode&>(library_node);
    nlohmann::json j;
    serializer::JSONSerializer serializer;

    j["code"] = const_paddding_node.code().value();
    j["pads"] = nlohmann::json::array();
    for (const auto& pad : const_paddding_node.pads()) {
        j["pads"].push_back(serializer.expression(pad));
    }
    const_paddding_node.y_layout().serialize_to_json(j["y_layout"]);
    const_paddding_node.x_layout().serialize_to_json(j["x_layout"]);

    return j;
}

data_flow::LibraryNode& ConstPaddingNodeSerializer::deserialize(
    const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
) {
    assert(j.contains("element_id"));
    assert(j.contains("code"));
    assert(j.contains("debug_info"));
    assert(j.contains("pads"));
    assert(j.contains("y_layout"));
    assert(j.contains("x_layout"));
    serializer::JSONSerializer serializer;

    DebugInfo debug_info = serializer.json_to_debug_info(j.at("debug_info"));
    symbolic::MultiExpression pads;
    for (const auto& pad : j.at("pads")) {
        pads.push_back(serializer.json_to_expr(pad));
    }
    TensorLayout y_layout = TensorLayout::deserialize_from_json(j.at("y_layout"));
    TensorLayout x_layout = TensorLayout::deserialize_from_json(j.at("x_layout"));

    return builder.add_library_node<ConstPaddingNode>(parent, debug_info, pads, y_layout, x_layout);
}

} // namespace tensor
} // namespace math
} // namespace sdfg
