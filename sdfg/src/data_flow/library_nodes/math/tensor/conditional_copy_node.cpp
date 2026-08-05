#include "sdfg/data_flow/library_nodes/math/tensor/conditional_copy_node.h"

#include <cstddef>
#include <memory>
#include <nlohmann/json_fwd.hpp>
#include <sstream>
#include <string>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/data_flow_graph.h"
#include "sdfg/data_flow/data_flow_node.h"
#include "sdfg/data_flow/library_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/tensor_layout.h"
#include "sdfg/data_flow/library_nodes/math/tensor/tensor_node.h"
#include "sdfg/data_flow/memlet.h"
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

namespace sdfg {
namespace math {
namespace tensor {

void ConditionalTensorCopyNode::validate_equal_shapes(const TensorLayout& layout1, const TensorLayout& layout2) const {
    int dims = layout1.dims();
    if (dims == 0) {
        return;
    }
    if (dims != layout2.dims()) {
        throw InvalidSDFGException(
            "ConditionalTensorCopyNode: Shapes mismatch: " + layout1.toStr() + " != " + layout2.toStr()
        );
    }
    for (int i = 0; i < dims; i++) {
        if (!symbolic::eq(layout1.get_dim(i), layout2.get_dim(i)) &&
            !symbolic::eq(layout1.get_dim(i), symbolic::one())) {
            throw InvalidSDFGException(
                "ConditionalTensorCopyNode: Shapes mismatch: " + layout1.toStr() + " != " + layout2.toStr()
            );
        }
    }
}

ConditionalTensorCopyNode::ConditionalTensorCopyNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const TensorLayout& layout_mask,
    const TensorLayout& layout_x1,
    const TensorLayout& layout_x2,
    const TensorLayout& layout_y,
    const data_flow::ImplementationType& impl_type
)
    : TensorNode(
          element_id,
          debug_info,
          vertex,
          parent,
          LibraryNodeType_ConditionalTensorCopy,
          {},
          {"Mask", "X1", "X2", "Y"},
          impl_type
      ),
      layout_mask_(layout_mask), layout_x1_(layout_x1), layout_x2_(layout_x2), layout_y_(layout_y) {}


const TensorLayout& ConditionalTensorCopyNode::layout_mask() const { return this->layout_mask_; }

const TensorLayout& ConditionalTensorCopyNode::layout_x1() const { return this->layout_x1_; }

const TensorLayout& ConditionalTensorCopyNode::layout_x2() const { return this->layout_x2_; }

const TensorLayout& ConditionalTensorCopyNode::layout_y() const { return this->layout_y_; }

void ConditionalTensorCopyNode::validate(const Function& function) const {
    auto& graph = this->get_parent();

    // Check presence of in and out edges
    if (graph.out_degree(*this) != 0) {
        throw InvalidSDFGException(
            "ConditionalTensorCopyNode: Expected no outputs but got: " + std::to_string(graph.out_degree(*this))
        );
    }
    const data_flow::Memlet* mask_iedge = graph.in_edge_for_connector(*this, "Mask");
    if (!mask_iedge) {
        throw InvalidSDFGException("ConditionalTensorCopyNode: No memlet connected at connector: Mask");
    }
    const data_flow::Memlet* x1_iedge = graph.in_edge_for_connector(*this, "X1");
    if (!x1_iedge) {
        throw InvalidSDFGException("ConditionalTensorCopyNode: No memlet connected at connector: X1");
    }
    const data_flow::Memlet* x2_iedge = graph.in_edge_for_connector(*this, "X2");
    if (!x2_iedge) {
        throw InvalidSDFGException("ConditionalTensorCopyNode: No memlet connected at connector: X2");
    }
    const data_flow::Memlet* y_iedge = graph.in_edge_for_connector(*this, "Y");
    if (!y_iedge) {
        throw InvalidSDFGException("ConditionalTensorCopyNode: No memlet connected at connector: Y");
    }

    // Check that the in edges have tensor types as base types
    if (mask_iedge->base_type().type_id() != types::TypeID::Tensor) {
        throw InvalidSDFGException(
            "ConditionalTensorCopyNode: Expected tensor type at connector 'Mask' but got: " +
            mask_iedge->base_type().print()
        );
    }
    if (x1_iedge->base_type().type_id() != types::TypeID::Tensor) {
        throw InvalidSDFGException(
            "ConditionalTensorCopyNode: Expected tensor type at connector 'X1' but got: " +
            x1_iedge->base_type().print()
        );
    }
    if (x2_iedge->base_type().type_id() != types::TypeID::Tensor) {
        throw InvalidSDFGException(
            "ConditionalTensorCopyNode: Expected tensor type at connector 'X2' but got: " +
            x2_iedge->base_type().print()
        );
    }
    if (y_iedge->base_type().type_id() != types::TypeID::Tensor) {
        throw InvalidSDFGException(
            "ConditionalTensorCopyNode: Expected tensor type at connector 'Y' but got: " + y_iedge->base_type().print()
        );
    }

    const types::Tensor& mask_tensor = static_cast<const types::Tensor&>(mask_iedge->base_type());
    const types::Tensor& x1_tensor = static_cast<const types::Tensor&>(x1_iedge->base_type());
    const types::Tensor& x2_tensor = static_cast<const types::Tensor&>(x2_iedge->base_type());
    const types::Tensor& y_tensor = static_cast<const types::Tensor&>(y_iedge->base_type());

    // Check that the tensor layouts match with the tensor types on the edges
    if (mask_tensor.layout() != this->layout_mask_) {
        throw InvalidSDFGException(
            "ConditionalTensorCopyNode: Provided tensor layout does not match the memlet tensor type for connector "
            "'Mask': " +
            mask_tensor.layout().toStr() + " != " + this->layout_mask_.toStr()
        );
    }
    if (x1_tensor.layout() != this->layout_x1_) {
        throw InvalidSDFGException(
            "ConditionalTensorCopyNode: Provided tensor layout does not match the memlet tensor type for connector "
            "'X1': " +
            x1_tensor.layout().toStr() + " != " + this->layout_x1_.toStr()
        );
    }
    if (x2_tensor.layout() != this->layout_x2_) {
        throw InvalidSDFGException(
            "ConditionalTensorCopyNode: Provided tensor layout does not match the memlet tensor type for connector "
            "'X2': " +
            x2_tensor.layout().toStr() + " != " + this->layout_x2_.toStr()
        );
    }
    if (y_tensor.layout() != this->layout_y_) {
        throw InvalidSDFGException(
            "ConditionalTensorCopyNode: Provided tensor layout does not match the memlet tensor type for connector "
            "'Y': " +
            y_tensor.layout().toStr() + " != " + this->layout_y_.toStr()
        );
    }

    // Check that all tensor layouts have the same shape
    this->validate_equal_shapes(this->layout_mask_, this->layout_y_);
    this->validate_equal_shapes(this->layout_x1_, this->layout_y_);
    this->validate_equal_shapes(this->layout_x2_, this->layout_y_);

    // Check that the tensor element type of the memlet for connector Mask is bool
    if (mask_tensor.primitive_type() != types::PrimitiveType::Bool) {
        throw InvalidSDFGException(
            "ConditionalTensorCopyNode: Expected a boolean element type for tensor type at connector 'Mask' but got: " +
            mask_tensor.element_type().print()
        );
    }

    // Check that the other tensor have the same element types
    types::PrimitiveType prim = y_tensor.primitive_type();
    if (x1_tensor.primitive_type() != prim || x2_tensor.primitive_type() != prim) {
        throw InvalidSDFGException(
            "ConditionalTensorCopyNode: Expected the same primitive types but got: " +
            x1_tensor.element_type().print() + ", " + x2_tensor.element_type().print() + ", and " +
            y_tensor.element_type().print()
        );
    }
}

bool ConditionalTensorCopyNode::supports_integer_types() const { return true; }

using Dir = passes::LibNodeExpander::InputUse;

passes::LibNodeExpander::ExpandOutcome ConditionalTensorCopyNode::
    expand(passes::LibNodeExpander::ExpandContext& context, structured_control_flow::Block& block) {
    auto standalone =
        context
            .replacement_requires_access_nodes({Dir::IndirectRead, Dir::IndirectRead, Dir::IndirectRead, Dir::IndirectWrite}
            );

    if (!standalone) {
        return context.unable();
    }

    auto& builder = standalone->builder();
    auto& dfg = this->get_parent();
    const auto* iedge_mask = dfg.in_edge_for_connector(*this, "Mask");
    if (!iedge_mask) {
        throw InvalidSDFGException("ConditionalTensorCopyNode: Cannot get in edge for connector 'Mask'");
    }
    const auto* iedge_x1 = dfg.in_edge_for_connector(*this, "X1");
    if (!iedge_x1) {
        throw InvalidSDFGException("ConditionalTensorCopyNode: Cannot get in edge for connector 'X1'");
    }
    const auto* iedge_x2 = dfg.in_edge_for_connector(*this, "X2");
    if (!iedge_x2) {
        throw InvalidSDFGException("ConditionalTensorCopyNode: Cannot get in edge for connector 'X2'");
    }
    const auto* iedge_y = dfg.in_edge_for_connector(*this, "Y");
    if (!iedge_y) {
        throw InvalidSDFGException("ConditionalTensorCopyNode: Cannot get in edge for connector 'Y'");
    }

    // Add a graph after the current block
    auto& new_sequence = standalone->replace_with_sequence();

    // Add map nest over shape
    types::Scalar indvar_type(types::PrimitiveType::UInt64);
    structured_control_flow::Sequence* current_seq = &new_sequence;
    data_flow::Subset subset;
    subset.reserve(this->layout_y_.dims());
    for (auto dim : this->layout_y_.shape()) {
        auto indvar_container = builder.find_new_name("_i");
        builder.add_container(indvar_container, indvar_type);
        auto indvar = symbolic::symbol(indvar_container);
        subset.push_back(indvar);
        auto& map = builder.add_map(
            *current_seq,
            indvar,
            symbolic::Lt(indvar, dim),
            symbolic::zero(),
            symbolic::add(indvar, symbolic::one()),
            structured_control_flow::ScheduleType_Sequential::create(),
            this->debug_info_
        );
        current_seq = &map.root();
    }

    // Load local variable from mask
    int mask_dims = this->layout_mask_.dims();
    data_flow::Subset mask_subset;
    mask_subset.reserve(mask_dims);
    for (int i = 0; i < mask_dims; i++) {
        if (symbolic::eq(this->layout_mask_.get_dim(i), symbolic::one())) {
            mask_subset.push_back(symbolic::zero());
        } else {
            mask_subset.push_back(subset[i]);
        }
    }
    auto cond_copy_container = builder.find_new_name("tmp_cond_copy_");
    types::Scalar bool_type(types::PrimitiveType::Bool);
    builder.add_container(cond_copy_container, bool_type);
    auto& mask_block = builder.add_block(*current_seq, this->debug_info_);
    {
        auto& mask_access = standalone->add_scalar_input_access(mask_block, MASK_INPUT_IDX);
        auto& cond_copy_access = builder.add_access(mask_block, cond_copy_container, this->debug_info_);
        auto& tasklet =
            builder.add_tasklet(mask_block, data_flow::TaskletCode::assign, "_out", {"_in"}, this->debug_info_);
        builder.add_computational_memlet(
            mask_block, mask_access, tasklet, "_in", mask_subset, iedge_mask->base_type(), iedge_mask->debug_info()
        );
        builder.add_computational_memlet(mask_block, tasklet, "_out", cond_copy_access, {}, this->debug_info_);
    }

    // Add branch over local mask variable
    auto& if_else = builder.add_if_else(*current_seq, this->debug_info_);
    auto cond_copy = symbolic::symbol(cond_copy_container);
    auto& case_true = builder.add_case(if_else, symbolic::Eq(cond_copy, symbolic::__true__()), this->debug_info_);
    auto& case_false = builder.add_case(if_else, symbolic::Eq(cond_copy, symbolic::__false__()), this->debug_info_);

    // Fill true case: Copy value from X1 to Y
    int x1_dims = this->layout_x1_.dims();
    data_flow::Subset x1_subset;
    x1_subset.reserve(x1_dims);
    for (int i = 0; i < x1_dims; i++) {
        if (symbolic::eq(this->layout_x1_.get_dim(i), symbolic::one())) {
            x1_subset.push_back(symbolic::zero());
        } else {
            x1_subset.push_back(subset[i]);
        }
    }
    auto& block_true = builder.add_block(case_true, this->debug_info_);
    {
        auto& x1_access = standalone->add_scalar_input_access(block_true, X1_INPUT_IDX);
        auto& y_access = standalone->add_scalar_input_access(block_true, Y_INPUT_IDX);
        auto& tasklet =
            builder.add_tasklet(block_true, data_flow::TaskletCode::assign, "_out", {"_in"}, this->debug_info_);
        builder.add_computational_memlet(
            block_true, x1_access, tasklet, "_in", x1_subset, iedge_x1->base_type(), iedge_x1->debug_info()
        );
        builder.add_computational_memlet(
            block_true, tasklet, "_out", y_access, subset, iedge_y->base_type(), iedge_y->debug_info()
        );
    }

    // Fill false case: Copy value from X2 to Y
    int x2_dims = this->layout_x2_.dims();
    data_flow::Subset x2_subset;
    x2_subset.reserve(x2_dims);
    for (int i = 0; i < x2_dims; i++) {
        if (symbolic::eq(this->layout_x2_.get_dim(i), symbolic::one())) {
            x2_subset.push_back(symbolic::zero());
        } else {
            x2_subset.push_back(subset[i]);
        }
    }
    auto& block_false = builder.add_block(case_false, this->debug_info_);
    {
        auto& x2_access = standalone->add_scalar_input_access(block_false, X2_INPUT_IDX);
        auto& y_access = standalone->add_scalar_input_access(block_false, Y_INPUT_IDX);
        auto& tasklet =
            builder.add_tasklet(block_false, data_flow::TaskletCode::assign, "_out", {"_in"}, this->debug_info_);
        builder.add_computational_memlet(
            block_false, x2_access, tasklet, "_in", x2_subset, iedge_x2->base_type(), iedge_x2->debug_info()
        );
        builder.add_computational_memlet(
            block_false, tasklet, "_out", y_access, subset, iedge_y->base_type(), iedge_y->debug_info()
        );
    }

    return standalone->successfully_expanded();
}

std::string ConditionalTensorCopyNode::toStr() const {
    std::stringstream stream;

    stream << "ConditionalTensorCopyNode(shape: ";
    TensorLayout::emit_symbolic_list(stream, this->layout_y_.shape());
    stream << ")";

    return stream.str();
}

symbolic::SymbolSet ConditionalTensorCopyNode::symbols() const {
    symbolic::SymbolSet syms;
    this->layout_mask_.collect_symbols(syms);
    this->layout_x1_.collect_symbols(syms);
    this->layout_x2_.collect_symbols(syms);
    this->layout_y_.collect_symbols(syms);
    return syms;
}

symbolic::Expression ConditionalTensorCopyNode::flop() const { return symbolic::zero(); }

std::unique_ptr<data_flow::DataFlowNode> ConditionalTensorCopyNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::make_unique<ConditionalTensorCopyNode>(
        element_id,
        this->debug_info_,
        vertex,
        parent,
        this->layout_mask_,
        this->layout_x1_,
        this->layout_x2_,
        this->layout_y_
    );
}

void ConditionalTensorCopyNode::
    replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) {
    this->layout_mask_.replace_symbols(old_expression, new_expression);
    this->layout_x1_.replace_symbols(old_expression, new_expression);
    this->layout_x2_.replace_symbols(old_expression, new_expression);
    this->layout_y_.replace_symbols(old_expression, new_expression);
}

void ConditionalTensorCopyNode::replace(const symbolic::ExpressionMapping& replacements) {
    this->layout_mask_.replace_symbols(replacements);
    this->layout_x1_.replace_symbols(replacements);
    this->layout_x2_.replace_symbols(replacements);
    this->layout_y_.replace_symbols(replacements);
}

nlohmann::json ConditionalTensorCopyNodeSerializer::serialize(const data_flow::LibraryNode& library_node) {
    const ConditionalTensorCopyNode& conditional_copy_node = static_cast<const ConditionalTensorCopyNode&>(library_node
    );
    nlohmann::json j;

    j["code"] = conditional_copy_node.code().value();
    conditional_copy_node.layout_mask().serialize_to_json(j["layout_mask"]);
    conditional_copy_node.layout_x1().serialize_to_json(j["layout_x1"]);
    conditional_copy_node.layout_x2().serialize_to_json(j["layout_x2"]);
    conditional_copy_node.layout_y().serialize_to_json(j["layout_y"]);

    return j;
}

data_flow::LibraryNode& ConditionalTensorCopyNodeSerializer::deserialize(
    const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
) {
    assert(j.contains("element_id"));
    assert(j.contains("code"));
    assert(j.contains("layout_mask"));
    assert(j.contains("layout_x1"));
    assert(j.contains("layout_x2"));
    assert(j.contains("layout_y"));
    assert(j.contains("debug_info"));

    TensorLayout layout_mask = TensorLayout::deserialize_from_json(j.at("layout_mask"));
    TensorLayout layout_x1 = TensorLayout::deserialize_from_json(j.at("layout_x1"));
    TensorLayout layout_x2 = TensorLayout::deserialize_from_json(j.at("layout_x2"));
    TensorLayout layout_y = TensorLayout::deserialize_from_json(j.at("layout_y"));

    serializer::JSONSerializer serializer;
    DebugInfo debug_info = serializer.json_to_debug_info(j.at("debug_info"));

    return builder
        .add_library_node<ConditionalTensorCopyNode>(parent, debug_info, layout_mask, layout_x1, layout_x2, layout_y);
}

} // namespace tensor
} // namespace math
} // namespace sdfg
