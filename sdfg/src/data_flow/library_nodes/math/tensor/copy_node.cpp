#include "sdfg/data_flow/library_nodes/math/tensor/copy_node.h"
#include <cstddef>
#include <list>
#include <memory>
#include <nlohmann/json_fwd.hpp>
#include <string>
#include <unordered_set>
#include <vector>
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
#include "sdfg/types/type.h"
#include "symengine/symengine_rcp.h"

namespace sdfg {
namespace math {
namespace tensor {

void TensorCopyNode::expand_identity_mode(
    passes::LibNodeExpander::AccessNodeExpand& standalone,
    builder::StructuredSDFGBuilder& builder,
    structured_control_flow::Sequence& sequence
) {
    auto& dfg = this->get_parent();
    structured_control_flow::Sequence* current_seq = &sequence;
    int dims = this->layout_x_.dims();
    types::Scalar indvar_type(types::PrimitiveType::UInt64);

    data_flow::Subset indvars;
    indvars.reserve(dims);
    for (int i = 0; i < dims; i++) {
        auto indvar_container = builder.find_new_name("_i");
        builder.add_container(indvar_container, indvar_type);
        auto indvar = symbolic::symbol(indvar_container);
        indvars.push_back(indvar);
        auto& map = builder.add_map(
            *current_seq,
            indvar,
            symbolic::Lt(indvar, this->layout_x_.get_dim(i)),
            symbolic::zero(),
            symbolic::add(indvar, symbolic::one()),
            structured_control_flow::ScheduleType_Sequential::create(),
            this->debug_info_
        );
        current_seq = &map.root();
    }

    auto& block = builder.add_block(*current_seq, {}, this->debug_info_);
    auto& x_access = standalone.add_scalar_input_access(block, X_INPUT_IDX);
    auto& y_access = standalone.add_scalar_input_access(block, Y_INPUT_IDX);

    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"}, this->debug_info_);

    auto* iedge_X = dfg.in_edge_for_connector(*this, "X");
    if (!iedge_X) {
        throw InvalidSDFGException("TensorCopyNode: Cannot get in edge for connector X");
    }
    builder
        .add_computational_memlet(block, x_access, tasklet, "_in", indvars, iedge_X->base_type(), iedge_X->debug_info());

    auto* iedge_Y = dfg.in_edge_for_connector(*this, "Y");
    if (!iedge_Y) {
        throw InvalidSDFGException("TensorCopyNode: Cannot get in edge for connector Y");
    }
    builder
        .add_computational_memlet(block, tasklet, "_out", y_access, indvars, iedge_Y->base_type(), iedge_Y->debug_info());
}

void TensorCopyNode::expand_permutation_mode(
    passes::LibNodeExpander::AccessNodeExpand& standalone,
    builder::StructuredSDFGBuilder& builder,
    structured_control_flow::Sequence& sequence
) {
    int dims = this->layout_x_.dims();
    std::vector<int> mask;
    std::unordered_set<int> used;
    mask.reserve(dims);
    for (int i = 0; i < dims; i++) {
        bool found = false;
        for (int j = 0; j < dims; j++) {
            if (!used.contains(j) && symbolic::eq(this->layout_y_.get_dim(i), this->layout_x_.get_dim(j))) {
                mask.push_back(j);
                used.insert(j);
                found = true;
                break;
            }
        }
        if (!found) {
            throw InvalidSDFGException(
                "TensorCopyNode: Could not find dimension in layout x matching to: " +
                this->layout_y_.get_dim(i)->__str__()
            );
        }
    }

    auto& dfg = this->get_parent();
    structured_control_flow::Sequence* current_seq = &sequence;
    types::Scalar indvar_type(types::PrimitiveType::UInt64);

    data_flow::Subset indvars_y;
    indvars_y.reserve(dims);
    for (int i = 0; i < dims; i++) {
        auto indvar_container = builder.find_new_name("_i");
        builder.add_container(indvar_container, indvar_type);
        auto indvar = symbolic::symbol(indvar_container);
        indvars_y.push_back(indvar);
        auto& map = builder.add_map(
            *current_seq,
            indvar,
            symbolic::Lt(indvar, this->layout_y_.get_dim(i)),
            symbolic::zero(),
            symbolic::add(indvar, symbolic::one()),
            structured_control_flow::ScheduleType_Sequential::create(),
            this->debug_info_
        );
        current_seq = &map.root();
    }

    data_flow::Subset indvars_x(dims, SymEngine::null);
    for (int i = 0; i < dims; i++) {
        indvars_x[mask[i]] = indvars_y[i];
    }

    auto& block = builder.add_block(*current_seq, this->debug_info_);
    auto& x_access = standalone.add_scalar_input_access(block, X_INPUT_IDX);
    auto& y_access = standalone.add_scalar_input_access(block, Y_INPUT_IDX);

    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"}, this->debug_info_);

    auto* iedge_X = dfg.in_edge_for_connector(*this, "X");
    if (!iedge_X) {
        throw InvalidSDFGException("TensorCopyNode: Cannot get in edge for connector X");
    }
    builder
        .add_computational_memlet(block, x_access, tasklet, "_in", indvars_x, iedge_X->base_type(), iedge_X->debug_info());

    auto* iedge_Y = dfg.in_edge_for_connector(*this, "Y");
    if (!iedge_Y) {
        throw InvalidSDFGException("TensorCopyNode: Cannot get in edge for connector Y");
    }
    builder.add_computational_memlet(
        block, tasklet, "_out", y_access, indvars_y, iedge_Y->base_type(), iedge_Y->debug_info()
    );
}

void TensorCopyNode::expand_squeeze_mode(
    passes::LibNodeExpander::AccessNodeExpand& standalone,
    builder::StructuredSDFGBuilder& builder,
    structured_control_flow::Sequence& sequence
) {
    bool x_bigger = this->layout_x_.dims() > this->layout_y_.dims();
    TensorLayout& bigger = x_bigger ? this->layout_x_ : this->layout_y_;
    TensorLayout& smaller = x_bigger ? this->layout_y_ : this->layout_x_;

    auto& dfg = this->get_parent();
    structured_control_flow::Sequence* current_seq = &sequence;
    int smaller_dims = smaller.dims();
    types::Scalar indvar_type(types::PrimitiveType::UInt64);

    data_flow::Subset indvars_smaller;
    indvars_smaller.reserve(smaller_dims);
    for (int i = 0; i < smaller_dims; i++) {
        auto indvar_container = builder.find_new_name("_i");
        builder.add_container(indvar_container, indvar_type);
        auto indvar = symbolic::symbol(indvar_container);
        indvars_smaller.push_back(indvar);
        auto& map = builder.add_map(
            *current_seq,
            indvar,
            symbolic::Lt(indvar, smaller.get_dim(i)),
            symbolic::zero(),
            symbolic::add(indvar, symbolic::one()),
            structured_control_flow::ScheduleType_Sequential::create(),
            this->debug_info_
        );
        current_seq = &map.root();
    }

    int bigger_dims = bigger.dims();
    data_flow::Subset indvars_bigger;
    indvars_bigger.reserve(bigger_dims);
    for (int i = 0, j = 0; i < bigger_dims; i++) {
        if (j >= smaller_dims || !symbolic::eq(bigger.get_dim(i), smaller.get_dim(j))) {
            if (symbolic::eq(bigger.get_dim(i), symbolic::one())) {
                indvars_bigger.push_back(symbolic::zero());
            } else {
                throw InvalidSDFGException(
                    "TensorCopyNode: Got not matching dimension that is not one: " + std::to_string(i) + " in " +
                    bigger.toStr()
                );
            }
        } else {
            indvars_bigger.push_back(indvars_smaller[j]);
            j++;
        }
    }

    auto& block = builder.add_block(*current_seq, {}, this->debug_info_);
    auto& x_access = standalone.add_scalar_input_access(block, X_INPUT_IDX);
    auto& y_access = standalone.add_scalar_input_access(block, Y_INPUT_IDX);

    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"}, this->debug_info_);

    auto* iedge_X = dfg.in_edge_for_connector(*this, "X");
    if (!iedge_X) {
        throw InvalidSDFGException("TensorCopyNode: Cannot get in edge for connector X");
    }
    builder.add_computational_memlet(
        block,
        x_access,
        tasklet,
        "_in",
        (x_bigger ? indvars_bigger : indvars_smaller),
        iedge_X->base_type(),
        iedge_X->debug_info()
    );

    auto* iedge_Y = dfg.in_edge_for_connector(*this, "Y");
    if (!iedge_Y) {
        throw InvalidSDFGException("TensorCopyNode: Cannot get in edge for connector Y");
    }
    builder.add_computational_memlet(
        block,
        tasklet,
        "_out",
        y_access,
        (x_bigger ? indvars_smaller : indvars_bigger),
        iedge_Y->base_type(),
        iedge_Y->debug_info()
    );
}

void TensorCopyNode::expand_reshape_mode(
    passes::LibNodeExpander::AccessNodeExpand& standalone,
    builder::StructuredSDFGBuilder& builder,
    structured_control_flow::Sequence& sequence
) {
    auto total_elements = this->layout_x_.total_elements();
    if (!symbolic::eq(total_elements, this->layout_y_.total_elements())) {
        throw InvalidSDFGException(
            "TensorCopyNode: Cannot expand because number of elements of layouts do not match: " +
            total_elements->__str__() + " != " + this->layout_y_.total_elements()->__str__()
        );
    }

    auto& dfg = this->get_parent();
    types::Scalar indvar_type(types::PrimitiveType::UInt64);
    auto indvar_container = builder.find_new_name("_i");
    builder.add_container(indvar_container, indvar_type);
    auto indvar = symbolic::symbol(indvar_container);
    auto& map = builder.add_map(
        sequence,
        indvar,
        symbolic::Lt(indvar, total_elements),
        symbolic::zero(),
        symbolic::add(indvar, symbolic::one()),
        structured_control_flow::ScheduleType_Sequential::create(),
        this->debug_info_
    );

    int x_dims = this->layout_x_.dims();
    data_flow::Subset indvars_x;
    indvars_x.reserve(x_dims);
    if (x_dims > 1) {
        std::list<symbolic::Expression> indvars_x_list;
        symbolic::Expression divisor = symbolic::one();
        for (int i = x_dims - 1; i >= 0; i--) {
            auto dim = this->layout_x_.get_dim(i);
            auto divison = symbolic::div(indvar, divisor);
            if (i == 0) {
                indvars_x_list.push_front(divison);
            } else {
                indvars_x_list.push_front(symbolic::mod(divison, dim));
            }
            divisor = symbolic::mul(divisor, dim);
        }
        indvars_x.insert(indvars_x.end(), indvars_x_list.begin(), indvars_x_list.end());
    } else {
        indvars_x.push_back(indvar);
    }

    int y_dims = this->layout_y_.dims();
    data_flow::Subset indvars_y;
    indvars_y.reserve(y_dims);
    if (y_dims > 1) {
        std::list<symbolic::Expression> indvars_y_list;
        symbolic::Expression divisor = symbolic::one();
        for (int i = y_dims - 1; i >= 0; i--) {
            auto dim = this->layout_y_.get_dim(i);
            auto division = symbolic::div(indvar, divisor);
            if (i == 0) {
                indvars_y_list.push_front(division);
            } else {
                indvars_y_list.push_front(symbolic::mod(division, dim));
            }
            divisor = symbolic::mul(divisor, dim);
        }
        indvars_y.insert(indvars_y.end(), indvars_y_list.begin(), indvars_y_list.end());
    } else {
        indvars_y.push_back(indvar);
    }

    auto& block = builder.add_block(map.root(), {}, this->debug_info_);
    auto& x_access = standalone.add_scalar_input_access(block, X_INPUT_IDX);
    auto& y_access = standalone.add_scalar_input_access(block, Y_INPUT_IDX);

    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"}, this->debug_info_);

    auto* iedge_X = dfg.in_edge_for_connector(*this, "X");
    if (!iedge_X) {
        throw InvalidSDFGException("TensorCopyNode: Cannot get in edge for connector X");
    }
    builder
        .add_computational_memlet(block, x_access, tasklet, "_in", indvars_x, iedge_X->base_type(), iedge_X->debug_info());

    auto* iedge_Y = dfg.in_edge_for_connector(*this, "Y");
    if (!iedge_Y) {
        throw InvalidSDFGException("TensorCopyNode: Cannot get in edge for connector Y");
    }
    builder.add_computational_memlet(
        block, tasklet, "_out", y_access, indvars_y, iedge_Y->base_type(), iedge_Y->debug_info()
    );
}

TensorCopyNode::TensorCopyNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const TensorLayout& layout_x,
    const TensorLayout& layout_y,
    const data_flow::ImplementationType& impl_type
)
    : TensorNode(element_id, debug_info, vertex, parent, LibraryNodeType_TensorCopy, {}, {"X", "Y"}, impl_type),
      layout_x_(layout_x), layout_y_(layout_y) {}

const TensorLayout& TensorCopyNode::layout_x() const { return this->layout_x_; }

const TensorLayout& TensorCopyNode::layout_y() const { return this->layout_y_; }

bool TensorCopyNode::is_identity_mode() const {
    int dims = this->layout_x_.dims();
    if (dims != this->layout_y_.dims()) {
        return false;
    }
    for (int i = 0; i < dims; i++) {
        if (!symbolic::eq(this->layout_x_.get_dim(i), this->layout_y_.get_dim(i))) {
            return false;
        }
    }
    return true;
}

bool TensorCopyNode::is_permutation_mode() const {
    if (this->is_identity_mode()) {
        return false;
    }
    int dims = this->layout_x_.dims();
    if (dims != this->layout_y_.dims()) {
        return false;
    }
    for (int i = 0; i < dims; i++) {
        bool found = false;
        for (int j = 0; j < dims; j++) {
            if (symbolic::eq(this->layout_x_.get_dim(i), this->layout_y_.get_dim(j))) {
                found = true;
                break;
            }
        }
        if (!found) {
            return false;
        }
    }
    return true;
}

bool TensorCopyNode::is_squeeze_mode() const {
    int dims_x = this->layout_x_.dims();
    int dims_y = this->layout_y_.dims();
    symbolic::MultiExpression bigger_shape, smaller_shape;
    if (dims_x < dims_y) {
        bigger_shape = this->layout_y_.shape();
        smaller_shape = this->layout_x_.shape();
    } else if (dims_x > dims_y) {
        bigger_shape = this->layout_x_.shape();
        smaller_shape = this->layout_y_.shape();
    } else {
        return false;
    }

    int offset = 0;
    for (int i = 0; i < bigger_shape.size(); i++) {
        if (i - offset >= smaller_shape.size() || !symbolic::eq(bigger_shape[i], smaller_shape[i - offset])) {
            if (symbolic::eq(bigger_shape[i], symbolic::one())) {
                offset++;
            } else {
                return false;
            }
        }
    }
    return true;
}

bool TensorCopyNode::is_reshape_mode() const {
    return !this->is_identity_mode() && !this->is_permutation_mode() && !this->is_squeeze_mode();
}

void TensorCopyNode::validate(const Function& function) const {
    TensorNode::validate(function);

    auto& graph = this->get_parent();

    if (graph.in_degree(*this) != 2) {
        throw InvalidSDFGException(
            "TensorCopyNode: Expected exactly 2 inputs (X, Y) but got: " + std::to_string(graph.in_degree(*this))
        );
    }
    if (graph.out_degree(*this) != 0) {
        throw InvalidSDFGException(
            "TensorCopyNode: Expected no outputs but got: " + std::to_string(graph.out_degree(*this))
        );
    }

    // Check that both layouts have the same number of elements
    auto x_num_elements = this->layout_x_.total_elements();
    auto y_num_elements = this->layout_y_.total_elements();
    if (!symbolic::eq(x_num_elements, y_num_elements)) {
        throw InvalidSDFGException(
            "TensorCopyNode: Number of elements of layouts do not match: " + x_num_elements->__str__() +
            " != " + y_num_elements->__str__()
        );
    }
}

bool TensorCopyNode::supports_integer_types() const { return true; }

using Dir = passes::LibNodeExpander::InputUse;

passes::LibNodeExpander::ExpandOutcome TensorCopyNode::
    expand(passes::LibNodeExpander::ExpandContext& context, structured_control_flow::Block& block) {
    auto standalone = context.replacement_requires_access_nodes({Dir::IndirectRead, Dir::IndirectWrite});

    if (!standalone) {
        return context.unable();
    }

    auto& builder = standalone->builder();

    // Add a graph after the current block
    auto& new_sequence = standalone->replace_with_sequence();

    if (this->is_identity_mode()) {
        this->expand_identity_mode(*standalone, builder, new_sequence);
    } else if (this->is_permutation_mode()) {
        this->expand_permutation_mode(*standalone, builder, new_sequence);
    } else if (this->is_squeeze_mode()) {
        this->expand_squeeze_mode(*standalone, builder, new_sequence);
    } else {
        this->expand_reshape_mode(*standalone, builder, new_sequence);
    }

    return standalone->successfully_expanded();
}

std::string TensorCopyNode::toStr() const {
    return "TensorCopyNode(X: " + this->layout_x_.toStr() + ", Y: " + this->layout_y_.toStr() + ")";
}

symbolic::SymbolSet TensorCopyNode::symbols() const {
    symbolic::SymbolSet syms;
    this->layout_x_.collect_symbols(syms);
    this->layout_y_.collect_symbols(syms);
    return syms;
}

symbolic::Expression TensorCopyNode::flop() const { return symbolic::zero(); }

std::unique_ptr<data_flow::DataFlowNode> TensorCopyNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::make_unique<TensorCopyNode>(
        element_id, this->debug_info_, vertex, parent, this->layout_x_, this->layout_y_, this->implementation_type_
    );
}

void TensorCopyNode::replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) {
    this->layout_x_.replace_symbols(old_expression, new_expression);
    this->layout_y_.replace_symbols(old_expression, new_expression);
}

void TensorCopyNode::replace(const symbolic::ExpressionMapping& replacements) {
    this->layout_x_.replace_symbols(replacements);
    this->layout_y_.replace_symbols(replacements);
}

nlohmann::json TensorCopyNodeSerializer::serialize(const data_flow::LibraryNode& library_node) {
    const auto& copy_node = static_cast<const TensorCopyNode&>(library_node);
    nlohmann::json j;

    j["code"] = copy_node.code().value();

    copy_node.layout_x().serialize_to_json(j["layout_x"]);
    copy_node.layout_y().serialize_to_json(j["layout_y"]);

    return j;
}

data_flow::LibraryNode& TensorCopyNodeSerializer::deserialize(
    const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
) {
    assert(j.contains("element_id"));
    assert(j.contains("code"));
    assert(j.contains("layout_x"));
    assert(j.contains("layout_y"));
    assert(j.contains("debug_info"));

    TensorLayout layout_x = TensorLayout::deserialize_from_json(j.at("layout_x"));
    TensorLayout layout_y = TensorLayout::deserialize_from_json(j.at("layout_y"));

    sdfg::serializer::JSONSerializer serializer;
    DebugInfo debug_info = serializer.json_to_debug_info(j.at("debug_info"));

    return builder.add_library_node<TensorCopyNode>(parent, debug_info, layout_x, layout_y);
}

} // namespace tensor
} // namespace math
} // namespace sdfg
