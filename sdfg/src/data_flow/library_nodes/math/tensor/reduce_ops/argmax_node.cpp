#include "sdfg/data_flow/library_nodes/math/tensor/reduce_ops/argmax_node.h"

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/types/type.h"
#include "sdfg/types/utils.h"

namespace sdfg {
namespace math {
namespace tensor {

ArgMaxNode::ArgMaxNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const std::vector<symbolic::Expression>& shape,
    const std::vector<int64_t>& axes,
    bool keepdims
)
    : ReduceNode(element_id, debug_info, vertex, parent, LibraryNodeType_ArgMax, shape, axes, keepdims) {
    single_axis();
}

void ArgMaxNode::validate(const Function& function) const {
    // Not TensorNode::validate: X and the Int64 indices Y differ in type.
    MathNode::validate(function);
    single_axis();
    validate_shapes();
    auto* output = get_parent().in_edge_for_connector(*this, inputs_.at(RESULT_PTR_IDX));
    if (output->base_type().primitive_type() != types::PrimitiveType::Int64) {
        throw InvalidSDFGException("Library Node: " + code().value() + " writes Int64 indices");
    }
}

passes::LibNodeExpander::ExpandOutcome ArgMaxNode::expand(passes::LibNodeExpander::ExpandContext& context, Block& block) {
    auto& dataflow = this->get_parent();
    auto* in_edge = dataflow.in_edge_for_connector(*this, inputs_.at(X_INPUT_IDX));
    auto* out_edge = dataflow.in_edge_for_connector(*this, inputs_.at(RESULT_PTR_IDX));
    if (dataflow.in_degree(*this) != 2 || !in_edge || !out_edge) {
        return context.unable();
    }
    auto element = in_edge->base_type().primitive_type();
    if (!types::is_floating_point(element) && !types::is_integer(element)) {
        return context.unable();
    }
    auto expansion = context.replacement_requires_access_nodes(
        {passes::LibNodeExpander::InputUse::IndirectReadWrite, passes::LibNodeExpander::InputUse::IndirectRead}
    );
    if (!expansion) {
        return context.unable();
    }
    auto& builder = expansion->builder();
    auto& debug_info = this->debug_info();
    auto axis = single_axis();
    auto [body, index] = add_outer_maps(builder, expansion->replace_with_sequence(), {axis});
    auto at = [&, index = index](const symbolic::Expression& position) {
        auto subset = index;
        subset[axis] = position;
        return data_flow::Subset(subset.begin(), subset.end());
    };
    auto is_true = [](const std::string& flag) {
        return symbolic::Ne(symbolic::symbol(flag), symbolic::zero());
    };
    types::Scalar index_type(types::PrimitiveType::Int64);

    // best = 0; for k in [1, n): best = k if X[k] beats X[best]
    auto best = add_scalar(builder, "_argmax_best", types::PrimitiveType::Int64);
    auto& init = builder.add_block(*body, {}, debug_info);
    add_index_assignment(builder, init, builder.add_access(init, best, debug_info), {}, index_type, "0", true);

    auto [k, scan] = add_counted_loop(builder, *body, symbolic::one(), shape_[axis], false);
    auto& test = builder.add_block(*scan, {}, debug_info);
    auto& input_type = in_edge->base_type();
    auto candidate = at(k);
    auto current = at(symbolic::symbol(best));
    symbolic::Condition beats;
    if (types::is_floating_point(element)) {
        // Greater, or a NaN against a number; never against an earlier NaN.
        auto greater =
            add_input_comparison(*expansion, test, data_flow::TaskletCode::fp_ugt, candidate, current, input_type);
        auto number =
            add_input_comparison(*expansion, test, data_flow::TaskletCode::fp_ord, current, current, input_type);
        beats = symbolic::And(is_true(greater), is_true(number));
    } else {
        auto code = types::is_signed(element) ? data_flow::TaskletCode::int_sgt : data_flow::TaskletCode::int_ugt;
        beats = is_true(add_input_comparison(*expansion, test, code, candidate, current, input_type));
    }
    auto& branch = builder.add_if_else(*scan, debug_info);
    auto& taken = builder.add_block(builder.add_case(branch, beats, debug_info), {}, debug_info);
    add_index_assignment(builder, taken, builder.add_access(taken, best, debug_info), {}, index_type, k->get_name(), false);

    data_flow::Subset output;
    for (size_t d = 0; d < shape_.size(); ++d) {
        if (static_cast<int64_t>(d) != axis) {
            output.push_back(index[d]);
        } else if (keepdims_) {
            output.push_back(symbolic::zero());
        }
    }
    auto& store = builder.add_block(*body, {}, debug_info);
    add_index_assignment(
        builder,
        store,
        expansion->add_indirect_write_access(store, RESULT_PTR_IDX),
        output,
        out_edge->base_type(),
        best,
        false
    );
    return expansion->successfully_expanded();
}

std::unique_ptr<data_flow::DataFlowNode> ArgMaxNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::make_unique<ArgMaxNode>(element_id, this->debug_info(), vertex, parent, shape_, axes_, keepdims_);
}

} // namespace tensor
} // namespace math
} // namespace sdfg
