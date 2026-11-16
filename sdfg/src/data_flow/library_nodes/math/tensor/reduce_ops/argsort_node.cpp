#include "sdfg/data_flow/library_nodes/math/tensor/reduce_ops/argsort_node.h"

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/types/type.h"
#include "sdfg/types/utils.h"

namespace sdfg {
namespace math {
namespace tensor {

ArgSortNode::ArgSortNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const std::vector<symbolic::Expression>& shape,
    const std::vector<int64_t>& axes,
    bool keepdims
)
    : ReduceNode(element_id, debug_info, vertex, parent, LibraryNodeType_ArgSort, shape, axes, false) {
    single_axis();
}

void ArgSortNode::validate(const Function& function) const {
    // Not TensorNode::validate: X and the Int64 indices Y differ in type.
    MathNode::validate(function);
    single_axis();
    validate_shapes();
    auto* output = get_parent().in_edge_for_connector(*this, inputs_.at(RESULT_PTR_IDX));
    if (output->base_type().primitive_type() != types::PrimitiveType::Int64) {
        throw InvalidSDFGException("Library Node: " + code().value() + " writes Int64 indices");
    }
}

passes::LibNodeExpander::ExpandOutcome ArgSortNode::expand(passes::LibNodeExpander::ExpandContext& context, Block& block) {
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
    auto [outer, index] = add_outer_maps(builder, expansion->replace_with_sequence(), {axis});
    auto at = [&, index = index](const symbolic::Expression& position) {
        auto subset = index;
        subset[axis] = position;
        return data_flow::Subset(subset.begin(), subset.end());
    };
    auto is_true = [](const std::string& flag) {
        return symbolic::Ne(symbolic::symbol(flag), symbolic::zero());
    };
    types::Scalar index_type(types::PrimitiveType::Int64);

    // Each element's rank is the number of elements that sort before it; it goes to Y[rank].
    auto [i, body] = add_counted_loop(builder, *outer, symbolic::zero(), shape_[axis], true);
    auto rank = add_scalar(builder, "_argsort_rank", types::PrimitiveType::Int64);
    auto& init = builder.add_block(*body, {}, debug_info);
    add_index_assignment(builder, init, builder.add_access(init, rank, debug_info), {}, index_type, "0", true);

    auto [j, count] = add_counted_loop(builder, *body, symbolic::zero(), shape_[axis], false);
    auto& test = builder.add_block(*count, {}, debug_info);
    auto& input_type = in_edge->base_type();
    auto other = at(j);
    auto self = at(i);
    auto earlier = symbolic::Lt(j, i);
    symbolic::Condition before;
    if (types::is_floating_point(element)) {
        using data_flow::TaskletCode;
        auto less = add_input_comparison(*expansion, test, TaskletCode::fp_olt, other, self, input_type);
        auto equal = add_input_comparison(*expansion, test, TaskletCode::fp_oeq, other, self, input_type);
        auto self_nan = add_input_comparison(*expansion, test, TaskletCode::fp_uno, self, self, input_type);
        auto other_nan = add_input_comparison(*expansion, test, TaskletCode::fp_uno, other, other, input_type);
        // NaNs sort after every number and among themselves by position.
        before = symbolic::Or(
            symbolic::Or(is_true(less), symbolic::And(is_true(self_nan), symbolic::Not(is_true(other_nan)))),
            symbolic::And(earlier, symbolic::Or(is_true(equal), symbolic::And(is_true(self_nan), is_true(other_nan))))
        );
    } else {
        auto code = types::is_signed(element) ? data_flow::TaskletCode::int_slt : data_flow::TaskletCode::int_ult;
        auto less = add_input_comparison(*expansion, test, code, other, self, input_type);
        auto equal = add_input_comparison(*expansion, test, data_flow::TaskletCode::int_eq, other, self, input_type);
        before = symbolic::Or(is_true(less), symbolic::And(earlier, is_true(equal)));
    }
    auto& branch = builder.add_if_else(*count, debug_info);
    auto& increment = builder.add_block(builder.add_case(branch, before, debug_info), {}, debug_info);
    auto& tasklet =
        builder.add_tasklet(increment, data_flow::TaskletCode::int_add, "_out", {"_in1", "_in2"}, debug_info);
    builder.add_computational_memlet(
        increment, builder.add_access(increment, rank, debug_info), tasklet, "_in1", {}, index_type, debug_info
    );
    builder.add_computational_memlet(
        increment,
        builder.add_constant(increment, "1", index_type, debug_info),
        tasklet,
        "_in2",
        {},
        index_type,
        debug_info
    );
    builder.add_computational_memlet(
        increment, tasklet, "_out", builder.add_access(increment, rank, debug_info), {}, index_type, debug_info
    );

    auto& store = builder.add_block(*body, {}, debug_info);
    add_index_assignment(
        builder,
        store,
        expansion->add_indirect_write_access(store, RESULT_PTR_IDX),
        at(symbolic::symbol(rank)),
        out_edge->base_type(),
        i->get_name(),
        false
    );
    return expansion->successfully_expanded();
}

std::unique_ptr<data_flow::DataFlowNode> ArgSortNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::make_unique<ArgSortNode>(element_id, this->debug_info(), vertex, parent, shape_, axes_, keepdims_);
}

} // namespace tensor
} // namespace math
} // namespace sdfg
