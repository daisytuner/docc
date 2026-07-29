#include "sdfg/data_flow/library_nodes/math/tensor/slice_node.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/structured_control_flow/for.h"

namespace sdfg {
namespace math {
namespace tensor {

SliceNode::SliceNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const std::vector<symbolic::Expression>& input_shape,
    long long dim,
    long long start,
    long long end,
    long long step,
    const data_flow::ImplementationType& impl_type
)
    : TensorNode(element_id, debug_info, vertex, parent, LibraryNodeType_Slice, {}, {"Y", "X"}, impl_type),
      input_shape_(input_shape), dim_(dim), start_(start), end_(end), step_(step) {}

const std::vector<symbolic::Expression>& SliceNode::input_shape() const { return input_shape_; }

long long SliceNode::dim() const { return dim_; }

long long SliceNode::start() const { return start_; }

long long SliceNode::end() const { return end_; }

long long SliceNode::step() const { return step_; }

bool SliceNode::supports_integer_types() const { return true; }

void SliceNode::validate(const Function& function) const {
    TensorNode::validate(function);

    if (dim_ < 0 || static_cast<size_t>(dim_) >= input_shape_.size()) {
        throw InvalidSDFGException(
            "SliceNode: dim out of range. dim: " + std::to_string(dim_) +
            " rank: " + std::to_string(input_shape_.size())
        );
    }
    if (step_ <= 0) {
        throw InvalidSDFGException("SliceNode: step must be positive but got " + std::to_string(step_));
    }
    if (start_ < 0 || end_ < start_) {
        throw InvalidSDFGException(
            "SliceNode: expected 0 <= start <= end but got start: " + std::to_string(start_) +
            " end: " + std::to_string(end_)
        );
    }
}

symbolic::SymbolSet SliceNode::symbols() const {
    symbolic::SymbolSet syms;
    for (const auto& dim : input_shape_) {
        for (auto& atom : symbolic::atoms(dim)) {
            syms.insert(atom);
        }
    }
    return syms;
}

void SliceNode::replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) {
    for (auto& dim : input_shape_) {
        dim = symbolic::subs(dim, old_expression, new_expression);
    }
}

void SliceNode::replace(const symbolic::ExpressionMapping& replacements) {
    for (auto& dim : input_shape_) {
        dim = symbolic::subs(dim, replacements);
    }
}

passes::LibNodeExpander::ExpandOutcome SliceNode::
    expand(passes::LibNodeExpander::ExpandContext& context, structured_control_flow::Block& block) {
    auto& dataflow = this->get_parent();

    if (dataflow.in_degree(*this) != 2 || dataflow.out_degree(*this) != 0) {
        return context.unable();
    }

    auto edges = dataflow.in_edges_by_connector(*this);
    auto& in_edge = *edges.at(X_INPUT_IDX);
    auto& result_ptr_edge = *edges.at(RESULT_PTR_IDX);

    using Use = passes::LibNodeExpander::InputUse;
    auto standalone = context.replacement_requires_access_nodes({Use::IndirectWrite, Use::IndirectRead});

    if (!standalone) {
        return context.unable();
    }

    // Output shape equals the input shape except along the sliced dimension.
    std::vector<symbolic::Expression> output_shape = input_shape_;
    long long sliced_dim = (end_ - start_ + step_ - 1) / step_;
    output_shape[dim_] = symbolic::integer(sliced_dim);

    symbolic::MultiExpression loop_vars;
    auto& builder = standalone->builder();
    structured_control_flow::Sequence* inner_scope = nullptr;

    for (size_t i = 0; i < output_shape.size(); ++i) {
        std::string var_name = builder.find_new_name("_i" + std::to_string(i));
        builder.add_container(var_name, types::Scalar(types::PrimitiveType::Int64));

        auto sym_var = symbolic::symbol(var_name);
        auto condition = symbolic::Lt(sym_var, output_shape[i]);
        auto init = symbolic::zero();
        auto update = symbolic::add(sym_var, symbolic::one());

        if (i == 0) {
            auto& loop = standalone->replace_with_structured_loop(
                passes::LibNodeExpander::AccessNodeExpand::LoopType::Map,
                sym_var,
                condition,
                init,
                update,
                structured_control_flow::ScheduleType_Sequential::create()
            );
            inner_scope = &loop.root();
        } else {
            auto& loop = builder.add_map(
                *inner_scope,
                sym_var,
                condition,
                init,
                update,
                structured_control_flow::ScheduleType_Sequential::create(),
                this->debug_info()
            );
            inner_scope = &loop.root();
        }
        loop_vars.push_back(sym_var);
    }

    auto& tasklet_block = builder.add_block(*inner_scope, {}, this->debug_info());

    auto& in_acc = standalone->add_indirect_read_access(tasklet_block, X_INPUT_IDX);
    auto& out_acc = standalone->add_indirect_write_access(tasklet_block, RESULT_PTR_IDX);

    // Source subset: `start + _i * step` along the sliced dimension, identity otherwise.
    symbolic::MultiExpression input_subset;
    input_subset.reserve(output_shape.size());
    for (size_t i = 0; i < output_shape.size(); ++i) {
        if (static_cast<long long>(i) == dim_) {
            input_subset
                .push_back(symbolic::add(symbolic::integer(start_), symbolic::mul(loop_vars[i], symbolic::integer(step_)))
                );
        } else {
            input_subset.push_back(loop_vars[i]);
        }
    }

    auto& tasklet =
        builder.add_tasklet(tasklet_block, data_flow::TaskletCode::assign, "_out", {"_in"}, this->debug_info());

    builder.add_computational_memlet(
        tasklet_block, in_acc, tasklet, "_in", input_subset, in_edge.base_type(), this->debug_info()
    );
    builder.add_computational_memlet(
        tasklet_block, tasklet, "_out", out_acc, loop_vars, result_ptr_edge.base_type(), this->debug_info()
    );

    return standalone->successfully_expanded();
}

std::unique_ptr<data_flow::DataFlowNode> SliceNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::unique_ptr<data_flow::DataFlowNode>(
        new SliceNode(element_id, this->debug_info(), vertex, parent, input_shape_, dim_, start_, end_, step_)
    );
}

data_flow::PointerAccessType SliceNode::pointer_access_type(int input_idx) const {
    if (input_idx == RESULT_PTR_IDX) {
        return data_flow::PointerAccessMeta::create_full_write_only(symbolic::__nullptr__(), true);
    } else if (input_idx == X_INPUT_IDX) {
        return data_flow::PointerAccessMeta::create_read_only(symbolic::__nullptr__(), true);
    } else {
        return TensorNode::pointer_access_type(input_idx);
    }
}

nlohmann::json SliceNodeSerializer::serialize(const data_flow::LibraryNode& library_node) {
    const SliceNode& slice_node = static_cast<const SliceNode&>(library_node);
    nlohmann::json j;

    j["code"] = slice_node.code().value();

    serializer::JSONSerializer serializer;
    j["input_shape"] = nlohmann::json::array();
    for (auto& dim : slice_node.input_shape()) {
        j["input_shape"].push_back(serializer.expression(dim));
    }

    j["dim"] = slice_node.dim();
    j["start"] = slice_node.start();
    j["end"] = slice_node.end();
    j["step"] = slice_node.step();

    return j;
}

data_flow::LibraryNode& SliceNodeSerializer::deserialize(
    const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
) {
    // Assertions for required fields
    assert(j.contains("element_id"));
    assert(j.contains("code"));
    assert(j.contains("debug_info"));
    assert(j.contains("input_shape"));
    assert(j.contains("dim"));
    assert(j.contains("start"));
    assert(j.contains("end"));
    assert(j.contains("step"));

    std::vector<symbolic::Expression> input_shape;
    for (const auto& dim : j["input_shape"]) {
        input_shape.push_back(symbolic::parse(dim.get<std::string>()));
    }

    long long dim = j["dim"].get<long long>();
    long long start = j["start"].get<long long>();
    long long end = j["end"].get<long long>();
    long long step = j["step"].get<long long>();

    // Extract debug info using JSONSerializer
    sdfg::serializer::JSONSerializer serializer;
    DebugInfo debug_info = serializer.json_to_debug_info(j["debug_info"]);

    return builder.add_library_node<SliceNode>(parent, debug_info, input_shape, dim, start, end, step);
}

} // namespace tensor
} // namespace math
} // namespace sdfg
