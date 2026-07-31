#include "sdfg/data_flow/library_nodes/math/tensor/index_node.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/structured_control_flow/for.h"

namespace sdfg {
namespace math {
namespace tensor {

static std::vector<std::string> make_index_inputs(long long num_indices) {
    std::vector<std::string> inputs = {"Y", "X"};
    for (long long j = 0; j < num_indices; ++j) {
        inputs.push_back("I" + std::to_string(j));
    }
    return inputs;
}

IndexNode::IndexNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const std::vector<symbolic::Expression>& input_shape,
    const std::vector<symbolic::Expression>& index_shape,
    long long dim_offset,
    long long num_indices,
    const data_flow::ImplementationType& impl_type
)
    : TensorNode(
          element_id, debug_info, vertex, parent, LibraryNodeType_Index, {}, make_index_inputs(num_indices), impl_type
      ),
      input_shape_(input_shape), index_shape_(index_shape), dim_offset_(dim_offset), num_indices_(num_indices) {}

const std::vector<symbolic::Expression>& IndexNode::input_shape() const { return input_shape_; }

const std::vector<symbolic::Expression>& IndexNode::index_shape() const { return index_shape_; }

long long IndexNode::dim_offset() const { return dim_offset_; }

long long IndexNode::num_indices() const { return num_indices_; }

bool IndexNode::supports_integer_types() const { return true; }

void IndexNode::validate(const Function& function) const {
    // NOTE: The base TensorNode::validate enforces that all connected memlets share one
    // primitive type. That does not hold here: the index tensors are integer-typed while
    // the data tensors may be floating-point. We therefore validate at the MathNode level
    // and add our own structural checks.
    MathNode::validate(function);

    if (num_indices_ < 1) {
        throw InvalidSDFGException("IndexNode: expected at least one index tensor but got " + std::to_string(num_indices_));
    }
    if (index_shape_.empty()) {
        throw InvalidSDFGException("IndexNode: index_shape must not be empty");
    }
    if (dim_offset_ < 0 || dim_offset_ + num_indices_ > static_cast<long long>(input_shape_.size())) {
        throw InvalidSDFGException(
            "IndexNode: indexed dimension block out of range. dim_offset: " + std::to_string(dim_offset_) +
            " num_indices: " + std::to_string(num_indices_) + " rank: " + std::to_string(input_shape_.size())
        );
    }
}

symbolic::SymbolSet IndexNode::symbols() const {
    symbolic::SymbolSet syms;
    for (const auto& dim : input_shape_) {
        for (auto& atom : symbolic::atoms(dim)) {
            syms.insert(atom);
        }
    }
    for (const auto& dim : index_shape_) {
        for (auto& atom : symbolic::atoms(dim)) {
            syms.insert(atom);
        }
    }
    return syms;
}

void IndexNode::replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) {
    for (auto& dim : input_shape_) {
        dim = symbolic::subs(dim, old_expression, new_expression);
    }
    for (auto& dim : index_shape_) {
        dim = symbolic::subs(dim, old_expression, new_expression);
    }
}

void IndexNode::replace(const symbolic::ExpressionMapping& replacements) {
    for (auto& dim : input_shape_) {
        dim = symbolic::subs(dim, replacements);
    }
    for (auto& dim : index_shape_) {
        dim = symbolic::subs(dim, replacements);
    }
}

passes::LibNodeExpander::ExpandOutcome IndexNode::
    expand(passes::LibNodeExpander::ExpandContext& context, structured_control_flow::Block& block) {
    auto& dataflow = this->get_parent();

    size_t expected_inputs = 2 + static_cast<size_t>(num_indices_);
    if (dataflow.in_degree(*this) != expected_inputs || dataflow.out_degree(*this) != 0) {
        return context.unable();
    }

    auto edges = dataflow.in_edges_by_connector(*this);
    auto& result_ptr_edge = *edges.at(RESULT_PTR_IDX);
    auto& in_edge = *edges.at(X_INPUT_IDX);
    std::vector<data_flow::Memlet*> index_edges;
    index_edges.reserve(num_indices_);
    for (long long j = 0; j < num_indices_; ++j) {
        index_edges.push_back(edges.at(FIRST_INDEX_IDX + j));
    }

    using Use = passes::LibNodeExpander::InputUse;
    std::vector<Use> uses = {Use::IndirectWrite, Use::IndirectRead};
    for (long long j = 0; j < num_indices_; ++j) {
        uses.push_back(Use::IndirectRead);
    }
    auto standalone = context.replacement_requires_access_nodes(uses);

    if (!standalone) {
        return context.unable();
    }

    // Output shape = leading dims ++ index (broadcast) shape ++ trailing dims.
    size_t nB = index_shape_.size();
    std::vector<symbolic::Expression> output_shape;
    output_shape.reserve(dim_offset_ + nB + (input_shape_.size() - dim_offset_ - num_indices_));
    for (long long d = 0; d < dim_offset_; ++d) {
        output_shape.push_back(input_shape_[d]);
    }
    for (const auto& s : index_shape_) {
        output_shape.push_back(s);
    }
    for (long long d = dim_offset_ + num_indices_; d < static_cast<long long>(input_shape_.size()); ++d) {
        output_shape.push_back(input_shape_[d]);
    }

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

    // The broadcast loop variables index into the (1:1 shaped) index tensors.
    symbolic::MultiExpression broadcast_vars(loop_vars.begin() + dim_offset_, loop_vars.begin() + dim_offset_ + nB);

    // First load each index value into a scalar symbol so it can be used as a
    // data-dependent coordinate in the gathering read below.
    std::vector<symbolic::Symbol> idx_syms;
    idx_syms.reserve(num_indices_);
    auto& load_block = builder.add_block(*inner_scope, {}, this->debug_info());
    for (long long j = 0; j < num_indices_; ++j) {
        std::string idx_name = builder.find_new_name("_idx" + std::to_string(j));
        builder.add_container(idx_name, types::Scalar(types::PrimitiveType::Int64));
        idx_syms.push_back(symbolic::symbol(idx_name));

        auto& idx_in_acc = standalone->add_indirect_read_access(load_block, FIRST_INDEX_IDX + j);
        auto& idx_out_acc = builder.add_access(load_block, idx_name, this->debug_info());
        auto& load_tasklet =
            builder.add_tasklet(load_block, data_flow::TaskletCode::assign, "_out", {"_in"}, this->debug_info());
        builder.add_computational_memlet(
            load_block, idx_in_acc, load_tasklet, "_in", broadcast_vars, index_edges[j]->base_type(), this->debug_info()
        );
        builder.add_computational_memlet(
            load_block,
            load_tasklet,
            "_out",
            idx_out_acc,
            {},
            types::Scalar(types::PrimitiveType::Int64),
            this->debug_info()
        );
    }

    // Gather: out[l..., b..., t...] = X[l..., idx_0, ..., idx_{k-1}, t...]
    auto& tasklet_block = builder.add_block(*inner_scope, {}, this->debug_info());
    auto& in_acc = standalone->add_indirect_read_access(tasklet_block, X_INPUT_IDX);
    auto& out_acc = standalone->add_indirect_write_access(tasklet_block, RESULT_PTR_IDX);

    symbolic::MultiExpression input_subset;
    input_subset.reserve(input_shape_.size());
    for (long long d = 0; d < dim_offset_; ++d) {
        input_subset.push_back(loop_vars[d]);
    }
    for (long long j = 0; j < num_indices_; ++j) {
        input_subset.push_back(idx_syms[j]);
    }
    for (size_t i = dim_offset_ + nB; i < output_shape.size(); ++i) {
        input_subset.push_back(loop_vars[i]);
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

std::unique_ptr<data_flow::DataFlowNode> IndexNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::unique_ptr<data_flow::DataFlowNode>(new IndexNode(
        element_id, this->debug_info(), vertex, parent, input_shape_, index_shape_, dim_offset_, num_indices_
    ));
}

data_flow::PointerAccessType IndexNode::pointer_access_type(int input_idx) const {
    if (input_idx == RESULT_PTR_IDX) {
        return data_flow::PointerAccessMeta::create_full_write_only(symbolic::__nullptr__(), true);
    } else if (input_idx >= X_INPUT_IDX && input_idx < FIRST_INDEX_IDX + num_indices_) {
        return data_flow::PointerAccessMeta::create_read_only(symbolic::__nullptr__(), true);
    } else {
        return TensorNode::pointer_access_type(input_idx);
    }
}

nlohmann::json IndexNodeSerializer::serialize(const data_flow::LibraryNode& library_node) {
    const IndexNode& index_node = static_cast<const IndexNode&>(library_node);
    nlohmann::json j;

    j["code"] = index_node.code().value();

    serializer::JSONSerializer serializer;
    j["input_shape"] = nlohmann::json::array();
    for (auto& dim : index_node.input_shape()) {
        j["input_shape"].push_back(serializer.expression(dim));
    }
    j["index_shape"] = nlohmann::json::array();
    for (auto& dim : index_node.index_shape()) {
        j["index_shape"].push_back(serializer.expression(dim));
    }

    j["dim_offset"] = index_node.dim_offset();
    j["num_indices"] = index_node.num_indices();

    return j;
}

data_flow::LibraryNode& IndexNodeSerializer::deserialize(
    const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
) {
    assert(j.contains("element_id"));
    assert(j.contains("code"));
    assert(j.contains("debug_info"));
    assert(j.contains("input_shape"));
    assert(j.contains("index_shape"));
    assert(j.contains("dim_offset"));
    assert(j.contains("num_indices"));

    std::vector<symbolic::Expression> input_shape;
    for (const auto& dim : j["input_shape"]) {
        input_shape.push_back(symbolic::parse(dim.get<std::string>()));
    }
    std::vector<symbolic::Expression> index_shape;
    for (const auto& dim : j["index_shape"]) {
        index_shape.push_back(symbolic::parse(dim.get<std::string>()));
    }

    long long dim_offset = j["dim_offset"].get<long long>();
    long long num_indices = j["num_indices"].get<long long>();

    sdfg::serializer::JSONSerializer serializer;
    DebugInfo debug_info = serializer.json_to_debug_info(j["debug_info"]);

    return builder.add_library_node<IndexNode>(parent, debug_info, input_shape, index_shape, dim_offset, num_indices);
}

} // namespace tensor
} // namespace math
} // namespace sdfg
