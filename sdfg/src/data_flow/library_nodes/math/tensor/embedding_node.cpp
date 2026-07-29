#include "sdfg/data_flow/library_nodes/math/tensor/embedding_node.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/structured_control_flow/for.h"

namespace sdfg {
namespace math {
namespace tensor {

EmbeddingNode::EmbeddingNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const std::vector<symbolic::Expression>& weight_shape,
    const std::vector<symbolic::Expression>& index_shape,
    const data_flow::ImplementationType& impl_type
)
    : TensorNode(element_id, debug_info, vertex, parent, LibraryNodeType_Embedding, {}, {"Y", "W", "I"}, impl_type),
      weight_shape_(weight_shape), index_shape_(index_shape) {}

const std::vector<symbolic::Expression>& EmbeddingNode::weight_shape() const { return weight_shape_; }

const std::vector<symbolic::Expression>& EmbeddingNode::index_shape() const { return index_shape_; }

bool EmbeddingNode::supports_integer_types() const { return true; }

void EmbeddingNode::validate(const Function& function) const {
    // NOTE: The base TensorNode::validate enforces that all connected memlets share one
    // primitive type. That does not hold here: the index tensor is integer-typed while
    // the weight/output tensors may be floating-point. We therefore validate at the
    // MathNode level and add our own structural checks.
    MathNode::validate(function);

    if (weight_shape_.size() != 2) {
        throw InvalidSDFGException(
            "EmbeddingNode: weight must be 2-dimensional but got rank " + std::to_string(weight_shape_.size())
        );
    }
    if (index_shape_.empty()) {
        throw InvalidSDFGException("EmbeddingNode: index_shape must not be empty");
    }
}

symbolic::SymbolSet EmbeddingNode::symbols() const {
    symbolic::SymbolSet syms;
    for (const auto& dim : weight_shape_) {
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

void EmbeddingNode::replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) {
    for (auto& dim : weight_shape_) {
        dim = symbolic::subs(dim, old_expression, new_expression);
    }
    for (auto& dim : index_shape_) {
        dim = symbolic::subs(dim, old_expression, new_expression);
    }
}

void EmbeddingNode::replace(const symbolic::ExpressionMapping& replacements) {
    for (auto& dim : weight_shape_) {
        dim = symbolic::subs(dim, replacements);
    }
    for (auto& dim : index_shape_) {
        dim = symbolic::subs(dim, replacements);
    }
}

passes::LibNodeExpander::ExpandOutcome EmbeddingNode::
    expand(passes::LibNodeExpander::ExpandContext& context, structured_control_flow::Block& block) {
    auto& dataflow = this->get_parent();

    if (dataflow.in_degree(*this) != 3 || dataflow.out_degree(*this) != 0) {
        return context.unable();
    }

    auto edges = dataflow.in_edges_by_connector(*this);
    auto& result_ptr_edge = *edges.at(RESULT_PTR_IDX);
    auto& weight_edge = *edges.at(W_INPUT_IDX);
    auto& index_edge = *edges.at(INDEX_IDX);

    using Use = passes::LibNodeExpander::InputUse;
    std::vector<Use> uses = {Use::IndirectWrite, Use::IndirectRead, Use::IndirectRead};
    auto standalone = context.replacement_requires_access_nodes(uses);

    if (!standalone) {
        return context.unable();
    }

    // Output shape = index (broadcast) shape ++ [embedding_dim].
    size_t nB = index_shape_.size();
    std::vector<symbolic::Expression> output_shape;
    output_shape.reserve(nB + 1);
    for (const auto& s : index_shape_) {
        output_shape.push_back(s);
    }
    output_shape.push_back(weight_shape_[1]);

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

    // The leading loop variables index into the index tensor.
    symbolic::MultiExpression index_vars(loop_vars.begin(), loop_vars.begin() + nB);

    // First load the index value into a scalar symbol so it can be used as the
    // data-dependent row coordinate in the gathering read below.
    std::string idx_name = builder.find_new_name("_idx");
    builder.add_container(idx_name, types::Scalar(types::PrimitiveType::Int64));
    auto idx_sym = symbolic::symbol(idx_name);

    auto& load_block = builder.add_block(*inner_scope, {}, this->debug_info());
    auto& idx_in_acc = standalone->add_indirect_read_access(load_block, INDEX_IDX);
    auto& idx_out_acc = builder.add_access(load_block, idx_name, this->debug_info());
    auto& load_tasklet =
        builder.add_tasklet(load_block, data_flow::TaskletCode::assign, "_out", {"_in"}, this->debug_info());
    builder.add_computational_memlet(
        load_block, idx_in_acc, load_tasklet, "_in", index_vars, index_edge.base_type(), this->debug_info()
    );
    builder.add_computational_memlet(
        load_block, load_tasklet, "_out", idx_out_acc, {}, types::Scalar(types::PrimitiveType::Int64), this->debug_info()
    );

    // Gather: out[b..., j] = W[idx, j]
    auto& tasklet_block = builder.add_block(*inner_scope, {}, this->debug_info());
    auto& in_acc = standalone->add_indirect_read_access(tasklet_block, W_INPUT_IDX);
    auto& out_acc = standalone->add_indirect_write_access(tasklet_block, RESULT_PTR_IDX);

    symbolic::MultiExpression weight_subset = {idx_sym, loop_vars[nB]};

    auto& tasklet =
        builder.add_tasklet(tasklet_block, data_flow::TaskletCode::assign, "_out", {"_in"}, this->debug_info());
    builder.add_computational_memlet(
        tasklet_block, in_acc, tasklet, "_in", weight_subset, weight_edge.base_type(), this->debug_info()
    );
    builder.add_computational_memlet(
        tasklet_block, tasklet, "_out", out_acc, loop_vars, result_ptr_edge.base_type(), this->debug_info()
    );

    return standalone->successfully_expanded();
}

std::unique_ptr<data_flow::DataFlowNode> EmbeddingNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::unique_ptr<data_flow::DataFlowNode>(
        new EmbeddingNode(element_id, this->debug_info(), vertex, parent, weight_shape_, index_shape_)
    );
}

data_flow::PointerAccessType EmbeddingNode::pointer_access_type(int input_idx) const {
    if (input_idx == RESULT_PTR_IDX) {
        return data_flow::PointerAccessMeta::create_full_write_only(symbolic::__nullptr__(), true);
    } else if (input_idx == W_INPUT_IDX || input_idx == INDEX_IDX) {
        return data_flow::PointerAccessMeta::create_read_only(symbolic::__nullptr__(), true);
    } else {
        return TensorNode::pointer_access_type(input_idx);
    }
}

nlohmann::json EmbeddingNodeSerializer::serialize(const data_flow::LibraryNode& library_node) {
    const EmbeddingNode& embedding_node = static_cast<const EmbeddingNode&>(library_node);
    nlohmann::json j;

    j["code"] = embedding_node.code().value();

    serializer::JSONSerializer serializer;
    j["weight_shape"] = nlohmann::json::array();
    for (auto& dim : embedding_node.weight_shape()) {
        j["weight_shape"].push_back(serializer.expression(dim));
    }
    j["index_shape"] = nlohmann::json::array();
    for (auto& dim : embedding_node.index_shape()) {
        j["index_shape"].push_back(serializer.expression(dim));
    }

    return j;
}

data_flow::LibraryNode& EmbeddingNodeSerializer::deserialize(
    const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
) {
    assert(j.contains("element_id"));
    assert(j.contains("code"));
    assert(j.contains("debug_info"));
    assert(j.contains("weight_shape"));
    assert(j.contains("index_shape"));

    std::vector<symbolic::Expression> weight_shape;
    for (const auto& dim : j["weight_shape"]) {
        weight_shape.push_back(symbolic::parse(dim.get<std::string>()));
    }
    std::vector<symbolic::Expression> index_shape;
    for (const auto& dim : j["index_shape"]) {
        index_shape.push_back(symbolic::parse(dim.get<std::string>()));
    }

    sdfg::serializer::JSONSerializer serializer;
    DebugInfo debug_info = serializer.json_to_debug_info(j["debug_info"]);

    return builder.add_library_node<EmbeddingNode>(parent, debug_info, weight_shape, index_shape);
}

} // namespace tensor
} // namespace math
} // namespace sdfg
