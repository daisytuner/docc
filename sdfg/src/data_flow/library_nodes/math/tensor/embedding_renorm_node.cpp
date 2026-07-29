#include "sdfg/data_flow/library_nodes/math/tensor/embedding_renorm_node.h"

#include <cmath>
#include <iomanip>
#include <sstream>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_nodes/math/cmath/cmath_node.h"
#include "sdfg/structured_control_flow/for.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/types/scalar.h"

namespace sdfg {
namespace math {
namespace tensor {

EmbeddingRenormNode::EmbeddingRenormNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const std::vector<symbolic::Expression>& weight_shape,
    const std::vector<symbolic::Expression>& index_shape,
    double max_norm,
    double norm_type,
    const data_flow::ImplementationType& impl_type
)
    : TensorNode(element_id, debug_info, vertex, parent, LibraryNodeType_EmbeddingRenorm, {}, {"W", "I"}, impl_type),
      weight_shape_(weight_shape), index_shape_(index_shape), max_norm_(max_norm), norm_type_(norm_type) {}

const std::vector<symbolic::Expression>& EmbeddingRenormNode::weight_shape() const { return weight_shape_; }

const std::vector<symbolic::Expression>& EmbeddingRenormNode::index_shape() const { return index_shape_; }

double EmbeddingRenormNode::max_norm() const { return max_norm_; }

double EmbeddingRenormNode::norm_type() const { return norm_type_; }

bool EmbeddingRenormNode::supports_integer_types() const { return false; }

void EmbeddingRenormNode::validate(const Function& function) const {
    // The index tensor is integer-typed while the weight is floating-point, so the
    // uniform-primitive-type check in TensorNode::validate does not apply. Validate at
    // the MathNode level and add our own structural checks (mirrors EmbeddingNode).
    MathNode::validate(function);

    if (weight_shape_.size() != 2) {
        throw InvalidSDFGException(
            "EmbeddingRenormNode: weight must be 2-dimensional but got rank " + std::to_string(weight_shape_.size())
        );
    }
    if (index_shape_.empty()) {
        throw InvalidSDFGException("EmbeddingRenormNode: index_shape must not be empty");
    }
}

symbolic::SymbolSet EmbeddingRenormNode::symbols() const {
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

void EmbeddingRenormNode::replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) {
    for (auto& dim : weight_shape_) {
        dim = symbolic::subs(dim, old_expression, new_expression);
    }
    for (auto& dim : index_shape_) {
        dim = symbolic::subs(dim, old_expression, new_expression);
    }
}

void EmbeddingRenormNode::replace(const symbolic::ExpressionMapping& replacements) {
    for (auto& dim : weight_shape_) {
        dim = symbolic::subs(dim, replacements);
    }
    for (auto& dim : index_shape_) {
        dim = symbolic::subs(dim, replacements);
    }
}

passes::LibNodeExpander::ExpandOutcome EmbeddingRenormNode::
    expand(passes::LibNodeExpander::ExpandContext& context, structured_control_flow::Block& block) {
    auto& dataflow = this->get_parent();

    if (dataflow.in_degree(*this) != 2 || dataflow.out_degree(*this) != 0) {
        return context.unable();
    }

    auto edges = dataflow.in_edges_by_connector(*this);
    auto& weight_edge = *edges.at(W_INPUT_IDX);
    auto& index_edge = *edges.at(INDEX_IDX);

    using Use = passes::LibNodeExpander::InputUse;
    std::vector<Use> uses = {Use::IndirectReadWrite, Use::IndirectRead};
    auto standalone = context.replacement_requires_access_nodes(uses);
    if (!standalone) {
        return context.unable();
    }

    auto& builder = standalone->builder();
    const types::PrimitiveType prim = weight_edge.base_type().primitive_type();
    const types::Scalar scalar_type(prim);
    const symbolic::Expression embedding_dim = weight_shape_[1];
    const bool p_is_inf = std::isinf(norm_type_);
    const bool p_is_one = (norm_type_ == 1.0);
    const bool p_is_two = (norm_type_ == 2.0);

    auto fmt = [](double v) {
        std::ostringstream ss;
        ss << std::setprecision(17) << v;
        return ss.str();
    };
    auto fresh_scalar = [&](const std::string& prefix) {
        std::string name = builder.find_new_name(prefix);
        builder.add_container(name, scalar_type);
        return name;
    };

    // Sequential loops over the index tensor. Sequential ordering makes the in-place
    // renormalization idempotent for duplicate indices (see class docs).
    size_t nB = index_shape_.size();
    symbolic::MultiExpression loop_vars;
    structured_control_flow::Sequence* inner_scope = nullptr;
    for (size_t i = 0; i < nB; ++i) {
        std::string var_name = builder.find_new_name("_i" + std::to_string(i));
        builder.add_container(var_name, types::Scalar(types::PrimitiveType::Int64));
        auto sym_var = symbolic::symbol(var_name);
        auto condition = symbolic::Lt(sym_var, index_shape_[i]);
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

    // Load the data-dependent row coordinate idx = I[loop_vars] into a scalar symbol.
    std::string idx_name = builder.find_new_name("_idx");
    builder.add_container(idx_name, types::Scalar(types::PrimitiveType::Int64));
    auto idx_sym = symbolic::symbol(idx_name);
    {
        auto& load_block = builder.add_block(*inner_scope, {}, this->debug_info());
        auto& idx_in_acc = standalone->add_indirect_read_access(load_block, INDEX_IDX);
        auto& idx_out_acc = builder.add_access(load_block, idx_name, this->debug_info());
        auto& load_tasklet =
            builder.add_tasklet(load_block, data_flow::TaskletCode::assign, "_out", {"_in"}, this->debug_info());
        builder.add_computational_memlet(
            load_block, idx_in_acc, load_tasklet, "_in", loop_vars, index_edge.base_type(), this->debug_info()
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

    // Accumulator for the row norm.
    std::string acc_name = fresh_scalar("_norm_acc");
    {
        auto& init_block = builder.add_block(*inner_scope, {}, this->debug_info());
        auto& zero_const = builder.add_constant(init_block, "0.0", scalar_type, this->debug_info());
        auto& acc_out = builder.add_access(init_block, acc_name, this->debug_info());
        auto& init_tasklet =
            builder.add_tasklet(init_block, data_flow::TaskletCode::assign, "_out", {"_in"}, this->debug_info());
        builder
            .add_computational_memlet(init_block, zero_const, init_tasklet, "_in", {}, scalar_type, this->debug_info());
        builder.add_computational_memlet(init_block, init_tasklet, "_out", acc_out, {}, scalar_type, this->debug_info());
    }

    // Sequential accumulation loop over the embedding dimension.
    {
        std::string j_name = builder.find_new_name("_j");
        builder.add_container(j_name, types::Scalar(types::PrimitiveType::Int64));
        auto j_sym = symbolic::symbol(j_name);
        auto& acc_loop = builder.add_for(
            *inner_scope,
            j_sym,
            symbolic::Lt(j_sym, embedding_dim),
            symbolic::zero(),
            symbolic::add(j_sym, symbolic::one()),
            this->debug_info()
        );
        auto& body = acc_loop.root();
        symbolic::MultiExpression w_subset = {idx_sym, j_sym};

        // aw = |W[idx, j]|
        auto& b = builder.add_block(body, {}, this->debug_info());
        auto& w_in = standalone->add_indirect_read_access(b, W_INPUT_IDX);
        std::string aw_name = fresh_scalar("_aw");
        auto& aw_acc = builder.add_access(b, aw_name, this->debug_info());
        auto& abs_op =
            builder.add_library_node<cmath::CMathNode>(b, this->debug_info(), cmath::CMathFunction::fabs, prim);
        builder.add_computational_memlet(b, w_in, abs_op, "_in1", w_subset, weight_edge.base_type(), this->debug_info());
        builder.add_computational_memlet(b, abs_op, "_out", aw_acc, {}, scalar_type, this->debug_info());

        // term = aw^p (or aw for p==1). For p==inf we take the running maximum instead.
        std::string term_name;
        if (p_is_one || p_is_inf) {
            term_name = aw_name;
        } else if (p_is_two) {
            term_name = fresh_scalar("_term");
            auto& term_acc = builder.add_access(b, term_name, this->debug_info());
            auto& sq_op = builder.add_tasklet(b, data_flow::fp_mul, "_out", {"_in1", "_in2"}, this->debug_info());
            builder.add_computational_memlet(b, aw_acc, sq_op, "_in1", {}, scalar_type, this->debug_info());
            builder.add_computational_memlet(b, aw_acc, sq_op, "_in2", {}, scalar_type, this->debug_info());
            builder.add_computational_memlet(b, sq_op, "_out", term_acc, {}, scalar_type, this->debug_info());
        } else {
            term_name = fresh_scalar("_term");
            auto& term_acc = builder.add_access(b, term_name, this->debug_info());
            auto& p_const = builder.add_constant(b, fmt(norm_type_), scalar_type, this->debug_info());
            auto& pow_op =
                builder.add_library_node<cmath::CMathNode>(b, this->debug_info(), cmath::CMathFunction::pow, prim);
            builder.add_computational_memlet(b, aw_acc, pow_op, "_in1", {}, scalar_type, this->debug_info());
            builder.add_computational_memlet(b, p_const, pow_op, "_in2", {}, scalar_type, this->debug_info());
            builder.add_computational_memlet(b, pow_op, "_out", term_acc, {}, scalar_type, this->debug_info());
        }

        // acc = acc <op> term, where op is fmax for the infinity norm and + otherwise.
        auto& acc_read = builder.add_access(b, acc_name, this->debug_info());
        // For p==1 and p==inf the term IS the abs value: read it from the exact access
        // node abs_op wrote to (aw_acc) so the read-after-write dependency is established
        // within the block. For the other norms the term was written to a fresh container
        // (term_name) by the sq/pow op above, which reads aw_acc and is therefore ordered.
        auto& term_read = (p_is_one || p_is_inf) ? aw_acc : builder.add_access(b, term_name, this->debug_info());
        auto& acc_write = builder.add_access(b, acc_name, this->debug_info());
        if (p_is_inf) {
            auto& max_op =
                builder.add_library_node<cmath::CMathNode>(b, this->debug_info(), cmath::CMathFunction::fmax, prim);
            builder.add_computational_memlet(b, acc_read, max_op, "_in1", {}, scalar_type, this->debug_info());
            builder.add_computational_memlet(b, term_read, max_op, "_in2", {}, scalar_type, this->debug_info());
            builder.add_computational_memlet(b, max_op, "_out", acc_write, {}, scalar_type, this->debug_info());
        } else {
            auto& add_op = builder.add_tasklet(b, data_flow::fp_add, "_out", {"_in1", "_in2"}, this->debug_info());
            builder.add_computational_memlet(b, acc_read, add_op, "_in1", {}, scalar_type, this->debug_info());
            builder.add_computational_memlet(b, term_read, add_op, "_in2", {}, scalar_type, this->debug_info());
            builder.add_computational_memlet(b, add_op, "_out", acc_write, {}, scalar_type, this->debug_info());
        }
    }

    // norm = acc^(1/p): identity for p in {1, inf}, sqrt for p==2, pow(acc, 1/p) otherwise.
    std::string norm_name;
    if (p_is_one || p_is_inf) {
        norm_name = acc_name;
    } else {
        norm_name = fresh_scalar("_norm");
        auto& b = builder.add_block(*inner_scope, {}, this->debug_info());
        auto& acc_acc = builder.add_access(b, acc_name, this->debug_info());
        auto& norm_acc = builder.add_access(b, norm_name, this->debug_info());
        if (p_is_two) {
            auto& sqrt_op =
                builder.add_library_node<cmath::CMathNode>(b, this->debug_info(), cmath::CMathFunction::sqrt, prim);
            builder.add_computational_memlet(b, acc_acc, sqrt_op, "_in1", {}, scalar_type, this->debug_info());
            builder.add_computational_memlet(b, sqrt_op, "_out", norm_acc, {}, scalar_type, this->debug_info());
        } else {
            auto& inv_p_const = builder.add_constant(b, fmt(1.0 / norm_type_), scalar_type, this->debug_info());
            auto& pow_op =
                builder.add_library_node<cmath::CMathNode>(b, this->debug_info(), cmath::CMathFunction::pow, prim);
            builder.add_computational_memlet(b, acc_acc, pow_op, "_in1", {}, scalar_type, this->debug_info());
            builder.add_computational_memlet(b, inv_p_const, pow_op, "_in2", {}, scalar_type, this->debug_info());
            builder.add_computational_memlet(b, pow_op, "_out", norm_acc, {}, scalar_type, this->debug_info());
        }
    }

    // scale = min(1, max_norm / (norm + 1e-7)). The clamp leaves within-norm rows unchanged.
    std::string scale_name = fresh_scalar("_scale");
    {
        auto& b = builder.add_block(*inner_scope, {}, this->debug_info());

        // denom = norm + 1e-7
        std::string denom_name = fresh_scalar("_denom");
        auto& norm_read = builder.add_access(b, norm_name, this->debug_info());
        auto& eps_const = builder.add_constant(b, "1e-7", scalar_type, this->debug_info());
        auto& denom_acc = builder.add_access(b, denom_name, this->debug_info());
        auto& denom_op = builder.add_tasklet(b, data_flow::fp_add, "_out", {"_in1", "_in2"}, this->debug_info());
        builder.add_computational_memlet(b, norm_read, denom_op, "_in1", {}, scalar_type, this->debug_info());
        builder.add_computational_memlet(b, eps_const, denom_op, "_in2", {}, scalar_type, this->debug_info());
        builder.add_computational_memlet(b, denom_op, "_out", denom_acc, {}, scalar_type, this->debug_info());

        // ratio = max_norm / denom
        std::string ratio_name = fresh_scalar("_ratio");
        auto& maxnorm_const = builder.add_constant(b, fmt(max_norm_), scalar_type, this->debug_info());
        auto& ratio_acc = builder.add_access(b, ratio_name, this->debug_info());
        auto& div_op = builder.add_tasklet(b, data_flow::fp_div, "_out", {"_in1", "_in2"}, this->debug_info());
        builder.add_computational_memlet(b, maxnorm_const, div_op, "_in1", {}, scalar_type, this->debug_info());
        builder.add_computational_memlet(b, denom_acc, div_op, "_in2", {}, scalar_type, this->debug_info());
        builder.add_computational_memlet(b, div_op, "_out", ratio_acc, {}, scalar_type, this->debug_info());

        // scale = min(1, ratio)
        auto& one_const = builder.add_constant(b, "1.0", scalar_type, this->debug_info());
        auto& scale_acc = builder.add_access(b, scale_name, this->debug_info());
        auto& min_op =
            builder.add_library_node<cmath::CMathNode>(b, this->debug_info(), cmath::CMathFunction::fmin, prim);
        builder.add_computational_memlet(b, one_const, min_op, "_in1", {}, scalar_type, this->debug_info());
        builder.add_computational_memlet(b, ratio_acc, min_op, "_in2", {}, scalar_type, this->debug_info());
        builder.add_computational_memlet(b, min_op, "_out", scale_acc, {}, scalar_type, this->debug_info());
    }

    // In-place scaling loop: W[idx, j] *= scale.
    {
        std::string j_name = builder.find_new_name("_j_scale");
        builder.add_container(j_name, types::Scalar(types::PrimitiveType::Int64));
        auto j_sym = symbolic::symbol(j_name);
        auto& scale_loop = builder.add_map(
            *inner_scope,
            j_sym,
            symbolic::Lt(j_sym, embedding_dim),
            symbolic::zero(),
            symbolic::add(j_sym, symbolic::one()),
            structured_control_flow::ScheduleType_Sequential::create(),
            this->debug_info()
        );
        auto& body = scale_loop.root();
        symbolic::MultiExpression w_subset = {idx_sym, j_sym};

        auto& b = builder.add_block(body, {}, this->debug_info());
        auto& w_in = standalone->add_indirect_read_access(b, W_INPUT_IDX);
        auto& scale_read = builder.add_access(b, scale_name, this->debug_info());
        auto& w_out = standalone->add_indirect_write_access(b, W_INPUT_IDX);
        auto& mul_op = builder.add_tasklet(b, data_flow::fp_mul, "_out", {"_in1", "_in2"}, this->debug_info());
        builder.add_computational_memlet(b, w_in, mul_op, "_in1", w_subset, weight_edge.base_type(), this->debug_info());
        builder.add_computational_memlet(b, scale_read, mul_op, "_in2", {}, scalar_type, this->debug_info());
        builder
            .add_computational_memlet(b, mul_op, "_out", w_out, w_subset, weight_edge.base_type(), this->debug_info());
    }

    return standalone->successfully_expanded();
}

std::unique_ptr<data_flow::DataFlowNode> EmbeddingRenormNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::unique_ptr<data_flow::DataFlowNode>(new EmbeddingRenormNode(
        element_id, this->debug_info(), vertex, parent, weight_shape_, index_shape_, max_norm_, norm_type_
    ));
}

data_flow::PointerAccessType EmbeddingRenormNode::pointer_access_type(int input_idx) const {
    if (input_idx == W_INPUT_IDX) {
        return data_flow::PointerAccessMeta::
            create_generic(data_flow::MemoryAccessPatternType(), data_flow::MemoryAccessPatternType(), true);
    } else if (input_idx == INDEX_IDX) {
        return data_flow::PointerAccessMeta::create_read_only(symbolic::__nullptr__(), true);
    } else {
        return TensorNode::pointer_access_type(input_idx);
    }
}

nlohmann::json EmbeddingRenormNodeSerializer::serialize(const data_flow::LibraryNode& library_node) {
    const EmbeddingRenormNode& renorm_node = static_cast<const EmbeddingRenormNode&>(library_node);
    nlohmann::json j;

    j["code"] = renorm_node.code().value();

    serializer::JSONSerializer serializer;
    j["weight_shape"] = nlohmann::json::array();
    for (auto& dim : renorm_node.weight_shape()) {
        j["weight_shape"].push_back(serializer.expression(dim));
    }
    j["index_shape"] = nlohmann::json::array();
    for (auto& dim : renorm_node.index_shape()) {
        j["index_shape"].push_back(serializer.expression(dim));
    }
    j["max_norm"] = renorm_node.max_norm();
    j["norm_type"] = renorm_node.norm_type();

    return j;
}

data_flow::LibraryNode& EmbeddingRenormNodeSerializer::deserialize(
    const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
) {
    assert(j.contains("element_id"));
    assert(j.contains("code"));
    assert(j.contains("debug_info"));
    assert(j.contains("weight_shape"));
    assert(j.contains("index_shape"));
    assert(j.contains("max_norm"));
    assert(j.contains("norm_type"));

    std::vector<symbolic::Expression> weight_shape;
    for (const auto& dim : j["weight_shape"]) {
        weight_shape.push_back(symbolic::parse(dim.get<std::string>()));
    }
    std::vector<symbolic::Expression> index_shape;
    for (const auto& dim : j["index_shape"]) {
        index_shape.push_back(symbolic::parse(dim.get<std::string>()));
    }
    double max_norm = j["max_norm"].get<double>();
    double norm_type = j["norm_type"].get<double>();

    sdfg::serializer::JSONSerializer serializer;
    DebugInfo debug_info = serializer.json_to_debug_info(j["debug_info"]);

    return builder
        .add_library_node<EmbeddingRenormNode>(parent, debug_info, weight_shape, index_shape, max_norm, norm_type);
}

} // namespace tensor
} // namespace math
} // namespace sdfg
