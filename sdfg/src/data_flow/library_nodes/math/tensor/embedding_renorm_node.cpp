#include "sdfg/data_flow/library_nodes/math/tensor/embedding_renorm_node.h"

#include <string>
#include <vector>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/library_nodes/math/cmath/cmath_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/copy_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/tensor_layout.h"
#include "sdfg/data_flow/pointer_metadata.h"
#include "sdfg/exceptions.h"
#include "sdfg/structured_control_flow/for.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/types/scalar.h"
#include "sdfg/types/type.h"

namespace sdfg {
namespace math {
namespace tensor {

EmbeddingRenormNode::EmbeddingRenormNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const TensorLayout& y_layout,
    const TensorLayout& weight_layout,
    const TensorLayout& indices_layout,
    const data_flow::ImplementationType& impl_type
)
    : TensorNode(
          element_id,
          debug_info,
          vertex,
          parent,
          LibraryNodeType_EmbeddingRenorm,
          {},
          {"Y", "Weight", "Indices", "MaxNorm", "NormType"},
          impl_type
      ),
      y_layout_(y_layout), weight_layout_(weight_layout), indices_layout_(indices_layout) {}

const TensorLayout& EmbeddingRenormNode::y_layout() const { return this->y_layout_; }

const TensorLayout& EmbeddingRenormNode::weight_layout() const { return this->weight_layout_; }

const TensorLayout& EmbeddingRenormNode::indices_layout() const { return this->indices_layout_; }

bool EmbeddingRenormNode::supports_integer_types() const { return false; }

void EmbeddingRenormNode::validate(const Function& function) const {
    auto& graph = this->get_parent();

    // The index tensor is integer-typed while the weight is floating-point, so the
    // uniform-primitive-type check in TensorNode::validate does not apply. Validate at
    // the MathNode level and add our own structural checks (mirrors EmbeddingNode).
    MathNode::validate(function);

    // Check presence of in and out edges
    if (graph.out_degree(*this) != 0) {
        throw InvalidSDFGException(
            "EmbeddingRenormNode: Expected no outputs but got: " + std::to_string(graph.out_degree(*this))
        );
    }
    if (graph.in_degree(*this) != 5) {
        throw InvalidSDFGException(
            "EmbeddingRenormNode: Expexted 5 inputs but got: " + std::to_string(graph.in_degree(*this))
        );
    }
    const auto* y_edge = graph.in_edge_for_connector(*this, "Y");
    if (!y_edge) {
        throw InvalidSDFGException("EmbeddingRenormNode: No memlet connected at connector: Y");
    }
    const auto* weight_edge = graph.in_edge_for_connector(*this, "Weight");
    if (!weight_edge) {
        throw InvalidSDFGException("EmbeddingRenormNode: No memlet connected at connector: Weight");
    }
    const auto* indices_edge = graph.in_edge_for_connector(*this, "Indices");
    if (!indices_edge) {
        throw InvalidSDFGException("EmbeddingRenormNode: No memlet connected at connector: Indices");
    }
    const auto* max_norm_edge = graph.in_edge_for_connector(*this, "MaxNorm");
    if (!max_norm_edge) {
        throw InvalidSDFGException("EmbeddingRenormNode: No memlet connected at connector: MaxNorm");
    }
    const auto* norm_type_edge = graph.in_edge_for_connector(*this, "NormType");
    if (!norm_type_edge) {
        throw InvalidSDFGException("EmbeddingRenormNode: No memlet connected at connector: NormType");
    }

    // Check that the in edges for Y, Weight, and Indices have tensor types as base types
    if (y_edge->base_type().type_id() != types::TypeID::Tensor) {
        throw InvalidSDFGException(
            "EmbeddingRenormNode: Expected tensor type at connector 'Y' but got: " + y_edge->base_type().print()
        );
    }
    const types::Tensor& y_tensor = static_cast<const types::Tensor&>(y_edge->base_type());
    if (weight_edge->base_type().type_id() != types::TypeID::Tensor) {
        throw InvalidSDFGException(
            "EmbeddingRenormNode: Expected tensor type at connector 'Weight' but got: " +
            weight_edge->base_type().print()
        );
    }
    const types::Tensor& weight_tensor = static_cast<const types::Tensor&>(weight_edge->base_type());
    if (indices_edge->base_type().type_id() != types::TypeID::Tensor) {
        throw InvalidSDFGException(
            "EmbeddingRenormNode: Expected tensor type at connector 'Indices' but got: " +
            indices_edge->base_type().print()
        );
    }
    const types::Tensor& indices_tensor = static_cast<const types::Tensor&>(indices_edge->base_type());

    // Check that the tensor layouts match with the tensor types on the edges
    if (y_tensor.layout() != this->y_layout_) {
        throw InvalidSDFGException(
            "EmbeddingRenormNode: Provided tensor layout does not match the memlet tensor type for connector 'Y': " +
            y_tensor.layout().toStr() + " != " + this->y_layout_.toStr()
        );
    }
    if (weight_tensor.layout() != this->weight_layout_) {
        throw InvalidSDFGException(
            "EmbeddingRenormNode: Provided tensor layout does not match the memlet tensor type for connector "
            "'Weight': " +
            weight_tensor.layout().toStr() + " != " + this->weight_layout_.toStr()
        );
    }
    if (indices_tensor.layout() != this->indices_layout_) {
        throw InvalidSDFGException(
            "EmbeddingRenormNode: Provided tensor layout does not match the memlet tensor type for connector "
            "'Indices': " +
            indices_tensor.layout().toStr() + " != " + this->indices_layout_.toStr()
        );
    }

    // Check shapes
    if (this->y_layout_.dims() != 2) {
        throw InvalidSDFGException(
            "EmbeddingRenormNode: Y must be 2-dimensional but got rank " + std::to_string(this->y_layout_.dims())
        );
    }
    if (this->weight_layout_.dims() != 2) {
        throw InvalidSDFGException(
            "EmbeddingRenormNode: Weight must be 2-dimensional but got rank " +
            std::to_string(this->weight_layout_.dims())
        );
    }

    // Check that the primitive type of Y, Weight, MaxNorm, and NormType are floating point
    if (!types::is_floating_point(y_tensor.primitive_type())) {
        throw InvalidSDFGException(
            "EmbeddingRenormNode: Expected Y tensor to be floating point but got: " +
            std::string(types::primitive_type_to_string(y_tensor.primitive_type()))
        );
    }
    if (!types::is_floating_point(weight_tensor.primitive_type())) {
        throw InvalidSDFGException(
            "EmbeddingRenormNode: Expected Weight tensor to be floating point but got: " +
            std::string(types::primitive_type_to_string(weight_tensor.primitive_type()))
        );
    }
    if (!types::is_floating_point(max_norm_edge->base_type().primitive_type())) {
        throw InvalidSDFGException(
            "EmbeddingRenormNode: Expected MaxNorm type to be floating point but got: " +
            std::string(types::primitive_type_to_string(max_norm_edge->base_type().primitive_type()))
        );
    }
    if (!types::is_floating_point(norm_type_edge->base_type().primitive_type())) {
        throw InvalidSDFGException(
            "EmbeddingRenormNode: Expected NormType type to be floating point but got: " +
            std::string(types::primitive_type_to_string(norm_type_edge->base_type().primitive_type()))
        );
    }

    // Check that the primitive type of Indices is integer
    if (!types::is_integer(indices_tensor.primitive_type())) {
        throw InvalidSDFGException(
            "EmbeddingRenormNode: Expected Indices tensor to be integer but got: " +
            std::string(types::primitive_type_to_string(indices_tensor.primitive_type()))
        );
    }
}

symbolic::SymbolSet EmbeddingRenormNode::symbols() const {
    symbolic::SymbolSet syms;
    this->y_layout_.collect_symbols(syms);
    this->weight_layout_.collect_symbols(syms);
    this->indices_layout_.collect_symbols(syms);
    return syms;
}

void EmbeddingRenormNode::replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) {
    this->y_layout_.replace_symbols(old_expression, new_expression);
    this->weight_layout_.replace_symbols(old_expression, new_expression);
    this->indices_layout_.replace_symbols(old_expression, new_expression);
}

void EmbeddingRenormNode::replace(const symbolic::ExpressionMapping& replacements) {
    this->y_layout_.replace_symbols(replacements);
    this->weight_layout_.replace_symbols(replacements);
    this->indices_layout_.replace_symbols(replacements);
}

passes::LibNodeExpander::ExpandOutcome EmbeddingRenormNode::
    expand(passes::LibNodeExpander::ExpandContext& context, structured_control_flow::Block& block) {
    auto& dataflow = this->get_parent();

    if (dataflow.in_degree(*this) != 5 || dataflow.out_degree(*this) != 0) {
        return context.unable();
    }

    auto edges = dataflow.in_edges_by_connector(*this);
    auto& y_edge = *edges.at(Y_INPUT_IDX);
    auto& weight_edge = *edges.at(WEIGHT_INPUT_IDX);
    auto& indices_edge = *edges.at(INDICES_INPUT_IDX);
    auto& max_norm_edge = *edges.at(MAX_NORM_INPUT_IDX);
    auto& norm_type_edge = *edges.at(NORM_TYPE_INPUT_IDX);

    using Use = passes::LibNodeExpander::InputUse;
    std::vector<Use> uses = {Use::IndirectWrite, Use::IndirectRead, Use::IndirectRead, Use::Scalar, Use::Scalar};
    auto standalone = context.replacement_requires_access_nodes(uses);
    if (!standalone) {
        return context.unable();
    }

    auto& builder = standalone->builder();
    auto& new_sequence = standalone->replace_with_sequence();
    const types::PrimitiveType prim = weight_edge.base_type().primitive_type();
    const types::Scalar scalar_type(prim);
    const symbolic::Expression embedding_dim = this->weight_layout_.get_dim(1);
    auto& norm_type_old_access = static_cast<data_flow::AccessNode&>(norm_type_edge.src());
    const bool p_is_inf = norm_type_old_access.data() == "INFINITY" || norm_type_old_access.data() == "-INFINITY";
    const bool p_is_one = data_flow::AccessNode::has_constant_value(norm_type_old_access, 1);
    const bool p_is_two = data_flow::AccessNode::has_constant_value(norm_type_old_access, 2);

    auto fresh_scalar = [&](const std::string& prefix) {
        std::string name = builder.find_new_name(prefix);
        builder.add_container(name, scalar_type);
        return name;
    };

    // Copy weight in y
    {
        auto& block = builder.add_block(new_sequence, this->debug_info_);
        auto& weight_access = standalone->add_indirect_read_access(block, WEIGHT_INPUT_IDX);
        auto& y_access = standalone->add_indirect_write_access(block, Y_INPUT_IDX);
        auto& copy_node =
            builder.add_library_node<TensorCopyNode>(block, this->debug_info_, this->weight_layout_, this->y_layout_);
        builder.add_computational_memlet(
            block, weight_access, copy_node, "X", {}, weight_edge.base_type(), weight_edge.debug_info()
        );
        builder.add_computational_memlet(block, y_access, copy_node, "Y", {}, y_edge.base_type(), y_edge.debug_info());
    }

    // Sequential loops over the index tensor. Sequential ordering makes the in-place
    // renormalization idempotent for duplicate indices (see class docs).
    size_t nB = this->indices_layout_.dims();
    symbolic::MultiExpression loop_vars;
    structured_control_flow::Sequence* inner_scope = &new_sequence;
    for (size_t i = 0; i < nB; ++i) {
        std::string var_name = builder.find_new_name("_i" + std::to_string(i));
        builder.add_container(var_name, types::Scalar(types::PrimitiveType::Int64));
        auto sym_var = symbolic::symbol(var_name);
        auto condition = symbolic::Lt(sym_var, this->indices_layout_.get_dim(i));
        auto init = symbolic::zero();
        auto update = symbolic::add(sym_var, symbolic::one());
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
        loop_vars.push_back(sym_var);
    }

    // Load the data-dependent row coordinate idx = I[loop_vars] into a scalar symbol.
    std::string idx_name = builder.find_new_name("_idx");
    builder.add_container(idx_name, types::Scalar(types::PrimitiveType::Int64));
    auto idx_sym = symbolic::symbol(idx_name);
    {
        auto& load_block = builder.add_block(*inner_scope, {}, this->debug_info());
        auto& idx_in_acc = standalone->add_indirect_read_access(load_block, INDICES_INPUT_IDX);
        auto& idx_out_acc = builder.add_access(load_block, idx_name, this->debug_info());
        auto& load_tasklet =
            builder.add_tasklet(load_block, data_flow::TaskletCode::assign, "_out", {"_in"}, this->debug_info());
        builder.add_computational_memlet(
            load_block, idx_in_acc, load_tasklet, "_in", loop_vars, indices_edge.base_type(), indices_edge.debug_info()
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
        auto& w_in = standalone->add_indirect_read_access(b, WEIGHT_INPUT_IDX);
        std::string aw_name = fresh_scalar("_aw");
        auto& aw_acc = builder.add_access(b, aw_name, this->debug_info());
        auto& abs_op =
            builder.add_library_node<cmath::CMathNode>(b, this->debug_info(), cmath::CMathFunction::fabs, prim);
        builder.add_computational_memlet(
            b, w_in, abs_op, "_in1", w_subset, weight_edge.base_type(), weight_edge.debug_info()
        );
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
            auto& p_const = standalone->add_scalar_input_access(b, NORM_TYPE_INPUT_IDX);
            auto& pow_op =
                builder.add_library_node<cmath::CMathNode>(b, this->debug_info(), cmath::CMathFunction::pow, prim);
            builder.add_computational_memlet(b, aw_acc, pow_op, "_in1", {}, scalar_type, this->debug_info());
            builder.add_computational_memlet(
                b, p_const, pow_op, "_in2", {}, norm_type_edge.base_type(), norm_type_edge.debug_info()
            );
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
            auto inv_p_container = fresh_scalar("_inv_p");
            auto& constant_one = builder.add_constant(b, "1.0", scalar_type);
            auto& p_const = standalone->add_scalar_input_access(b, NORM_TYPE_INPUT_IDX);
            auto& inv_p_const = builder.add_access(b, inv_p_container, this->debug_info_);
            auto& div_op =
                builder.add_tasklet(b, data_flow::TaskletCode::fp_div, "_out", {"_in1", "_in2"}, this->debug_info_);
            builder.add_computational_memlet(b, constant_one, div_op, "_in1", {}, this->debug_info_);
            builder.add_computational_memlet(
                b, p_const, div_op, "_in2", {}, norm_type_edge.base_type(), norm_type_edge.debug_info()
            );
            builder.add_computational_memlet(b, div_op, "_out", inv_p_const, {}, this->debug_info_);
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
        auto& maxnorm_const = standalone->add_scalar_input_access(b, MAX_NORM_INPUT_IDX);
        auto& ratio_acc = builder.add_access(b, ratio_name, this->debug_info());
        auto& div_op = builder.add_tasklet(b, data_flow::fp_div, "_out", {"_in1", "_in2"}, this->debug_info());
        builder.add_computational_memlet(
            b, maxnorm_const, div_op, "_in1", {}, max_norm_edge.base_type(), max_norm_edge.debug_info()
        );
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
        auto& w_in = standalone->add_indirect_read_access(b, WEIGHT_INPUT_IDX);
        auto& scale_read = builder.add_access(b, scale_name, this->debug_info());
        auto& w_out = standalone->add_indirect_write_access(b, Y_INPUT_IDX);
        auto& mul_op = builder.add_tasklet(b, data_flow::fp_mul, "_out", {"_in1", "_in2"}, this->debug_info());
        builder.add_computational_memlet(
            b, w_in, mul_op, "_in1", w_subset, weight_edge.base_type(), weight_edge.debug_info()
        );
        builder.add_computational_memlet(b, scale_read, mul_op, "_in2", {}, scalar_type, this->debug_info());
        builder.add_computational_memlet(b, mul_op, "_out", w_out, w_subset, y_edge.base_type(), y_edge.debug_info());
    }

    return standalone->successfully_expanded();
}

std::unique_ptr<data_flow::DataFlowNode> EmbeddingRenormNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::unique_ptr<data_flow::DataFlowNode>(new EmbeddingRenormNode(
        element_id,
        this->debug_info_,
        vertex,
        parent,
        this->y_layout_,
        this->weight_layout_,
        this->indices_layout_,
        this->implementation_type_
    ));
}

data_flow::PointerAccessType EmbeddingRenormNode::pointer_access_type(int input_idx) const {
    switch (input_idx) {
        case Y_INPUT_IDX:
            return data_flow::PointerAccessMeta::create_full_write_only(this->y_layout_.total_elements(), true);
        case WEIGHT_INPUT_IDX:
            return data_flow::PointerAccessMeta::create_read_only(this->weight_layout_.total_elements(), true);
        case INDICES_INPUT_IDX:
            return data_flow::PointerAccessMeta::create_read_only(this->indices_layout_.total_elements(), true);
        default:
            return nullptr;
    }
}

nlohmann::json EmbeddingRenormNodeSerializer::serialize(const data_flow::LibraryNode& library_node) {
    const EmbeddingRenormNode& renorm_node = static_cast<const EmbeddingRenormNode&>(library_node);
    nlohmann::json j;

    j["code"] = renorm_node.code().value();

    renorm_node.y_layout().serialize_to_json(j["y_layout"]);
    renorm_node.weight_layout().serialize_to_json(j["weight_layout"]);
    renorm_node.indices_layout().serialize_to_json(j["indices_layout"]);

    return j;
}

data_flow::LibraryNode& EmbeddingRenormNodeSerializer::deserialize(
    const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
) {
    assert(j.contains("element_id"));
    assert(j.contains("code"));
    assert(j.contains("debug_info"));
    assert(j.contains("y_layout"));
    assert(j.contains("weight_layout"));
    assert(j.contains("indices_layout"));

    auto y_layout = TensorLayout::deserialize_from_json(j.at("y_layout"));
    auto weight_layout = TensorLayout::deserialize_from_json(j.at("weight_layout"));
    auto indices_layout = TensorLayout::deserialize_from_json(j.at("indices_layout"));

    sdfg::serializer::JSONSerializer serializer;
    DebugInfo debug_info = serializer.json_to_debug_info(j["debug_info"]);

    return builder.add_library_node<EmbeddingRenormNode>(parent, debug_info, y_layout, weight_layout, indices_layout);
}

} // namespace tensor
} // namespace math
} // namespace sdfg
