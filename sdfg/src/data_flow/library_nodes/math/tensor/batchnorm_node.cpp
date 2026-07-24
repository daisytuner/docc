#include "sdfg/data_flow/library_nodes/math/tensor/batchnorm_node.h"

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/library_nodes/math/cmath/cmath_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/tensor_expansion_utils.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/structured_loop.h"

namespace sdfg::math::tensor {


BatchNormNode::BatchNormNode(
    size_t element_id,
    const DebugInfo& debug_info,
    graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    TensorLayout layout,
    QuantizationType quantization,
    data_flow::ImplementationType impl_type
)
    : TensorNode(
          element_id,
          debug_info,
          vertex,
          parent,
          LibraryNodeType_BatchNorm,
          {},
          {"Batch", "Var", "E", "Gamma", "Beta", "epsilon", "B_out"},
          std::move(impl_type)
      ),
      layout_(std::move(layout)), quantization_(quantization) {}

symbolic::SymbolSet BatchNormNode::symbols() const {
    symbolic::SymbolSet syms;
    layout_.collect_symbols(syms);
    return syms;
}

types::PrimitiveType BatchNormNode::quantization() const { return quantization_; }

void BatchNormNode::set_quantization(const types::PrimitiveType quant) { quantization_ = quant; }

void BatchNormNode::replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) {
    layout_.replace_symbols(old_expression, new_expression);
}

void BatchNormNode::replace(const symbolic::ExpressionMapping& replacements) { layout_.replace_symbols(replacements); }

std::unique_ptr<data_flow::DataFlowNode> BatchNormNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::unique_ptr<data_flow::DataFlowNode>(new BatchNormNode(
        element_id, debug_info(), vertex, parent, this->layout_, this->quantization_, this->implementation_type_
    ));
}

std::string BatchNormNode::toStr() const { return "BatchNorm(" + layout_.toStr() + ")"; }

passes::LibNodeExpander::ExpandOutcome BatchNormNode::
    expand(passes::LibNodeExpander::ExpandContext& context, structured_control_flow::Block& block) {
    auto& dataflow = this->get_parent();

    auto* batch_iedge = dataflow.in_edge_for_connector(*this, "Batch");
    auto& data_type = batch_iedge->base_type();
    types::Scalar scalar_type(data_type.primitive_type());
    types::Tensor tensor_1d(scalar_type, {num_features()}, {symbolic::one()});
    std::string temp_var_prefix = "_batchn_tmp";
    int tmp_idx = 0;

    //{"Batch", "Var", "E", "Gamma", "Beta", "epsilon", "B_out"},

    constexpr auto BATCH_IDX = 0;
    constexpr auto VAR_IDX = 1;
    constexpr auto E_IDX = 2;
    constexpr auto GAMMA_IDX = 3;
    constexpr auto BETA_IDX = 4;
    constexpr auto EPS_IDX = 5;
    constexpr auto B_OUT_IDX = 6;
    using Use = passes::LibNodeExpander::InputUse;
    auto standalone = context.replacement_requires_access_nodes(
        {Use::IndirectRead,
         Use::IndirectRead,
         Use::IndirectRead,
         Use::IndirectRead,
         Use::IndirectRead,
         Use::Scalar,
         Use::IndirectWrite}
    );

    if (!standalone) {
        return context.unable();
    }

    auto& new_sequence = standalone->replace_with_sequence();
    auto& builder = standalone->builder();

    auto loop_dims = create_maps(builder, layout_.shape(), new_sequence);

    // GPU implementation of batchnorm:
    // Move sqrt and division into the innermost loop to enable more parallelism.

    auto& c_dim = loop_dims.at(1);
    std::vector<symbolic::Expression> c_subset{c_dim.indvar};

    auto& innermost_dim = loop_dims.at(layout_.dims() - 1);

    std::vector<symbolic::Expression> innermost_subset;
    for (auto& builder_map_dim : loop_dims) {
        innermost_subset.push_back(builder_map_dim.indvar);
    }

    auto& innermost_block = builder.add_block(innermost_dim.seq);

    // Access nodes
    auto& x_in = standalone->add_indirect_read_access(innermost_block, BATCH_IDX);
    auto& var_elem_in = standalone->add_indirect_read_access(innermost_block, VAR_IDX);
    data_flow::AccessNode& epsilon_const = standalone->add_scalar_input_access(innermost_block, EPS_IDX);
    auto& e_elem_in = standalone->add_indirect_read_access(innermost_block, E_IDX);
    auto& gamma_elem_in = standalone->add_indirect_read_access(innermost_block, GAMMA_IDX);
    auto& beta_elem_in = standalone->add_indirect_read_access(innermost_block, BETA_IDX);
    auto& result_ptr_out_elem = standalone->add_indirect_write_access(innermost_block, B_OUT_IDX);

    // var[c] + eps
    auto& add_eps_op = builder.add_tasklet(innermost_block, data_flow::fp_add, "_out", {"var", "eps"}, debug_info());
    builder.add_computational_memlet(innermost_block, var_elem_in, add_eps_op, "var", c_subset, tensor_1d);
    builder.add_computational_memlet(innermost_block, epsilon_const, add_eps_op, "eps", {}, scalar_type);
    auto tmp_eps_name = create_temp_var(builder, temp_var_prefix, tmp_idx++, scalar_type);
    auto& tmp_eps = builder.add_access(innermost_block, tmp_eps_name);
    builder.add_computational_memlet(innermost_block, add_eps_op, "_out", tmp_eps, {}, scalar_type);

    // sqrt(var[c] + eps)
    auto tmp_sqrt_name = create_temp_var(builder, temp_var_prefix, tmp_idx++, scalar_type);
    auto& tmp_sqrt = builder.add_access(innermost_block, tmp_sqrt_name);
    auto& sqrt_op = builder.add_library_node<
        cmath::CMathNode>(innermost_block, debug_info(), cmath::CMathFunction::sqrt, data_type.primitive_type());
    builder.add_computational_memlet(innermost_block, tmp_eps, sqrt_op, "_in1", {}, scalar_type);
    builder.add_computational_memlet(innermost_block, sqrt_op, "_out", tmp_sqrt, {}, scalar_type);

    // 1.0 / sqrt(var[c] + eps)
    auto& one_const = builder.add_constant(innermost_block, "1.0", scalar_type);
    auto& div_op = builder.add_tasklet(innermost_block, data_flow::fp_div, "_out", {"one", "sqrt"});
    builder.add_computational_memlet(innermost_block, one_const, div_op, "one", {}, scalar_type);
    builder.add_computational_memlet(innermost_block, tmp_sqrt, div_op, "sqrt", {}, scalar_type);
    auto interm_name = create_temp_var(builder, temp_var_prefix, tmp_idx++, scalar_type);
    auto& interm_store = builder.add_access(innermost_block, interm_name);
    builder.add_computational_memlet(innermost_block, div_op, "_out", interm_store, {}, scalar_type);

    // x - e[c]
    auto& sub_op = builder.add_tasklet(innermost_block, data_flow::fp_sub, "_out", {"x", "e"}, debug_info());
    builder.add_computational_memlet(innermost_block, x_in, sub_op, "x", innermost_subset, data_type);
    builder.add_computational_memlet(innermost_block, e_elem_in, sub_op, "e", c_subset, tensor_1d);
    auto tmp_sub_name = create_temp_var(builder, temp_var_prefix, tmp_idx++, scalar_type);
    auto& tmp_sub = builder.add_access(innermost_block, tmp_sub_name);
    builder.add_computational_memlet(innermost_block, sub_op, "_out", tmp_sub, {}, scalar_type);

    // (x - e[c]) * (1/sqrt(var[c]+eps))
    auto& mul_interm_op = builder.add_tasklet(innermost_block, data_flow::fp_mul, "_out", {"num", "den"}, debug_info());
    builder.add_computational_memlet(innermost_block, tmp_sub, mul_interm_op, "num", {}, scalar_type);
    builder.add_computational_memlet(innermost_block, interm_store, mul_interm_op, "den", {}, scalar_type);
    auto tmp_interm = create_temp_var(builder, temp_var_prefix, tmp_idx++, scalar_type);
    auto& tmp_mul_interm = builder.add_access(innermost_block, tmp_interm);
    builder.add_computational_memlet(innermost_block, mul_interm_op, "_out", tmp_mul_interm, {}, scalar_type);

    // * gamma[c]
    auto& mul_gamma_op = builder.add_tasklet(innermost_block, data_flow::fp_mul, "_out", {"frac", "g"}, debug_info());
    builder.add_computational_memlet(innermost_block, tmp_mul_interm, mul_gamma_op, "frac", {}, scalar_type);
    builder.add_computational_memlet(innermost_block, gamma_elem_in, mul_gamma_op, "g", c_subset, tensor_1d);
    auto tmp_gamma = create_temp_var(builder, temp_var_prefix, tmp_idx++, scalar_type);
    auto& tmp_mul_gamma = builder.add_access(innermost_block, tmp_gamma);
    builder.add_computational_memlet(innermost_block, mul_gamma_op, "_out", tmp_mul_gamma, {}, scalar_type);

    // + beta[c]
    auto& add_beta_op = builder.add_tasklet(innermost_block, data_flow::fp_add, "_out", {"_in", "b"}, debug_info());
    builder.add_computational_memlet(innermost_block, tmp_mul_gamma, add_beta_op, "_in", {}, scalar_type);
    builder.add_computational_memlet(innermost_block, beta_elem_in, add_beta_op, "b", c_subset, tensor_1d);
    builder
        .add_computational_memlet(innermost_block, add_beta_op, "_out", result_ptr_out_elem, innermost_subset, data_type);

    return standalone->successfully_expanded();
}

symbolic::Expression BatchNormNode::flop() const {
    auto inner_elems = symbolic::mul(layout_.get_dim_innermost(0), layout_.get_dim_innermost(1));
    auto outer_elems = symbolic::mul(layout_.shape().at(0), layout_.shape().at(1));

    // (x-e) * sqrt_pre_calc * g + b = 4 flops
    auto inner_flops = symbolic::mul(symbolic::integer(4), inner_elems);
    // sqrt_pre_calc = 1/sqrt(var + eps) // 3 flops
    auto outer_flops = symbolic::mul(symbolic::add(inner_flops, symbolic::integer(3)), outer_elems);
    return outer_flops;
}

data_flow::PointerAccessType BatchNormNode::pointer_access_type(int input_idx) const {
    if (input_idx >= 0 && input_idx <= 4) {
        return data_flow::PointerAccessMeta::create_read_only(symbolic::__nullptr__(), true);
    } else if (input_idx == 6) {
        return data_flow::PointerAccessMeta::create_full_write_only(symbolic::__nullptr__(), true);
    } else {
        return TensorNode::pointer_access_type(input_idx);
    }
}

nlohmann::json BatchNormNodeSerializer::serialize(const data_flow::LibraryNode& library_node) {
    auto& node = static_cast<const BatchNormNode&>(library_node);
    nlohmann::json j;

    j["code"] = node.code().value();

    node.batch_layout().serialize_to_json(j["batch_layout"]);

    j["batch_quant"] = node.quantization();

    return j;
}

data_flow::LibraryNode& BatchNormNodeSerializer::deserialize(
    const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
) {
    auto layout = TensorLayout::deserialize_from_json(j.at("batch_layout"));
    auto quant = j.at("batch_quant").get<types::PrimitiveType>();

    serializer::JSONSerializer serializer;
    auto deb_info = serializer.json_to_debug_info(j.at("debug_info"));

    return builder.add_library_node<BatchNormNode>(parent, deb_info, layout, quant);
}

} // namespace sdfg::math::tensor
