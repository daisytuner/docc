#include "sdfg/data_flow/library_nodes/math/tensor/batchnorm_node.h"

#include "sdfg/analysis/scope_analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/library_nodes/math/cmath/cmath_node.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/structured_loop.h"

namespace sdfg::math::tensor {


BatchNormNode::BatchNormNode(
    size_t element_id,
    const DebugInfo& debug_info,
    graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    TensorLayout layout,
    types::PrimitiveType quantization,
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

std::unique_ptr<data_flow::DataFlowNode> BatchNormNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::unique_ptr<data_flow::DataFlowNode>(new BatchNormNode(
        element_id, debug_info(), vertex, parent, this->layout_, this->quantization_, this->implementation_type_
    ));
}

std::string BatchNormNode::toStr() const { return "BatchNorm(" + layout_.toStr() + ")"; }

struct InputContainerInfo {
    std::string name;
    bool is_const = false;
    const data_flow::Memlet* memlet;
    const data_flow::AccessNode* access_to_remove = nullptr;

    void remove_old(builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& block) const {
        if (memlet) {
            builder.remove_memlet(block, *memlet);
        }
        if (access_to_remove) {
            builder.remove_node(block, *access_to_remove);
        }
    }
};

InputContainerInfo find_usable_input_access_node(
    data_flow::DataFlowGraph& dataflow, data_flow::LibraryNode& node, const std::string& input_conn
) {
    auto* edge = dataflow.in_edge_for_connector(node, input_conn);
    if (!edge) {
        throw InvalidSDFGException(node.toStr() + " requires input on " + input_conn);
    }
    auto* access_node = dynamic_cast<const data_flow::AccessNode*>(&edge->src());
    if (!access_node) {
        throw InvalidSDFGException(node.toStr() + " requires input on " + input_conn + " to be an access node");
    }

    return {
        .name = access_node->data(),
        .is_const = !!dynamic_cast<const data_flow::ConstantNode*>(&edge->src()),
        .memlet = edge,
        .access_to_remove = access_node
    };
}

struct BuilderMapDim {
    symbolic::Expression indvar;
    structured_control_flow::StructuredLoop& loop;
    structured_control_flow::Sequence& seq;
};

std::vector<BuilderMapDim> create_maps(
    builder::StructuredSDFGBuilder& builder,
    const std::vector<symbolic::Expression>& sizes,
    structured_control_flow::Sequence& block
) {
    std::vector<BuilderMapDim> scopes;

    Sequence* last_scope = &block;

    for (size_t i = 0; i < sizes.size(); i++) {
        std::string indvar_str = builder.find_new_name("_i");
        builder.add_container(indvar_str, types::Scalar(types::PrimitiveType::UInt64));

        auto indvar = symbolic::symbol(indvar_str);
        auto init = symbolic::zero();
        auto update = symbolic::add(indvar, symbolic::one());
        auto condition = symbolic::Lt(indvar, sizes.at(i));
        auto& last_map = builder.add_map(
            *last_scope,
            indvar,
            condition,
            init,
            update,
            structured_control_flow::ScheduleType_Sequential::create(),
            {},
            block.debug_info()
        );
        auto& seq = last_map.root();
        last_scope = &seq;

        scopes.push_back(
            {.indvar = indvar, .loop = dynamic_cast<structured_control_flow::StructuredLoop&>(last_map), .seq = seq}
        );
    }

    return scopes;
}

std::string
create_temp_var(builder::StructuredSDFGBuilder& builder, const std::string& prefix, int gen, const types::IType& type) {
    std::string n = prefix + "_" + std::to_string(gen);
    auto name = builder.find_new_name(n);
    builder.add_container(name, type);
    return name;
}

bool BatchNormNode::expand(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    auto& dataflow = this->get_parent();
    auto& block = static_cast<structured_control_flow::Block&>(*dataflow.get_parent());

    auto& scope_analysis = analysis_manager.get<analysis::ScopeAnalysis>();
    auto& parent = static_cast<structured_control_flow::Sequence&>(*scope_analysis.parent_scope(&block));
    int index = parent.index(block);
    auto& transition = parent.at(index).second;

    auto batch_in = find_usable_input_access_node(dataflow, *this, "Batch");
    auto& data_type = batch_in.memlet->base_type();
    types::Scalar scalar_type(data_type.primitive_type());
    types::Tensor tensor_1d(scalar_type, {num_features()}, {symbolic::one()}); // TODO verify / get from inputs
    std::string temp_var_prefix = "_batchn_tmp";
    int tmp_idx = 0;
    auto var_in = find_usable_input_access_node(dataflow, *this, "Var");
    auto e_in = find_usable_input_access_node(dataflow, *this, "E");
    auto gamma_in = find_usable_input_access_node(dataflow, *this, "Gamma");
    auto beta_in = find_usable_input_access_node(dataflow, *this, "Beta");
    auto result_ptr_in = find_usable_input_access_node(dataflow, *this, "B_out");
    auto eps_in = find_usable_input_access_node(dataflow, *this, "epsilon");

    auto& new_sequence = builder.add_sequence_before(parent, block, transition.assignments(), debug_info());

    auto loop_dims = create_maps(builder, layout_.shape(), new_sequence);

    auto& c_dim = loop_dims.at(1);
    std::vector<symbolic::Expression> c_subset{c_dim.indvar};
    auto interm_name = builder.find_new_name("_b_sqrt_div");
    builder.add_container(interm_name, scalar_type);
    auto& inter_block = builder.add_block_before(
        c_dim.seq, static_cast<structured_control_flow::ControlFlowNode&>(loop_dims.at(2).loop), {}, DebugInfo()
    );

    auto& var_elem_in = builder.add_access(inter_block, var_in.name);
    data_flow::AccessNode& epsilon_const = eps_in.is_const ? builder.add_constant(inter_block, eps_in.name, scalar_type)
                                                           : builder.add_access(inter_block, eps_in.name);

    auto& add_eps_op = builder.add_tasklet(inter_block, data_flow::fp_add, "_out", {"var", "eps"}, debug_info());

    builder.add_computational_memlet(inter_block, var_elem_in, add_eps_op, "var", c_subset, tensor_1d);
    builder.add_computational_memlet(inter_block, epsilon_const, add_eps_op, "eps", {}, scalar_type);

    auto tmp_eps_name = create_temp_var(builder, temp_var_prefix, tmp_idx++, scalar_type);
    auto& tmp_eps = builder.add_access(inter_block, tmp_eps_name);

    builder.add_computational_memlet(inter_block, add_eps_op, "_out", tmp_eps, {}, scalar_type);

    auto tmp_sqrt_name = create_temp_var(builder, temp_var_prefix, tmp_idx++, scalar_type);
    auto& tmp_sqrt = builder.add_access(inter_block, tmp_sqrt_name);

    auto& sqrt_op = builder.add_library_node<
        cmath::CMathNode>(inter_block, debug_info(), cmath::CMathFunction::sqrt, data_type.primitive_type());

    builder.add_computational_memlet(inter_block, tmp_eps, sqrt_op, "_in1", {}, scalar_type);

    builder.add_computational_memlet(inter_block, sqrt_op, "_out", tmp_sqrt, {}, scalar_type);

    auto& one_const = builder.add_constant(inter_block, "1.0", scalar_type);
    auto& div_op = builder.add_tasklet(inter_block, data_flow::fp_div, "_out", {"one", "sqrt"});
    builder.add_computational_memlet(inter_block, one_const, div_op, "one", {}, scalar_type);
    builder.add_computational_memlet(inter_block, tmp_sqrt, div_op, "sqrt", {}, scalar_type);

    auto& interm_store = builder.add_access(inter_block, interm_name);
    builder.add_computational_memlet(inter_block, div_op, "_out", interm_store, {}, scalar_type);

    auto& innermost_dim = loop_dims.at(layout_.dims() - 1);

    std::vector<symbolic::Expression> innermost_subset;
    for (auto& builder_map_dim : loop_dims) {
        innermost_subset.push_back(builder_map_dim.indvar);
    }

    auto& innermost_block = builder.add_block(innermost_dim.seq);
    auto& x_in = builder.add_access(innermost_block, batch_in.name);
    auto& interm_in = builder.add_access(innermost_block, interm_name);
    auto& e_elem_in = builder.add_access(innermost_block, e_in.name);
    auto& gamma_elem_in = builder.add_access(innermost_block, gamma_in.name);
    auto& beta_elem_in = builder.add_access(innermost_block, beta_in.name);

    auto& result_ptr_out_elem = builder.add_access(innermost_block, result_ptr_in.name);

    auto& sub_op = builder.add_tasklet(innermost_block, data_flow::fp_sub, "_out", {"x", "e"}, debug_info());

    builder.add_computational_memlet(innermost_block, x_in, sub_op, "x", innermost_subset, data_type);
    builder.add_computational_memlet(innermost_block, e_elem_in, sub_op, "e", c_subset, tensor_1d);
    auto tmp_sub_name = create_temp_var(builder, temp_var_prefix, tmp_idx++, scalar_type);
    auto& tmp_sub = builder.add_access(innermost_block, tmp_sub_name);
    builder.add_computational_memlet(innermost_block, sub_op, "_out", tmp_sub, {}, scalar_type);

    auto& mul_interm_op = builder.add_tasklet(innermost_block, data_flow::fp_mul, "_out", {"num", "den"}, debug_info());

    builder.add_computational_memlet(innermost_block, tmp_sub, mul_interm_op, "num", {}, scalar_type);
    builder.add_computational_memlet(innermost_block, interm_in, mul_interm_op, "den", {}, scalar_type);
    auto tmp_interm = create_temp_var(builder, temp_var_prefix, tmp_idx++, scalar_type);
    auto& tmp_mul_interm = builder.add_access(innermost_block, tmp_interm);
    builder.add_computational_memlet(innermost_block, mul_interm_op, "_out", tmp_mul_interm, {}, scalar_type);

    auto& mul_gamma_op = builder.add_tasklet(innermost_block, data_flow::fp_mul, "_out", {"frac", "g"}, debug_info());

    builder.add_computational_memlet(innermost_block, tmp_mul_interm, mul_gamma_op, "frac", {}, scalar_type);
    builder.add_computational_memlet(innermost_block, gamma_elem_in, mul_gamma_op, "g", c_subset, tensor_1d);

    auto tmp_gamma = create_temp_var(builder, temp_var_prefix, tmp_idx++, scalar_type);
    auto& tmp_mul_gamma = builder.add_access(innermost_block, tmp_gamma);
    builder.add_computational_memlet(innermost_block, mul_gamma_op, "_out", tmp_mul_gamma, {}, scalar_type);

    auto& add_beta_op = builder.add_tasklet(innermost_block, data_flow::fp_add, "_out", {"_in", "b"}, debug_info());

    builder.add_computational_memlet(innermost_block, tmp_mul_gamma, add_beta_op, "_in", {}, scalar_type);
    builder.add_computational_memlet(innermost_block, beta_elem_in, add_beta_op, "b", c_subset, tensor_1d);
    builder
        .add_computational_memlet(innermost_block, add_beta_op, "_out", result_ptr_out_elem, innermost_subset, data_type);

    batch_in.remove_old(builder, block);
    var_in.remove_old(builder, block);
    e_in.remove_old(builder, block);
    eps_in.remove_old(builder, block);
    gamma_in.remove_old(builder, block);
    beta_in.remove_old(builder, block);
    result_ptr_in.remove_old(builder, block);

    builder.remove_node(block, *this);
    assert(dataflow.nodes().size() == 0 && "At expand time, no other nodes may be in the same graph");
    builder.remove_child(parent, index + 1);

    return true;
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
