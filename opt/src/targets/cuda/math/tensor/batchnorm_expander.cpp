#include "sdfg/targets/cuda/math/tensor/batchnorm_expander.h"
#include "sdfg/analysis/scope_analysis.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/library_nodes/math/cmath/cmath_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/tensor_expansion_utils.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/types/scalar.h"
#include "sdfg/types/tensor.h"

namespace sdfg {
namespace offloading {

bool CudaBatchNormExpander::expand(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    // GPU implementation of batchnorm:
    // Move sqrt and division into the innermost loop to enable more parallelism.
    using math::tensor::create_maps;
    using math::tensor::create_temp_var;
    using math::tensor::find_usable_input_access_node;

    auto& dataflow = node_.get_parent();
    auto& block = static_cast<structured_control_flow::Block&>(*dataflow.get_parent());

    auto& scope_analysis = analysis_manager.get<analysis::ScopeAnalysis>();
    auto& parent = static_cast<structured_control_flow::Sequence&>(*scope_analysis.parent_scope(&block));
    int index = parent.index(block);
    auto& transition = parent.at(index).second;

    auto batch_in = find_usable_input_access_node(dataflow, node_, "Batch");
    auto& data_type = batch_in.memlet->base_type();
    types::Scalar scalar_type(data_type.primitive_type());
    types::Tensor tensor_1d(scalar_type, {node_.num_features()}, {symbolic::one()});
    std::string temp_var_prefix = "_batchn_tmp";
    int tmp_idx = 0;
    auto var_in = find_usable_input_access_node(dataflow, node_, "Var");
    auto e_in = find_usable_input_access_node(dataflow, node_, "E");
    auto gamma_in = find_usable_input_access_node(dataflow, node_, "Gamma");
    auto beta_in = find_usable_input_access_node(dataflow, node_, "Beta");
    auto result_ptr_in = find_usable_input_access_node(dataflow, node_, "B_out");
    auto eps_in = find_usable_input_access_node(dataflow, node_, "epsilon");

    auto& new_sequence = builder.add_sequence_before(parent, block, transition.assignments(), node_.debug_info());

    auto loop_dims = create_maps(builder, node_.batch_layout().shape(), new_sequence);

    auto& c_dim = loop_dims.at(1);
    std::vector<symbolic::Expression> c_subset{c_dim.indvar};

    auto& innermost_dim = loop_dims.at(node_.batch_layout().dims() - 1);

    std::vector<symbolic::Expression> innermost_subset;
    for (auto& builder_map_dim : loop_dims) {
        innermost_subset.push_back(builder_map_dim.indvar);
    }
    auto& innermost_block = builder.add_block(innermost_dim.seq);

    // Access nodes
    auto& x_in = builder.add_access(innermost_block, batch_in.name);
    auto& var_elem_in = builder.add_access(innermost_block, var_in.name);
    data_flow::AccessNode& epsilon_const = eps_in.is_const
                                               ? builder.add_constant(innermost_block, eps_in.name, scalar_type)
                                               : builder.add_access(innermost_block, eps_in.name);
    auto& e_elem_in = builder.add_access(innermost_block, e_in.name);
    auto& gamma_elem_in = builder.add_access(innermost_block, gamma_in.name);
    auto& beta_elem_in = builder.add_access(innermost_block, beta_in.name);
    auto& result_ptr_out_elem = builder.add_access(innermost_block, result_ptr_in.name);

    // var[c] + eps
    auto& add_eps_op =
        builder.add_tasklet(innermost_block, data_flow::fp_add, "_out", {"var", "eps"}, node_.debug_info());
    builder.add_computational_memlet(innermost_block, var_elem_in, add_eps_op, "var", c_subset, tensor_1d);
    builder.add_computational_memlet(innermost_block, epsilon_const, add_eps_op, "eps", {}, scalar_type);
    auto tmp_eps_name = create_temp_var(builder, temp_var_prefix, tmp_idx++, scalar_type);
    auto& tmp_eps = builder.add_access(innermost_block, tmp_eps_name);
    builder.add_computational_memlet(innermost_block, add_eps_op, "_out", tmp_eps, {}, scalar_type);

    // sqrt(var[c] + eps)
    auto tmp_sqrt_name = create_temp_var(builder, temp_var_prefix, tmp_idx++, scalar_type);
    auto& tmp_sqrt = builder.add_access(innermost_block, tmp_sqrt_name);
    auto& sqrt_op = builder.add_library_node<math::cmath::CMathNode>(
        innermost_block, node_.debug_info(), math::cmath::CMathFunction::sqrt, data_type.primitive_type()
    );
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
    auto& sub_op = builder.add_tasklet(innermost_block, data_flow::fp_sub, "_out", {"x", "e"}, node_.debug_info());
    builder.add_computational_memlet(innermost_block, x_in, sub_op, "x", innermost_subset, data_type);
    builder.add_computational_memlet(innermost_block, e_elem_in, sub_op, "e", c_subset, tensor_1d);
    auto tmp_sub_name = create_temp_var(builder, temp_var_prefix, tmp_idx++, scalar_type);
    auto& tmp_sub = builder.add_access(innermost_block, tmp_sub_name);
    builder.add_computational_memlet(innermost_block, sub_op, "_out", tmp_sub, {}, scalar_type);

    // (x - e[c]) * (1/sqrt(var[c]+eps))
    auto& mul_interm_op =
        builder.add_tasklet(innermost_block, data_flow::fp_mul, "_out", {"num", "den"}, node_.debug_info());
    builder.add_computational_memlet(innermost_block, tmp_sub, mul_interm_op, "num", {}, scalar_type);
    builder.add_computational_memlet(innermost_block, interm_store, mul_interm_op, "den", {}, scalar_type);
    auto tmp_interm = create_temp_var(builder, temp_var_prefix, tmp_idx++, scalar_type);
    auto& tmp_mul_interm = builder.add_access(innermost_block, tmp_interm);
    builder.add_computational_memlet(innermost_block, mul_interm_op, "_out", tmp_mul_interm, {}, scalar_type);

    // * gamma[c]
    auto& mul_gamma_op =
        builder.add_tasklet(innermost_block, data_flow::fp_mul, "_out", {"frac", "g"}, node_.debug_info());
    builder.add_computational_memlet(innermost_block, tmp_mul_interm, mul_gamma_op, "frac", {}, scalar_type);
    builder.add_computational_memlet(innermost_block, gamma_elem_in, mul_gamma_op, "g", c_subset, tensor_1d);
    auto tmp_gamma = create_temp_var(builder, temp_var_prefix, tmp_idx++, scalar_type);
    auto& tmp_mul_gamma = builder.add_access(innermost_block, tmp_gamma);
    builder.add_computational_memlet(innermost_block, mul_gamma_op, "_out", tmp_mul_gamma, {}, scalar_type);

    // + beta[c]
    auto& add_beta_op =
        builder.add_tasklet(innermost_block, data_flow::fp_add, "_out", {"_in", "b"}, node_.debug_info());
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
    builder.remove_node(block, node_);
    assert(dataflow.nodes().size() == 0 && "At expand time, no other nodes may be in the same graph");
    builder.remove_child(parent, index + 1);
    return true;
}

} // namespace offloading
} // namespace sdfg
