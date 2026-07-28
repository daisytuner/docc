#include "sdfg/data_flow/library_nodes/math/tensor/elementwise_ops/gelu_node.h"

#include <cstddef>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <nlohmann/json_fwd.hpp>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/data_flow_graph.h"
#include "sdfg/data_flow/data_flow_node.h"
#include "sdfg/data_flow/library_node.h"
#include "sdfg/data_flow/library_nodes/math/cmath/cmath_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/elementwise_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/tensor_node.h"
#include "sdfg/element.h"
#include "sdfg/graph/graph.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/scalar.h"
#include "sdfg/types/type.h"

namespace sdfg {
namespace math {
namespace tensor {

GELUNode::GELUNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const std::vector<symbolic::Expression>& shape,
    bool tanh_approx,
    QuantizationType quantization,
    const data_flow::ImplementationType& impl_type
)
    : ElementWiseDataflowTensorNode(
          element_id, debug_info, vertex, parent, LibraryNodeType_GELU, shape, "Y", {"X"}, quantization, impl_type
      ),
      tanh_approx_(tanh_approx) {}

bool GELUNode::tanh_approx() const { return this->tanh_approx_; }

ElementWiseDataflowTensorNode::ElementOutput GELUNode::expand_operation_dataflow_precise(
    builder::StructuredSDFGBuilder& builder,
    structured_control_flow::Block& block,
    std::vector<ElementWiseDataflowTensorNode::ElementInput>& needed_inputs,
    types::PrimitiveType expected_type
) {
    auto& input = needed_inputs.at(0);
    types::Scalar scalar_type(input.required_type);

    // tmp_gelu_x = x
    auto& output_node_assign = this->create_tmp_access_node(builder, block, "tmp_gelu_x_", scalar_type);
    auto& assign_tasklet =
        builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"}, this->debug_info_);
    input.consumer = &assign_tasklet;
    input.input_conn_index = 0;
    builder.add_computational_memlet(block, assign_tasklet, "_out", output_node_assign, {}, this->debug_info_);

    // tmp_gelu_div1 = tmp_gelu_x / sqrt(2) (= tmp_gelu_x * sqrt(1/2))
    auto& sqrt_one_halt_const = builder.add_constant(block, "0.70710678118654752440", scalar_type, this->debug_info_);
    auto& output_node_first_div = this->create_tmp_access_node(builder, block, "tmp_gelu_div1_", scalar_type);
    auto& first_div_tasklet =
        builder.add_tasklet(block, data_flow::TaskletCode::fp_mul, "_out", {"_in1", "_in2"}, this->debug_info_);
    builder.add_computational_memlet(block, output_node_assign, first_div_tasklet, "_in1", {}, this->debug_info_);
    builder.add_computational_memlet(block, sqrt_one_halt_const, first_div_tasklet, "_in2", {}, this->debug_info_);
    builder.add_computational_memlet(block, first_div_tasklet, "_out", output_node_first_div, {}, this->debug_info_);

    // tmp_gelu_erf = erf(tmp_gelu_div1)
    auto& output_node_erf = this->create_tmp_access_node(builder, block, "tmp_gelu_erf_", scalar_type);
    auto& libnode = builder.add_library_node<
        cmath::CMathNode>(block, this->debug_info_, cmath::CMathFunction::erf, scalar_type.primitive_type());
    builder.add_computational_memlet(block, output_node_first_div, libnode, "_in1", {}, scalar_type, this->debug_info_);
    builder.add_computational_memlet(block, libnode, "_out", output_node_erf, {}, scalar_type, this->debug_info_);

    // tmp_gelu_add = 1 + tmp_gelu_erf
    auto& one_const = builder.add_constant(block, "1.0", scalar_type, this->debug_info_);
    auto& output_node_add = this->create_tmp_access_node(builder, block, "tmp_gelu_add_", scalar_type);
    auto& add_tasklet =
        builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"}, this->debug_info_);
    builder.add_computational_memlet(block, one_const, add_tasklet, "_in1", {}, this->debug_info_);
    builder.add_computational_memlet(block, output_node_erf, add_tasklet, "_in2", {}, this->debug_info_);
    builder.add_computational_memlet(block, add_tasklet, "_out", output_node_add, {}, this->debug_info_);

    // tmp_gelu_div2 = tmp_gelu_x / 2 (= tmp_gelu_x * 0.5)
    auto& one_half_const = builder.add_constant(block, "0.5", scalar_type, this->debug_info_);
    auto& output_node_second_div = this->create_tmp_access_node(builder, block, "tmp_gelu_div2_", scalar_type);
    auto& second_div_tasklet =
        builder.add_tasklet(block, data_flow::TaskletCode::fp_mul, "_out", {"_in1", "_in2"}, this->debug_info_);
    builder.add_computational_memlet(block, output_node_assign, second_div_tasklet, "_in1", {}, this->debug_info_);
    builder.add_computational_memlet(block, one_half_const, second_div_tasklet, "_in2", {}, this->debug_info_);
    builder.add_computational_memlet(block, second_div_tasklet, "_out", output_node_second_div, {}, this->debug_info_);

    // y = tmp_gelu_div2 * tmp_gelu_add
    auto& mul_tasklet =
        builder.add_tasklet(block, data_flow::TaskletCode::fp_mul, "_out", {"_in1", "_in2"}, this->debug_info_);
    builder.add_computational_memlet(block, output_node_second_div, mul_tasklet, "_in1", {}, this->debug_info_);
    builder.add_computational_memlet(block, output_node_add, mul_tasklet, "_in2", {}, this->debug_info_);

    return {.producer = &mul_tasklet, .output_conn_index = 0, .type = input.required_type};
}

ElementWiseDataflowTensorNode::ElementOutput GELUNode::expand_operation_dataflow_tanh_approx(
    builder::StructuredSDFGBuilder& builder,
    structured_control_flow::Block& block,
    std::vector<ElementWiseDataflowTensorNode::ElementInput>& needed_inputs,
    types::PrimitiveType expected_type
) {
    auto& input = needed_inputs.at(0);
    types::Scalar scalar_type(input.required_type);

    // tmp_gelu_x = x
    auto& output_node_assign = this->create_tmp_access_node(builder, block, "tmp_gelu_x_", scalar_type);
    auto& assign_tasklet =
        builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"}, this->debug_info_);
    input.consumer = &assign_tasklet;
    input.input_conn_index = 0;
    builder.add_computational_memlet(block, assign_tasklet, "_out", output_node_assign, {}, this->debug_info_);

    // tmp_gelu_mul1 = 0.044715 * tmp_gelu_x
    auto& magic_const = builder.add_constant(block, "0.044715", scalar_type, this->debug_info_);
    auto& output_node_mul1 = this->create_tmp_access_node(builder, block, "tmp_gelu_mul1_", scalar_type);
    auto& mul1_tasklet =
        builder.add_tasklet(block, data_flow::TaskletCode::fp_mul, "_out", {"_in1", "_in2"}, this->debug_info_);
    builder.add_computational_memlet(block, magic_const, mul1_tasklet, "_in1", {}, this->debug_info_);
    builder.add_computational_memlet(block, output_node_assign, mul1_tasklet, "_in2", {}, this->debug_info_);
    builder.add_computational_memlet(block, mul1_tasklet, "_out", output_node_mul1, {}, this->debug_info_);

    // tmp_gelu_mul2 = tmp_gelu_x * tmp_gelu_x
    auto& output_node_mul2 = this->create_tmp_access_node(builder, block, "tmp_gelu_mul2_", scalar_type);
    auto& mul2_tasklet =
        builder.add_tasklet(block, data_flow::TaskletCode::fp_mul, "_out", {"_in1", "_in2"}, this->debug_info_);
    builder.add_computational_memlet(block, output_node_assign, mul2_tasklet, "_in1", {}, this->debug_info_);
    builder.add_computational_memlet(block, output_node_assign, mul2_tasklet, "_in2", {}, this->debug_info_);
    builder.add_computational_memlet(block, mul2_tasklet, "_out", output_node_mul2, {}, this->debug_info_);

    // tmp_gelu_mul3 = tmp_gelu_mul1 * tmp_gelu_mul2
    auto& output_node_mul3 = this->create_tmp_access_node(builder, block, "tmp_gelu_mul3_", scalar_type);
    auto& mul3_tasklet =
        builder.add_tasklet(block, data_flow::TaskletCode::fp_mul, "_out", {"_in1", "_in2"}, this->debug_info_);
    builder.add_computational_memlet(block, output_node_mul1, mul3_tasklet, "_in1", {}, this->debug_info_);
    builder.add_computational_memlet(block, output_node_mul2, mul3_tasklet, "_in2", {}, this->debug_info_);
    builder.add_computational_memlet(block, mul3_tasklet, "_out", output_node_mul3, {}, this->debug_info_);

    // tmp_gelu_add1 = tmp_gelu_x + tmp_gelu_mul3
    auto& output_node_add1 = this->create_tmp_access_node(builder, block, "tmp_gelu_add1_", scalar_type);
    auto& add1_tasklet =
        builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"}, this->debug_info_);
    builder.add_computational_memlet(block, output_node_assign, add1_tasklet, "_in1", {}, this->debug_info_);
    builder.add_computational_memlet(block, output_node_mul3, add1_tasklet, "_in2", {}, this->debug_info_);
    builder.add_computational_memlet(block, add1_tasklet, "_out", output_node_add1, {}, this->debug_info_);

    // tmp_gelu_mul4 = sqrt(2 / pi) * tmp_gelu_add1
    auto& sqrt_2_pi_const = builder.add_constant(block, "0.797884560802865", scalar_type, this->debug_info_);
    auto& output_node_mul4 = this->create_tmp_access_node(builder, block, "tmp_gelu_mul4_", scalar_type);
    auto& mul4_tasklet =
        builder.add_tasklet(block, data_flow::TaskletCode::fp_mul, "_out", {"_in1", "_in2"}, this->debug_info_);
    builder.add_computational_memlet(block, sqrt_2_pi_const, mul4_tasklet, "_in1", {}, this->debug_info_);
    builder.add_computational_memlet(block, output_node_add1, mul4_tasklet, "_in2", {}, this->debug_info_);
    builder.add_computational_memlet(block, mul4_tasklet, "_out", output_node_mul4, {}, this->debug_info_);

    // tmp_gelu_tanh = tanh(tmp_gelu_mul4)
    auto& output_node_tanh = this->create_tmp_access_node(builder, block, "tmp_gelu_tanh_", scalar_type);
    auto& libnode = builder.add_library_node<
        cmath::CMathNode>(block, this->debug_info_, cmath::CMathFunction::tanh, scalar_type.primitive_type());
    builder.add_computational_memlet(block, output_node_mul4, libnode, "_in1", {}, scalar_type, this->debug_info_);
    builder.add_computational_memlet(block, libnode, "_out", output_node_tanh, {}, scalar_type, this->debug_info_);

    // tmp_gelu_add2 = 1 + tmp_gelu_tanh
    auto& one_const = builder.add_constant(block, "1.0", scalar_type, this->debug_info_);
    auto& output_node_add2 = this->create_tmp_access_node(builder, block, "tmp_gelu_add2_", scalar_type);
    auto& add2_tasklet =
        builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"}, this->debug_info_);
    builder.add_computational_memlet(block, one_const, add2_tasklet, "_in1", {}, this->debug_info_);
    builder.add_computational_memlet(block, output_node_tanh, add2_tasklet, "_in2", {}, this->debug_info_);
    builder.add_computational_memlet(block, add2_tasklet, "_out", output_node_add2, {}, this->debug_info_);

    // tmp_gelu_div = tmp_gelu_x / 2 (= tmp_gelu_x * 0.5)
    auto& one_half_const = builder.add_constant(block, "0.5", scalar_type, this->debug_info_);
    auto& output_node_div = this->create_tmp_access_node(builder, block, "tmp_gelu_div_", scalar_type);
    auto& div_tasklet =
        builder.add_tasklet(block, data_flow::TaskletCode::fp_mul, "_out", {"_in1", "_in2"}, this->debug_info_);
    builder.add_computational_memlet(block, output_node_assign, div_tasklet, "_in1", {}, this->debug_info_);
    builder.add_computational_memlet(block, one_half_const, div_tasklet, "_in2", {}, this->debug_info_);
    builder.add_computational_memlet(block, div_tasklet, "_out", output_node_div, {}, this->debug_info_);

    // y = tmp_gelu_div * tmp_gelu_add2
    auto& mul5_tasklet =
        builder.add_tasklet(block, data_flow::TaskletCode::fp_mul, "_out", {"_in1", "_in2"}, this->debug_info_);
    builder.add_computational_memlet(block, output_node_div, mul5_tasklet, "_in1", {}, this->debug_info_);
    builder.add_computational_memlet(block, output_node_add2, mul5_tasklet, "_in2", {}, this->debug_info_);

    return {.producer = &mul5_tasklet, .output_conn_index = 0, .type = input.required_type};
}

ElementWiseDataflowTensorNode::ElementOutput GELUNode::expand_operation_dataflow(
    builder::StructuredSDFGBuilder& builder,
    structured_control_flow::Block& block,
    std::vector<ElementWiseDataflowTensorNode::ElementInput>& needed_inputs,
    types::PrimitiveType expected_type
) {
    if (this->tanh_approx_) {
        return this->expand_operation_dataflow_tanh_approx(builder, block, needed_inputs, expected_type);
    } else {
        return this->expand_operation_dataflow_precise(builder, block, needed_inputs, expected_type);
    }
}

std::unique_ptr<data_flow::DataFlowNode> GELUNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::unique_ptr<data_flow::DataFlowNode>(new GELUNode(
        element_id,
        this->debug_info(),
        vertex,
        parent,
        this->shape_,
        this->tanh_approx_,
        fixed_quantization_,
        implementation_type_
    ));
}

std::string GELUNode::toStr() const {
    std::stringstream stream;

    stream << "GELU(shape: [";
    for (size_t i = 0; i < this->shape_.size(); i++) {
        if (i > 0) {
            stream << ", ";
        }
        stream << this->shape_[i]->__str__();
    }
    stream << "], tanh_approx: " << this->tanh_approx_
           << ", quant: " << types::primitive_type_to_string(this->fixed_quantization_) << ")";

    return stream.str();
}

nlohmann::json GELUNodeSerializer::serialize(const data_flow::LibraryNode& library_node) {
    const auto& elem_node = static_cast<const GELUNode&>(library_node);
    nlohmann::json j = BaseElementWiseDataflowTensorNodeSerializer::serialize(library_node);

    j["tanh_approx"] = elem_node.tanh_approx();

    return j;
}

data_flow::LibraryNode& GELUNodeSerializer::deserialize(
    const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
) {
    auto base = deserialize_base_values(j);

    // Assertions for required fields
    assert(j.contains("tanh_approx"));

    bool tanh_approx = j["tanh_approx"].get<bool>();

    return builder.add_library_node<GELUNode>(parent, base.debug_info, base.shape, tanh_approx, base.quantization);
}

} // namespace tensor
} // namespace math
} // namespace sdfg
