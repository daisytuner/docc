#include "sdfg/data_flow/library_nodes/math/tensor/elementwise_ops/cmath_node.h"

#include <cstddef>
#include <memory>
#include <nlohmann/json_fwd.hpp>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "sdfg/analysis/analysis.h"
#include "sdfg/analysis/scope_analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/data_flow_graph.h"
#include "sdfg/data_flow/data_flow_node.h"
#include "sdfg/data_flow/library_node.h"
#include "sdfg/data_flow/library_nodes/math/cmath/cmath_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/tensor_node.h"
#include "sdfg/element.h"
#include "sdfg/exceptions.h"
#include "sdfg/graph/graph.h"
#include "sdfg/serializer/json_serializer.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/scalar.h"
#include "sdfg/types/tensor.h"

namespace sdfg {
namespace math {
namespace tensor {

CMathTensorNode::CMathTensorNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const cmath::CMathFunction cmath_function,
    const std::string& modified_tensor_conn,
    const std::vector<std::string>& tensor_inputs,
    const std::vector<symbolic::Expression>& shape,
    QuantizationType quantization,
    const data_flow::ImplementationType& impl_type
)
    : ElementWiseDataflowTensorNode(
          element_id,
          debug_info,
          vertex,
          parent,
          LibraryNodeType_TensorCMath,
          shape,
          modified_tensor_conn,
          tensor_inputs,
          quantization,
          impl_type
      ),
      cmath_function_(cmath_function) {}

void CMathTensorNode::validate(const Function& function) const {
    auto& graph = this->get_parent();

    validate_target_tensor(graph);
    validate_all_input_tensors(graph);

    auto actual_inputs = this->inputs().size() - 1;
    // Validate: inputs match arity
    if (cmath::cmath_function_to_arity(this->cmath_function()) != actual_inputs) {
        throw InvalidSDFGException(
            "CMathTensorNode (Code: " + std::string(cmath::cmath_function_to_stem(this->cmath_function())) +
            "): Invalid number of inputs. Expected " +
            std::to_string(cmath::cmath_function_to_arity(this->cmath_function())) + ", got " +
            std::to_string(actual_inputs)
        );
    }
}

cmath::CMathFunction CMathTensorNode::cmath_function() const { return this->cmath_function_; }

ElementWiseDataflowTensorNode::ElementOutput CMathTensorNode::expand_operation_dataflow(
    builder::StructuredSDFGBuilder& builder,
    analysis::AnalysisManager& analysis_manager,
    Block& block,
    std::vector<ElementInput>& needed_inputs,
    types::PrimitiveType expected_type
) {
    if (cmath::cmath_function_to_arity(this->cmath_function()) > needed_inputs.size()) {
        return {}; // not mappable, probably invalid
    }

    auto prim_type = needed_inputs.at(0).required_type;

    auto& libnode = builder.add_library_node<cmath::CMathNode>(block, debug_info_, this->cmath_function(), prim_type);
    auto& inputs = libnode.inputs();
    for (size_t i = 0; i < inputs.size(); i++) {
        auto& tensor_input = needed_inputs.at(i);
        tensor_input.consumer = &libnode;
        tensor_input.input_conn_index = i;
    }

    // validate that expected_type is also output by cmath function

    return {.producer = &libnode, .output_conn_index = 0, .type = expected_type};
}

bool CMathTensorNode::supports_integer_types() const {
    return this->cmath_function() == cmath::CMathFunction::lrint ||
           this->cmath_function() == cmath::CMathFunction::llrint ||
           this->cmath_function() == cmath::CMathFunction::lround ||
           this->cmath_function() == cmath::CMathFunction::llround;
}

std::unique_ptr<data_flow::DataFlowNode> CMathTensorNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::unique_ptr<data_flow::DataFlowNode>(new CMathTensorNode(
        element_id,
        this->debug_info(),
        vertex,
        parent,
        this->cmath_function(),
        inputs_.at(0),
        std::vector<std::string>(inputs_.cbegin() + 1, inputs_.cend()),
        this->shape(),
        fixed_quantization_,
        implementation_type_
    ));
}

std::string CMathTensorNode::toStr() const {
    std::stringstream stream;

    const auto* iedge = this->get_parent().in_edge_for_connector(*this, this->input(0));
    stream << this->code().value() << "("
           << cmath::get_cmath_intrinsic_name(this->cmath_function(), iedge->base_type().primitive_type()) << ")";

    return stream.str();
}

nlohmann::json CMathTensorNodeSerializer::serialize(const data_flow::LibraryNode& library_node) {
    const auto& elem_node = static_cast<const CMathTensorNode&>(library_node);
    nlohmann::json j = BaseElementWiseDataflowTensorNodeSerializer::serialize(library_node);

    auto input_arr = nlohmann::json::array();
    for (auto& input : elem_node.inputs()) {
        input_arr.push_back(input);
    }
    j["inputs"] = input_arr;

    j["cmath_function"] = cmath::cmath_function_to_stem(elem_node.cmath_function());

    return j;
}

data_flow::LibraryNode& CMathTensorNodeSerializer::deserialize(
    const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
) {
    auto base = deserialize_base_values(j);

    // Assertions for required fields
    assert(j.contains("inputs"));
    assert(j.contains("cmath_function"));

    std::vector<std::string> inputs;
    for (const auto& input : j["inputs"]) {
        inputs.push_back(input.get<std::string>());
    }

    auto cmath_function = cmath::string_to_cmath_function(j["cmath_function"].get<std::string>());

    std::vector<std::string> tensor_inputs(inputs.cbegin() + 1, inputs.cend());

    return static_cast<CMathTensorNode&>(builder.add_library_node<CMathTensorNode>(
        parent, base.debug_info, cmath_function, inputs.at(0), tensor_inputs, base.shape, base.quantization
    ));
}

} // namespace tensor
} // namespace math
} // namespace sdfg
