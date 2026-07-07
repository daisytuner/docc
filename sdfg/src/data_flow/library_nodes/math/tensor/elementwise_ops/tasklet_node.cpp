#include "sdfg/data_flow/library_nodes/math/tensor/elementwise_ops/tasklet_node.h"

#include <cstddef>
#include <memory>
#include <nlohmann/json_fwd.hpp>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/data_flow_graph.h"
#include "sdfg/data_flow/data_flow_node.h"
#include "sdfg/data_flow/library_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/tensor_node.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/element.h"
#include "sdfg/exceptions.h"
#include "sdfg/graph/graph.h"
#include "sdfg/serializer/json_serializer.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/scalar.h"
#include "sdfg/types/tensor.h"
#include "sdfg/types/type.h"

namespace sdfg {
namespace math {
namespace tensor {

TaskletTensorNode::TaskletTensorNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const data_flow::TaskletCode tasklet_code,
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
          LibraryNodeType_TensorTasklet,
          shape,
          modified_tensor_conn,
          tensor_inputs,
          quantization,
          impl_type
      ),
      tasklet_code_(tasklet_code) {}

void TaskletTensorNode::validate(const Function& function) const {
    ElementWiseDataflowTensorNode::validate(function);
    auto& graph = this->get_parent();

    // Validate: inputs match arity
    auto actual_inputs = this->inputs_.size() - 1;
    if (data_flow::arity(this->tasklet_code()) != actual_inputs) {
        throw InvalidSDFGException(
            "TaskletTensorNode #" + std::to_string(element_id_) + ": (Code: " + std::to_string(this->tasklet_code()) +
            "): Invalid number of inputs. Expected " + std::to_string(data_flow::arity(this->tasklet_code())) +
            ", got " + std::to_string(actual_inputs)
        );
    }

    // Validate: inputs match type of operation
    for (auto& iedge : graph.in_edges(*this)) {
        auto input_type = iedge.result_type(function);
        if (is_integer(this->tasklet_code()) && !types::is_integer(input_type->primitive_type())) {
            throw InvalidSDFGException(
                "TaskletTensorNode #" + std::to_string(element_id_) +
                ": (Code: " + std::to_string(this->tasklet_code()) + "): Integer operation with non-integer input type"
            );
        }
        if (is_floating_point(this->tasklet_code()) && !types::is_floating_point(input_type->primitive_type())) {
            throw InvalidSDFGException(
                "TaskletTensorNode #" + std::to_string(element_id_) + ": (Code: " +
                std::to_string(this->tasklet_code()) + "): Floating point operation with integer input type"
            );
        }
    }
}

data_flow::TaskletCode TaskletTensorNode::tasklet_code() const { return this->tasklet_code_; }

bool TaskletTensorNode::supports_integer_types() const { return data_flow::is_integer(this->tasklet_code()); }

ElementWiseDataflowTensorNode::ElementOutput TaskletTensorNode::expand_operation_dataflow(
    builder::StructuredSDFGBuilder& builder,
    Block& block,
    std::vector<ElementInput>& needed_inputs,
    types::PrimitiveType expected_type
) {
    auto code = this->tasklet_code();
    auto code_arity = data_flow::arity(code);
    if (code_arity > needed_inputs.size()) {
        return {}; // not mappable, probably invalid
    }

    auto prim_type = needed_inputs.at(0).required_type;
    std::vector<std::string> tasklet_inputs;
    tasklet_inputs.reserve(code_arity);
    for (int i = 0; i < code_arity; ++i) {
        tasklet_inputs.push_back(inputs_.at(1 + i));
    }

    auto& tasklet = builder.add_tasklet(block, code, "_out", tasklet_inputs, debug_info_);
    auto& inputs = tasklet.inputs();
    for (size_t i = 0; i < code_arity; i++) {
        auto& tensor_input = needed_inputs.at(i);
        tensor_input.consumer = &tasklet;
        tensor_input.input_conn_index = i;
    }

    // validate that expected_type is also output by tasklet function

    return {.producer = &tasklet, .output_conn_index = 0, .type = expected_type};
}

std::unique_ptr<data_flow::DataFlowNode> TaskletTensorNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::unique_ptr<data_flow::DataFlowNode>(new TaskletTensorNode(
        element_id,
        this->debug_info(),
        vertex,
        parent,
        this->tasklet_code(),
        inputs_.at(0),
        std::vector<std::string>(inputs_.cbegin() + 1, inputs_.cend()),
        this->shape(),
        fixed_quantization_,
        implementation_type_
    ));
}

std::string TaskletTensorNode::toStr() const {
    std::stringstream stream;

    stream << this->code().value() << ": " << std::to_string(this->tasklet_code()) << ", [";
    for (size_t i = 0; i < this->shape().size(); i++) {
        if (i > 0) {
            stream << ", ";
        }
        stream << this->shape().at(i)->__str__();
    }
    stream << "]";

    return stream.str();
}

nlohmann::json TaskletTensorNodeSerializer::serialize(const data_flow::LibraryNode& library_node) {
    const auto& elem_node = static_cast<const TaskletTensorNode&>(library_node);
    nlohmann::json j = BaseElementWiseDataflowTensorNodeSerializer::serialize(library_node);

    auto input_arr = nlohmann::json::array();
    for (auto& input : elem_node.inputs()) {
        input_arr.push_back(input);
    }
    j["inputs"] = input_arr;

    j["tasklet_code"] = elem_node.tasklet_code();

    return j;
}

data_flow::LibraryNode& TaskletTensorNodeSerializer::deserialize(
    const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
) {
    auto base = deserialize_base_values(j);

    // Assertions for required fields
    assert(j.contains("inputs"));
    assert(j.contains("tasklet_code"));

    std::vector<std::string> inputs;
    for (const auto& input : j["inputs"]) {
        inputs.push_back(input.get<std::string>());
    }

    auto tasklet_code = static_cast<data_flow::TaskletCode>(j["tasklet_code"].get<int>());

    std::vector<std::string> tensor_inputs(inputs.cbegin() + 1, inputs.cend());

    return static_cast<TaskletTensorNode&>(builder.add_library_node<TaskletTensorNode>(
        parent, base.debug_info, tasklet_code, inputs.at(0), tensor_inputs, base.shape, base.quantization
    ));
}

} // namespace tensor
} // namespace math
} // namespace sdfg
