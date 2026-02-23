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
#include "sdfg/analysis/scope_analysis.h"
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

namespace sdfg {
namespace math {
namespace tensor {

TaskletTensorNode::TaskletTensorNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const data_flow::TaskletCode tasklet_code,
    const std::vector<std::string>& outputs,
    const std::vector<std::string>& inputs,
    const std::vector<symbolic::Expression>& shape
)
    : TensorNode(
          element_id,
          debug_info,
          vertex,
          parent,
          LibraryNodeType_TensorTasklet,
          outputs,
          inputs,
          data_flow::ImplementationType_NONE
      ),
      tasklet_code_(tasklet_code), shape_(shape) {}

void TaskletTensorNode::validate(const Function& function) const {
    TensorNode::validate(function);

    auto& graph = this->get_parent();

    // Validate: inputs match arity
    if (data_flow::arity(this->tasklet_code()) != this->inputs_.size()) {
        throw InvalidSDFGException(
            "TaskletTensorNode (Code: " + std::to_string(this->tasklet_code()) +
            "): Invalid number of inputs. Expected " + std::to_string(data_flow::arity(this->tasklet_code())) +
            ", got " + std::to_string(this->inputs_.size())
        );
    }

    // Validate: inputs match type of operation
    for (auto& iedge : graph.in_edges(*this)) {
        auto input_type = iedge.result_type(function);
        if (is_integer(this->tasklet_code()) && !types::is_integer(input_type->primitive_type())) {
            throw InvalidSDFGException(
                "TaskletTensorNode (Code: " + std::to_string(this->tasklet_code()) +
                "): Integer operation with non-integer input type"
            );
        }
        if (is_floating_point(this->tasklet_code()) && !types::is_floating_point(input_type->primitive_type())) {
            throw InvalidSDFGException(
                "TaskletTensorNode (Code: " + std::to_string(this->tasklet_code()) +
                "): Floating point operation with integer input type"
            );
        }
    }

    auto& oedge = *graph.out_edges(*this).begin();
    auto& tensor_output = static_cast<const types::Tensor&>(oedge.base_type());
    if (tensor_output.shape().size() != this->shape_.size()) {
        throw InvalidSDFGException(
            "Library Node: Output tensor shape must match node shape. Output shape: " +
            std::to_string(tensor_output.shape().size()) + " Node shape: " + std::to_string(this->shape_.size())
        );
    }
    for (size_t i = 0; i < this->shape_.size(); ++i) {
        if (!symbolic::eq(tensor_output.shape().at(i), this->shape_.at(i))) {
            throw InvalidSDFGException(
                "Library Node: Output tensor shape does not match expected shape. Output shape: " +
                tensor_output.shape().at(i)->__str__() + " Expected shape: " + this->shape_.at(i)->__str__()
            );
        }
    }

    for (auto& iedge : graph.in_edges(*this)) {
        auto& tensor_input = static_cast<const types::Tensor&>(iedge.base_type());
        // Case 1: Scalar input is allowed as secondary input
        if (tensor_input.is_scalar()) {
            continue;
        }

        // Case 2: Tensor input
        if (tensor_input.shape().size() != this->shape_.size()) {
            throw InvalidSDFGException(
                "Library Node: Input tensor shape must match node shape. Input shape: " +
                std::to_string(tensor_input.shape().size()) + " Node shape: " + std::to_string(this->shape_.size())
            );
        }
        for (size_t i = 0; i < this->shape_.size(); ++i) {
            if (!symbolic::eq(tensor_input.shape().at(i), this->shape_.at(i))) {
                throw InvalidSDFGException(
                    "Library Node: Input tensor shape does not match expected shape. Input shape: " +
                    tensor_input.shape().at(i)->__str__() + " Expected shape: " + this->shape_.at(i)->__str__()
                );
            }
        }
    }
}

data_flow::TaskletCode TaskletTensorNode::tasklet_code() const { return this->tasklet_code_; }

const std::vector<symbolic::Expression>& TaskletTensorNode::shape() const { return this->shape_; }

symbolic::SymbolSet TaskletTensorNode::symbols() const {
    symbolic::SymbolSet syms;
    for (const auto& dim : this->shape_) {
        for (auto& atom : symbolic::atoms(dim)) {
            syms.insert(atom);
        }
    }
    return syms;
}

void TaskletTensorNode::replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) {
    for (auto& dim : shape_) {
        dim = symbolic::subs(dim, old_expression, new_expression);
    }
}

bool TaskletTensorNode::expand(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    auto& dataflow = this->get_parent();
    auto& block = static_cast<structured_control_flow::Block&>(*dataflow.get_parent());
    if (dataflow.in_degree(*this) != data_flow::arity(this->tasklet_code()) || dataflow.out_degree(*this) != 1) {
        return false;
    }
    auto& scope_analysis = analysis_manager.get<analysis::ScopeAnalysis>();
    auto& parent = static_cast<structured_control_flow::Sequence&>(*scope_analysis.parent_scope(&block));
    int index = parent.index(block);
    auto& transition = parent.at(index).second;

    auto iedges = dataflow.in_edges_by_connector(*this);
    auto& oedge = *dataflow.out_edges(*this).begin();

    // Checks if legal
    std::unordered_set<data_flow::AccessNode*> input_nodes;
    for (auto* iedge : iedges) {
        input_nodes.insert(static_cast<data_flow::AccessNode*>(&iedge->src()));
    }
    auto& output_node = static_cast<data_flow::AccessNode&>(oedge.dst());
    for (auto* input_node : input_nodes) {
        if (dataflow.in_degree(*input_node) != 0) {
            return false;
        }
    }
    if (dataflow.out_degree(output_node) != 0) {
        return false;
    }

    // Add new graph after the current block
    auto& new_sequence = builder.add_sequence_before(parent, block, transition.assignments(), block.debug_info());

    // Add maps
    structured_control_flow::Sequence* last_scope = &new_sequence;
    structured_control_flow::Map* last_map = nullptr;
    std::vector<symbolic::Expression> loop_vars;

    for (size_t i = 0; i < this->shape_.size(); i++) {
        std::string indvar_str = builder.find_new_name("_i");
        builder.add_container(indvar_str, types::Scalar(types::PrimitiveType::UInt64));

        auto indvar = symbolic::symbol(indvar_str);
        auto init = symbolic::zero();
        auto update = symbolic::add(indvar, symbolic::one());
        auto condition = symbolic::Lt(indvar, this->shape_[i]);
        last_map = &builder.add_map(
            *last_scope,
            indvar,
            condition,
            init,
            update,
            structured_control_flow::ScheduleType_Sequential::create(),
            {},
            block.debug_info()
        );
        last_scope = &last_map->root();

        loop_vars.push_back(indvar);
    }

    // Add tasklet block
    auto& code_block = builder.add_block(*last_scope);
    auto& tasklet = builder.add_tasklet(code_block, this->tasklet_code(), this->output(0), this->inputs());
    std::unordered_map<std::string, data_flow::AccessNode*> container_map;
    for (size_t i = 0; i < iedges.size(); i++) {
        auto* access_node = static_cast<data_flow::AccessNode*>(&iedges[i]->src());
        auto& data = access_node->data();
        data_flow::AccessNode* new_access_node = nullptr;
        if (container_map.contains(data)) {
            new_access_node = container_map.at(data);
        } else {
            if (auto* constant_node = dynamic_cast<data_flow::ConstantNode*>(access_node)) {
                new_access_node =
                    &builder.add_constant(code_block, data, constant_node->type(), constant_node->debug_info());
            } else {
                new_access_node = &builder.add_access(code_block, data, access_node->debug_info());
            }
            container_map.insert({data, new_access_node});
        }
        const auto& iedge_base_type = static_cast<const types::Tensor&>(iedges[i]->base_type());
        if (builder.subject().exists(data) && iedge_base_type.is_scalar()) {
            builder.add_computational_memlet(
                code_block, *new_access_node, tasklet, this->input(i), {}, iedge_base_type, iedges[i]->debug_info()
            );
        } else {
            builder.add_computational_memlet(
                code_block,
                *new_access_node,
                tasklet,
                this->input(i),
                loop_vars,
                iedge_base_type,
                iedges[i]->debug_info()
            );
        }
    }
    auto& out_node = builder.add_access(code_block, output_node.data(), output_node.debug_info());
    builder.add_computational_memlet(
        code_block, tasklet, this->output(0), out_node, loop_vars, oedge.base_type(), oedge.debug_info()
    );

    // Clean up block
    for (auto* iedge : iedges) {
        builder.remove_memlet(block, *iedge);
    }
    builder.remove_memlet(block, oedge);
    for (auto* input_node : input_nodes) {
        builder.remove_node(block, *input_node);
    }
    builder.remove_node(block, output_node);
    builder.remove_node(block, *this);
    builder.remove_child(parent, index + 1);

    return true;
}

bool TaskletTensorNode::supports_integer_types() const { return data_flow::is_integer(this->tasklet_code()); }

std::unique_ptr<data_flow::DataFlowNode> TaskletTensorNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::unique_ptr<data_flow::DataFlowNode>(new TaskletTensorNode(
        element_id,
        this->debug_info(),
        vertex,
        parent,
        this->tasklet_code(),
        this->outputs(),
        this->inputs(),
        this->shape()
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
    nlohmann::json j;

    j["code"] = elem_node.code().value();

    j["output"] = elem_node.output(0);

    j["inputs"] = nlohmann::json::array();
    for (auto& input : elem_node.inputs()) {
        j["inputs"].push_back(input);
    }

    j["tasklet_code"] = elem_node.tasklet_code();

    serializer::JSONSerializer serializer;
    j["shape"] = nlohmann::json::array();
    for (auto& dim : elem_node.shape()) {
        j["shape"].push_back(serializer.expression(dim));
    }

    return j;
}

data_flow::LibraryNode& TaskletTensorNodeSerializer::deserialize(
    const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
) {
    // Assertions for required fields
    assert(j.contains("element_id"));
    assert(j.contains("code"));
    assert(j.contains("debug_info"));
    assert(j.contains("output"));
    assert(j.contains("inputs"));
    assert(j.contains("tasklet_code"));
    assert(j.contains("shape"));

    auto code = j["code"].get<std::string>();

    std::vector<std::string> outputs({j["output"].get<std::string>()});

    std::vector<std::string> inputs;
    for (const auto& input : j["inputs"]) {
        inputs.push_back(input.get<std::string>());
    }

    auto tasklet_code = static_cast<data_flow::TaskletCode>(j["tasklet_code"].get<int>());

    std::vector<symbolic::Expression> shape;
    for (const auto& dim : j["shape"]) {
        shape.push_back(symbolic::parse(dim.get<std::string>()));
    }

    // Extract debug info using JSONSerializer
    sdfg::serializer::JSONSerializer serializer;
    DebugInfo debug_info = serializer.json_to_debug_info(j["debug_info"]);

    return static_cast<TaskletTensorNode&>(builder.add_library_node<
                                           TaskletTensorNode>(parent, debug_info, tasklet_code, outputs, inputs, shape)
    );
}

} // namespace tensor
} // namespace math
} // namespace sdfg
