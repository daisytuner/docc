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
    const std::vector<std::string>& outputs,
    const std::vector<std::string>& inputs,
    const std::vector<symbolic::Expression>& shape
)
    : TensorNode(
          element_id,
          debug_info,
          vertex,
          parent,
          LibraryNodeType_TensorCMath,
          outputs,
          inputs,
          data_flow::ImplementationType_NONE
      ),
      cmath_function_(cmath_function), shape_(shape) {}

void CMathTensorNode::validate(const Function& function) const {
    auto& graph = this->get_parent();

    // Check that all input memlets are tensor of scalar
    for (auto& iedge : graph.in_edges(*this)) {
        if (iedge.base_type().type_id() != types::TypeID::Tensor) {
            throw InvalidSDFGException(
                "TensorNode: Input memlet must be of tensor type. Found type: " + iedge.base_type().print()
            );
        }
    }

    // Check that all output memlets are tensor of scalar
    for (auto& oedge : graph.out_edges(*this)) {
        if (oedge.base_type().type_id() != types::TypeID::Tensor) {
            throw InvalidSDFGException(
                "TensorNode: Output memlet must be of tensor type. Found type: " + oedge.base_type().print()
            );
        }
    }

    // Validate: inputs match arity
    if (cmath::cmath_function_to_arity(this->cmath_function()) != this->inputs_.size()) {
        throw InvalidSDFGException(
            "CMathTensorNode (Code: " + std::string(cmath::cmath_function_to_stem(this->cmath_function())) +
            "): Invalid number of inputs. Expected " +
            std::to_string(cmath::cmath_function_to_arity(this->cmath_function())) + ", got " +
            std::to_string(this->inputs_.size())
        );
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

cmath::CMathFunction CMathTensorNode::cmath_function() const { return this->cmath_function_; }

const std::vector<symbolic::Expression>& CMathTensorNode::shape() const { return this->shape_; }

symbolic::SymbolSet CMathTensorNode::symbols() const {
    symbolic::SymbolSet syms;
    for (const auto& dim : this->shape_) {
        for (auto& atom : symbolic::atoms(dim)) {
            syms.insert(atom);
        }
    }
    return syms;
}

void CMathTensorNode::replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) {
    for (auto& dim : shape_) {
        dim = symbolic::subs(dim, old_expression, new_expression);
    }
}

bool CMathTensorNode::expand(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    auto& dataflow = this->get_parent();
    auto& block = static_cast<structured_control_flow::Block&>(*dataflow.get_parent());
    if (dataflow.in_degree(*this) != cmath::cmath_function_to_arity(this->cmath_function()) ||
        dataflow.out_degree(*this) != 1) {
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

    // Add library node block
    auto& code_block = builder.add_block(*last_scope);
    auto& libnode = builder.add_library_node<cmath::CMathNode>(
        code_block, this->debug_info(), this->cmath_function(), iedges.at(0)->base_type().primitive_type()
    );
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
                code_block, *new_access_node, libnode, libnode.input(i), {}, iedge_base_type, iedges[i]->debug_info()
            );
        } else {
            builder.add_computational_memlet(
                code_block,
                *new_access_node,
                libnode,
                libnode.input(i),
                loop_vars,
                iedge_base_type,
                iedges[i]->debug_info()
            );
        }
    }
    auto& out_node = builder.add_access(code_block, output_node.data(), output_node.debug_info());
    builder.add_computational_memlet(
        code_block, libnode, libnode.output(0), out_node, loop_vars, oedge.base_type(), oedge.debug_info()
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
        this->outputs(),
        this->inputs(),
        this->shape()
    ));
}

std::string CMathTensorNode::toStr() const {
    std::stringstream stream;

    const auto& oedge = *this->get_parent().out_edges(*this).begin();
    stream << this->output(0) << " = "
           << cmath::get_cmath_intrinsic_name(this->cmath_function(), oedge.base_type().primitive_type()) << "(";
    for (size_t i = 0; i < this->inputs().size(); i++) {
        if (i > 0) {
            stream << ", ";
        }
        stream << this->input(i);
    }
    stream << ")";

    return stream.str();
}

nlohmann::json CMathTensorNodeSerializer::serialize(const data_flow::LibraryNode& library_node) {
    const auto& elem_node = static_cast<const CMathTensorNode&>(library_node);
    nlohmann::json j;

    j["code"] = elem_node.code().value();

    j["output"] = elem_node.output(0);

    j["inputs"] = nlohmann::json::array();
    for (auto& input : elem_node.inputs()) {
        j["inputs"].push_back(input);
    }

    j["cmath_function"] = cmath::cmath_function_to_stem(elem_node.cmath_function());

    serializer::JSONSerializer serializer;
    j["shape"] = nlohmann::json::array();
    for (auto& dim : elem_node.shape()) {
        j["shape"].push_back(serializer.expression(dim));
    }

    return j;
}

data_flow::LibraryNode& CMathTensorNodeSerializer::deserialize(
    const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
) {
    // Assertions for required fields
    assert(j.contains("element_id"));
    assert(j.contains("code"));
    assert(j.contains("debug_info"));
    assert(j.contains("output"));
    assert(j.contains("inputs"));
    assert(j.contains("cmath_function"));
    assert(j.contains("shape"));

    auto code = j["code"].get<std::string>();

    std::vector<std::string> outputs({j["output"].get<std::string>()});

    std::vector<std::string> inputs;
    for (const auto& input : j["inputs"]) {
        inputs.push_back(input.get<std::string>());
    }

    auto cmath_function = cmath::string_to_cmath_function(j["cmath_function"].get<std::string>());

    std::vector<symbolic::Expression> shape;
    for (const auto& dim : j["shape"]) {
        shape.push_back(symbolic::parse(dim.get<std::string>()));
    }

    // Extract debug info using JSONSerializer
    sdfg::serializer::JSONSerializer serializer;
    DebugInfo debug_info = serializer.json_to_debug_info(j["debug_info"]);

    return static_cast<CMathTensorNode&>(builder.add_library_node<
                                         CMathTensorNode>(parent, debug_info, cmath_function, outputs, inputs, shape));
}

} // namespace tensor
} // namespace math
} // namespace sdfg
