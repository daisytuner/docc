#include "sdfg/data_flow/library_nodes/math/tensor/elementwise_node.h"

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/types/type.h"

#include "sdfg/analysis/scope_analysis.h"

namespace sdfg {
namespace math {
namespace tensor {

ElementWiseDataflowTensorNode::ElementWiseDataflowTensorNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const data_flow::LibraryNodeCode& code,
    const std::vector<symbolic::Expression>& shape,
    const std::string& modified_tensor_conn,
    const std::vector<std::string>& tensor_inputs,
    QuantizationType quantization,
    const data_flow::ImplementationType& impl_type
)
    : TensorNode(
          element_id,
          debug_info,
          vertex,
          parent,
          code,
          {},
          build_input_conns(modified_tensor_conn, tensor_inputs),
          impl_type
      ),
      fixed_quantization_(quantization), shape_(shape) {}

std::vector<std::string> ElementWiseDataflowTensorNode::
    build_input_conns(const std::string& modified_tensor_conn, const std::vector<std::string>& tensor_inputs) {
    std::vector<std::string> input_conns;
    input_conns.reserve(1 + input_conns.size());
    input_conns.push_back(modified_tensor_conn);
    input_conns.insert(input_conns.end(), tensor_inputs.begin(), tensor_inputs.end());
    return input_conns;
}

types::PrimitiveType ElementWiseDataflowTensorNode::fixed_quantization() const { return fixed_quantization_; }

types::PrimitiveType ElementWiseDataflowTensorNode::quantization(const data_flow::DataFlowGraph& data_flow_graph
) const {
    if (fixed_quantization_ != QUANTIZATION_MATCH_INPUTS) {
        return fixed_quantization_;
    } else {
        return this->primitive_type(data_flow_graph);
    }
}

std::optional<types::PrimitiveType> ElementWiseDataflowTensorNode::uniform_quantization(const data_flow::DataFlowGraph&
                                                                                            data_flow_graph) const {
    if (fixed_quantization_ != QUANTIZATION_MATCH_INPUTS) {
        auto inferred = this->primitive_type(data_flow_graph);
        if (inferred == fixed_quantization_) {
            return fixed_quantization_;
        } else {
            return std::nullopt;
        }
    } else {
        return this->primitive_type(data_flow_graph);
    }
}

void ElementWiseDataflowTensorNode::validate_shape_matches(
    const std::vector<symbolic::Expression>& required_shape, const TensorLayout& layout, const std::string& name
) const {
    if (layout.shape().size() != required_shape.size()) {
        throw InvalidSDFGException(
            "On libNode #" + std::to_string(element_id()) + ": " + name +
            " tensor shape must match node shape dims: Given: " + std::to_string(layout.shape().size()) +
            " Required: " + std::to_string(this->shape_.size())
        );
    }
    auto& given_shape = layout.shape();
    for (size_t i = 0; i < required_shape.size(); ++i) {
        if (!symbolic::eq(layout.shape().at(i), this->shape_.at(i))) {
            throw InvalidSDFGException(
                "On libNode #" + std::to_string(element_id()) + ": " + name +
                " tensor shape must match shape: Given: " + layout.shape().at(i)->__str__() +
                " Expected shape: " + this->shape_.at(i)->__str__()
            );
        }
    }
}

void ElementWiseDataflowTensorNode::validate_target_tensor(const data_flow::DataFlowGraph& graph) const {
    auto* target_ptr_edge = graph.in_edge_for_connector(*this, inputs_.at(0));
    auto& tensor_output = static_cast<const types::Tensor&>(target_ptr_edge->base_type());

    validate_shape_matches(shape_, tensor_output.layout(), "output tensor");
}

void ElementWiseDataflowTensorNode::validate_all_input_tensors(const data_flow::DataFlowGraph& graph) const {
    for (int i = 1; i < tensor_input_count(); ++i) {
        auto* iedge = graph.in_edge_for_connector(*this, inputs_.at(i));
        if (!iedge) {
            throw InvalidSDFGException(
                "On libNode #" + std::to_string(element_id()) + ": input " + inputs_.at(i) + " is not connected"
            );
        }
        if (iedge->base_type().type_id() == types::TypeID::Scalar) {
            continue;
        }
        auto& tensor_input = static_cast<const types::Tensor&>(iedge->base_type());
        // Case 1: Scalar input is allowed as secondary input
        if (tensor_input.is_scalar()) {
            continue;
        }

        // currently no arbitrary broadcast support! but could be added
        validate_shape_matches(shape_, tensor_input.layout(), "input " + inputs_.at(i));
    }
}

void ElementWiseDataflowTensorNode::validate_non_tensor_inputs(const data_flow::DataFlowGraph& graph) const {
    for (int i = tensor_input_count(); i < inputs_.size(); ++i) {
        auto* iedge = graph.in_edge_for_connector(*this, inputs_.at(i));
        if (!iedge) {
            if (i < mandatory_input_count()) {
                throw InvalidSDFGException(
                    "On libNode #" + std::to_string(element_id()) + ": input " + inputs_.at(i) + " is not connected"
                );
            } else {
                continue;
            }
        }
        if (iedge->base_type().type_id() != types::TypeID::Scalar) {
            throw InvalidSDFGException(
                "On libNode #" + std::to_string(element_id()) + ": input " + inputs_.at(i) + " is not scalar"
            );
        }
    }
}

void ElementWiseDataflowTensorNode::validate(const Function& function) const {
    TensorNode::validate(function);

    auto& graph = this->get_parent();

    validate_target_tensor(graph);

    validate_all_input_tensors(graph);

    validate_non_tensor_inputs(graph);
}

symbolic::SymbolSet ElementWiseDataflowTensorNode::symbols() const {
    symbolic::SymbolSet syms;
    for (const auto& dim : shape_) {
        for (auto& atom : symbolic::atoms(dim)) {
            syms.insert(atom);
        }
    }
    return syms;
}

void ElementWiseDataflowTensorNode::
    replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) {
    for (auto& dim : shape_) {
        dim = symbolic::subs(dim, old_expression, new_expression);
    }
}

std::pair<structured_control_flow::Sequence*, std::vector<symbolic::Expression>> ElementWiseDataflowTensorNode::
    add_eltwise_scope(
        builder::StructuredSDFGBuilder& builder,
        const DebugInfo& scope_deb_info,
        Sequence& parent,
        const std::vector<symbolic::Expression>& shape
    ) {
    // Add maps
    data_flow::Subset new_subset;
    std::vector<symbolic::Expression> loop_vars;
    structured_control_flow::Sequence* last_scope = &parent;
    structured_control_flow::Map* last_map = nullptr;

    for (size_t i = 0; i < shape.size(); i++) {
        std::string indvar_str = builder.find_new_name("_i");
        builder.add_container(indvar_str, types::Scalar(types::PrimitiveType::UInt64));

        auto indvar = symbolic::symbol(indvar_str);
        auto init = symbolic::zero();
        auto update = symbolic::add(indvar, symbolic::one());
        auto condition = symbolic::Lt(indvar, shape.at(i));
        last_map = &builder.add_map(
            *last_scope,
            indvar,
            condition,
            init,
            update,
            structured_control_flow::ScheduleType_Sequential::create(),
            {},
            scope_deb_info
        );
        last_scope = &last_map->root();

        loop_vars.push_back(indvar);
    }
    return {last_scope, loop_vars};
}

std::unique_ptr<types::IType> ElementWiseDataflowTensorNode::access_type(const std::pair<
                                                                         types::PrimitiveType,
                                                                         const TensorLayout*>& pair) {
    if (pair.second) {
        return std::make_unique<types::Tensor>(pair.first, *pair.second);
    } else {
        return std::make_unique<types::Scalar>(pair.first);
    }
}

bool ElementWiseDataflowTensorNode::create_input(
    builder::StructuredSDFGBuilder& builder,
    structured_control_flow::Block& block,
    const data_flow::AccessNode& org_src,
    const std::pair<types::PrimitiveType, const TensorLayout*>& src_type,
    const ElementInput& needed_input,
    const std::vector<symbolic::Expression>& eltwise_subset
) {
    auto* new_consumer = needed_input.consumer;
    if (new_consumer) {
        if (src_type.first != needed_input.required_type) {
            throw InvalidSDFGException(
                "Input " + std::to_string(needed_input.input_conn_index) + " on node #" +
                std::to_string(new_consumer->element_id()) + " is required as " +
                types::primitive_type_to_string(needed_input.required_type) + " but provided as " +
                types::primitive_type_to_string(src_type.first)
            );
        }
        if (org_src.is_constant()) {
            types::Scalar const_type(src_type.first);
            auto& input_node = builder.add_constant(block, org_src.data(), const_type);
            auto new_type = access_type(src_type);
            builder.add_computational_memlet(
                block, input_node, *new_consumer, new_consumer->input(needed_input.input_conn_index), {}, *new_type
            );
        } else {
            auto& input_node = builder.add_access(block, org_src.data());
            auto new_type = access_type(src_type);
            std::vector<symbolic::Expression> subset;
            if (src_type.second && !src_type.second->is_scalar()) {
                subset = eltwise_subset;
            }
            builder.add_computational_memlet(
                block, input_node, *new_consumer, new_consumer->input(needed_input.input_conn_index), subset, *new_type
            );
        }
        return true;
    } else {
        return false;
    }
}

void ElementWiseDataflowTensorNode::create_output(
    builder::StructuredSDFGBuilder& builder,
    structured_control_flow::Block& block,
    const data_flow::AccessNode& org_dst,
    const types::Tensor& dst_type,
    const ElementOutput& provided_output,
    const std::vector<symbolic::Expression>& eltwise_subset
) {
    auto* producer = provided_output.producer;
    if (dst_type.primitive_type() != provided_output.type) {
        throw InvalidSDFGException(
            "Output " + std::to_string(provided_output.output_conn_index) + " on node #" +
            std::to_string(producer->element_id()) + " is provided as " +
            types::primitive_type_to_string(provided_output.type) + " but required as " +
            types::primitive_type_to_string(dst_type.primitive_type())
        );
    }
    auto& output_node = builder.add_access(block, org_dst.data());
    builder.add_computational_memlet(
        block, *producer, producer->output(provided_output.output_conn_index), output_node, eltwise_subset, dst_type
    );
}

bool ElementWiseDataflowTensorNode::
    expand(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    auto& dataflow = this->get_parent();
    auto& org_block = static_cast<structured_control_flow::Block&>(*dataflow.get_parent());

    auto* output_tensor_iedge = dataflow.in_edge_for_connector(*this, inputs_.at(0));
    if (!output_tensor_iedge) {
        return false;
    }
    auto& target_tensor = static_cast<const types::Tensor&>(output_tensor_iedge->base_type());
    std::vector<const data_flow::Memlet*> iedges;
    std::vector<const data_flow::AccessNode*> inputs_sa;
    std::vector<std::pair<types::PrimitiveType, const TensorLayout*>> input_types;
    iedges.reserve(inputs_.size() - 1);
    for (int i = 1; i < this->inputs_.size(); ++i) {
        auto* iedge = dataflow.in_edge_for_connector(*this, inputs_.at(i));
        if (!iedge) {
            if (i < mandatory_input_count()) {
                return false;
            } else {
                continue;
            }
        }
        iedges.push_back(iedge);
        auto* input_sa = dataflow.find_standalone_entry(iedge);
        if (!input_sa) {
            return false;
        }
        inputs_sa.push_back(input_sa);
        auto& input_type = iedge->base_type();
        if (input_type.type_id() == types::TypeID::Scalar) {
            input_types.emplace_back(input_type.primitive_type(), nullptr);
        } else {
            auto& tensor_type = static_cast<const types::Tensor&>(iedge->base_type());
            input_types.emplace_back(input_type.primitive_type(), &tensor_type.layout());
        }
    }

    auto* output_tensor_sa = dataflow.find_standalone_entry(output_tensor_iedge);
    if (!output_tensor_sa) {
        return false;
    }

    auto& scope_analysis = analysis_manager.get<analysis::ScopeAnalysis>();
    auto& parent = static_cast<structured_control_flow::Sequence&>(*scope_analysis.parent_scope(&org_block));
    int index = parent.index(org_block);
    auto& transition = parent.at(index).second;

    // Add new graph after the current block
    auto& new_sequence =
        builder.add_sequence_before(parent, org_block, transition.assignments(), org_block.debug_info());

    auto [eltw_scope, loop_vars] = add_eltwise_scope(builder, org_block.debug_info(), new_sequence, shape_);

    std::vector<tensor::ElementWiseDataflowTensorNode::ElementInput> eltwise_inputs;
    eltwise_inputs.reserve(inputs_.size() - 1);
    for (int i = 0; i < input_types.size(); ++i) {
        eltwise_inputs.push_back({.required_type = input_types.at(i).first});
    }

    auto& new_block = builder.add_block(*eltw_scope);

    auto produced_output =
        expand_operation_dataflow(builder, analysis_manager, new_block, eltwise_inputs, target_tensor.primitive_type());
    if (!produced_output.producer) {
        return false;
    }

    // for all old input edge, remove old, create new
    for (int i = 0; i < iedges.size(); ++i) {
        create_input(builder, new_block, *inputs_sa.at(i), input_types.at(i), eltwise_inputs.at(i), loop_vars);
    }
    create_output(builder, new_block, *output_tensor_sa, target_tensor, produced_output, loop_vars);
    // careful, many pointers in the input vectors become invalid beyond this point
    for (int i = 0; i < iedges.size(); ++i) {
        builder.remove_memlet(org_block, *iedges.at(i));
        builder.remove_node(org_block, *inputs_sa.at(i));
    }
    // Clean up block
    builder.remove_memlet(org_block, *output_tensor_iedge);
    builder.remove_node(org_block, *output_tensor_sa);
    builder.remove_node(org_block, *this);
    builder.remove_child(parent, index + 1);

    return true;
}

nlohmann::json BaseElementWiseDataflowTensorNodeSerializer::serialize(const data_flow::LibraryNode& library_node) {
    const ElementWiseDataflowTensorNode& elem_node = static_cast<const ElementWiseDataflowTensorNode&>(library_node);
    nlohmann::json j;

    j["code"] = elem_node.code().value();

    serializer::JSONSerializer serializer;
    j["shape"] = nlohmann::json::array();
    for (auto& dim : elem_node.shape()) {
        j["shape"].push_back(serializer.expression(dim));
    }

    j["result_quant"] = elem_node.fixed_quantization();

    return j;
}

BaseElementWiseDataflowTensorNodeSerializer::BaseDeser BaseElementWiseDataflowTensorNodeSerializer::
    deserialize_base_values(const nlohmann::json& j) {
    assert(j.contains("element_id"));
    assert(j.contains("code"));
    assert(j.contains("debug_info"));

    std::vector<symbolic::Expression> shape;
    if (j.contains("shape")) {
        for (const auto& dim : j["shape"]) {
            shape.push_back(symbolic::parse(dim.get<std::string>()));
        }
    }

    serializer::JSONSerializer serializer;
    auto debug_info = serializer.json_to_debug_info(j["debug_info"]);
    return {
        .shape = shape,
        .quantization = deserialize_quantization(j, "result_quant", QUANTIZATION_MATCH_INPUTS),
        .debug_info = debug_info
    };
}


ElementWiseUnaryNode::ElementWiseUnaryNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const data_flow::LibraryNodeCode& code,
    const std::vector<symbolic::Expression>& shape,
    QuantizationType quantization,
    const data_flow::ImplementationType& impl_type
)
    : TensorNode(element_id, debug_info, vertex, parent, code, {}, {"Y", "X"}, impl_type), shape_(shape) {}

void ElementWiseUnaryNode::validate(const Function& function) const {
    TensorNode::validate(function);

    auto& graph = this->get_parent();

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

symbolic::SymbolSet ElementWiseUnaryNode::symbols() const {
    symbolic::SymbolSet syms;
    for (const auto& dim : shape_) {
        for (auto& atom : symbolic::atoms(dim)) {
            syms.insert(atom);
        }
    }
    return syms;
}

void ElementWiseUnaryNode::replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) {
    for (auto& dim : shape_) {
        dim = symbolic::subs(dim, old_expression, new_expression);
    }
}

bool ElementWiseUnaryNode::expand(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    auto& dataflow = this->get_parent();
    auto& block = static_cast<structured_control_flow::Block&>(*dataflow.get_parent());
    if (dataflow.in_degree(*this) != 1 || dataflow.out_degree(*this) != 1) {
        return false;
    }

    auto& scope_analysis = analysis_manager.get<analysis::ScopeAnalysis>();
    auto& parent = static_cast<structured_control_flow::Sequence&>(*scope_analysis.parent_scope(&block));
    int index = parent.index(block);
    auto& transition = parent.at(index).second;

    auto& input = this->inputs_.at(0);
    auto& output = this->outputs_.at(0);

    auto& iedge = *dataflow.in_edges(*this).begin();
    auto& oedge = *dataflow.out_edges(*this).begin();

    // Checks if legal
    auto& input_node = static_cast<data_flow::AccessNode&>(iedge.src());
    auto& output_node = static_cast<data_flow::AccessNode&>(oedge.dst());
    if (dataflow.in_degree(input_node) != 0 || dataflow.out_degree(output_node) != 0) {
        return false;
    }

    // Add new graph after the current block
    auto& new_sequence = builder.add_sequence_before(parent, block, transition.assignments(), block.debug_info());

    // Add maps
    data_flow::Subset new_subset;
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

    bool success = this->expand_operation(
        builder,
        analysis_manager,
        *last_scope,
        input_node.data(),
        output_node.data(),
        static_cast<const types::Tensor&>(iedge.base_type()),
        static_cast<const types::Tensor&>(oedge.base_type()),
        loop_vars
    );
    if (!success) {
        return false;
    }

    // Clean up block
    builder.remove_memlet(block, iedge);
    builder.remove_memlet(block, oedge);
    builder.remove_node(block, input_node);
    builder.remove_node(block, output_node);
    builder.remove_node(block, *this);
    builder.remove_child(parent, index + 1);

    return true;
}

ElementWiseBinaryNode::ElementWiseBinaryNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const data_flow::LibraryNodeCode& code,
    const std::vector<symbolic::Expression>& shape
)
    : TensorNode(element_id, debug_info, vertex, parent, code, {"C"}, {"A", "B"}, data_flow::ImplementationType_NONE),
      shape_(shape) {}

void ElementWiseBinaryNode::validate(const Function& function) const {
    TensorNode::validate(function);

    auto& graph = this->get_parent();

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

symbolic::SymbolSet ElementWiseBinaryNode::symbols() const {
    symbolic::SymbolSet syms;
    for (const auto& dim : shape_) {
        for (auto& atom : symbolic::atoms(dim)) {
            syms.insert(atom);
        }
    }
    return syms;
}

void ElementWiseBinaryNode::replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) {
    for (auto& dim : shape_) {
        dim = symbolic::subs(dim, old_expression, new_expression);
    }
}

void ElementWiseBinaryNode::create_input_memlet(
    builder::StructuredSDFGBuilder& builder,
    const std::string& input_conn,
    const std::string& input_name,
    const types::Tensor& input_type,
    const data_flow::Subset& subset,
    structured_control_flow::Block& code_block,
    data_flow::CodeNode& code_node
) {
    if (builder.subject().exists(input_name)) {
        auto& input_node = builder.add_access(code_block, input_name);
        if (input_type.is_scalar()) {
            builder.add_computational_memlet(code_block, input_node, code_node, input_conn, {}, input_type);
        } else {
            builder.add_computational_memlet(code_block, input_node, code_node, input_conn, subset, input_type);
        }
    } else {
        types::Scalar const_type(input_type.primitive_type());
        auto& input_node = builder.add_constant(code_block, input_name, const_type);
        builder.add_computational_memlet(code_block, input_node, code_node, input_conn, {}, input_type);
    }
}

bool ElementWiseBinaryNode::expand(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    auto& dataflow = this->get_parent();
    auto& block = static_cast<structured_control_flow::Block&>(*dataflow.get_parent());
    if (dataflow.in_degree(*this) != 2 || dataflow.out_degree(*this) != 1) {
        return false;
    }
    auto& scope_analysis = analysis_manager.get<analysis::ScopeAnalysis>();
    auto& parent = static_cast<structured_control_flow::Sequence&>(*scope_analysis.parent_scope(&block));
    int index = parent.index(block);
    auto& transition = parent.at(index).second;

    auto& input_a = this->inputs_.at(0);
    auto& input_b = this->inputs_.at(1);
    auto& output = this->outputs_.at(0);

    auto iedge_a = &(*dataflow.in_edges(*this).begin());
    auto iedge_b = &(*(++dataflow.in_edges(*this).begin()));
    if (iedge_a->dst_conn() != "A") {
        std::swap(iedge_a, iedge_b);
    }
    auto& oedge = *dataflow.out_edges(*this).begin();

    // Checks if legal
    auto& input_node_a = static_cast<data_flow::AccessNode&>(iedge_a->src());
    auto& input_node_b = static_cast<data_flow::AccessNode&>(iedge_b->src());
    auto& output_node = static_cast<data_flow::AccessNode&>(oedge.dst());
    if (dataflow.in_degree(input_node_a) != 0 || dataflow.in_degree(input_node_b) != 0 ||
        dataflow.out_degree(output_node) != 0) {
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
    bool success = this->expand_operation(
        builder,
        analysis_manager,
        *last_scope,
        input_node_a.data(),
        input_node_b.data(),
        output_node.data(),
        static_cast<const types::Tensor&>(iedge_a->base_type()),
        static_cast<const types::Tensor&>(iedge_b->base_type()),
        static_cast<const types::Tensor&>(oedge.base_type()),
        loop_vars
    );
    if (!success) {
        return false;
    }

    // Clean up block
    builder.remove_memlet(block, *iedge_a);
    builder.remove_memlet(block, *iedge_b);
    builder.remove_memlet(block, oedge);
    builder.remove_node(block, input_node_a);
    // Only remove input_node_b if it's different from input_node_a
    if (&input_node_b != &input_node_a) {
        builder.remove_node(block, input_node_b);
    }
    builder.remove_node(block, output_node);
    builder.remove_node(block, *this);
    builder.remove_child(parent, index + 1);

    return true;
}

} // namespace tensor
} // namespace math
} // namespace sdfg
