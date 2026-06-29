#include "sdfg/data_flow/library_nodes/math/tensor/elementwise_node.h"

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/types/type.h"

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

void ElementWiseDataflowTensorNode::set_fixed_quantization(const QuantizationType quant) {
    fixed_quantization_ = quant;
}

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

void ElementWiseDataflowTensorNode::replace(const symbolic::ExpressionMapping& replacements) {
    for (auto& dim : shape_) {
        dim = symbolic::subs(dim, replacements);
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
    const std::vector<symbolic::Expression>& eltwise_subset,
    std::unordered_map<const data_flow::AccessNode*, data_flow::AccessNode*>& new_node_mapping
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
        auto existing_input_it = new_node_mapping.find(&org_src);
        data_flow::AccessNode* input_node;
        std::vector<symbolic::Expression> empty_subset;
        const std::vector<symbolic::Expression>* memlet_subset;
        if (src_type.second && !src_type.second->is_scalar()) {
            memlet_subset = &eltwise_subset;
        } else {
            memlet_subset = &empty_subset;
        }
        auto new_type = access_type(src_type);
        if (existing_input_it != new_node_mapping.end()) {
            input_node = existing_input_it->second;
        } else {
            if (org_src.is_constant()) {
                types::Scalar const_type(src_type.first);
                input_node = &builder.add_constant(block, org_src.data(), const_type);
            } else {
                input_node = &builder.add_access(block, org_src.data());
            }
            new_node_mapping.emplace(&org_src, input_node);
        }

        builder.add_computational_memlet(
            block,
            *input_node,
            *new_consumer,
            new_consumer->input(needed_input.input_conn_index),
            *memlet_subset,
            *new_type
        );
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

    auto& parent = static_cast<structured_control_flow::Sequence&>(*org_block.get_parent());
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

    std::unordered_map<const data_flow::AccessNode*, data_flow::AccessNode*> new_node_mapping;

    // for all old input edge, remove old, create new
    for (int i = 0; i < iedges.size(); ++i) {
        create_input(
            builder, new_block, *inputs_sa.at(i), input_types.at(i), eltwise_inputs.at(i), loop_vars, new_node_mapping
        );
    }
    create_output(builder, new_block, *output_tensor_sa, target_tensor, produced_output, loop_vars);
    builder.clear_code_node_legacy(org_block, *this);
    // WARNING: this has been deallocated at this point!!
    builder.remove_child(parent, index + 1);

    return true;
}

data_flow::PointerAccessType ElementWiseDataflowTensorNode::pointer_access_type(int input_idx) const {
    if (input_idx == 0) {
        return data_flow::PointerAccessMeta::create_full_write_only(symbolic::__nullptr__(), true);
    } else if (input_idx < tensor_input_count()) {
        return data_flow::PointerAccessMeta::create_read_only(symbolic::__nullptr__(), true);
    } else {
        return TensorNode::pointer_access_type(input_idx);
    }
}

data_flow::AccessNode& ElementWiseDataflowTensorNode::create_tmp_access_node(
    builder::StructuredSDFGBuilder& builder,
    structured_control_flow::Block& block,
    const std::string& prefix,
    const types::IType& type
) const {
    auto cont = builder.find_new_name(prefix);
    builder.add_container(cont, type);
    auto& output_node_add = builder.add_access(block, cont);
    return output_node_add;
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

} // namespace tensor
} // namespace math
} // namespace sdfg
