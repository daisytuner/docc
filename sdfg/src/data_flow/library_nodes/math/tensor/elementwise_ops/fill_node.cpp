#include "sdfg/data_flow/library_nodes/math/tensor/elementwise_ops/fill_node.h"

#include "sdfg/analysis/analysis.h"
#include "sdfg/analysis/scope_analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/library_nodes/math/math_node.h"
#include "sdfg/types/type.h"

namespace sdfg {
namespace math {
namespace tensor {

FillNode::FillNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const std::vector<symbolic::Expression>& shape
)
    : ElementWiseUnaryNode(element_id, debug_info, vertex, parent, LibraryNodeType_Fill, shape) {}

bool FillNode::expand_operation(
    builder::StructuredSDFGBuilder& builder,
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::Sequence& body,
    const std::string& input_name,
    const std::string& output_name,
    const types::Tensor& input_type,
    const types::Tensor& output_type,
    const data_flow::Subset& subset
) {
    // Add code
    auto& code_block = builder.add_block(body);

    auto& input_node_new = builder.add_access(code_block, input_name);
    auto& output_node_new = builder.add_access(code_block, output_name);

    auto& tasklet = builder.add_tasklet(code_block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(code_block, input_node_new, tasklet, "_in", subset, input_type);
    builder.add_computational_memlet(code_block, tasklet, "_out", output_node_new, subset, output_type);

    return true;
}

std::unique_ptr<data_flow::DataFlowNode> FillNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::unique_ptr<
        data_flow::DataFlowNode>(new FillNode(element_id, this->debug_info(), vertex, parent, this->shape_));
}

bool FillNode::expand(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    std::cerr << "expand" << std::endl;
    auto& dataflow = this->get_parent();
    auto& block = static_cast<structured_control_flow::Block&>(*dataflow.get_parent());
    if (dataflow.in_degree(*this) != 1 || dataflow.out_degree(*this) != 1) {
        return false;
    }

    auto& scope_analysis = analysis_manager.get<analysis::ScopeAnalysis>();
    auto& parent = static_cast<structured_control_flow::Sequence&>(*scope_analysis.parent_scope(&block));
    int index = parent.index(block);
    auto& transition = parent.at(index).second;

    auto& iedge = *dataflow.in_edges(*this).begin();
    auto& oedge = *dataflow.out_edges(*this).begin();

    auto input_node_const = dynamic_cast<data_flow::ConstantNode*>(&iedge.src());
    auto input_access = dynamic_cast<data_flow::AccessNode*>(&iedge.src());
    auto output_node = dynamic_cast<data_flow::AccessNode*>(&oedge.dst());
    if (dataflow.out_degree(*output_node) != 0) {
        return false;
    }
    if (input_access) {
        if (dataflow.in_degree(*input_access) != 0) {
            return false;
        }
    }

    // Add new graph after the current block
    auto& new_sequence = builder.add_sequence_before(parent, block, transition.assignments(), block.debug_info());

    // Add maps over output shape
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

    // Add the fill operation: read scalar input, write to each output element
    auto& code_block = builder.add_block(*last_scope);

    auto& tasklet = builder.add_tasklet(code_block, data_flow::TaskletCode::assign, "_out", {"_in"});

    if (input_node_const) {
        auto& input_node_new = builder.add_constant(code_block, input_node_const->data(), input_node_const->type());
        builder.add_computational_memlet(code_block, input_node_new, tasklet, "_in", {}, input_node_const->type());
    } else {
        auto& input_node_new = builder.add_access(code_block, input_access->data());
        builder.add_computational_memlet(code_block, input_node_new, tasklet, "_in", {}, iedge.base_type());
    }

    auto& output_node_new = builder.add_access(code_block, output_node->data());
    builder.add_computational_memlet(
        code_block, tasklet, "_out", output_node_new, loop_vars, static_cast<const types::Tensor&>(oedge.base_type())
    );

    // Clean up block
    builder.remove_memlet(block, iedge);
    builder.remove_memlet(block, oedge);
    if (input_node_const) {
        builder.remove_node(block, *input_node_const);
    } else {
        builder.remove_node(block, *input_access);
    }
    builder.remove_node(block, *output_node);
    builder.remove_node(block, *this);
    builder.remove_child(parent, index + 1);

    return true;
}

void FillNode::validate(const Function& function) const {
    MathNode::validate(function);

    // Validate that the input is a scalar
    auto& graph = this->get_parent();
    for (auto& iedge : graph.in_edges(*this)) {
        if (iedge.base_type().type_id() != types::TypeID::Scalar) {
            throw InvalidSDFGException(
                "FillNode: Input memlet must be of scalar type. Found type: " + iedge.base_type().print()
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

    // Validate that all memlets have the same primitive type
    types::PrimitiveType prim_type = primitive_type(graph);

    // Check if this operation supports integer types
    if (!supports_integer_types() && types::is_integer(prim_type)) {
        throw InvalidSDFGException(
            "TensorNode: This operation does not support integer types. Found type: " +
            std::string(types::primitive_type_to_string(prim_type))
        );
    }
}

} // namespace tensor
} // namespace math
} // namespace sdfg
