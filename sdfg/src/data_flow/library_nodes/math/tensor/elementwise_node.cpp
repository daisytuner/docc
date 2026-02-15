#include "sdfg/data_flow/library_nodes/math/tensor/elementwise_node.h"

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"

#include "sdfg/analysis/scope_analysis.h"

namespace sdfg {
namespace math {
namespace tensor {

ElementWiseUnaryNode::ElementWiseUnaryNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const data_flow::LibraryNodeCode& code,
    const std::vector<symbolic::Expression>& shape
)
    : TensorNode(element_id, debug_info, vertex, parent, code, {"Y"}, {"X"}, data_flow::ImplementationType_NONE),
      shape_(shape) {}

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
