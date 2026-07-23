#include "sdfg/data_flow/library_nodes/math/tensor/broadcast_node.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_nodes/math/tensor/matmul_node.h"
#include "sdfg/structured_control_flow/for.h"

namespace sdfg {
namespace math {
namespace tensor {

BroadcastNode::BroadcastNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const std::vector<symbolic::Expression>& input_shape,
    const std::vector<symbolic::Expression>& output_shape
)
    : TensorNode(
          element_id,
          debug_info,
          vertex,
          parent,
          LibraryNodeType_Broadcast,
          {},
          {"Y", "X"},
          data_flow::ImplementationType_NONE
      ),
      input_shape_(input_shape), output_shape_(output_shape) {}

void BroadcastNode::validate(const Function& function) const {
    TensorNode::validate(function);

    auto& graph = this->get_parent();

    auto* iedge = graph.in_edge_for_connector(*this, inputs_.at(X_INPUT_IDX));
    auto& shape = static_cast<const types::Tensor&>(iedge->base_type());
    if (!shape.is_scalar()) {
        if (shape.shape().size() != this->input_shape_.size()) {
            throw InvalidSDFGException(
                "Library Node: Tensor shape must match node shape. Tensor shape: " +
                std::to_string(shape.shape().size()) + " Node shape: " + std::to_string(this->input_shape_.size())
            );
        }
        for (size_t i = 0; i < this->input_shape_.size(); ++i) {
            if (!symbolic::eq(shape.shape().at(i), this->input_shape_.at(i))) {
                throw InvalidSDFGException(
                    "Library Node: Tensor shape does not match expected shape. Tensor shape: " +
                    shape.shape().at(i)->__str__() + " Expected shape: " + this->input_shape_.at(i)->__str__()
                );
            }
        }
    }

    auto* oedge = graph.in_edge_for_connector(*this, inputs_.at(RESULT_PTR_IDX));
    auto& output_shape = static_cast<const types::Tensor&>(oedge->base_type());
    if (output_shape.shape().size() != this->output_shape_.size()) {
        throw InvalidSDFGException(
            "Library Node: Output tensor shape must match node shape. Output tensor shape: " +
            std::to_string(output_shape.shape().size()) + " Node shape: " + std::to_string(this->output_shape_.size())
        );
    }

    for (size_t i = 0; i < this->output_shape_.size(); ++i) {
        if (!symbolic::eq(output_shape.shape().at(i), this->output_shape_.at(i))) {
            throw InvalidSDFGException(
                "Library Node: Output tensor shape does not match expected shape. Output tensor shape: " +
                output_shape.shape().at(i)->__str__() + " Expected shape: " + this->output_shape_.at(i)->__str__()
            );
        }
    }
}

symbolic::SymbolSet BroadcastNode::symbols() const {
    symbolic::SymbolSet syms;
    for (const auto& dim : input_shape_) {
        for (auto& atom : symbolic::atoms(dim)) {
            syms.insert(atom);
        }
    }
    for (const auto& dim : output_shape_) {
        for (auto& atom : symbolic::atoms(dim)) {
            syms.insert(atom);
        }
    }
    return syms;
}

void BroadcastNode::replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) {
    for (auto& dim : input_shape_) {
        dim = symbolic::subs(dim, old_expression, new_expression);
    }
    for (auto& dim : output_shape_) {
        dim = symbolic::subs(dim, old_expression, new_expression);
    }
}

void BroadcastNode::replace(const symbolic::ExpressionMapping& replacements) {
    for (auto& dim : input_shape_) {
        dim = symbolic::subs(dim, replacements);
    }
    for (auto& dim : output_shape_) {
        dim = symbolic::subs(dim, replacements);
    }
}

passes::LibNodeExpander::ExpandOutcome BroadcastNode::
    expand(passes::LibNodeExpander::ExpandContext& context, structured_control_flow::Block& block) {
    auto& dataflow = this->get_parent();

    if (dataflow.in_degree(*this) != 2 || dataflow.out_degree(*this) != 0) {
        return context.unable();
    }

    auto edges = dataflow.in_edges_by_connector(*this);
    auto& in_edge = *edges.at(X_INPUT_IDX);
    auto& result_ptr_edge = *edges.at(RESULT_PTR_IDX);


    using Use = passes::LibNodeExpander::InputUse;
    auto standalone = context.replacement_requires_access_nodes({Use::IndirectWrite, Use::IndirectRead});

    if (!standalone) {
        return context.unable();
    }

    symbolic::MultiExpression loop_vars;
    auto& builder = standalone->builder();
    structured_control_flow::Sequence* inner_scope = nullptr;

    for (size_t i = 0; i < output_shape_.size(); ++i) {
        std::string var_name = builder.find_new_name("_i" + std::to_string(i));
        builder.add_container(var_name, types::Scalar(types::PrimitiveType::Int64));

        auto sym_var = symbolic::symbol(var_name);
        auto condition = symbolic::Lt(sym_var, output_shape_[i]);
        auto init = symbolic::zero();
        auto update = symbolic::add(sym_var, symbolic::one());

        if (i == 0) {
            auto& loop = standalone->replace_with_structured_loop(
                passes::LibNodeExpander::AccessNodeExpand::LoopType::Map,
                sym_var,
                condition,
                init,
                update,
                structured_control_flow::ScheduleType_Sequential::create()
            );
            inner_scope = &loop.root();
        } else {
            auto& loop = builder.add_map(
                *inner_scope,
                sym_var,
                condition,
                init,
                update,
                structured_control_flow::ScheduleType_Sequential::create(),
                {},
                this->debug_info()
            );
            inner_scope = &loop.root();
        }
        loop_vars.push_back(sym_var);
    }

    auto& tasklet_block = builder.add_block(*inner_scope, {}, this->debug_info());

    auto& in_acc = standalone->add_indirect_read_access(tasklet_block, X_INPUT_IDX);
    auto& out_acc = standalone->add_indirect_write_access(tasklet_block, RESULT_PTR_IDX);

    symbolic::MultiExpression input_subset = {};
    size_t j = 0;
    for (size_t i = 0; i < output_shape_.size(); ++i) {
        if (j >= input_shape_.size()) {
            break;
        } else if (symbolic::eq(output_shape_[i], input_shape_[j])) {
            input_subset.push_back(loop_vars[i]);
            j++;
        }
    }
    if (j < input_shape_.size()) {
        throw InvalidSDFGException("BroadcastNode: Could not resolve indvars for inputs");
    }
    auto& iedge_tensor = static_cast<const types::Tensor&>(in_edge.base_type());
    if (iedge_tensor.is_scalar()) {
        input_subset = {};
    }

    auto& tasklet =
        builder.add_tasklet(tasklet_block, data_flow::TaskletCode::assign, "_out", {"_in"}, this->debug_info());

    builder.add_computational_memlet(
        tasklet_block, in_acc, tasklet, "_in", input_subset, in_edge.base_type(), this->debug_info()
    );
    builder.add_computational_memlet(
        tasklet_block, tasklet, "_out", out_acc, loop_vars, result_ptr_edge.base_type(), this->debug_info()
    );

    return standalone->successfully_expanded();
}

std::unique_ptr<data_flow::DataFlowNode> BroadcastNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::unique_ptr<data_flow::DataFlowNode>(
        new BroadcastNode(element_id, this->debug_info(), vertex, parent, input_shape_, output_shape_)
    );
}

data_flow::PointerAccessType BroadcastNode::pointer_access_type(int input_idx) const {
    if (input_idx == RESULT_PTR_IDX) {
        return data_flow::PointerAccessMeta::create_full_write_only(symbolic::__nullptr__(), true);
    } else if (input_idx == X_INPUT_IDX) {
        return data_flow::PointerAccessMeta::create_read_only(symbolic::__nullptr__(), true);
    } else {
        return TensorNode::pointer_access_type(input_idx);
    }
}

} // namespace tensor
} // namespace math
} // namespace sdfg
