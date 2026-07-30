#include "sdfg/data_flow/library_nodes/math/tensor/elementwise_ops/arange_node.h"

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/library_nodes/math/math_node.h"
#include "sdfg/types/type.h"

namespace sdfg {
namespace math {
namespace tensor {

ArangeNode::ArangeNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const std::vector<symbolic::Expression>& shape,
    const data_flow::ImplementationType& impl_type
)
    : TensorNode(element_id, debug_info, vertex, parent, LibraryNodeType_Arange, {"_out"}, {"_start", "_step"}, impl_type),
      shape_(shape) {}

const std::vector<symbolic::Expression>& ArangeNode::shape() const { return shape_; }

void ArangeNode::validate(const Function& function) const {
    TensorNode::validate(function);

    // Validate that _start and _step inputs are scalars
    auto& dataflow = this->get_parent();
    auto edges = dataflow.in_edges_by_connector(*this);
    if (edges.count(START_IDX)) {
        auto* start_edge = edges.at(START_IDX);
        if (start_edge->base_type().type_id() != types::TypeID::Scalar) {
            throw InvalidSDFGException(
                "ArangeNode: _start input must be of scalar type. Found type: " + start_edge->base_type().print()
            );
        }
    }
    if (edges.count(STEP_IDX)) {
        auto* step_edge = edges.at(STEP_IDX);
        if (step_edge->base_type().type_id() != types::TypeID::Scalar) {
            throw InvalidSDFGException(
                "ArangeNode: _step input must be of scalar type. Found type: " + step_edge->base_type().print()
            );
        }
    }
}

symbolic::SymbolSet ArangeNode::symbols() const {
    symbolic::SymbolSet syms = TensorNode::symbols();
    for (const auto& dim : shape_) {
        symbolic::add_symbols(syms, dim);
    }
    return syms;
}

void ArangeNode::replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) {
    TensorNode::replace(old_expression, new_expression);
    for (auto& dim : shape_) {
        dim = symbolic::replace(dim, old_expression, new_expression);
    }
}

void ArangeNode::replace(const symbolic::ExpressionMapping& replacements) {
    TensorNode::replace(replacements);
    for (auto& dim : shape_) {
        dim = symbolic::replace(dim, replacements);
    }
}

bool ArangeNode::supports_integer_types() const { return true; }

passes::LibNodeExpander::ExpandOutcome ArangeNode::
    expand(passes::LibNodeExpander::ExpandContext& context, structured_control_flow::Block& block) {
    auto& dataflow = this->get_parent();

    if (dataflow.in_degree(*this) != 3 || dataflow.out_degree(*this) != 0) {
        return context.unable();
    }

    auto edges = dataflow.in_edges_by_connector(*this);
    auto& result_ptr_edge = *edges.at(RESULT_PTR_IDX);
    auto& start_edge = *edges.at(START_IDX);
    auto& step_edge = *edges.at(STEP_IDX);

    using Use = passes::LibNodeExpander::InputUse;
    auto standalone =
        context.replacement_requires_access_nodes({Use::IndirectWrite, Use::IndirectRead, Use::IndirectRead});

    if (!standalone) {
        return context.unable();
    }

    symbolic::MultiExpression loop_vars;
    auto& builder = standalone->builder();
    structured_control_flow::Sequence* inner_scope = nullptr;

    for (size_t i = 0; i < shape_.size(); ++i) {
        std::string var_name = builder.find_new_name("__i" + std::to_string(i));
        builder.add_container(var_name, types::Scalar(types::PrimitiveType::Int64));

        auto sym_var = symbolic::symbol(var_name);
        auto condition = symbolic::Lt(sym_var, shape_[i]);
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
                structured_control_flow::ScheduleType_Sequential::create()
            );
            inner_scope = &loop.root();
        }

        loop_vars.push_back(sym_var);
    }

    auto& tasklet_block = inner_scope ? *inner_scope : builder.add_sequence(block);

    auto& out_acc = builder.add_access_node(tasklet_block, result_ptr_edge.data());
    auto& start_acc = builder.add_access_node(tasklet_block, start_edge.data());
    auto& step_acc = builder.add_access_node(tasklet_block, step_edge.data());

    // Only works for 1D arange right now which is standard
    std::string tasklet_code = "_out = _start + _step * " + loop_vars.at(0)->__str__();

    auto& tasklet = builder.add_tasklet(
        tasklet_block, data_flow::TaskletCode::assign, "_out", {"_start", "_step"}, tasklet_code, this->debug_info()
    );

    builder.add_computational_memlet(
        tasklet_block, start_acc, tasklet, "_start", {}, start_edge.base_type(), this->debug_info()
    );
    builder.add_computational_memlet(
        tasklet_block, step_acc, tasklet, "_step", {}, step_edge.base_type(), this->debug_info()
    );
    builder.add_computational_memlet(
        tasklet_block, tasklet, "_out", out_acc, loop_vars, result_ptr_edge.base_type(), this->debug_info()
    );

    return standalone->successfully_expanded();
}

std::unique_ptr<data_flow::DataFlowNode> ArangeNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::unique_ptr<data_flow::DataFlowNode>(
        new ArangeNode(element_id, this->debug_info(), vertex, parent, shape_, implementation_type_)
    );
}

data_flow::PointerAccessType ArangeNode::pointer_access_type(int input_idx) const {
    if (input_idx == RESULT_PTR_IDX) {
        return data_flow::PointerAccessMeta::create_full_write_only(symbolic::__nullptr__(), true);
    } else if (input_idx == START_IDX || input_idx == STEP_IDX) {
        return data_flow::PointerAccessMeta::create_read_only(symbolic::__nullptr__(), true);
    } else {
        return TensorNode::pointer_access_type(input_idx);
    }
}

nlohmann::json ArangeNodeSerializer::serialize(const data_flow::LibraryNode& library_node) {
    const ArangeNode& arange_node = static_cast<const ArangeNode&>(library_node);
    nlohmann::json j;

    j["code"] = arange_node.code().value();

    serializer::JSONSerializer serializer;
    j["shape"] = nlohmann::json::array();
    for (auto& dim : arange_node.shape()) {
        j["shape"].push_back(serializer.expression(dim));
    }

    return j;
}

data_flow::LibraryNode& ArangeNodeSerializer::deserialize(
    const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
) {
    auto debug_info = builder.parse_debug_info(j);

    serializer::JSONSerializer serializer;
    std::vector<symbolic::Expression> shape;
    for (auto& dim : j["shape"]) {
        shape.push_back(serializer.parse_expression(dim));
    }

    return builder.add_library_node<ArangeNode>(parent, debug_info, shape);
}

} // namespace tensor
} // namespace math
} // namespace sdfg
