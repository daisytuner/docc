#include "sdfg/data_flow/library_nodes/math/tensor/arange_node.h"

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
    : TensorNode(
          element_id,
          debug_info,
          vertex,
          parent,
          LibraryNodeType_Arange,
          {},
          {"_out", "_start", "_end", "_step"},
          impl_type
      ),
      shape_(shape) {}

const std::vector<symbolic::Expression>& ArangeNode::shape() const { return shape_; }

void ArangeNode::validate(const Function& function) const {
    TensorNode::validate(function);

    auto& dataflow = this->get_parent();
    auto edges = dataflow.in_edges_by_connector(*this);

    if (edges.size() > RESULT_PTR_IDX && edges[RESULT_PTR_IDX] != nullptr) {
        auto* result_edge = edges.at(RESULT_PTR_IDX);
        auto type_id = result_edge->base_type().type_id();
        if (type_id != types::TypeID::Tensor && type_id != types::TypeID::Pointer) {
            throw InvalidSDFGException(
                "ArangeNode: _out input must be of tensor or pointer type. Found type: " +
                result_edge->base_type().print()
            );
        }
    }

    if (edges.size() > START_IDX && edges[START_IDX] != nullptr) {
        auto* start_edge = edges.at(START_IDX);
        if (start_edge->base_type().type_id() != types::TypeID::Scalar) {
            throw InvalidSDFGException(
                "ArangeNode: _start input must be of scalar type. Found type: " + start_edge->base_type().print()
            );
        }
    }
    if (edges.size() > END_IDX && edges[END_IDX] != nullptr) {
        auto* end_edge = edges.at(END_IDX);
        if (end_edge->base_type().type_id() != types::TypeID::Scalar) {
            throw InvalidSDFGException(
                "ArangeNode: _end input must be of scalar type. Found type: " + end_edge->base_type().print()
            );
        }
    }
    if (edges.size() > STEP_IDX && edges[STEP_IDX] != nullptr) {
        auto* step_edge = edges.at(STEP_IDX);
        if (step_edge->base_type().type_id() != types::TypeID::Scalar) {
            throw InvalidSDFGException(
                "ArangeNode: _step input must be of scalar type. Found type: " + step_edge->base_type().print()
            );
        }
    }
}

symbolic::SymbolSet ArangeNode::symbols() const {
    symbolic::SymbolSet syms;
    for (const auto& dim : shape_) {
        for (auto& atom : symbolic::atoms(dim)) {
            syms.insert(atom);
        }
    }
    return syms;
}

void ArangeNode::replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) {
    for (auto& dim : shape_) {
        dim = symbolic::subs(dim, old_expression, new_expression);
    }
}

void ArangeNode::replace(const symbolic::ExpressionMapping& replacements) {
    for (auto& dim : shape_) {
        dim = symbolic::subs(dim, replacements);
    }
}

bool ArangeNode::supports_integer_types() const { return true; }

passes::LibNodeExpander::ExpandOutcome ArangeNode::
    expand(passes::LibNodeExpander::ExpandContext& context, structured_control_flow::Block& block) {
    auto& dataflow = this->get_parent();

    if (dataflow.in_degree(*this) != 4 || dataflow.out_degree(*this) != 0) {
        return context.unable();
    }

    auto edges = dataflow.in_edges_by_connector(*this);
    auto& result_ptr_edge = *edges.at(RESULT_PTR_IDX);
    auto& start_edge = *edges.at(START_IDX);
    auto& step_edge = *edges.at(STEP_IDX);

    using Use = passes::LibNodeExpander::InputUse;
    // _end is Skip: it is captured symbolically in shape_ and not needed in the expansion body
    auto standalone =
        context.replacement_requires_access_nodes({Use::IndirectWrite, Use::IndirectRead, Use::Skip, Use::IndirectRead}
        );

    if (!standalone) {
        return context.unable();
    }

    symbolic::MultiExpression loop_vars;
    auto& builder = standalone->builder();
    structured_control_flow::Sequence* inner_scope = nullptr;

    for (size_t i = 0; i < shape_.size(); ++i) {
        std::string var_name = builder.find_new_name("_i" + std::to_string(i));
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

    auto& tasklet_block = builder.add_block(*inner_scope, {}, this->debug_info());

    auto& out_acc = standalone->add_indirect_write_access(tasklet_block, RESULT_PTR_IDX);
    auto& start_acc = standalone->add_indirect_read_access(tasklet_block, START_IDX);
    auto& step_acc = standalone->add_indirect_read_access(tasklet_block, STEP_IDX);

    bool is_float = false;
    if (auto* tensor_type = dynamic_cast<const types::Tensor*>(&result_ptr_edge.base_type())) {
        is_float = types::is_floating_point(tensor_type->primitive_type());
    } else if (auto* ptr_type = dynamic_cast<const types::Pointer*>(&result_ptr_edge.base_type())) {
        is_float = types::is_floating_point(ptr_type->primitive_type());
    }

    auto& i0_acc = builder.add_access(tasklet_block, loop_vars.at(0)->__str__(), this->debug_info());

    if (is_float) {
        std::string cast_tmp_name = builder.find_new_name("_i_cast");
        builder.add_container(cast_tmp_name, types::Scalar(result_ptr_edge.base_type().primitive_type()));
        auto& cast_tmp_acc = builder.add_access(tasklet_block, cast_tmp_name, this->debug_info());

        auto& cast_tasklet =
            builder.add_tasklet(tasklet_block, data_flow::TaskletCode::assign, "_out", {"_in"}, this->debug_info());

        builder.add_computational_memlet(
            tasklet_block,
            i0_acc,
            cast_tasklet,
            "_in",
            {},
            types::Scalar(types::PrimitiveType::Int64),
            this->debug_info()
        );

        builder.add_computational_memlet(
            tasklet_block,
            cast_tasklet,
            "_out",
            cast_tmp_acc,
            {},
            types::Scalar(result_ptr_edge.base_type().primitive_type()),
            this->debug_info()
        );

        auto& tasklet = builder.add_tasklet(
            tasklet_block, data_flow::TaskletCode::fp_fma, "_out", {"_step", "_i", "_start"}, this->debug_info()
        );

        builder.add_computational_memlet(
            tasklet_block, step_acc, tasklet, "_step", {}, step_edge.base_type(), this->debug_info()
        );
        builder.add_computational_memlet(
            tasklet_block,
            cast_tmp_acc,
            tasklet,
            "_i",
            {},
            types::Scalar(result_ptr_edge.base_type().primitive_type()),
            this->debug_info()
        );
        builder.add_computational_memlet(
            tasklet_block, start_acc, tasklet, "_start", {}, start_edge.base_type(), this->debug_info()
        );

        builder.add_computational_memlet(
            tasklet_block, tasklet, "_out", out_acc, loop_vars, result_ptr_edge.base_type(), this->debug_info()
        );
    } else {
        std::string tmp_name = builder.find_new_name("_arange_tmp");
        builder.add_container(tmp_name, types::Scalar(result_ptr_edge.base_type().primitive_type()));
        auto& tmp_acc = builder.add_access(tasklet_block, tmp_name, this->debug_info());

        auto& tasklet_mul =
            builder
                .add_tasklet(tasklet_block, data_flow::TaskletCode::int_mul, "_out", {"_step", "_i"}, this->debug_info());
        builder.add_computational_memlet(
            tasklet_block, step_acc, tasklet_mul, "_step", {}, step_edge.base_type(), this->debug_info()
        );
        builder.add_computational_memlet(
            tasklet_block, i0_acc, tasklet_mul, "_i", {}, types::Scalar(types::PrimitiveType::Int64), this->debug_info()
        );
        builder.add_computational_memlet(
            tasklet_block,
            tasklet_mul,
            "_out",
            tmp_acc,
            {},
            types::Scalar(result_ptr_edge.base_type().primitive_type()),
            this->debug_info()
        );

        auto& tasklet_add = builder.add_tasklet(
            tasklet_block, data_flow::TaskletCode::int_add, "_out", {"_tmp", "_start"}, this->debug_info()
        );
        builder.add_computational_memlet(
            tasklet_block,
            tmp_acc,
            tasklet_add,
            "_tmp",
            {},
            types::Scalar(result_ptr_edge.base_type().primitive_type()),
            this->debug_info()
        );
        builder.add_computational_memlet(
            tasklet_block, start_acc, tasklet_add, "_start", {}, start_edge.base_type(), this->debug_info()
        );
        builder.add_computational_memlet(
            tasklet_block, tasklet_add, "_out", out_acc, loop_vars, result_ptr_edge.base_type(), this->debug_info()
        );
    }

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
    } else if (input_idx == START_IDX || input_idx == END_IDX || input_idx == STEP_IDX) {
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
    sdfg::serializer::JSONSerializer serializer;
    DebugInfo debug_info = serializer.json_to_debug_info(j["debug_info"]);

    std::vector<symbolic::Expression> shape;
    for (auto& dim : j["shape"]) {
        shape.push_back(symbolic::parse(dim.get<std::string>()));
    }

    return builder.add_library_node<ArangeNode>(parent, debug_info, shape);
}

} // namespace tensor
} // namespace math
} // namespace sdfg
