#include "sdfg/data_flow/library_nodes/math/tensor/arange_node.h"

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/library_nodes/math/math_node.h"
#include "sdfg/data_flow/memlet.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/type.h"
#include "sdfg/types/utils.h"

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
        if (type_id != types::TypeID::Tensor) {
            throw InvalidSDFGException(
                "ArangeNode: _out input must be of tensor type. Found type: " + result_edge->base_type().print()
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

    bool start_zero =
        isa<data_flow::AccessNode>(start_edge.src()) &&
        data_flow::AccessNode::has_constant_value(static_cast<data_flow::AccessNode&>(start_edge.src()), 0ll);
    bool step_one =
        isa<data_flow::AccessNode>(step_edge.src()) &&
        data_flow::AccessNode::has_constant_value(static_cast<data_flow::AccessNode&>(step_edge.src()), 1ll);

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
        auto& dim = shape_[i];
        builder.add_container(var_name, types::Scalar(types::get_primitive_type_to_hold_upper_bound(dim)));

        auto sym_var = symbolic::symbol(var_name);
        auto condition = symbolic::Lt(sym_var, dim);
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
                this->debug_info_
            );
            inner_scope = &loop.root();
        }

        loop_vars.push_back(sym_var);
    }

    auto& new_block = builder.add_block(*inner_scope, {}, this->debug_info_);
    types::Scalar base_type(result_ptr_edge.base_type().primitive_type());
    bool is_float = types::is_floating_point(base_type.primitive_type());

    // Create access to index variable and cast it if necessary
    auto* i_access = &builder.add_access(new_block, loop_vars.at(0)->__str__(), this->debug_info_);
    if (is_float) {
        std::string cast_tmp_name = builder.find_new_name("_i_cast");
        builder.add_container(cast_tmp_name, base_type);
        auto& cast_tmp_access = builder.add_access(new_block, cast_tmp_name, this->debug_info_);
        auto& cast_tasklet =
            builder.add_tasklet(new_block, data_flow::TaskletCode::assign, "_out", {"_in"}, this->debug_info_);
        builder.add_computational_memlet(
            new_block, *i_access, cast_tasklet, "_in", {}, types::Scalar(types::PrimitiveType::Int64), this->debug_info_
        );
        builder
            .add_computational_memlet(new_block, cast_tasklet, "_out", cast_tmp_access, {}, base_type, this->debug_info_);
        i_access = &cast_tmp_access;
    }

    // Special case: Use FMA
    if (!start_zero && !step_one && is_float) {
        auto& start_access = standalone->add_indirect_read_access(new_block, START_IDX);
        auto& step_access = standalone->add_indirect_read_access(new_block, STEP_IDX);
        auto& result_access = standalone->add_indirect_write_access(new_block, RESULT_PTR_IDX);
        auto& fma_tasklet = builder.add_tasklet(
            new_block, data_flow::TaskletCode::fp_fma, "_out", {"_in1", "_in2", "_in3"}, this->debug_info_
        );
        builder.add_computational_memlet(new_block, *i_access, fma_tasklet, "_in1", {}, base_type, this->debug_info_);
        builder
            .add_computational_memlet(new_block, step_access, fma_tasklet, "_in2", {}, base_type, step_edge.debug_info());
        builder
            .add_computational_memlet(new_block, start_access, fma_tasklet, "_in3", {}, base_type, step_edge.debug_info());
        builder.add_computational_memlet(
            new_block,
            fma_tasklet,
            "_out",
            result_access,
            loop_vars,
            result_ptr_edge.base_type(),
            result_ptr_edge.debug_info()
        );
        return standalone->successfully_expanded();
    }

    // Special case: Use assign
    if (start_zero && step_one) {
        auto& result_access = standalone->add_indirect_write_access(new_block, RESULT_PTR_IDX);
        auto& assign_tasklet =
            builder.add_tasklet(new_block, data_flow::TaskletCode::assign, "_out", {"_in"}, this->debug_info_);
        builder.add_computational_memlet(new_block, *i_access, assign_tasklet, "_in", {}, base_type);
        builder.add_computational_memlet(
            new_block,
            assign_tasklet,
            "_out",
            result_access,
            loop_vars,
            result_ptr_edge.base_type(),
            result_ptr_edge.debug_info()
        );
        return standalone->successfully_expanded();
    }

    data_flow::AccessNode* tmp_access = nullptr;
    data_flow::Subset tmp_subset;
    const types::IType* tmp_type = nullptr;
    data_flow::AccessNode* result_access = nullptr;
    data_flow::Subset result_subset;
    const types::IType* result_type = nullptr;
    if (start_zero && !step_one) {
        tmp_access = &standalone->add_indirect_write_access(new_block, RESULT_PTR_IDX);
        tmp_subset = loop_vars;
        tmp_type = &result_ptr_edge.base_type();
    } else if (!start_zero && step_one) {
        tmp_access = i_access;
        tmp_type = &base_type;
        result_access = &standalone->add_indirect_write_access(new_block, RESULT_PTR_IDX);
        result_subset = loop_vars;
        result_type = &result_ptr_edge.base_type();
    } else {
        auto tmp_step_container = builder.find_new_name("_arange_tmp");
        builder.add_container(tmp_step_container, base_type);
        tmp_access = &builder.add_access(new_block, tmp_step_container, this->debug_info_);
        tmp_type = &base_type;
        result_access = &standalone->add_indirect_write_access(new_block, RESULT_PTR_IDX);
        result_subset = loop_vars;
        result_type = &result_ptr_edge.base_type();
    }

    // Multiply index variable with step if necessary
    if (!step_one) {
        auto& step_access = standalone->add_indirect_read_access(new_block, STEP_IDX);
        auto& mul_tasklet = builder.add_tasklet(
            new_block,
            (is_float ? data_flow::TaskletCode::fp_mul : data_flow::TaskletCode::int_mul),
            "_out",
            {"_in1", "_in2"},
            this->debug_info_
        );
        builder.add_computational_memlet(new_block, *i_access, mul_tasklet, "_in1", {}, base_type, this->debug_info_);
        builder
            .add_computational_memlet(new_block, step_access, mul_tasklet, "_in2", {}, base_type, step_edge.debug_info());
        builder.add_computational_memlet(
            new_block, mul_tasklet, "_out", *tmp_access, tmp_subset, *tmp_type, this->debug_info_
        );
    }

    // Add start if necessary
    if (!start_zero) {
        auto& start_access = standalone->add_indirect_read_access(new_block, START_IDX);
        auto& add_tasklet = builder.add_tasklet(
            new_block,
            (is_float ? data_flow::TaskletCode::fp_add : data_flow::TaskletCode::int_add),
            "_out",
            {"_in1", "_in2"},
            this->debug_info_
        );
        builder.add_computational_memlet(
            new_block, *tmp_access, add_tasklet, "_in1", tmp_subset, *tmp_type, this->debug_info_
        );
        builder.add_computational_memlet(
            new_block, start_access, add_tasklet, "_in2", {}, base_type, start_edge.debug_info()
        );
        builder.add_computational_memlet(
            new_block, add_tasklet, "_out", *result_access, result_subset, *result_type, result_ptr_edge.debug_info()
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

symbolic::Expression ArangeNode::flop() const {
    auto edges = this->get_parent().in_edges_by_connector(*this);
    auto& start_edge = *edges.at(START_IDX);
    auto& step_edge = *edges.at(STEP_IDX);
    bool start_zero =
        isa<data_flow::AccessNode>(start_edge.src()) &&
        data_flow::AccessNode::has_constant_value(static_cast<const data_flow::AccessNode&>(start_edge.src()), 0ll);
    bool step_one =
        isa<data_flow::AccessNode>(step_edge.src()) &&
        data_flow::AccessNode::has_constant_value(static_cast<const data_flow::AccessNode&>(step_edge.src()), 1ll);

    long long inner = 0;
    if (!start_zero) {
        ++inner;
    }
    if (!step_one) {
        ++inner;
    }
    return symbolic::mul(SymEngine::mul(this->shape_), symbolic::integer(inner));
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
