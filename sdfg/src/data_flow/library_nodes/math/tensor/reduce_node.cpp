#include "sdfg/data_flow/library_nodes/math/tensor/reduce_node.h"

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"

#include <algorithm>

#include "sdfg/types/utils.h"

namespace sdfg {
namespace math {
namespace tensor {

ReduceNode::ReduceNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const data_flow::LibraryNodeCode& code,
    const std::vector<symbolic::Expression>& shape,
    const std::vector<int64_t>& axes,
    bool keepdims
)
    : TensorNode(element_id, debug_info, vertex, parent, code, {}, {"Y", "X"}, data_flow::ImplementationType_NONE),
      shape_(shape), axes_(axes), keepdims_(keepdims) {
}

void ReduceNode::validate(const Function& function) const {
    TensorNode::validate(function);
    validate_shapes();
}

std::vector<int64_t> ReduceNode::normalized_axes() const {
    auto rank = static_cast<int64_t>(shape_.size());
    std::vector<int64_t> axes = axes_;
    for (auto& axis : axes) {
        if (axis < 0) {
            axis += rank;
        }
        if (axis < 0 || axis >= rank) {
            throw InvalidSDFGException(
                "Library Node: Axis value out of bounds. Axis: " + std::to_string(axis) +
                " Shape size: " + std::to_string(shape_.size())
            );
        }
    }
    std::sort(axes.begin(), axes.end());
    return axes;
}

int64_t ReduceNode::single_axis() const {
    if (axes_.size() != 1) {
        throw InvalidSDFGException("Library Node: " + code().value() + " takes exactly one axis");
    }
    return normalized_axes().front();
}

std::vector<symbolic::Expression> ReduceNode::output_shape() const {
    auto axes = normalized_axes();
    std::vector<symbolic::Expression> shape;
    for (size_t i = 0; i < shape_.size(); ++i) {
        if (std::find(axes.begin(), axes.end(), static_cast<int64_t>(i)) == axes.end()) {
            shape.push_back(shape_[i]);
        } else if (keepdims_) {
            shape.push_back(symbolic::one());
        }
    }
    return shape;
}

void ReduceNode::validate_shapes() const {
    auto& graph = this->get_parent();
    auto* iedge_input = graph.in_edge_for_connector(*this, inputs_.at(X_INPUT_IDX));
    auto* iedge_result = graph.in_edge_for_connector(*this, inputs_.at(RESULT_PTR_IDX));
    if (!iedge_input || !iedge_result) {
        throw InvalidSDFGException("Library Node: " + code().value() + " needs X and Y");
    }
    if (auto* tensor = dynamic_cast<const types::Tensor*>(&iedge_input->base_type())) {
        validate_shape_matches(shape_, tensor->layout(), "input");
    }
    if (auto* tensor = dynamic_cast<const types::Tensor*>(&iedge_result->base_type())) {
        validate_shape_matches(output_shape(), tensor->layout(), "output");
    }
}

std::pair<structured_control_flow::Sequence*, std::vector<symbolic::Expression>> ReduceNode::add_outer_maps(
    builder::StructuredSDFGBuilder& builder, structured_control_flow::Sequence& scope, const std::vector<int64_t>& axes
) const {
    std::vector<symbolic::Expression> index(shape_.size(), SymEngine::null);
    auto* body = &scope;
    for (size_t i = 0; i < shape_.size(); ++i) {
        if (std::find(axes.begin(), axes.end(), static_cast<int64_t>(i)) != axes.end()) {
            continue;
        }
        auto& limit = shape_[i];
        auto indvar = symbolic::symbol(add_scalar(builder, "_i", types::get_primitive_type_to_hold_upper_bound(limit)));
        auto& map = builder.add_map(
            *body,
            indvar,
            symbolic::Lt(indvar, limit),
            symbolic::zero(),
            symbolic::add(indvar, symbolic::one()),
            structured_control_flow::ScheduleType_Sequential::create(),
            this->debug_info()
        );
        body = &map.root();
        index[i] = indvar;
    }
    return {body, index};
}

std::string ReduceNode::
    add_scalar(builder::StructuredSDFGBuilder& builder, const std::string& base, types::PrimitiveType type) {
    auto name = builder.find_new_name(base);
    builder.add_container(name, types::Scalar(type));
    return name;
}

std::pair<symbolic::Symbol, structured_control_flow::Sequence*> ReduceNode::add_counted_loop(
    builder::StructuredSDFGBuilder& builder,
    structured_control_flow::Sequence& scope,
    const symbolic::Expression& init,
    const symbolic::Expression& limit,
    bool map
) const {
    auto indvar = symbolic::symbol(add_scalar(builder, map ? "_i" : "_k", types::PrimitiveType::Int64));
    auto condition = symbolic::Lt(indvar, limit);
    auto update = symbolic::add(indvar, symbolic::one());
    if (map) {
        auto& node = builder.add_map(
            scope,
            indvar,
            condition,
            init,
            update,
            structured_control_flow::ScheduleType_Sequential::create(),
            this->debug_info()
        );
        return {indvar, &node.root()};
    }
    auto& node = builder.add_for(scope, indvar, condition, init, update, this->debug_info());
    return {indvar, &node.root()};
}

void ReduceNode::add_index_assignment(
    builder::StructuredSDFGBuilder& builder,
    structured_control_flow::Block& block,
    data_flow::AccessNode& target,
    const data_flow::Subset& subset,
    const types::IType& target_type,
    const std::string& value,
    bool constant
) const {
    auto& debug_info = this->debug_info();
    types::Scalar index(types::PrimitiveType::Int64);
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"}, debug_info);
    auto& source = constant ? static_cast<data_flow::AccessNode&>(builder.add_constant(block, value, index, debug_info))
                            : builder.add_access(block, value, debug_info);
    builder.add_computational_memlet(block, source, tasklet, "_in", {}, index, debug_info);
    builder.add_computational_memlet(block, tasklet, "_out", target, subset, target_type, debug_info);
}

std::string ReduceNode::add_input_comparison(
    passes::LibNodeExpander::AccessNodeExpand& expansion,
    structured_control_flow::Block& block,
    data_flow::TaskletCode code,
    const data_flow::Subset& left,
    const data_flow::Subset& right,
    const types::IType& input_type
) const {
    auto& builder = expansion.builder();
    auto& debug_info = this->debug_info();
    auto result = add_scalar(builder, "_cmp", types::PrimitiveType::Bool);
    auto& tasklet = builder.add_tasklet(block, code, "_out", {"_in1", "_in2"}, debug_info);
    builder.add_computational_memlet(
        block, expansion.add_indirect_read_access(block, X_INPUT_IDX), tasklet, "_in1", left, input_type, debug_info
    );
    builder.add_computational_memlet(
        block, expansion.add_indirect_read_access(block, X_INPUT_IDX), tasklet, "_in2", right, input_type, debug_info
    );
    builder.add_computational_memlet(
        block,
        tasklet,
        "_out",
        builder.add_access(block, result, debug_info),
        {},
        types::Scalar(types::PrimitiveType::Bool),
        debug_info
    );
    return result;
}

symbolic::SymbolSet ReduceNode::symbols() const {
    symbolic::SymbolSet syms;
    for (const auto& dim : shape_) {
        for (auto& atom : symbolic::atoms(dim)) {
            syms.insert(atom);
        }
    }
    return syms;
}

void ReduceNode::replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) {
    for (auto& dim : shape_) {
        dim = symbolic::subs(dim, old_expression, new_expression);
    }
}

void ReduceNode::replace(const symbolic::ExpressionMapping& replacements) {
    for (auto& dim : shape_) {
        dim = symbolic::subs(dim, replacements);
    }
}

data_flow::PointerAccessType ReduceNode::pointer_access_type(int input_idx) const {
    if (input_idx == 0) {
        return data_flow::PointerAccessMeta::create_full_write_only(symbolic::__nullptr__(), true);
    } else if (input_idx == 1) {
        return data_flow::PointerAccessMeta::create_read_only(symbolic::__nullptr__(), true);
    } else {
        return TensorNode::pointer_access_type(input_idx);
    }
}

std::ostream& operator<<(std::ostream& os, const std::vector<int64_t>& list) {
    os << "[";
    for (size_t i = 0; i < list.size(); ++i) {
        if (i > 0) {
            os << ", ";
        }
        os << list[i];
    }
    os << "]";
    return os;
}

std::string ReduceNode::toStr() const {
    std::stringstream ss;
    ss << this->code_.value();
    ss << "(shape=";
    TensorLayout::emit_symbolic_list(ss, shape_);
    ss << ", axes=" << axes_;
    ss << ", keep=" << this->keepdims_;
    ss << ")";
    return ss.str();
}

passes::LibNodeExpander::ExpandOutcome ReduceNode::expand_inner(
    passes::LibNodeExpander::AccessNodeExpand& expansion,
    structured_control_flow::Block& block,
    const data_flow::Memlet* iedge_input,
    const data_flow::Memlet* iedge_result,
    const std::vector<symbolic::Expression>& output_shape,
    const std::vector<int64_t>& sorted_axes
) {
    sdfg::types::Scalar element_type(iedge_result->base_type().primitive_type());
    types::Tensor scalar_tensor(element_type.primitive_type(), {});

    // Add new sequence
    auto& new_sequence = expansion.replace_with_sequence();
    auto& builder = expansion.builder();
    std::string accum_container;

    // 1. Initialization Loop
    {
        data_flow::Subset init_subset;
        structured_control_flow::Sequence* last_scope = &new_sequence;
        structured_control_flow::Map* last_map = nullptr;

        for (size_t i = 0; i < output_shape.size(); ++i) {
            std::string indvar_str = builder.find_new_name("_i_init");
            builder.add_container(indvar_str, types::Scalar(types::PrimitiveType::Int64));

            auto indvar = symbolic::symbol(indvar_str);
            auto init = symbolic::zero();
            auto update = symbolic::add(indvar, symbolic::one());
            auto condition = symbolic::Lt(indvar, output_shape[i]);

            last_map = &builder.add_map(
                *last_scope,
                indvar,
                condition,
                init,
                update,
                structured_control_flow::ScheduleType_Sequential::create(),
                block.debug_info()
            );
            last_scope = &last_map->root();
            init_subset.push_back(indvar);
        }

        // Add initialization tasklet
        auto& init_block = builder.add_block(*last_scope, {}, block.debug_info());
        auto& init_tasklet =
            builder.add_tasklet(init_block, data_flow::TaskletCode::assign, {"_out"}, {"_in"}, block.debug_info());

        auto& const_node =
            builder
                .add_constant(init_block, this->identity(element_type.primitive_type()), element_type, block.debug_info());
        auto& out_access = expansion.add_indirect_write_access(init_block, RESULT_PTR_IDX);

        builder
            .add_computational_memlet(init_block, const_node, init_tasklet, "_in", {}, scalar_tensor, block.debug_info());
        builder.add_computational_memlet(
            init_block, init_tasklet, "_out", out_access, init_subset, iedge_result->base_type(), block.debug_info()
        );
        accum_container = out_access.data();
    }

    // 2. Reduction Loop
    {
        data_flow::Subset input_subset;
        data_flow::Subset output_subset;

        // Outer parallel loops (Maps) over the dimensions that are not reduced
        auto [outer_scope, outer_index] = add_outer_maps(builder, new_sequence, sorted_axes);
        structured_control_flow::Sequence* last_scope = outer_scope;
        structured_control_flow::StructuredLoop* last_loop = nullptr;

        std::map<size_t, symbolic::Expression> loop_vars_map;
        for (size_t i = 0; i < outer_index.size(); ++i) {
            if (!outer_index[i].is_null()) {
                loop_vars_map[i] = outer_index[i];
            }
        }

        // Inner sequential loops (Fors / Reduces) over the reduced dimensions
        std::vector<size_t> inner_dims(sorted_axes.begin(), sorted_axes.end());
        inner_dims.erase(std::unique(inner_dims.begin(), inner_dims.end()), inner_dims.end());
        auto reduction_operation = this->reduction_operation();
        for (size_t dim_idx : inner_dims) {
            std::string indvar_str = builder.find_new_name("_k");
            auto& limit = shape_[dim_idx];
            builder.add_container(indvar_str, types::Scalar(types::get_primitive_type_to_hold_upper_bound(limit)));

            auto indvar = symbolic::symbol(indvar_str);
            auto init = symbolic::zero();
            auto update = symbolic::add(indvar, symbolic::one());
            auto condition = symbolic::Lt(indvar, limit);

            if (reduction_operation.has_value()) {
                last_loop = &builder.add_reduce(
                    *last_scope,
                    indvar,
                    condition,
                    init,
                    update,
                    {{.operation = reduction_operation.value(), .container = accum_container}},
                    structured_control_flow::ScheduleType_Sequential::create(),
                    block.debug_info()
                );
            } else {
                last_loop = &builder.add_for(*last_scope, indvar, condition, init, update, block.debug_info());
            }
            last_scope = &last_loop->root();
            loop_vars_map[dim_idx] = indvar;
        }

        // Construct output indices
        std::vector<symbolic::Expression> input_indices;
        std::vector<symbolic::Expression> output_indices;
        for (size_t i = 0; i < shape_.size(); ++i) {
            input_indices.push_back(loop_vars_map[i]);
            bool is_axis = false;
            for (auto axis : sorted_axes) {
                if (axis == (int64_t) i) {
                    is_axis = true;
                    break;
                }
            }

            if (is_axis) {
                if (keepdims_) {
                    output_indices.push_back(symbolic::zero());
                }
            } else {
                output_indices.push_back(loop_vars_map[i]);
            }
        }

        this->expand_reduction(
            expansion,
            builder,
            *last_scope,
            static_cast<const types::Tensor&>(iedge_input->base_type()),
            static_cast<const types::Tensor&>(iedge_result->base_type()),
            input_indices,
            output_indices
        );
    }

    return expansion.successfully_expanded();
}

passes::LibNodeExpander::ExpandOutcome ReduceNode::expand(passes::LibNodeExpander::ExpandContext& context, Block& block) {
    auto& dataflow = this->get_parent();

    if (dataflow.in_degree(*this) != 2) {
        return context.unable();
    }

    auto* iedge_input = dataflow.in_edge_for_connector(*this, inputs_.at(1));
    auto* iedge_result = dataflow.in_edge_for_connector(*this, inputs_.at(0));

    auto sorted_axes = normalized_axes();
    auto output_shape = this->output_shape();

    auto expansion = context.replacement_requires_access_nodes(
        {passes::LibNodeExpander::InputUse::IndirectReadWrite, passes::LibNodeExpander::InputUse::IndirectRead}
    );

    if (!expansion) {
        return context.unable();
    }

    return expand_inner(*expansion.get(), block, iedge_input, iedge_result, output_shape, sorted_axes);
}

} // namespace tensor
} // namespace math
} // namespace sdfg
