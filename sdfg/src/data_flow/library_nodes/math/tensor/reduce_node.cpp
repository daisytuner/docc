#include "sdfg/data_flow/library_nodes/math/tensor/reduce_node.h"

#include "sdfg/analysis/analysis.h"
#include "sdfg/analysis/scope_analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"

#include <algorithm>

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
      shape_(shape), axes_(axes), keepdims_(keepdims) {}

void ReduceNode::validate(const Function& function) const {
    TensorNode::validate(function);

    auto& graph = this->get_parent();

    auto* iedge = graph.in_edge_for_connector(*this, inputs_.at(1));
    auto& tensor_input = static_cast<const types::Tensor&>(iedge->base_type());
    validate_shape_matches(shape_, tensor_input.layout(), "input");

    // Calculate expected output shape based on axes and keepdims
    std::vector<int64_t> sorted_axes = axes_;
    // Normalize negative axes
    for (auto& axis : sorted_axes) {
        if (axis < 0) {
            axis = static_cast<int64_t>(shape_.size()) + axis;
        }
        // Validate axis is in bounds
        if (axis < 0 || axis >= static_cast<int64_t>(shape_.size())) {
            throw InvalidSDFGException(
                "Library Node: Axis value out of bounds. Axis: " + std::to_string(axis) +
                " Shape size: " + std::to_string(shape_.size())
            );
        }
    }
    std::sort(sorted_axes.begin(), sorted_axes.end());

    std::vector<symbolic::Expression> expected_output_shape;
    for (size_t i = 0; i < shape_.size(); ++i) {
        bool is_axis = false;
        for (auto axis : sorted_axes) {
            if (axis == (int64_t) i) {
                is_axis = true;
                break;
            }
        }

        if (is_axis) {
            if (keepdims_) {
                expected_output_shape.push_back(symbolic::one());
            }
        } else {
            expected_output_shape.push_back(shape_[i]);
        }
    }

    auto* iedge_result = graph.in_edge_for_connector(*this, inputs_.at(0));
    auto& tensor_output = static_cast<const types::Tensor&>(iedge_result->base_type());
    validate_shape_matches(expected_output_shape, tensor_output.layout(), "output");
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

data_flow::PointerAccessType ReduceNode::pointer_access_type(int input_idx) const {
    if (input_idx == 0) {
        return data_flow::PointerAccessMeta::create_full_write_only(symbolic::__nullptr__(), true);
    } else if (input_idx == 1) {
        return data_flow::PointerAccessMeta::create_read_only(symbolic::__nullptr__(), true);
    } else {
        return TensorNode::pointer_access_type(input_idx);
    }
}

bool ReduceNode::expand_inner(
    builder::StructuredSDFGBuilder& builder,
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::Block& block,
    data_flow::DataFlowGraph& dataflow,
    structured_control_flow::Sequence& parent,
    Transition& transition,
    const data_flow::Memlet* iedge_input,
    const data_flow::Memlet* iedge_result,
    const data_flow::AccessNode* input_node,
    const data_flow::AccessNode* output_node,
    const std::vector<symbolic::Expression>& output_shape,
    const std::vector<int64_t>& sorted_axes
) {
    auto org_idx = parent.index(block);

    sdfg::types::Scalar element_type(iedge_result->base_type().primitive_type());
    types::Tensor scalar_tensor(element_type.primitive_type(), {});

    // Add new sequence
    auto& new_sequence = builder.add_sequence_before(parent, block, transition.assignments(), block.debug_info());

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
                {},
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
        auto& out_access = builder.add_access(init_block, output_node->data(), block.debug_info());

        builder
            .add_computational_memlet(init_block, const_node, init_tasklet, "_in", {}, scalar_tensor, block.debug_info());
        builder.add_computational_memlet(
            init_block, init_tasklet, "_out", out_access, init_subset, iedge_result->base_type(), block.debug_info()
        );
    }

    // 2. Reduction Loop
    {
        data_flow::Subset input_subset;
        data_flow::Subset output_subset;

        structured_control_flow::Sequence* last_scope = &new_sequence;
        structured_control_flow::StructuredLoop* last_loop = nullptr;

        std::map<size_t, symbolic::Expression> loop_vars_map;
        std::vector<size_t> outer_dims;
        std::vector<size_t> inner_dims;

        // Partition dimensions
        for (size_t i = 0; i < shape_.size(); ++i) {
            bool is_axis = false;
            for (auto axis : sorted_axes) {
                if (axis == (int64_t) i) {
                    is_axis = true;
                    break;
                }
            }
            if (is_axis) {
                inner_dims.push_back(i);
            } else {
                outer_dims.push_back(i);
            }
        }

        // Generate outer parallel loops (Maps)
        for (size_t dim_idx : outer_dims) {
            std::string indvar_str = builder.find_new_name("_i");
            builder.add_container(indvar_str, types::Scalar(types::PrimitiveType::Int64));

            auto indvar = symbolic::symbol(indvar_str);
            auto init = symbolic::zero();
            auto update = symbolic::add(indvar, symbolic::one());
            auto condition = symbolic::Lt(indvar, shape_[dim_idx]);

            auto& map = builder.add_map(
                *last_scope,
                indvar,
                condition,
                init,
                update,
                structured_control_flow::ScheduleType_Sequential::create(),
                {},
                block.debug_info()
            );
            last_scope = &map.root();
            loop_vars_map[dim_idx] = indvar;
        }

        // Generate inner sequential loops (Fors)
        for (size_t dim_idx : inner_dims) {
            std::string indvar_str = builder.find_new_name("_k");
            builder.add_container(indvar_str, types::Scalar(types::PrimitiveType::Int64));

            auto indvar = symbolic::symbol(indvar_str);
            auto init = symbolic::zero();
            auto update = symbolic::add(indvar, symbolic::one());
            auto condition = symbolic::Lt(indvar, shape_[dim_idx]);

            last_loop = &builder.add_for(*last_scope, indvar, condition, init, update, {}, block.debug_info());
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
            builder,
            analysis_manager,
            *last_scope,
            input_node->data(),
            output_node->data(),
            static_cast<const types::Tensor&>(iedge_input->base_type()),
            static_cast<const types::Tensor&>(iedge_result->base_type()),
            input_indices,
            output_indices
        );
    }

    // Clean up block
    builder.clear_code_node_legacy(block, *this);
    // WARNING: this has been deallocated at this point!!
    builder.remove_child(parent, org_idx + 1);

    return true;
}

bool ReduceNode::expand(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    auto& dataflow = this->get_parent();
    auto& block = static_cast<structured_control_flow::Block&>(*dataflow.get_parent());

    if (dataflow.in_degree(*this) != 2) {
        return false;
    }

    auto& scope_analysis = analysis_manager.get<analysis::ScopeAnalysis>();
    auto& parent = static_cast<structured_control_flow::Sequence&>(*scope_analysis.parent_scope(&block));
    int index = parent.index(block);
    auto& transition = parent.at(index).second;

    auto* iedge_input = dataflow.in_edge_for_connector(*this, inputs_.at(1));
    auto* iedge_result = dataflow.in_edge_for_connector(*this, inputs_.at(0));


    auto* input_node = dataflow.find_standalone_entry(iedge_input);
    auto* output_node = dataflow.find_standalone_entry(iedge_result);

    if (!input_node || !output_node) {
        return false;
    }

    // Calculate output shape
    std::vector<symbolic::Expression> output_shape;
    std::vector<int64_t> sorted_axes = axes_;
    // Normalize negative axes
    for (auto& axis : sorted_axes) {
        if (axis < 0) {
            axis = static_cast<int64_t>(shape_.size()) + axis;
        }
        // Validate axis is in bounds
        if (axis < 0 || axis >= static_cast<int64_t>(shape_.size())) {
            throw InvalidSDFGException(
                "Library Node: Axis value out of bounds. Axis: " + std::to_string(axis) +
                " Shape size: " + std::to_string(shape_.size())
            );
        }
    }
    std::sort(sorted_axes.begin(), sorted_axes.end());

    for (size_t i = 0; i < shape_.size(); ++i) {
        bool is_axis = false;
        for (auto axis : sorted_axes) {
            if (axis == (int64_t) i) {
                is_axis = true;
                break;
            }
        }

        if (is_axis) {
            if (keepdims_) {
                output_shape.push_back(symbolic::one());
            }
        } else {
            output_shape.push_back(shape_[i]);
        }
    }


    return expand_inner(
        builder,
        analysis_manager,
        block,
        dataflow,
        parent,
        transition,
        iedge_input,
        iedge_result,
        input_node,
        output_node,
        output_shape,
        sorted_axes
    );
}

} // namespace tensor
} // namespace math
} // namespace sdfg
