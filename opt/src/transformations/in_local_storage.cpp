#include "sdfg/transformations/in_local_storage.h"

#include <cassert>
#include <cstddef>
#include <string>

#include "sdfg/analysis/assumptions_analysis.h"
#include "sdfg/analysis/loop_analysis.h"
#include "sdfg/analysis/scope_analysis.h"
#include "sdfg/analysis/type_analysis.h"
#include "sdfg/analysis/users.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/memlet.h"
#include "sdfg/passes/structured_control_flow/dead_cfg_elimination.h"
#include "sdfg/passes/structured_control_flow/sequence_fusion.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/transformations/utils.h"
#include "sdfg/types/array.h"
#include "sdfg/types/scalar.h"

namespace sdfg {
namespace transformations {

InLocalStorage::InLocalStorage(structured_control_flow::StructuredLoop& loop, std::string container)
    : loop_(loop), container_(std::move(container)) {}

std::string InLocalStorage::name() const { return "InLocalStorage"; }

bool InLocalStorage::can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    auto& sdfg = builder.subject();
    auto& body = this->loop_.root();

    // Criterion: Container must exist in the SDFG
    if (!sdfg.exists(this->container_)) {
        return false;
    }

    // Criterion: Container must be an array type
    auto& container_type = sdfg.type(this->container_);
    if (container_type.type_id() != types::TypeID::Pointer && container_type.type_id() != types::TypeID::Array) {
        return false;
    }

    // Criterion: Check if container is used in the loop
    auto& users = analysis_manager.get<analysis::Users>();
    analysis::UsersView body_users(users, body);
    if (body_users.uses(this->container_).empty()) {
        return false;
    }

    // Criterion: Container must be READ-ONLY within the loop (no writes)
    // This is what distinguishes InLocalStorage from OutLocalStorage
    if (!body_users.writes(this->container_).empty()) {
        return false;
    }

    // Criterion: All accesses must have the same dimensionality
    auto accesses = body_users.uses(this->container_);
    auto first_access = accesses.at(0);
    auto first_subsets = first_access->subsets();
    if (first_subsets.empty()) {
        return false;
    }
    auto& first_subset = first_subsets.at(0);
    if (first_subset.size() == 0) {
        return false;
    }

    for (auto* access : accesses) {
        for (auto& subset : access->subsets()) {
            if (subset.size() != first_subset.size()) {
                return false;
            }
        }
    }

    // Store representative subset for later use
    access_info_.representative_subset = first_subset;

    // Use LoopAnalysis to collect all nested loops within the target loop
    auto& loop_analysis = analysis_manager.get<analysis::LoopAnalysis>();

    // Collect all nested loops (descendants of this loop, plus this loop itself)
    std::vector<structured_control_flow::StructuredLoop*> nested_loops;
    nested_loops.push_back(&loop_);

    auto descendants = loop_analysis.descendants(&loop_);
    for (auto* desc : descendants) {
        if (auto* nested_loop = dynamic_cast<structured_control_flow::StructuredLoop*>(desc)) {
            nested_loops.push_back(nested_loop);
        }
    }

    // Analyze index expressions per dimension
    // For each dimension, check which nested loop indvars it depends on
    // and compute buffer size from their iteration counts
    access_info_.dimensions.clear();
    access_info_.bases.clear();

    // Track which loops contribute to buffer dimensions
    bool target_loop_contributes = false;
    bool descendant_loops_contribute = false;

    bool found_varying_dim = false;
    for (size_t d = 0; d < first_subset.size(); d++) {
        auto& index_expr = first_subset.at(d);
        auto atoms = symbolic::atoms(index_expr);

        // Check if any atom is an induction variable of a nested loop
        symbolic::Expression dim_size = symbolic::integer(1);
        symbolic::Expression dim_base = index_expr;
        bool has_loop_indvar = false;

        for (auto* nested_loop : nested_loops) {
            auto indvar = nested_loop->indvar();

            // Check if this indvar appears in the index expression
            bool found = false;
            for (auto& atom : atoms) {
                if (symbolic::eq(atom, indvar)) {
                    found = true;
                    break;
                }
            }

            if (found) {
                // Get iteration count for this loop
                auto iter_count = nested_loop->num_iterations();
                if (iter_count.is_null()) {
                    return false; // Need known iteration count
                }

                dim_size = symbolic::mul(dim_size, iter_count);

                // Base: substitute indvar with its initial value
                dim_base = symbolic::subs(dim_base, indvar, nested_loop->init());
                has_loop_indvar = true;

                // Track whether target or descendants contribute
                if (nested_loop == &loop_) {
                    target_loop_contributes = true;
                } else {
                    descendant_loops_contribute = true;
                }
            }
        }

        if (has_loop_indvar) {
            access_info_.dimensions.push_back(dim_size);
            access_info_.bases.push_back(dim_base);
            found_varying_dim = true;
        } else {
            // Constant dimension - size 1, base is the expression itself
            access_info_.dimensions.push_back(symbolic::integer(1));
            access_info_.bases.push_back(index_expr);
        }
    }

    // We need at least one varying dimension to make localization useful
    if (!found_varying_dim) {
        return false;
    }

    // Determine copy placement:
    // - copy_inside_loop = true: only descendants contribute (tiled case, per-iteration)
    // - copy_inside_loop = false: target loop contributes (simple case, copy once before)
    //   This includes the case where both target and descendants contribute.
    access_info_.copy_inside_loop = descendant_loops_contribute && !target_loop_contributes;

    return true;
}

void InLocalStorage::apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    auto& sdfg = builder.subject();
    auto& users = analysis_manager.get<analysis::Users>();
    auto& scope_analysis = analysis_manager.get<analysis::ScopeAnalysis>();

    auto parent_node = scope_analysis.parent_scope(&loop_);
    auto parent = dynamic_cast<structured_control_flow::Sequence*>(parent_node);
    if (!parent) {
        throw InvalidSDFGException("InLocalStorage: Parent of loop must be a Sequence!");
    }

    // Get type information for the original container
    auto& type = sdfg.type(this->container_);
    types::Scalar scalar_type(type.primitive_type());

    // Create local buffer name
    local_name_ = "__daisy_in_local_storage_" + this->container_;

    // Collect varying dimensions and compute strides for linearization
    std::vector<size_t> varying_dims;
    std::vector<symbolic::Expression> dim_sizes;
    for (size_t d = 0; d < access_info_.dimensions.size(); d++) {
        auto dim_size = access_info_.dimensions.at(d);
        if (!symbolic::eq(dim_size, symbolic::integer(1))) {
            varying_dims.push_back(d);
            dim_sizes.push_back(dim_size);
        }
    }

    // Compute strides: stride[i] = product of dim_sizes[i+1..]
    std::vector<symbolic::Expression> strides(varying_dims.size());
    symbolic::Expression total_size = symbolic::integer(1);
    for (int i = varying_dims.size() - 1; i >= 0; i--) {
        strides[i] = total_size;
        total_size = symbolic::mul(total_size, dim_sizes[i]);
    }

    // Create the local buffer array
    types::Array buffer_type(scalar_type, total_size);
    builder.add_container(local_name_, buffer_type);

    // Get access information from loop body
    analysis::UsersView body_users(users, loop_.root());
    auto accesses = body_users.uses(this->container_);
    auto first_access = accesses.at(0);
    auto& first_subset = access_info_.representative_subset;

    // Determine where to insert copy loops based on analysis
    // - copy_inside_loop == true: insert inside loop_.root() (per-iteration, for tiled case)
    // - copy_inside_loop == false: insert before loop_ in parent (once, for simple case)
    structured_control_flow::Sequence* copy_scope;
    structured_control_flow::ControlFlowNode* insert_before_node;

    if (access_info_.copy_inside_loop) {
        // Tiled case: insert inside loop, before its children
        copy_scope = &loop_.root();
        insert_before_node = (loop_.root().size() > 0) ? &loop_.root().at(0).first : nullptr;
    } else {
        // Simple case: insert before loop in its parent
        copy_scope = parent;
        insert_before_node = &loop_;
    }

    std::vector<symbolic::Symbol> copy_indvars;
    data_flow::Subset local_subset; // Indices for the local buffer

    bool first_copy_loop = true;
    for (size_t d = 0; d < access_info_.dimensions.size(); d++) {
        auto dim_size = access_info_.dimensions.at(d);

        // Skip dimensions with size 1 (constant)
        if (symbolic::eq(dim_size, symbolic::integer(1))) {
            local_subset.push_back(symbolic::integer(0));
            continue;
        }

        // Create loop index variable for this dimension
        auto indvar_name = "__daisy_ils_" + this->container_ + "_d" + std::to_string(d);
        types::Scalar indvar_type(types::PrimitiveType::UInt64);
        builder.add_container(indvar_name, indvar_type);
        auto indvar = symbolic::symbol(indvar_name);
        copy_indvars.push_back(indvar);

        // Loop: for indvar = 0; indvar < dim_size; indvar++
        auto init = symbolic::integer(0);
        auto condition = symbolic::Lt(indvar, dim_size);
        auto update = symbolic::add(indvar, symbolic::integer(1));

        if (first_copy_loop && insert_before_node) {
            // First copy loop: insert before the existing node
            auto& copy_loop = builder.add_for_before(
                *copy_scope, *insert_before_node, indvar, condition, init, update, {}, loop_.debug_info()
            );
            copy_scope = &copy_loop.root();
            first_copy_loop = false;
        } else if (first_copy_loop) {
            // No existing node to insert before - just add the loop
            auto& copy_loop = builder.add_for(*copy_scope, indvar, condition, init, update, {}, loop_.debug_info());
            copy_scope = &copy_loop.root();
            first_copy_loop = false;
        } else {
            // Nested copy loops: add inside the current copy scope
            auto& copy_loop = builder.add_for(*copy_scope, indvar, condition, init, update, {}, loop_.debug_info());
            copy_scope = &copy_loop.root();
        }

        // Index for local buffer is the loop indvar
        local_subset.push_back(indvar);
    }

    // Create the copy block in the innermost loop
    auto& copy_block = builder.add_block(*copy_scope);

    // Build the source subset - substitute index expressions with (base + local_indvar)
    data_flow::Subset src_subset;
    size_t indvar_idx = 0;
    for (size_t d = 0; d < first_subset.size(); d++) {
        auto dim_size = access_info_.dimensions.at(d);
        if (symbolic::eq(dim_size, symbolic::integer(1))) {
            // Constant dimension - use original expression
            src_subset.push_back(first_subset.at(d));
        } else {
            // Varying dimension - base + local_indvar
            auto base = access_info_.bases.at(d);
            auto local_indvar = copy_indvars.at(indvar_idx++);
            src_subset.push_back(symbolic::add(base, local_indvar));
        }
    }

    // Read from original container
    auto& copy_access_src = builder.add_access(copy_block, this->container_);
    // Write to local buffer
    auto& copy_access_dst = builder.add_access(copy_block, local_name_);
    // Tasklet: _out = _in
    auto& copy_tasklet = builder.add_tasklet(copy_block, data_flow::TaskletCode::assign, "_out", {"_in"});

    // Input memlet: read from original array
    builder.add_computational_memlet(copy_block, copy_access_src, copy_tasklet, "_in", src_subset, type);

    // Output memlet: write to local buffer
    builder.add_computational_memlet(copy_block, copy_tasklet, "_out", copy_access_dst, local_subset, buffer_type);

    // Now update all accesses in the main loop body to use the local buffer
    // Transform the access indices from original form to local indices
    for (auto* user : body_users.uses(this->container_)) {
        auto element = user->element();
        if (auto memlet = dynamic_cast<data_flow::Memlet*>(element)) {
            auto& old_subset = memlet->subset();

            // Build new subset: subtract base from each dimension
            data_flow::Subset new_subset;
            for (size_t d = 0; d < old_subset.size(); d++) {
                auto base = access_info_.bases.at(d);
                // new_index = old_index - base
                new_subset.push_back(symbolic::sub(old_subset.at(d), base));
            }
            memlet->set_subset(new_subset);
            memlet->set_base_type(buffer_type);
        }
    }

    // Replace container name in the loop body
    loop_.replace(symbolic::symbol(this->container_), symbolic::symbol(local_name_));

    // Cleanup
    analysis_manager.invalidate_all();

    passes::SequenceFusion sf_pass;
    passes::DeadCFGElimination dce_pass;
    bool applies = false;
    do {
        applies = false;
        applies |= dce_pass.run(builder, analysis_manager);
        applies |= sf_pass.run(builder, analysis_manager);
    } while (applies);
}

void InLocalStorage::to_json(nlohmann::json& j) const {
    std::string loop_type;
    if (dynamic_cast<structured_control_flow::For*>(&loop_)) {
        loop_type = "for";
    } else if (dynamic_cast<structured_control_flow::Map*>(&loop_)) {
        loop_type = "map";
    } else {
        throw std::runtime_error("Unsupported loop type for serialization of loop: " + loop_.indvar()->get_name());
    }
    j["subgraph"] = {{"0", {{"element_id", this->loop_.element_id()}, {"type", loop_type}}}};
    j["transformation_type"] = this->name();
    j["container"] = container_;
}

InLocalStorage InLocalStorage::from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& desc) {
    auto loop_id = desc["subgraph"]["0"]["element_id"].get<size_t>();
    std::string container = desc["container"].get<std::string>();
    auto element = builder.find_element_by_id(loop_id);
    if (!element) {
        throw InvalidTransformationDescriptionException("Element with ID " + std::to_string(loop_id) + " not found.");
    }
    auto loop = dynamic_cast<structured_control_flow::StructuredLoop*>(element);
    if (!loop) {
        throw InvalidTransformationDescriptionException(
            "Element with ID " + std::to_string(loop_id) + " is not a structured loop."
        );
    }

    return InLocalStorage(*loop, container);
}

} // namespace transformations
} // namespace sdfg
