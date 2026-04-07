#include "sdfg/transformations/loop_rotate.h"

#include <stdexcept>

#include "sdfg/analysis/scope_analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/scalar.h"

/**
 * Loop Rotate Transformation Implementation
 *
 * This transformation converts a loop with negative stride to positive stride.
 *
 * Algorithm:
 * 1. Extract the lower bound from the loop condition (canonical_bound_lower)
 * 2. Compute new loop parameters:
 *    - new_init = lower_bound + 1 (start just above the lower bound)
 *    - new_condition = indvar < old_init + 1 (go up to but not including old_init + 1)
 *    - new_update = indvar + 1 (positive unit stride)
 * 3. Create a container for the rotated index value
 * 4. Add assignment: __i_orig__ = old_init + new_init - indvar
 * 5. Replace all uses of indvar in body with __i_orig__
 *
 * Mathematical derivation:
 * Original: i = init, init-1, ..., bound+1 (stride = -1, condition: bound < i)
 * After: i' = bound+1, bound+2, ..., init (stride = +1, condition: i' < init+1)
 * Mapping: original_i = init + (bound+1) - i' = init + new_init - i'
 */

namespace sdfg {
namespace transformations {

LoopRotate::LoopRotate(structured_control_flow::StructuredLoop& loop) : loop_(loop) {}

std::string LoopRotate::name() const { return "LoopRotate"; }

bool LoopRotate::can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    // Check for negative unit stride
    auto stride = loop_.stride();
    if (stride.is_null()) {
        return false;
    }
    if (stride->as_int() != -1) {
        // Only support stride == -1 for now
        return false;
    }

    // Check that we can extract the lower bound
    auto lower_bound = loop_.canonical_bound_lower();
    if (lower_bound.is_null()) {
        return false;
    }

    return true;
}

void LoopRotate::apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    auto indvar = loop_.indvar();
    auto old_init = loop_.init();

    // Get lower bound (exclusive): condition is "bound < indvar"
    // So the last valid value is bound + 1
    auto lower_bound = loop_.canonical_bound_lower();

    // New init: start at lower_bound + 1
    auto new_init = symbolic::add(lower_bound, symbolic::one());
    new_init = symbolic::expand(new_init);
    new_init = symbolic::simplify(new_init);

    // New condition: indvar < old_init + 1
    auto new_rhs = symbolic::add(old_init, symbolic::one());
    new_rhs = symbolic::expand(new_rhs);
    new_rhs = symbolic::simplify(new_rhs);
    auto new_condition = symbolic::Lt(indvar, new_rhs);

    // New update: indvar + 1 (positive unit stride)
    auto new_update = symbolic::add(indvar, symbolic::one());

    // Create a new container for the rotated (original) value
    rotated_container_name_ = "__" + indvar->get_name() + "_orig__";

    // Find a unique name if it already exists
    int suffix = 0;
    while (builder.subject().exists(rotated_container_name_)) {
        rotated_container_name_ = "__" + indvar->get_name() + "_orig_" + std::to_string(suffix++) + "__";
    }

    // Add the container with the same type as the induction variable
    auto& indvar_type = builder.subject().type(indvar->get_name());
    builder.add_container(rotated_container_name_, indvar_type);

    auto rotated_var = symbolic::symbol(rotated_container_name_);

    // Compute the rotated value: original_i = old_init + new_init - indvar
    // This maps i' = new_init to original_i = old_init
    //           i' = new_init+1 to original_i = old_init - 1
    //           etc.
    auto rotated_value = symbolic::sub(symbolic::add(old_init, new_init), indvar);
    rotated_value = symbolic::expand(rotated_value);
    rotated_value = symbolic::simplify(rotated_value);

    // Update the loop parameters
    builder.update_loop(loop_, indvar, new_condition, new_init, new_update);

    // Replace all uses of indvar in loop body with the rotated variable
    loop_.root().replace(indvar, rotated_var);

    // Add an empty block before the first child to set the rotated variable in the transition
    if (loop_.root().size() > 0) {
        auto& first_child = loop_.root().at(0).first;
        builder.add_block_before(loop_.root(), first_child, control_flow::Assignments{{rotated_var, rotated_value}});
    } else {
        builder.add_block(loop_.root(), control_flow::Assignments{{rotated_var, rotated_value}});
    }

    // Reconstruct original indvar value after loop exit
    // After loop, indvar holds transformed final value; we restore: indvar = old_init + new_init - indvar
    auto& scope_analysis = analysis_manager.get<analysis::ScopeAnalysis>();
    auto parent_node = scope_analysis.parent_scope(&loop_);
    auto* parent = dynamic_cast<structured_control_flow::Sequence*>(parent_node);
    if (parent) {
        builder.add_block_after(*parent, loop_, {{indvar, rotated_value}}, loop_.debug_info());
    }
}

void LoopRotate::to_json(nlohmann::json& j) const {
    std::string loop_type = "for";
    if (dynamic_cast<const structured_control_flow::Map*>(&loop_)) {
        loop_type = "map";
    }

    j["transformation_type"] = this->name();
    j["subgraph"] = {{"0", {{"element_id", loop_.element_id()}, {"type", loop_type}}}};
    j["parameters"] = nlohmann::json::object();
}

LoopRotate LoopRotate::from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& desc) {
    auto loop_id = desc["subgraph"]["0"]["element_id"].get<size_t>();

    auto element = builder.find_element_by_id(loop_id);
    if (element == nullptr) {
        throw std::runtime_error("Element with ID " + std::to_string(loop_id) + " not found.");
    }

    auto loop = dynamic_cast<structured_control_flow::StructuredLoop*>(element);
    if (loop == nullptr) {
        throw std::runtime_error("Element with ID " + std::to_string(loop_id) + " is not a StructuredLoop.");
    }

    return LoopRotate(*loop);
}

} // namespace transformations
} // namespace sdfg
