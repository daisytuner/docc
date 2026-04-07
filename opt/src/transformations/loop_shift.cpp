#include "sdfg/transformations/loop_shift.h"

#include <stdexcept>

#include "sdfg/analysis/loop_analysis.h"
#include "sdfg/analysis/scope_analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/scalar.h"

/**
 * Loop Shift Transformation Implementation
 *
 * This transformation shifts a loop's iteration space to start from a different
 * initial value.
 *
 * Algorithm:
 * 1. Compute the shift amount: shift = old_init - target_init
 * 2. Create a new scalar container to hold the original iteration value
 * 3. Add an empty block at the beginning of the loop body
 * 4. Add assignment in the block's transition: shifted_var = indvar + shift
 * 5. Update loop bounds: init = target_init, condition adjusted
 * 6. User should run SymbolPropagation afterwards to propagate the assignment
 */

namespace sdfg {
namespace transformations {

LoopShift::LoopShift(structured_control_flow::StructuredLoop& loop) : loop_(loop), offset_(loop.init()) {}

LoopShift::LoopShift(structured_control_flow::StructuredLoop& loop, const symbolic::Expression& offset)
    : loop_(loop), offset_(offset) {}

std::string LoopShift::name() const { return "LoopShift"; }

bool LoopShift::can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    if (symbolic::eq(offset_, symbolic::zero())) {
        // No shift needed
        return false;
    }

    for (auto& atom : symbolic::atoms(offset_)) {
        if (symbolic::eq(atom, loop_.indvar())) {
            // Offset cannot contain the induction variable itself
            return false;
        }
    }
    return true;
}

void LoopShift::apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    auto indvar = loop_.indvar();
    auto old_init = loop_.init();
    auto old_condition = loop_.condition();
    auto old_update = loop_.update();

    // Compute the new init: new_init = old_init - offset
    auto new_init = symbolic::sub(old_init, offset_);
    new_init = symbolic::expand(new_init);
    new_init = symbolic::simplify(new_init);

    // Create a new container for the shifted (original) value
    // Use a unique name based on the original indvar
    shifted_container_name_ = "__" + indvar->get_name() + "_orig__";

    // Find a unique name if it already exists
    int suffix = 0;
    while (builder.subject().exists(shifted_container_name_)) {
        shifted_container_name_ = "__" + indvar->get_name() + "_orig_" + std::to_string(suffix++) + "__";
    }

    // Add the container with the same type as the induction variable
    auto& indvar_type = builder.subject().type(indvar->get_name());
    builder.add_container(shifted_container_name_, indvar_type);

    auto shifted_var = symbolic::symbol(shifted_container_name_);
    auto shifted_value = symbolic::add(indvar, offset_);
    shifted_value = symbolic::expand(shifted_value);
    shifted_value = symbolic::simplify(shifted_value);

    // We use symbolic substitution: replace old indvar with (indvar + shift) in condition
    auto new_condition = symbolic::subs(old_condition, indvar, shifted_value);

    // Update the loop
    builder.update_loop(loop_, indvar, new_condition, new_init, old_update);

    // Update the body to reference the original value via the new container
    loop_.root().replace(indvar, shifted_var);

    // Add an empty block before the first child to set the shifted variable in the transition
    if (loop_.root().size() > 0) {
        auto& first_child = loop_.root().at(0).first;
        builder.add_block_before(loop_.root(), first_child, control_flow::Assignments{{shifted_var, shifted_value}});
    } else {
        builder.add_block(loop_.root(), control_flow::Assignments{{shifted_var, shifted_value}});
    }

    // Reconstruct original indvar value after loop exit
    // After loop, indvar holds transformed final value; we restore: indvar = indvar + offset
    auto& scope_analysis = analysis_manager.get<analysis::ScopeAnalysis>();
    auto parent_node = scope_analysis.parent_scope(&loop_);
    auto* parent = dynamic_cast<structured_control_flow::Sequence*>(parent_node);
    if (parent) {
        builder.add_block_after(*parent, loop_, {{indvar, shifted_value}}, loop_.debug_info());
    }
}

void LoopShift::to_json(nlohmann::json& j) const {
    std::string loop_type = "for";
    if (dynamic_cast<const structured_control_flow::Map*>(&loop_)) {
        loop_type = "map";
    }

    j["transformation_type"] = this->name();
    j["subgraph"] = {{"0", {{"element_id", loop_.element_id()}, {"type", loop_type}}}};
    j["parameters"] = {{"offset", offset_->__str__()}};
}

LoopShift LoopShift::from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& desc) {
    auto loop_id = desc["subgraph"]["0"]["element_id"].get<size_t>();
    auto offset_str = desc["parameters"]["offset"].get<std::string>();

    auto element = builder.find_element_by_id(loop_id);
    if (element == nullptr) {
        throw std::runtime_error("Element with ID " + std::to_string(loop_id) + " not found.");
    }

    auto loop = dynamic_cast<structured_control_flow::StructuredLoop*>(element);
    if (loop == nullptr) {
        throw std::runtime_error("Element with ID " + std::to_string(loop_id) + " is not a StructuredLoop.");
    }

    auto offset = symbolic::parse(offset_str);
    return LoopShift(*loop, offset);
}

} // namespace transformations
} // namespace sdfg
