#include "sdfg/transformations/loop_skewing.h"
#include <stdexcept>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/symbolic/symbolic.h"

/**
 * Loop Skewing Transformation Implementation
 *
 * This simplified implementation of loop skewing modifies the inner loop's iteration
 * space to depend on the outer loop's iteration variable, creating a "skewed" pattern.
 *
 * The transformation:
 * - Works with any StructuredLoop (For or Map) as inner loop
 * - Adjusts inner loop init: init_j + skew_factor * (i - init_i)
 * - Adjusts inner loop condition using symbolic::subs
 * - Updates memory access patterns in loop body using root().replace()
 * - Uses builder.update_loop() to modify the loop in place
 *
 * Safety:
 * - Skewing is a pure re-indexing that preserves iteration order within the inner loop
 * - Inner loop bounds must not depend on outer loop variable
 * - Loops must be properly nested
 */

namespace sdfg {
namespace transformations {

LoopSkewing::LoopSkewing(
    structured_control_flow::StructuredLoop& outer_loop,
    structured_control_flow::StructuredLoop& inner_loop,
    int skew_factor
)
    : outer_loop_(outer_loop), inner_loop_(inner_loop), skew_factor_(skew_factor) {}

std::string LoopSkewing::name() const { return "LoopSkewing"; }

bool LoopSkewing::can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    // Criterion 0: Skew factor must be non-zero
    if (this->skew_factor_ == 0) {
        return false;
    }

    auto& outer_indvar = this->outer_loop_.indvar();

    // Criterion 2: Inner loop must not depend on outer loop indvar
    auto inner_loop_init = this->inner_loop_.init();
    auto inner_loop_condition = this->inner_loop_.condition();
    auto inner_loop_update = this->inner_loop_.update();

    if (symbolic::uses(inner_loop_init, outer_indvar->get_name()) ||
        symbolic::uses(inner_loop_condition, outer_indvar->get_name()) ||
        symbolic::uses(inner_loop_update, outer_indvar->get_name())) {
        return false;
    }

    // Criterion 3: Outer loop must have only the inner loop as a child
    if (outer_loop_.root().size() != 1) {
        return false;
    }
    if (outer_loop_.root().at(0).second.assignments().size() > 0) {
        return false;
    }
    if (&outer_loop_.root().at(0).first != &inner_loop_) {
        return false;
    }

    return true;
}

void LoopSkewing::apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    auto outer_indvar = this->outer_loop_.indvar();
    auto inner_indvar = this->inner_loop_.indvar();

    // Calculate the skewing offset: skew_factor * (outer_indvar - outer_init)
    auto skew_offset =
        symbolic::mul(symbolic::integer(this->skew_factor_), symbolic::sub(outer_indvar, this->outer_loop_.init()));

    // New inner loop init: inner_init + skew_offset
    auto new_inner_init = symbolic::add(this->inner_loop_.init(), skew_offset);

    // New inner loop condition:
    // Substitute j with (j - skew_offset) in the condition
    // This adjusts the upper bound: j < M becomes (j - offset) < M, i.e., j < M + offset
    auto new_inner_condition =
        symbolic::subs(this->inner_loop_.condition(), inner_indvar, symbolic::sub(inner_indvar, skew_offset));

    // Inner loop update remains the same
    auto new_inner_update = this->inner_loop_.update();

    // Update the inner loop in place using builder.update_loop
    builder.update_loop(this->inner_loop_, inner_indvar, new_inner_condition, new_inner_init, new_inner_update);

    // Adjust memory access patterns in the loop body
    // Replace j with (j - skew_offset) in all expressions
    this->inner_loop_.root().replace(inner_indvar, symbolic::sub(inner_indvar, skew_offset));

    analysis_manager.invalidate_all();
}

void LoopSkewing::to_json(nlohmann::json& j) const {
    // Determine loop types for serialization
    std::string outer_type = "for";
    std::string inner_type = "for";

    if (dynamic_cast<const structured_control_flow::Map*>(&this->outer_loop_)) {
        outer_type = "map";
    }
    if (dynamic_cast<const structured_control_flow::Map*>(&this->inner_loop_)) {
        inner_type = "map";
    }

    j["transformation_type"] = this->name();
    j["subgraph"] = {
        {"0", {{"element_id", this->outer_loop_.element_id()}, {"type", outer_type}}},
        {"1", {{"element_id", this->inner_loop_.element_id()}, {"type", inner_type}}}
    };
    j["parameters"] = {{"skew_factor", this->skew_factor_}};
}

LoopSkewing LoopSkewing::from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& desc) {
    auto outer_loop_id = desc["subgraph"]["0"]["element_id"].get<size_t>();
    auto inner_loop_id = desc["subgraph"]["1"]["element_id"].get<size_t>();
    int skew_factor = desc["parameters"]["skew_factor"].get<int>();

    auto outer_element = builder.find_element_by_id(outer_loop_id);
    auto inner_element = builder.find_element_by_id(inner_loop_id);

    if (outer_element == nullptr) {
        throw std::runtime_error("Element with ID " + std::to_string(outer_loop_id) + " not found.");
    }
    if (inner_element == nullptr) {
        throw std::runtime_error("Element with ID " + std::to_string(inner_loop_id) + " not found.");
    }

    auto outer_loop = dynamic_cast<structured_control_flow::StructuredLoop*>(outer_element);
    if (outer_loop == nullptr) {
        throw std::runtime_error("Element with ID " + std::to_string(outer_loop_id) + " is not a StructuredLoop.");
    }

    auto inner_loop = dynamic_cast<structured_control_flow::StructuredLoop*>(inner_element);
    if (inner_loop == nullptr) {
        throw std::runtime_error("Element with ID " + std::to_string(inner_loop_id) + " is not a StructuredLoop.");
    }

    return LoopSkewing(*outer_loop, *inner_loop, skew_factor);
}

} // namespace transformations
} // namespace sdfg
