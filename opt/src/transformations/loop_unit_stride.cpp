#include "sdfg/transformations/loop_unit_stride.h"

#include <stdexcept>

#include "sdfg/analysis/scope_analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/scalar.h"

/**
 * Loop Unit Stride Transformation Implementation
 *
 * This transformation converts a loop with non-unit stride to unit stride
 * using symbolic substitution on the condition. Direction is preserved.
 *
 * Precondition: init == 0 (apply LoopShift first if needed)
 *
 * Algorithm:
 * 1. Verify init == 0 and stride != 1 and stride != -1
 * 2. Compute strided expression: strided_expr = |stride| * i
 * 3. Substitute indvar with strided_expr in the condition
 * 4. Create new loop: for i = 0; cond(strided_expr); i++ or i--
 * 5. Add assignment: __i_orig__ = strided_expr
 * 6. Replace all uses of indvar with __i_orig__
 *
 * Example (positive stride s = 2):
 *   Original: for (i = 0; i < 8; i += 2)  // iterations: 0, 2, 4, 6
 *   After:    for (i = 0; 2*i < 8; i++)
 *             __i_orig__ = 2 * i
 *
 * Example (negative stride s = -2):
 *   Original: for (i = 0; -10 < i; i -= 2)  // iterations: 0, -2, -4, -6, -8
 *   After:    for (i = 0; -10 < 2*i; i--)   // Note: |stride| used, direction preserved
 *             __i_orig__ = 2 * i            // i' = 0, -1, -2, -3, -4 -> 0, -2, -4, -6, -8
 *
 */

namespace sdfg {
namespace transformations {

LoopUnitStride::LoopUnitStride(structured_control_flow::StructuredLoop& loop) : loop_(loop) {}

std::string LoopUnitStride::name() const { return "LoopUnitStride"; }

bool LoopUnitStride::can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    // Require init == 0 (apply LoopShift first)
    if (!symbolic::eq(loop_.init(), symbolic::zero())) {
        return false;
    }

    // Check for non-unit stride
    auto stride = loop_.stride();
    if (stride.is_null()) {
        return false;
    }

    int stride_val = stride->as_int();
    if (stride_val == 1 || stride_val == -1) {
        // Already unit stride, nothing to do
        // Use LoopRotate for stride = -1
        return false;
    }

    if (stride_val == 0) {
        // Zero stride is invalid
        return false;
    }

    return true;
}

void LoopUnitStride::apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    auto indvar = loop_.indvar();
    auto old_condition = loop_.condition();
    auto stride = loop_.stride();
    int stride_val = stride->as_int();
    bool positive = stride_val > 0;
    int stride_val_abs = std::abs(stride_val);

    // Compute the strided expression: stride * i
    // (init is guaranteed to be 0 by can_be_applied)
    auto strided_expr = symbolic::mul(symbolic::integer(stride_val_abs), indvar);

    // New loop parameters:
    // - init = 0
    // - condition = old_condition with indvar substituted by strided_expr
    // - update = i + 1 or i - 1
    auto new_init = symbolic::zero();
    auto new_condition = symbolic::subs(old_condition, indvar, strided_expr);
    auto new_update = positive ? symbolic::add(indvar, symbolic::one()) : symbolic::sub(indvar, symbolic::one());

    // Create a new container for the strided (original) value
    strided_container_name_ = "__" + indvar->get_name() + "_orig__";

    // Find a unique name if it already exists
    int suffix = 0;
    while (builder.subject().exists(strided_container_name_)) {
        strided_container_name_ = "__" + indvar->get_name() + "_orig_" + std::to_string(suffix++) + "__";
    }

    // Add the container with the same type as the induction variable
    auto& indvar_type = builder.subject().type(indvar->get_name());
    builder.add_container(strided_container_name_, indvar_type);

    auto strided_var = symbolic::symbol(strided_container_name_);

    // Update the loop parameters
    builder.update_loop(loop_, indvar, new_condition, new_init, new_update);

    // Replace all uses of indvar in loop body with the strided variable
    loop_.root().replace(indvar, strided_var);

    // Add an empty block before the first child to set the strided variable in the transition
    if (loop_.root().size() > 0) {
        auto& first_child = loop_.root().at(0).first;
        builder.add_block_before(loop_.root(), first_child, control_flow::Assignments{{strided_var, strided_expr}});
    } else {
        builder.add_block(loop_.root(), control_flow::Assignments{{strided_var, strided_expr}});
    }

    // Reconstruct original indvar value after loop exit
    // After loop, indvar holds transformed final value; we restore: indvar = |stride| * indvar
    auto& scope_analysis = analysis_manager.get<analysis::ScopeAnalysis>();
    auto parent_node = scope_analysis.parent_scope(&loop_);
    auto* parent = dynamic_cast<structured_control_flow::Sequence*>(parent_node);
    if (parent) {
        builder.add_block_after(*parent, loop_, {{indvar, strided_expr}}, loop_.debug_info());
    }
}

void LoopUnitStride::to_json(nlohmann::json& j) const {
    std::string loop_type = "for";
    if (dynamic_cast<const structured_control_flow::Map*>(&loop_)) {
        loop_type = "map";
    }

    j["transformation_type"] = this->name();
    j["subgraph"] = {{"0", {{"element_id", loop_.element_id()}, {"type", loop_type}}}};
}

LoopUnitStride LoopUnitStride::from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& j) {
    auto loop_id = j["subgraph"]["0"]["element_id"].get<size_t>();

    auto element = builder.find_element_by_id(loop_id);
    if (element == nullptr) {
        throw std::runtime_error("LoopUnitStride: Element with ID " + std::to_string(loop_id) + " not found.");
    }

    auto loop = dynamic_cast<structured_control_flow::StructuredLoop*>(element);
    if (loop == nullptr) {
        throw std::runtime_error("LoopUnitStride: Element with ID " + std::to_string(loop_id) + " is not a loop.");
    }

    return LoopUnitStride(*loop);
}

std::string LoopUnitStride::strided_container_name() const { return strided_container_name_; }

} // namespace transformations
} // namespace sdfg
