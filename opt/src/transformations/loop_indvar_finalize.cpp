#include "sdfg/transformations/loop_indvar_finalize.h"

#include <stdexcept>

#include "sdfg/analysis/scope_analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/symbolic/symbolic.h"

/**
 * Loop Indvar Finalize Transformation Implementation
 *
 * For a normalized loop, computes the closed-form final value of the original
 * induction variable and adds a block right after the loop with this assignment.
 *
 * closed_form = num_iterations
 *
 * SymbolPropagation will then propagate this value into any subsequent blocks
 * that reference the induction variable, breaking WAR dependencies.
 */

namespace sdfg {
namespace transformations {

LoopIndvarFinalize::LoopIndvarFinalize(structured_control_flow::StructuredLoop& loop) : loop_(loop) {}

std::string LoopIndvarFinalize::name() const { return "LoopIndvarFinalize"; }

bool LoopIndvarFinalize::
    can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    // Loop must be in normal form
    if (!loop_.is_loop_normal_form()) {
        return false;
    }

    return true;
}

void LoopIndvarFinalize::apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    auto indvar = loop_.indvar();

    // Use max(0, num_iterations) to handle the case where the loop doesn't execute
    // (when bound <= 0, the loop body is skipped and indvar stays at 0)
    auto closed_form = symbolic::simplify(symbolic::max(symbolic::zero(), loop_.num_iterations()));

    auto& scope_analysis = analysis_manager.get<analysis::ScopeAnalysis>();
    auto parent_node = scope_analysis.parent_scope(&loop_);
    auto* parent = dynamic_cast<structured_control_flow::Sequence*>(parent_node);

    // Add block with closed-form assignment right after the loop
    // SymbolPropagation will propagate into subsequent blocks
    builder.add_block_after(*parent, loop_, {{indvar, closed_form}}, loop_.debug_info());

    analysis_manager.invalidate_all();
}

void LoopIndvarFinalize::to_json(nlohmann::json& j) const {
    std::string loop_type = "for";
    if (dynamic_cast<const structured_control_flow::Map*>(&loop_)) {
        loop_type = "map";
    }

    j["transformation_type"] = this->name();
    j["subgraph"] = {{"0", {{"element_id", loop_.element_id()}, {"type", loop_type}}}};
    j["parameters"] = {};
}

LoopIndvarFinalize LoopIndvarFinalize::from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& j) {
    auto loop_id = j["subgraph"]["0"]["element_id"].get<size_t>();

    auto element = builder.find_element_by_id(loop_id);
    if (!element) {
        throw std::runtime_error("LoopIndvarFinalize: Element not found");
    }

    auto loop = dynamic_cast<structured_control_flow::StructuredLoop*>(element);
    if (!loop) {
        throw std::runtime_error("LoopIndvarFinalize: Element is not a loop");
    }

    return LoopIndvarFinalize(*loop);
}

} // namespace transformations
} // namespace sdfg
