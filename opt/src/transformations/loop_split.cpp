#include "sdfg/transformations/loop_split.h"

#include "sdfg/analysis/scope_analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/deepcopy/structured_sdfg_deep_copy.h"
#include "sdfg/structured_control_flow/for.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {
namespace transformations {

LoopSplit::LoopSplit(structured_control_flow::StructuredLoop& loop, const symbolic::Expression& split_point)
    : loop_(loop), split_point_(split_point) {};

std::string LoopSplit::name() const { return "LoopSplit"; };

bool LoopSplit::can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    // Loop must be contiguous (unit stride)
    if (!loop_.is_contiguous()) {
        return false;
    }

    // Loop must have a canonical bound (well-formed upper bound)
    auto bound = loop_.canonical_bound();
    if (bound == SymEngine::null) {
        return false;
    }

    // split_point must not be null
    if (split_point_ == SymEngine::null) {
        return false;
    }

    return true;
};

void LoopSplit::apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    auto& sdfg = builder.subject();

    auto indvar = loop_.indvar();
    auto condition = loop_.condition();
    auto init = loop_.init();
    auto update = loop_.update();
    auto bound = loop_.canonical_bound();

    // Get parent scope
    auto& scope_analysis = analysis_manager.get<analysis::ScopeAnalysis>();
    auto parent = static_cast<structured_control_flow::Sequence*>(scope_analysis.parent_scope(&loop_));

    // Create the first loop (before the original): for (i = init; i < split_point && original_cond; i++)
    //
    // We conjoin the original loop condition so that downstream symbolic
    // analysis (assumptions, MLA delinearization) sees the FULL bound on the
    // in-panel iteration space, not just the split point. Without this, when
    // `split_point` may exceed the original upper bound (a runtime-dependent
    // case), the in-panel loop's effective range would be unrepresentable in
    // the symbolic model.
    auto first_condition = symbolic::And(symbolic::Lt(indvar, split_point_), condition);

    structured_control_flow::StructuredLoop* first_loop = nullptr;
    if (auto map = dynamic_cast<structured_control_flow::Map*>(&loop_)) {
        first_loop = &builder.add_map_before(
            *parent, loop_, indvar, first_condition, init, update, map->schedule_type(), {}, loop_.debug_info()
        );
    } else {
        first_loop =
            &builder.add_for_before(*parent, loop_, indvar, first_condition, init, update, {}, loop_.debug_info());
    }

    // Deep copy the original loop body into the first loop
    deepcopy::StructuredSDFGDeepCopy deep_copy(builder, first_loop->root(), loop_.root());
    deep_copy.insert();

    // Give the first loop a fresh induction variable (to avoid name collision)
    std::string new_indvar_name = builder.find_new_name(indvar->get_name());
    builder.add_container(new_indvar_name, sdfg.type(indvar->get_name()));
    first_loop->replace(indvar, symbolic::symbol(new_indvar_name));

    // Update the original loop to start at split_point: for (i = split_point; i < bound; i++)
    builder.update_loop(loop_, indvar, condition, split_point_, update);

    analysis_manager.invalidate_all();
};

void LoopSplit::to_json(nlohmann::json& j) const {
    std::string loop_type = "for";
    if (dynamic_cast<const structured_control_flow::Map*>(&loop_)) {
        loop_type = "map";
    }

    j["transformation_type"] = this->name();
    j["subgraph"] = {{"0", {{"element_id", loop_.element_id()}, {"type", loop_type}}}};
    j["parameters"] = {{"split_point", split_point_->__str__()}};
};

LoopSplit LoopSplit::from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& desc) {
    auto loop_id = desc["subgraph"]["0"]["element_id"].get<size_t>();
    auto split_point_str = desc["parameters"]["split_point"].get<std::string>();

    auto element = builder.find_element_by_id(loop_id);
    if (element == nullptr) {
        throw InvalidTransformationDescriptionException("Element with ID " + std::to_string(loop_id) + " not found.");
    }

    auto loop = dynamic_cast<structured_control_flow::StructuredLoop*>(element);
    if (loop == nullptr) {
        throw InvalidTransformationDescriptionException(
            "Element with ID " + std::to_string(loop_id) + " is not a StructuredLoop."
        );
    }

    auto split_point = symbolic::parse(split_point_str);
    return LoopSplit(*loop, split_point);
};

} // namespace transformations
} // namespace sdfg
