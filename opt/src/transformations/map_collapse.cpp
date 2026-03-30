#include "sdfg/transformations/map_collapse.h"

#include "sdfg/analysis/assumptions_analysis.h"
#include "sdfg/analysis/loop_analysis.h"
#include "sdfg/analysis/scope_analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {
namespace transformations {

MapCollapse::MapCollapse(structured_control_flow::Map& loop, size_t count) : loop_(loop), count_(count) {}

std::string MapCollapse::name() const { return "MapCollapse"; }

bool MapCollapse::can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    // Criterion: count must be at least 2
    if (count_ < 2) {
        return false;
    }

    auto& assumptions_analysis = analysis_manager.get<analysis::AssumptionsAnalysis>();

    // Collect the chain of count_ perfectly-nested maps
    std::vector<structured_control_flow::Map*> maps;
    maps.push_back(&loop_);

    auto* current = &loop_;
    for (size_t i = 1; i < count_; ++i) {
        auto& body = current->root();

        // Criterion: All target maps must be perfectly nested (exactly one child that is a Map)
        if (body.size() != 1) {
            return false;
        }

        auto* next = dynamic_cast<structured_control_flow::Map*>(&body.at(0).first);
        if (!next) {
            return false;
        }

        // Criterion: The Sequence holding each map must have empty transitions
        if (!body.at(0).second.empty()) {
            return false;
        }

        maps.push_back(next);
        current = next;
    }

    // Criterion: All maps must be contiguous (stride 1)
    for (auto* map : maps) {
        if (!analysis::LoopAnalysis::is_contiguous(map, assumptions_analysis)) {
            return false;
        }
    }

    // Collect indvars of all maps being collapsed
    symbolic::SymbolSet indvars;
    for (auto* map : maps) {
        indvars.insert(map->indvar());
    }

    // Criterion: Map inits may not depend on any of the loop induction variables
    for (auto* map : maps) {
        auto init = map->init();
        for (auto& iv : indvars) {
            if (symbolic::uses(init, iv)) {
                return false;
            }
        }
    }

    // Criterion: Map bounds may not depend on any of the loop induction variables
    // of the maps being collapsed
    for (auto* map : maps) {
        auto bound = analysis::LoopAnalysis::canonical_bound(map, assumptions_analysis);
        if (bound == SymEngine::null) {
            return false;
        }
        for (auto& iv : indvars) {
            if (symbolic::uses(bound, iv)) {
                return false;
            }
        }
    }

    return true;
}

void MapCollapse::apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    auto& sdfg = builder.subject();
    auto& assumptions_analysis = analysis_manager.get<analysis::AssumptionsAnalysis>();
    auto& scope_analysis = analysis_manager.get<analysis::ScopeAnalysis>();

    // Step 1: Gather the maps to collapse and their bounds
    std::vector<structured_control_flow::Map*> maps;
    maps.push_back(&loop_);
    auto* current = &loop_;
    for (size_t i = 1; i < count_; ++i) {
        auto* next = dynamic_cast<structured_control_flow::Map*>(&current->root().at(0).first);
        maps.push_back(next);
        current = next;
    }

    std::vector<symbolic::Symbol> indvars;
    std::vector<symbolic::Expression> bounds;
    for (auto* map : maps) {
        indvars.push_back(map->indvar());
        bounds.push_back(analysis::LoopAnalysis::canonical_bound(map, assumptions_analysis));
    }

    // Step 2: Compute total iteration count = product of all bounds
    symbolic::Expression total_bound = bounds[0];
    for (size_t i = 1; i < bounds.size(); ++i) {
        total_bound = symbolic::mul(total_bound, bounds[i]);
    }

    // Step 3: Create the collapsed induction variable
    auto civ_name = builder.find_new_name(indvars[0]->get_name() + "_collapsed");
    builder.add_container(civ_name, sdfg.type(indvars[0]->get_name()));
    auto civ = symbolic::symbol(civ_name);

    // Step 4: Find the parent sequence of the outermost map
    auto parent = static_cast<structured_control_flow::Sequence*>(scope_analysis.parent_scope(&loop_));
    size_t index = parent->index(loop_);
    auto& transition = parent->at(index).second;

    // Step 5: Create the new collapsed map before the original
    auto& collapsed_map = builder.add_map_before(
        *parent,
        loop_,
        civ,
        symbolic::Lt(civ, total_bound),
        symbolic::integer(0),
        symbolic::add(civ, symbolic::integer(1)),
        loop_.schedule_type(),
        transition.assignments(),
        loop_.debug_info()
    );

    // Step 6: Add an empty block for indvar recovery before the original contents
    builder.add_block(collapsed_map.root());

    // Step 7: Move the body of the innermost map into the collapsed map
    auto* innermost = maps.back();
    builder.move_children(innermost->root(), collapsed_map.root());

    // Step 8: Add indvar recovery assignments to the transition of the empty
    // block so that all induction variables are defined before the original
    // loop contents.
    //
    // For maps [0..n-1] with bounds [B0, B1, ..., B_{n-1}]:
    //   indvar_0     = civ / (B1 * B2 * ... * B_{n-1})
    //   indvar_k     = (civ / (B_{k+1} * ... * B_{n-1})) % B_k
    //   indvar_{n-1} = civ % B_{n-1}
    auto& first_transition = collapsed_map.root().at(0).second;
    size_t n = indvars.size();

    for (size_t k = 0; k < n; ++k) {
        // Compute suffix product = B_{k+1} * ... * B_{n-1}
        symbolic::Expression suffix = symbolic::integer(1);
        for (size_t j = k + 1; j < n; ++j) {
            suffix = symbolic::mul(suffix, bounds[j]);
        }

        symbolic::Expression value;
        if (k == 0 && n == 1) {
            // Single-loop degenerate case (shouldn't happen with count>=2, but safe)
            value = civ;
        } else if (k == 0) {
            // Outermost: indvar_0 = civ / suffix
            value = symbolic::div(civ, suffix);
        } else if (k == n - 1) {
            // Innermost: indvar_{n-1} = civ % B_{n-1}
            value = symbolic::mod(civ, bounds[k]);
        } else {
            // Middle: indvar_k = (civ / suffix) % B_k
            value = symbolic::mod(symbolic::div(civ, suffix), bounds[k]);
        }

        first_transition.assignments()[indvars[k]] = value;
    }

    // Step 9: Remove the original nest
    // The index shifted by 1 because we inserted a map before
    transition.assignments().clear();
    builder.remove_child(*parent, parent->index(loop_));

    analysis_manager.invalidate_all();
    applied_ = true;
    collapsed_loop_ = &collapsed_map;
}

void MapCollapse::to_json(nlohmann::json& j) const {
    std::string loop_type;
    if (dynamic_cast<const structured_control_flow::Map*>(&loop_)) {
        loop_type = "map";
    } else {
        throw InvalidSDFGException("Unsupported loop type for serialization of map: " + loop_.indvar()->get_name());
    }

    j["transformation_type"] = this->name();
    j["subgraph"] = {{"0", {{"element_id", this->loop_.element_id()}, {"type", loop_type}}}};
    j["parameters"] = {{"count", count_}};
}

MapCollapse MapCollapse::from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& desc) {
    auto loop_id = desc["subgraph"]["0"]["element_id"].get<size_t>();
    size_t count = desc["parameters"]["count"].get<size_t>();
    auto element = builder.find_element_by_id(loop_id);
    if (!element) {
        throw InvalidTransformationDescriptionException("Element with ID " + std::to_string(loop_id) + " not found.");
    }
    auto loop = dynamic_cast<structured_control_flow::Map*>(element);

    return MapCollapse(*loop, count);
}

structured_control_flow::Map* MapCollapse::collapsed_loop() {
    if (!applied_) {
        return &loop_;
    }

    return collapsed_loop_;
}

} // namespace transformations
} // namespace sdfg
