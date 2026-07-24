#include "sdfg/transformations/map_collapse.h"

#include <string>
#include <unordered_set>
#include <vector>

#include "sdfg/analysis/users.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/structured_control_flow/if_else.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/structured_control_flow/sequence.h"
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

    // Fast path: a chain of perfectly-nested maps.
    if (this->check_perfect_nest()) {
        return true;
    }

    // Imperfect (CUDA-style) collapse is performed one level at a time.
    if (count_ == 2 && this->check_imperfect(analysis_manager)) {
        return true;
    }

    return false;
}

bool MapCollapse::check_perfect_nest() {
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

        auto* next = dyn_cast<structured_control_flow::Map*>(&body.at(0));
        if (!next) {
            return false;
        }

        maps.push_back(next);
        current = next;
    }

    // Criterion: All maps must be contiguous (stride 1)
    for (auto* map : maps) {
        if (!map->is_contiguous()) {
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
        auto bound = map->canonical_bound();
        if (bound.is_null()) {
            // If we can't even compute a closed-form bound, be conservative and disallow collapsing
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

bool MapCollapse::is_collapsible_inner_map(structured_control_flow::Map& map, const symbolic::Symbol& outer_indvar) {
    // Must increment by exactly 1 to participate in the flattened iteration space
    if (!map.is_contiguous()) {
        return false;
    }

    // Init may not depend on the outer induction variable
    if (symbolic::uses(map.init(), outer_indvar)) {
        return false;
    }

    // A closed-form bound is required and may not depend on the outer indvar
    auto bound = map.canonical_bound();
    if (bound.is_null()) {
        return false;
    }
    if (symbolic::uses(bound, outer_indvar)) {
        return false;
    }

    return true;
}

bool MapCollapse::check_imperfect(analysis::AnalysisManager& analysis_manager) {
    // The outer map must itself be collapsible (contiguous, closed-form bound
    // that does not depend on its own induction variable).
    auto outer_indvar = loop_.indvar();
    if (!loop_.is_contiguous()) {
        return false;
    }
    if (symbolic::uses(loop_.init(), outer_indvar)) {
        return false;
    }
    auto outer_bound = loop_.canonical_bound();
    if (outer_bound.is_null()) {
        return false;
    }
    if (symbolic::uses(outer_bound, outer_indvar)) {
        return false;
    }

    auto& body = loop_.root();
    const size_t n = body.size();

    std::vector<bool> is_collapsible(n, false);
    size_t num_collapsible = 0;
    for (size_t idx = 0; idx < n; ++idx) {
        auto* map = dyn_cast<structured_control_flow::Map*>(&body.at(idx));
        if (map != nullptr && this->is_collapsible_inner_map(*map, outer_indvar)) {
            is_collapsible[idx] = true;
            ++num_collapsible;
        }
        // Everything else (blocks, if-else, loops, non-collapsible maps) is a
        // "skipped" element that will be replicated on every inner thread.
    }

    // Need at least one collapsible inner map to flatten against.
    if (num_collapsible < 1) {
        return false;
    }

    // Data-dependency safety gate (replication model).
    //
    // The collapse flattens the outer map together with one inner level into a
    // single parallel iteration space. Collapsible inner maps run for the valid
    // portion of the flattened inner index (`inner < bound`); every other
    // ("skipped") body element is *replicated* on every inner thread.
    //
    // Replication is safe for a skipped element because it is a sibling of the
    // inner maps and therefore cannot reference the inner induction variable: its
    // reads and writes are identical for all inner threads of the same outer
    // iteration. Hence a value produced by a skipped element and consumed by a
    // later element (RAW) is reproduced independently on each thread - there is no
    // cross-thread dependency.
    //
    // What replication cannot make safe, and is therefore rejected:
    //   * A container written by a *collapsible* map and accessed (read or
    //     written) by any other body element: collapsible writes vary across the
    //     inner index, so the consumer would observe another thread's data
    //     without synchronization (covers RAW, WAR, WAW against collapsible maps).
    //   * A write-write conflict between two different body elements on the same
    //     container: ordering across threads is no longer guaranteed.
    auto& users = analysis_manager.get<analysis::Users>();

    std::vector<std::unordered_set<std::string>> writes(n);
    std::vector<std::unordered_set<std::string>> reads(n);
    for (size_t idx = 0; idx < n; ++idx) {
        analysis::UsersView view(users, body.at(idx));
        for (auto* u : view.writes()) {
            writes[idx].insert(u->container());
        }
        for (auto* u : view.moves()) {
            writes[idx].insert(u->container());
        }
        // Views alias memory; treat conservatively as both a read and a write.
        for (auto* u : view.views()) {
            writes[idx].insert(u->container());
            reads[idx].insert(u->container());
        }
        for (auto* u : view.reads()) {
            reads[idx].insert(u->container());
        }
    }

    for (size_t a = 0; a < n; ++a) {
        for (size_t b = 0; b < n; ++b) {
            if (a == b) {
                continue;
            }
            for (const auto& container : writes[a]) {
                const bool accessed_by_b = writes[b].count(container) != 0 || reads[b].count(container) != 0;
                if (!accessed_by_b) {
                    continue;
                }
                // Unsafe if the writer is a collapsible map (inner-varying write
                // shared with another element), or if both elements write the same
                // container (write-write conflict). A skipped writer feeding a
                // pure reader (RAW) is safe under replication and allowed.
                if (is_collapsible[a] || writes[b].count(container) != 0) {
                    return false;
                }
            }
        }
    }

    return true;
}

void MapCollapse::apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    if (this->check_perfect_nest()) {
        this->apply_perfect(builder, analysis_manager);
    } else {
        this->apply_imperfect(builder, analysis_manager);
    }
}

void MapCollapse::apply_perfect(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    auto& sdfg = builder.subject();

    // Step 1: Gather the maps to collapse and their bounds
    std::vector<structured_control_flow::Map*> maps;
    maps.push_back(&loop_);
    auto* current = &loop_;
    for (size_t i = 1; i < count_; ++i) {
        auto* next = dyn_cast<structured_control_flow::Map*>(&current->root().at(0));
        maps.push_back(next);
        current = next;
    }

    std::vector<symbolic::Symbol> indvars;
    std::vector<symbolic::Expression> bounds;
    for (auto* map : maps) {
        indvars.push_back(map->indvar());
        bounds.push_back(map->canonical_bound());
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
    auto parent = static_cast<structured_control_flow::Sequence*>(loop_.get_parent());

    // Step 5: Create the new collapsed map before the original
    auto& collapsed_map = builder.add_map_before(
        *parent,
        loop_,
        civ,
        symbolic::Lt(civ, total_bound),
        symbolic::integer(0),
        symbolic::add(civ, symbolic::integer(1)),
        loop_.schedule_type(),
        loop_.debug_info()
    );

    // Step 6: Add an empty block for indvar recovery before the original contents
    auto& recovery_assignments = builder.add_assignments(collapsed_map.root(), {});

    // Step 7: Move the body of the innermost map into the collapsed map
    auto* innermost = maps.back();
    builder.move_children(innermost->root(), collapsed_map.root());

    // Step 8: Add indvar recovery assignments so that all induction variables are defined before the original
    // loop contents.
    //
    // For maps [0..n-1] with bounds [B0, B1, ..., B_{n-1}]:
    //   indvar_0     = civ / (B1 * B2 * ... * B_{n-1})
    //   indvar_k     = (civ / (B_{k+1} * ... * B_{n-1})) % B_k
    //   indvar_{n-1} = civ % B_{n-1}
    size_t n = indvars.size();

    symbolic::ExpressionMapping recovery_map;
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

        recovery_assignments.assignments()[indvars[k]] = value;
        recovery_map[indvars[k]] = value;
    }

    // Step 8b: Inline the recovered induction variables directly into the collapsed body.
    // This substitutes each original indvar with its closed-form expression in memlet subsets,
    // tasklet code and nested control flow, so downstream analyses/codegen see the relation to
    // the collapsed induction variable without requiring a separate SymbolPropagation pass.
    // The recovery assignments above are kept as a fallback: uses that cannot take a complex
    // expression (e.g. an induction variable used as an access-node container name) are left
    // untouched by replace() and still resolved through the transition.
    collapsed_map.root().replace(recovery_map);

    // Step 9: Remove the original nest
    // The index shifted by 1 because we inserted a map before
    builder.remove_child(*parent, parent->index(loop_));

    applied_ = true;
    collapsed_loop_ = &collapsed_map;
}

void MapCollapse::apply_imperfect(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    auto& sdfg = builder.subject();
    auto& outer = loop_;
    auto outer_indvar = outer.indvar();
    auto& body = outer.root();

    // Step 1: Classify the outer body children (in order) and gather the bounds
    // of the collapsible inner maps.
    struct BodyItem {
        structured_control_flow::ControlFlowNode* node;
        structured_control_flow::Map* map; // non-null only if collapsible
        symbolic::Expression bound; // valid only if collapsible
    };

    std::vector<BodyItem> items;
    std::vector<symbolic::Expression> collapsible_bounds;
    for (size_t idx = 0; idx < body.size(); ++idx) {
        auto& child = body.at(idx);
        auto* map = dyn_cast<structured_control_flow::Map*>(&child);
        if (map != nullptr && this->is_collapsible_inner_map(*map, outer_indvar)) {
            auto bound = map->canonical_bound();
            items.push_back({&child, map, bound});
            collapsible_bounds.push_back(bound);
        } else {
            items.push_back({&child, nullptr, SymEngine::null});
        }
    }

    // Step 2: Virtual inner extent = max of all collapsible bounds.
    symbolic::Expression inner_extent = collapsible_bounds[0];
    for (size_t i = 1; i < collapsible_bounds.size(); ++i) {
        inner_extent = symbolic::max(inner_extent, collapsible_bounds[i]);
    }

    auto outer_bound = outer.canonical_bound();
    auto total_bound = symbolic::mul(outer_bound, inner_extent);

    // Step 3: Create the collapsed induction variable and the virtual inner index.
    auto civ_name = builder.find_new_name(outer_indvar->get_name() + "_collapsed");
    builder.add_container(civ_name, sdfg.type(outer_indvar->get_name()));
    auto civ = symbolic::symbol(civ_name);

    auto inner_name = builder.find_new_name(outer_indvar->get_name() + "_inner");
    builder.add_container(inner_name, sdfg.type(outer_indvar->get_name()));
    auto inner_index = symbolic::symbol(inner_name);

    // Step 4: Find the parent sequence of the outer map.
    auto parent = static_cast<structured_control_flow::Sequence*>(outer.get_parent());

    // Step 5: Create the collapsed map before the original outer map.
    auto& collapsed_map = builder.add_map_before(
        *parent,
        outer,
        civ,
        symbolic::Lt(civ, total_bound),
        symbolic::integer(0),
        symbolic::add(civ, symbolic::integer(1)),
        outer.schedule_type(),
        outer.debug_info()
    );

    // Step 6: Recovery block defining the outer index and the virtual inner index:
    //   outer_indvar = civ / inner_extent
    //   inner_index  = civ % inner_extent
    auto& recovery_assignments = builder.add_assignments(collapsed_map.root(), {});
    recovery_assignments.assignments()[outer_indvar] = symbolic::div(civ, inner_extent);
    recovery_assignments.assignments()[inner_index] = symbolic::mod(civ, inner_extent);

    // Induction variables to inline into the collapsed body (see Step 8b). The outer
    // induction variable maps directly to its closed-form; each collapsible inner induction
    // variable maps to the virtual inner index (civ % inner_extent).
    symbolic::ExpressionMapping recovery_map;
    recovery_map[outer_indvar] = symbolic::div(civ, inner_extent);

    // Step 7: Move the outer body into the collapsed map (after the recovery block),
    // preserving order.
    builder.move_children(outer.root(), collapsed_map.root());

    // Step 8: Process each original body element. Collapsible maps run for the
    // valid portion of the inner extent (`inner_index < bound`). Every other
    // ("skipped") element is replicated: it stays a direct child of the collapsed
    // body and therefore runs on every inner thread. Because a skipped element is
    // a sibling of the inner maps it cannot reference the inner index, so all
    // inner threads of an outer iteration execute it identically.
    for (auto& item : items) {
        auto* child = item.node;
        if (item.map != nullptr) {
            auto inner_iv = item.map->indvar();

            auto& if_else = builder.add_if_else_before(collapsed_map.root(), *child, outer.debug_info());
            auto& branch = builder.add_case(if_else, symbolic::Lt(inner_index, item.bound), outer.debug_info());

            // Recover the inner induction variable from the virtual index.
            builder.add_assignments(branch, {{inner_iv, inner_index}});

            // Move the map body into the guarded branch and drop the empty map shell.
            builder.move_children(item.map->root(), branch);
            builder.remove_child(collapsed_map.root(), collapsed_map.root().index(*child));

            recovery_map[inner_iv] = symbolic::mod(civ, inner_extent);
        }
        // Skipped elements are left in place (replicated on every inner thread).
    }

    // Step 8b: Inline the recovered induction variables directly into the collapsed body so
    // downstream analyses/codegen see the relation to the collapsed induction variable without
    // requiring a separate SymbolPropagation pass. The recovery assignments (outer recovery
    // block and per-branch inner recovery) are kept as a fallback for uses that cannot take a
    // complex expression (e.g. an induction variable used as an access-node container name).
    collapsed_map.root().replace(recovery_map);

    // Step 9: Remove the original outer map.
    builder.remove_child(*parent, parent->index(outer));

    applied_ = true;
    collapsed_loop_ = &collapsed_map;

    analysis_manager.invalidate_all();
}

void MapCollapse::to_json(nlohmann::json& j) const {
    j["transformation_type"] = this->name();
    j["parameters"] = nlohmann::json::object();
    j["parameters"] = {{"count", count_}};

    serializer::JSONSerializer ser_flat(false);
    j["subgraph"] = nlohmann::json::object();
    j["subgraph"]["0"] = nlohmann::json::object();
    ser_flat.serialize_node(j["subgraph"]["0"], loop_);
}

MapCollapse MapCollapse::from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& desc) {
    auto loop_id = desc["subgraph"]["0"]["element_id"].get<size_t>();
    size_t count = desc["parameters"]["count"].get<size_t>();
    auto element = builder.find_element_by_id(loop_id);
    if (!element) {
        throw InvalidTransformationDescriptionException("Element with ID " + std::to_string(loop_id) + " not found.");
    }
    auto loop = dyn_cast<structured_control_flow::Map*>(element);

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
