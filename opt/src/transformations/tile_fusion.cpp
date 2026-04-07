#include "sdfg/transformations/tile_fusion.h"

#include <cmath>
#include <stdexcept>

#include <symengine/solve.h>

#include "sdfg/analysis/arguments_analysis.h"
#include "sdfg/analysis/assumptions_analysis.h"
#include "sdfg/analysis/loop_analysis.h"
#include "sdfg/analysis/scope_analysis.h"
#include "sdfg/analysis/users.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/data_flow_graph.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/for.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/symbolic/delinearization.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/symbolic/utils.h"

namespace sdfg {
namespace transformations {

namespace {

/// Collect all Block nodes reachable from a Sequence, including through nested For/Map loops.
void collect_blocks(structured_control_flow::Sequence& seq, std::vector<structured_control_flow::Block*>& blocks) {
    for (size_t i = 0; i < seq.size(); ++i) {
        auto& child = seq.at(i).first;
        if (auto* block = dynamic_cast<structured_control_flow::Block*>(&child)) {
            blocks.push_back(block);
        } else if (auto* loop = dynamic_cast<structured_control_flow::StructuredLoop*>(&child)) {
            collect_blocks(loop->root(), blocks);
        }
    }
}

/// Get the first block from a sequence, recursively descending into loops.
structured_control_flow::Block* get_first_block(structured_control_flow::Sequence& seq) {
    for (size_t i = 0; i < seq.size(); ++i) {
        auto& child = seq.at(i).first;
        if (auto* block = dynamic_cast<structured_control_flow::Block*>(&child)) {
            return block;
        } else if (auto* loop = dynamic_cast<structured_control_flow::StructuredLoop*>(&child)) {
            auto* result = get_first_block(loop->root());
            if (result) return result;
        }
    }
    return nullptr;
}

} // namespace

TileFusion::TileFusion(structured_control_flow::Map& first_map, structured_control_flow::Map& second_map)
    : first_map_(first_map), second_map_(second_map) {}

std::string TileFusion::name() const { return "TileFusion"; }

int TileFusion::compute_radius(
    const data_flow::Subset& producer_write_subset,
    const std::vector<data_flow::Subset>& consumer_read_subsets,
    const std::vector<structured_control_flow::StructuredLoop*>& producer_loops,
    const std::vector<structured_control_flow::StructuredLoop*>& consumer_loops,
    const symbolic::Assumptions& producer_assumptions,
    const symbolic::Assumptions& consumer_assumptions
) {
    if (producer_loops.empty() || consumer_loops.empty()) {
        return -1;
    }

    // Delinearize the producer write subset
    auto producer_sub = producer_write_subset;
    if (producer_sub.size() == 1) {
        auto producer_result = symbolic::delinearize(producer_write_subset.at(0), producer_assumptions);
        if (producer_result.success) {
            producer_sub = producer_result.indices;
        }
    }

    // Extract the innermost producer loop indvar (the spatial iteration variable)
    auto* producer_inner = producer_loops.back();
    auto producer_indvar = producer_inner->indvar();

    // Extract the innermost consumer loop indvar
    auto* consumer_inner = consumer_loops.back();
    auto consumer_indvar = consumer_inner->indvar();

    int max_radius = 0;

    for (const auto& consumer_read_subset : consumer_read_subsets) {
        auto consumer_sub = consumer_read_subset;
        if (consumer_sub.size() == 1) {
            auto consumer_result = symbolic::delinearize(consumer_read_subset.at(0), consumer_assumptions);
            if (consumer_result.success) {
                consumer_sub = consumer_result.indices;
            }
        }
        if (consumer_sub.size() != producer_sub.size()) {
            return -1;
        }

        // Solve: producer_sub[d] = consumer_sub[d] for producer_indvar
        // This gives us: producer_indvar = f(consumer_indvar)
        SymEngine::vec_sym producer_vars;
        producer_vars.push_back(SymEngine::rcp_static_cast<const SymEngine::Symbol>(producer_indvar));

        SymEngine::vec_basic equations;
        for (size_t d = 0; d < producer_sub.size(); ++d) {
            auto diff = symbolic::sub(producer_sub.at(d), consumer_sub.at(d));
            if (symbolic::uses(diff, producer_indvar->get_name())) {
                // This dimension depends on the producer indvar — include in system
                equations.push_back(diff);
            }
            // Dimensions not involving the producer indvar are independent
            // (e.g., inner loop variables in 2D stencils) — skip them.
        }

        if (equations.size() != producer_vars.size()) {
            return -1;
        }

        SymEngine::vec_basic solution;
        try {
            solution = SymEngine::linsolve(equations, producer_vars);
        } catch (...) {
            return -1;
        }

        if (solution.size() != 1) {
            return -1;
        }

        auto& sol = solution[0];
        if (SymEngine::is_a<SymEngine::NaN>(*sol) || SymEngine::is_a<SymEngine::Infty>(*sol)) {
            return -1;
        }

        // The solution is: producer_indvar = f(consumer_indvar)
        // The offset is: f(consumer_indvar) - consumer_indvar
        // For a stencil B[1+i] and consumer reads B[j], B[1+j], B[2+j]:
        //   solve 1+i=j   -> i = j-1, offset = -1
        //   solve 1+i=1+j -> i = j,   offset = 0
        //   solve 1+i=2+j -> i = j+1, offset = +1
        auto offset_expr = symbolic::expand(symbolic::sub(sol, consumer_indvar));

        // The offset should be a constant integer
        if (!SymEngine::is_a<SymEngine::Integer>(*offset_expr)) {
            return -1;
        }

        int offset = std::abs(static_cast<int>(SymEngine::down_cast<const SymEngine::Integer&>(*offset_expr).as_int()));

        if (offset > max_radius) {
            max_radius = offset;
        }
    }

    return max_radius;
}

bool TileFusion::can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    candidates_.clear();
    radius_ = 0;

    // Get analyses upfront
    auto& scope_analysis = analysis_manager.get<analysis::ScopeAnalysis>();
    auto& loop_analysis = analysis_manager.get<analysis::LoopAnalysis>();
    auto& assumptions_analysis = analysis_manager.get<analysis::AssumptionsAnalysis>();
    auto& arguments_analysis = analysis_manager.get<analysis::ArgumentsAnalysis>();

    // ==========================================================================
    // Criterion 1: Both maps are in the same parent sequence, consecutive
    // ==========================================================================
    auto* first_parent = scope_analysis.parent_scope(&first_map_);
    auto* second_parent = scope_analysis.parent_scope(&second_map_);
    if (first_parent == nullptr || second_parent == nullptr) {
        return false;
    }
    if (first_parent != second_parent) {
        return false;
    }

    auto* parent_sequence = dynamic_cast<structured_control_flow::Sequence*>(first_parent);
    if (parent_sequence == nullptr) {
        return false;
    }

    int first_index = parent_sequence->index(first_map_);
    int second_index = parent_sequence->index(second_map_);
    if (first_index == -1 || second_index == -1) {
        return false;
    }
    if (second_index != first_index + 1) {
        return false;
    }

    // ==========================================================================
    // Criterion 2: Transition between them is empty
    // ==========================================================================
    auto& transition = parent_sequence->at(first_index).second;
    if (!transition.empty()) {
        return false;
    }

    // ==========================================================================
    // Criterion 3: Both are perfectly nested tile structures with Maps
    // Structure required: outer tile Map -> inner iteration Map(s) -> Block(s)
    // ==========================================================================
    auto first_loop_info = loop_analysis.loop_info(&first_map_);
    auto second_loop_info = loop_analysis.loop_info(&second_map_);

    // Must be perfectly nested (no side code between loop levels)
    if (!first_loop_info.is_perfectly_nested) {
        return false;
    }
    if (!second_loop_info.is_perfectly_nested) {
        return false;
    }

    // Must be all Maps (perfectly parallel) - this is a tile fusion on parallel maps
    if (!first_loop_info.is_perfectly_parallel) {
        return false;
    }
    if (!second_loop_info.is_perfectly_parallel) {
        return false;
    }

    // Must have at least depth 2 (outer tile + inner iteration)
    if (first_loop_info.max_depth < 2) {
        return false;
    }
    if (second_loop_info.max_depth < 2) {
        return false;
    }

    // The immediate child must be a Map (the inner iteration map)
    if (first_map_.root().size() != 1) {
        return false;
    }
    auto* first_inner_map = dynamic_cast<structured_control_flow::Map*>(&first_map_.root().at(0).first);
    if (first_inner_map == nullptr) {
        return false;
    }

    if (second_map_.root().size() != 1) {
        return false;
    }
    auto* second_inner_map = dynamic_cast<structured_control_flow::Map*>(&second_map_.root().at(0).first);
    if (second_inner_map == nullptr) {
        return false;
    }

    // ==========================================================================
    // Criterion 4: Compatible tile bounds (same init and stride)
    // ==========================================================================
    if (!symbolic::eq(first_map_.init(), second_map_.init())) {
        return false;
    }

    auto first_stride = first_map_.stride();
    auto second_stride = second_map_.stride();
    if (first_stride.is_null() || second_stride.is_null()) {
        return false;
    }
    if (!symbolic::eq(first_stride, second_stride)) {
        return false;
    }

    // Extract tile size as integer
    if (!SymEngine::is_a<SymEngine::Integer>(*first_stride)) {
        return false;
    }
    int tile_size = static_cast<int>(SymEngine::down_cast<const SymEngine::Integer&>(*first_stride).as_int());
    if (tile_size <= 0) {
        return false;
    }

    // ==========================================================================
    // Criterion 5: Shared intermediate container exists
    // First writes C, second reads C, second does NOT write C
    // ==========================================================================
    auto first_args = arguments_analysis.arguments(analysis_manager, first_map_);
    auto second_args = arguments_analysis.arguments(analysis_manager, second_map_);

    std::unordered_set<std::string> first_outputs;
    for (const auto& [name, arg] : first_args) {
        if (arg.is_output) {
            first_outputs.insert(name);
        }
    }

    std::unordered_set<std::string> shared_containers;
    for (const auto& [name, arg] : second_args) {
        if (first_outputs.contains(name)) {
            if (arg.is_output) {
                return false; // Consumer also writes the shared container
            }
            if (arg.is_input) {
                shared_containers.insert(name);
            }
        }
    }
    if (shared_containers.empty()) {
        return false;
    }

    // ==========================================================================
    // Criterion 6: Compute radius for each shared container
    // ==========================================================================
    // Collect the loop hierarchy for producer and consumer
    std::vector<structured_control_flow::StructuredLoop*> producer_loops = {&first_map_, first_inner_map};
    std::vector<structured_control_flow::StructuredLoop*> consumer_loops = {&second_map_, second_inner_map};

    // Get assumptions from the innermost block (recursively find it)
    auto* first_block = get_first_block(first_inner_map->root());
    if (first_block == nullptr) {
        return false;
    }
    auto& producer_assumptions = assumptions_analysis.get(*first_block);

    auto* second_block = get_first_block(second_inner_map->root());
    if (second_block == nullptr) {
        return false;
    }
    auto& consumer_assumptions = assumptions_analysis.get(*second_block);

    // Collect all blocks in producer and consumer
    std::vector<structured_control_flow::Block*> producer_blocks;
    collect_blocks(first_inner_map->root(), producer_blocks);
    std::vector<structured_control_flow::Block*> consumer_blocks;
    collect_blocks(second_inner_map->root(), consumer_blocks);

    int overall_radius = 0;

    for (const auto& container : shared_containers) {
        // Find the producer write subset for this container
        data_flow::Subset producer_write_subset;
        bool found_producer = false;

        for (auto* block : producer_blocks) {
            auto& dataflow = block->dataflow();
            for (auto& node : dataflow.nodes()) {
                auto* access = dynamic_cast<data_flow::AccessNode*>(&node);
                if (access == nullptr || access->data() != container) {
                    continue;
                }
                // It's a write if it has incoming edges and no outgoing edges
                if (dataflow.in_degree(*access) > 0 && dataflow.out_degree(*access) == 0) {
                    auto& iedge = *dataflow.in_edges(*access).begin();
                    if (iedge.type() != data_flow::MemletType::Computational) {
                        continue;
                    }
                    if (found_producer) {
                        return false; // Multiple writes to same container
                    }
                    producer_write_subset = iedge.subset();
                    found_producer = true;
                }
            }
        }
        if (!found_producer || producer_write_subset.empty()) {
            return false;
        }

        // Collect all consumer read subsets for this container
        std::vector<data_flow::Subset> consumer_read_subsets;
        for (auto* block : consumer_blocks) {
            auto& dataflow = block->dataflow();
            for (auto& node : dataflow.nodes()) {
                auto* access = dynamic_cast<data_flow::AccessNode*>(&node);
                if (access == nullptr || access->data() != container) {
                    continue;
                }
                // It's a read if it has outgoing edges
                if (dataflow.out_degree(*access) > 0) {
                    for (auto& memlet : dataflow.out_edges(*access)) {
                        if (memlet.type() != data_flow::MemletType::Computational) {
                            continue;
                        }
                        auto& subset = memlet.subset();
                        // Deduplicate
                        bool found = false;
                        for (const auto& existing : consumer_read_subsets) {
                            if (existing.size() != subset.size()) continue;
                            bool match = true;
                            for (size_t d = 0; d < existing.size(); ++d) {
                                if (!symbolic::eq(existing[d], subset[d])) {
                                    match = false;
                                    break;
                                }
                            }
                            if (match) {
                                found = true;
                                break;
                            }
                        }
                        if (!found) {
                            consumer_read_subsets.push_back(subset);
                        }
                    }
                }
            }
        }
        if (consumer_read_subsets.empty()) {
            return false;
        }

        int radius = compute_radius(
            producer_write_subset,
            consumer_read_subsets,
            producer_loops,
            consumer_loops,
            producer_assumptions,
            consumer_assumptions
        );
        if (radius < 0) {
            return false;
        }

        // ==========================================================================
        // Criterion 7: Radius must be less than tile size
        // ==========================================================================
        if (radius >= tile_size) {
            return false;
        }

        if (radius > overall_radius) {
            overall_radius = radius;
        }

        TileFusionCandidate candidate;
        candidate.container = container;
        candidate.radius = radius;
        candidates_.push_back(candidate);
    }

    radius_ = overall_radius;
    return true;
}

void TileFusion::apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    auto& sdfg = builder.subject();
    auto& scope_analysis = analysis_manager.get<analysis::ScopeAnalysis>();
    auto& assumptions_analysis = analysis_manager.get<analysis::AssumptionsAnalysis>();

    // Get parent sequence
    auto* parent = static_cast<structured_control_flow::Sequence*>(scope_analysis.parent_scope(&first_map_));
    size_t first_index = parent->index(first_map_);

    // Extract tile loop properties from first map (representative)
    auto tile_indvar = first_map_.indvar();
    auto tile_init = first_map_.init();
    auto tile_condition = first_map_.condition();
    auto tile_update = first_map_.update();

    // Extract tile size
    auto stride = first_map_.stride();
    auto tile_size_expr = stride;

    // Get references to inner maps before moving
    auto* first_inner_map = dynamic_cast<structured_control_flow::Map*>(&first_map_.root().at(0).first);
    auto* second_inner_map = dynamic_cast<structured_control_flow::Map*>(&second_map_.root().at(0).first);

    // Extract inner map properties before they get moved
    auto first_inner_indvar = first_inner_map->indvar();
    auto first_inner_init = first_inner_map->init();
    auto first_inner_condition = first_inner_map->condition();
    auto first_inner_update = first_inner_map->update();
    auto first_inner_schedule = first_inner_map->schedule_type();

    auto second_inner_indvar = second_inner_map->indvar();
    auto second_inner_init = second_inner_map->init();
    auto second_inner_condition = second_inner_map->condition();
    auto second_inner_update = second_inner_map->update();
    auto second_inner_schedule = second_inner_map->schedule_type();

    // Get the old tile indvars to substitute later
    auto first_tile_indvar = first_map_.indvar();
    auto second_tile_indvar = second_map_.indvar();

    // Step 1: Create a new For tile loop with the same bounds as the first tile Map
    // Use the condition from the first map (which uses first_tile_indvar)
    // We need a fresh indvar for the new tile loop
    auto new_tile_indvar_name = builder.find_new_name(tile_indvar->get_name());
    builder.add_container(new_tile_indvar_name, sdfg.type(tile_indvar->get_name()));
    auto new_tile_indvar = symbolic::symbol(new_tile_indvar_name);

    // Substitute the old tile indvar in the condition with the new one
    auto new_tile_condition = symbolic::subs(tile_condition, tile_indvar, new_tile_indvar);

    auto& fused_for = builder.add_for_before(
        *parent,
        first_map_,
        new_tile_indvar,
        new_tile_condition,
        tile_init,
        symbolic::subs(tile_update, tile_indvar, new_tile_indvar),
        {},
        first_map_.debug_info()
    );

    // Step 2: Create the producer inner Map inside the fused For
    // Its init and condition need to reference the new tile indvar
    auto new_first_inner_init = symbolic::subs(first_inner_init, first_tile_indvar, new_tile_indvar);
    auto new_first_inner_condition = symbolic::subs(first_inner_condition, first_tile_indvar, new_tile_indvar);

    // Extend the producer's range by the radius
    if (radius_ > 0) {
        // Extend init: max(0, tile - radius) -> take the original init and subtract radius
        // Original init is typically: tile_indvar (i.e., the iteration starts at the tile boundary)
        // New init: max(original_lower_bound, new_init - radius)
        auto radius_expr = symbolic::integer(radius_);
        auto extended_init = symbolic::max(symbolic::integer(0), symbolic::sub(new_first_inner_init, radius_expr));
        new_first_inner_init = extended_init;

        // Extend condition: the condition has form like And(i < tile + S, i < N)
        // We need to adjust the tile-relative bound: i < tile + S becomes i < tile + S + radius
        // We do this by substituting new_tile_indvar with (new_tile_indvar + radius) in the
        // tile-relative part. Since the condition is a conjunction, we can reconstruct it:
        // Original: And(inner_indvar < new_tile_indvar + S, inner_indvar < N)
        // Extended: And(inner_indvar < new_tile_indvar + S + radius, inner_indvar < N)
        auto extended_tile_bound =
            symbolic::Lt(first_inner_indvar, symbolic::add(new_tile_indvar, symbolic::add(tile_size_expr, radius_expr)));

        // Get the original non-tile bound from the canonical bound of the original inner map
        auto canonical = first_inner_map->canonical_bound();
        if (!canonical.is_null()) {
            auto original_bound = symbolic::Lt(first_inner_indvar, canonical);
            new_first_inner_condition = symbolic::And(extended_tile_bound, original_bound);
        } else {
            new_first_inner_condition = extended_tile_bound;
        }
    }

    auto& new_first_inner = builder.add_map(
        fused_for.root(),
        first_inner_indvar,
        new_first_inner_condition,
        new_first_inner_init,
        first_inner_update,
        first_inner_schedule
    );

    // Move the producer body into the new inner map
    builder.move_children(first_inner_map->root(), new_first_inner.root());

    // Step 3: Create the consumer inner Map inside the fused For, after the producer
    auto new_second_inner_init = symbolic::subs(second_inner_init, second_tile_indvar, new_tile_indvar);
    auto new_second_inner_condition = symbolic::subs(second_inner_condition, second_tile_indvar, new_tile_indvar);

    auto& new_second_inner = builder.add_map(
        fused_for.root(),
        second_inner_indvar,
        new_second_inner_condition,
        new_second_inner_init,
        second_inner_update,
        second_inner_schedule
    );

    // Move the consumer body into the new inner map
    builder.move_children(second_inner_map->root(), new_second_inner.root());

    // Step 4: Remove the original two tile Map nests
    // After we've moved the children out, remove the old maps from the parent
    // The fused_for was inserted before first_map_, so first_map_ is now at first_index + 1
    // and second_map_ is at first_index + 2
    // Remove second first (higher index) to avoid invalidating first's index
    size_t current_first_index = parent->index(first_map_);
    size_t current_second_index = parent->index(second_map_);
    builder.remove_child(*parent, current_second_index);
    builder.remove_child(*parent, current_first_index);

    analysis_manager.invalidate_all();
    applied_ = true;
    fused_loop_ = &fused_for;
}

void TileFusion::to_json(nlohmann::json& j) const {
    j["transformation_type"] = this->name();
    j["subgraph"] = {
        {"0", {{"element_id", first_map_.element_id()}, {"type", "map"}}},
        {"1", {{"element_id", second_map_.element_id()}, {"type", "map"}}}
    };
    j["parameters"] = {{"radius", radius_}};
}

TileFusion TileFusion::from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& desc) {
    auto first_map_id = desc["subgraph"]["0"]["element_id"].get<size_t>();
    auto second_map_id = desc["subgraph"]["1"]["element_id"].get<size_t>();

    auto first_element = builder.find_element_by_id(first_map_id);
    auto second_element = builder.find_element_by_id(second_map_id);

    if (first_element == nullptr) {
        throw InvalidTransformationDescriptionException("Element with ID " + std::to_string(first_map_id) + " not found.");
    }
    if (second_element == nullptr) {
        throw InvalidTransformationDescriptionException(
            "Element with ID " + std::to_string(second_map_id) + " not found."
        );
    }

    auto* first_map = dynamic_cast<structured_control_flow::Map*>(first_element);
    auto* second_map = dynamic_cast<structured_control_flow::Map*>(second_element);

    if (first_map == nullptr) {
        throw InvalidTransformationDescriptionException(
            "Element with ID " + std::to_string(first_map_id) + " is not a Map."
        );
    }
    if (second_map == nullptr) {
        throw InvalidTransformationDescriptionException(
            "Element with ID " + std::to_string(second_map_id) + " is not a Map."
        );
    }

    return TileFusion(*first_map, *second_map);
}

structured_control_flow::StructuredLoop* TileFusion::fused_loop() const {
    if (!applied_) {
        throw InvalidTransformationException("Accessing fused loop before apply.");
    }
    return fused_loop_;
}

int TileFusion::radius() const { return radius_; }

} // namespace transformations
} // namespace sdfg
