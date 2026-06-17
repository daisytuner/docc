#include "sdfg/passes/new_map_fusion_pass.h"

#include "sdfg/analysis/assumptions_analysis.h"
#include "sdfg/data_flow/library_nodes/stdlib/malloc.h"
#include "sdfg/deepcopy/structured_sdfg_deep_copy.h"
#include "sdfg/visitor/structured_sdfg_visitor.h"
#include "symengine/subs.h"

namespace sdfg::passes {

FusionLoopCandidate* NewMapFusionPass::State::get_next_level_map_stack(FusionLoopCandidate& current) {
    auto& children = loop_analysis->children(current.loop);
    if (children.empty()) {
        return nullptr;
    }

    auto* next = children.at(0);
    return &fuse_candidates.at(next->element_id());
}

bool NewMapFusionPass::run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    fused_count = 0;

    auto loop_ana = std::make_unique<analysis::LoopAnalysis>(builder.subject());
    loop_ana->run(analysis_manager);

    State state(builder, analysis_manager, std::move(loop_ana));

    auto& assumption_analysis = analysis_manager.get<analysis::AssumptionsAnalysis>();
    auto& arguments_analysis = analysis_manager.get<analysis::ArgumentsAnalysis>();

    auto outermost = state.loop_analysis->outermost_loops();

    for (auto* control_flow_node : state.loop_analysis->loops()) {
        if (auto* map = dynamic_cast<Map*>(control_flow_node)) {
            auto& indvar = map->indvar();
            auto& assumpts = assumption_analysis.get(map->root(), true);
            auto* indvar_boundaries = find_indvar_boundaries(indvar, assumpts);

            if (indvar_boundaries && !indvar_boundaries->tight_lower_bound().is_null() &&
                !indvar_boundaries->tight_upper_bound().is_null() && !indvar_boundaries->map().is_null()) {
                auto& args = arguments_analysis.arguments(analysis_manager, *map);
                state.fuse_candidates[control_flow_node->element_id()] = {map, indvar_boundaries, &args};
            }
        }
    }

    MapFusionHandler handler(state);

    NeighboringPatternVisitor v(handler);
    v.dispatch(builder.subject().root());

    return fused_count;
}

const symbolic::Assumption* NewMapFusionPass::
    find_indvar_boundaries(const symbolic::Symbol& indvar, const symbolic::Assumptions& assumptions) {
    auto it = assumptions.find(indvar);
    if (it != assumptions.end()) {
        return &it->second;
    }

    return nullptr;
}

MapFusionHandler::MapFusionHandler(NewMapFusionPass::State& state) : state_(state) {}

bool MapFusionHandler::fuse_contents(
    FusionLoopCandidate* first_current,
    FusionLoopCandidate* second_current,
    SymEngine::map_basic_basic indvar_mapping,
    Sequence& target_root
) {
    Sequence* append_root = nullptr;
    if (target_root.size() == 0) {
        append_root = &target_root;
    } else { // there currently is no way to prepend or limit replace, so add to new sequence, then move
        append_root = &state_.builder.add_sequence_before(target_root, target_root.at(0).first, {}, {});
    }

    deepcopy::StructuredSDFGDeepCopy copier(state_.builder, *append_root, first_current->loop->root());
    auto copy_mapping = copier.insert();

    update_fused_seq(*append_root, indvar_mapping);

    if (append_root != &target_root) { // need to fixup / flatten the copied sequence into the target sequence
        state_.builder.move_children(*append_root, target_root, 0);
    }

    auto& first_children = state_.loop_analysis->children(first_current->loop);
    bool keep_visiting_second = !state_.loop_analysis->children(second_current->loop).empty() ||
                                !first_children.empty();
    if (!first_children.empty()) {
        for (auto& child : first_children) {
            state_.loop_analysis->copied_loop(
                child,
                second_current->loop,
                const_cast<structured_control_flow::ControlFlowNode*>(copy_mapping.at(child))
            );
        }
    }
    return keep_visiting_second;
}

PatternHandler::MatchResult MapFusionHandler::match(Map& first, Map& second, bool no_uses_between) {
    auto first_it = state_.fuse_candidates.find(first.element_id());
    if (first_it == state_.fuse_candidates.end()) {
        return {};
    }
    auto* first_current = &first_it->second;

    auto second_it = state_.fuse_candidates.find(second.element_id());
    if (second_it == state_.fuse_candidates.end()) {
        return {};
    }
    auto* second_current = &second_it->second;

    bool no_data_conflicts = this->no_data_conflicts(*first_current, *second_current);
    if (!no_data_conflicts) {
        return {};
    }

    SymEngine::map_basic_basic indvar_mapping;
    bool level_match;
    int current_level = -1;
    int last_matched_level = -1;
    auto second_loop_info = state_.loop_analysis->loop_info(&second);
    auto max_map_stack_depth =
        std::min(state_.loop_analysis->loop_info(&first).map_stack_depth, second_loop_info.map_stack_depth) - 1;
    bool keep_looking;

    do {
        keep_looking = false;
        ++current_level;
        indvar_mapping[first_current->loop->indvar()] = second_current->loop->indvar();

        level_match = this->loop_match(*first_current, *second_current, indvar_mapping);
        if (level_match) {
            last_matched_level = current_level;
            if (current_level < max_map_stack_depth) {
                keep_looking = true;
                first_current = state_.get_next_level_map_stack(*first_current);
                second_current = state_.get_next_level_map_stack(*second_current);
            }
        }
    } while (keep_looking);

    if (last_matched_level >= 0) {
        DEBUG_PRINTLN(
            "Can fuse map stack (" << last_matched_level + 1 << " lvls): #" << first.element_id() << " | #"
                                   << first_current->loop->element_id() << ", #" << second.element_id() << " | #"
                                   << second_current->loop->element_id()
        );

        auto& target_root = second_current->loop->root();
        bool keep_visiting_second =
            fuse_contents(first_current, second_current, indvar_mapping, target_root, no_uses_between);

        // if there are further loops inside the now fused body, visit those as well
        return {.removed_first = false, .visit_second_body = keep_visiting_second};
    }

    return {};
}

bool MapFusionHandler::
    loop_match(FusionLoopCandidate& first, FusionLoopCandidate& second, SymEngine::map_basic_basic& canonical_indvars) {
    bool lower_match =
        symbolic::eq(first.indvar_boundaries->tight_lower_bound(), second.indvar_boundaries->tight_lower_bound());
    if (!lower_match) {
        return false;
    }
    bool upper_match =
        symbolic::eq(first.indvar_boundaries->tight_upper_bound(), second.indvar_boundaries->tight_upper_bound());
    if (!upper_match) {
        return false;
    }
    auto first_canonicalized_map = SymEngine::subs(first.indvar_boundaries->map(), canonical_indvars);
    bool map_match = symbolic::eq(first_canonicalized_map, second.indvar_boundaries->map());
    if (!map_match) {
        return false;
    }

    return true;
}

bool MapFusionHandler::
    no_data_conflicts(const FusionLoopCandidate& first_candidate, const FusionLoopCandidate& second_candidate) {
    auto& first_args = *first_candidate.args;
    auto& second_args = *second_candidate.args;
    for (auto& [name, prod_meta] : first_args) {
        auto cons_it = second_candidate.args->find(name);
        if (cons_it != second_args.end()) {
            // argument appears in both
            auto& cons_meta = cons_it->second;
            // any potential conflicts to check if both are maps?
        }
    }

    return true;
}

void MapFusionHandler::update_fused_seq(Sequence& sequence, const symbolic::ExpressionMapping& replacements) {
    sequence.replace(replacements);
}

NeighboringPatternVisitor::NeighboringPatternVisitor(PatternHandler& handler) : handler_(handler) {}

bool NeighboringPatternVisitor::visit(sdfg::structured_control_flow::Sequence& node) {
    if (node.size() < 2) { // impossible to find a match, just descend into it
        return ActualStructuredSDFGVisitor::visit(node);
    }

    // Iterate over sequence looking for consecutive (Map, StructuredLoop) pairs
    size_t i = 0;
    while (i + 1 < node.size()) {
        auto& child_node = node.at(i).first;
        auto* first = dynamic_cast<structured_control_flow::Map*>(&child_node);
        if (!first) {
            i++;
            dispatch(child_node);
            continue;
        }
        if (first->root().size() == 0) {
            i++;
            continue;
        }

        if (auto* second = dynamic_cast<structured_control_flow::Map*>(&node.at(i + 1).first)) {
            if (second->root().size() == 0) {
                i++;
                continue;
            }

            auto result = handler_.match(*first, *second, TODO);

            if (!result.removed_first) {
                dispatch(child_node);
            }
            if (result.visit_second_body) {
                auto* second_updated_child = result.second_root_replacement ? result.second_root_replacement : second;
                dispatch(*second_updated_child);
            }
        } else if (i + 2 < node.size()) {
            auto* mid_block = dynamic_cast<structured_control_flow::Block*>(&node.at(i + 1).first);
            if (mid_block && mid_block->is_a_library_node<stdlib::MallocNode>()) {
                if (auto* second = dynamic_cast<structured_control_flow::Map*>(&node.at(i + 2).first)) {
                    if (second->root().size() == 0) {
                        i++;
                        continue;
                    }

                    // until we support matching this, just keep visiting
                    dispatch(child_node);
                    dispatch(*second);
                    // transformations::MapFusion transformation(*first, *second, false);
                    // if (transformation.can_be_applied(builder_, analysis_manager_)) {
                    //     auto first_name = first->indvar()->get_name();
                    //     auto second_name = second->indvar()->get_name();
                    //     transformation.apply(builder_, analysis_manager_);
                    //     DEBUG_PRINTLN(
                    //         "Applied MapFusion to map " + first_name + " and loop " + second_name +
                    //         " with intermediate malloc block"
                    //     );
                    //     applied = true;
                    //
                    //     i = i + 2; // Skip over the newly moved malloc block and the second loop that was just fused
                    //     continue;
                    // }
                }
            }
        }
        i++;
    }

    return true;
}

} // namespace sdfg::passes
