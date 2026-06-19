#include "sdfg/passes/map_fusion_by_domain_pass.h"

#include "sdfg/analysis/assumptions_analysis.h"
#include "sdfg/analysis/base_user_visitor.h"
#include "sdfg/data_flow/library_nodes/stdlib/malloc.h"
#include "sdfg/deepcopy/structured_sdfg_deep_copy.h"
#include "sdfg/symbolic/utils.h"
#include "sdfg/visitor/structured_sdfg_visitor.h"
#include "symengine/subs.h"

namespace sdfg::passes {

class LoopIndirectAccessFinder : public analysis::BaseUserVisitor {
    analysis::LoopAnalysis& loop_analysis_;
    std::unordered_map<analysis::ElementId, std::unique_ptr<FusionLoopCandidate>>& fuse_candidates_;
    struct LoopEntry {
        ControlFlowNode* loop;
        analysis::LocalLoopInfo::LoopType type;
        std::unordered_set<std::string> indvars;
    };
    std::deque<LoopEntry> loop_stack_;

    LoopEntry* get_current_loop() {
        if (loop_stack_.empty()) {
            return nullptr;
        }
        return &loop_stack_.back();
    }

public:
    LoopIndirectAccessFinder(
        analysis::LoopAnalysis& loops,
        std::unordered_map<analysis::ElementId, std::unique_ptr<FusionLoopCandidate>>& fuse_candidates
    )
        : loop_analysis_(loops), fuse_candidates_(fuse_candidates) {}

    bool visit(sdfg::structured_control_flow::For& node) override {
        loop_stack_.emplace_back(&node, analysis::LocalLoopInfo::LoopType::For);
        loop_stack_.back().indvars.emplace(node.indvar()->get_name());
        auto res = ActualStructuredSDFGVisitor::visit(node);
        loop_stack_.pop_back();
        return res;
    }

    bool visit(sdfg::structured_control_flow::While& node) override {
        loop_stack_.emplace_back(&node, analysis::LocalLoopInfo::LoopType::For);
        auto res = ActualStructuredSDFGVisitor::visit(node);
        loop_stack_.pop_back();
        return res;
    }

    bool visit(sdfg::structured_control_flow::Map& node) override {
        loop_stack_.emplace_back(&node, analysis::LocalLoopInfo::LoopType::For);
        loop_stack_.back().indvars.emplace(node.indvar()->get_name());
        auto res = ActualStructuredSDFGVisitor::visit(node);
        loop_stack_.pop_back();
        return res;
    }

    void use_as_symbol_read(
        const std::string& container,
        const ControlFlowNode* node,
        const Element* user,
        SymbolReadLocation loc,
        int loc_index,
        symbolic::Expression expr
    ) override {}

    void use_as_dst_node(
        const std::string& container,
        const data_flow::AccessNode& node,
        const data_flow::Memlet& edge,
        const Block& block
    ) override {
        auto current = get_current_loop();
        if (current && edge.is_dst_pointed_to_write()) {
            auto cand_it = fuse_candidates_.find(current->loop->element_id());
            if (cand_it != fuse_candidates_.end()) {
                auto& cand = *cand_it->second;
                auto arg_it = cand.args.find(container);
                if (arg_it != cand.args.end()) {
                    auto& fusion_arg = arg_it->second;
                    if (fusion_arg.subset.has_value() &&
                        !symbolic::vectors_of_expressions_match(fusion_arg.subset.value(), edge.subset())) {
                        fusion_arg.not_understood = true;
                    } else if (!fusion_arg.subset.has_value() && !fusion_arg.not_understood) {
                        fusion_arg.subset = edge.subset();
                    }
                }
            }
        }
    }
    void use_as_return_src(const std::string& container, const Return& ret) override {}
    /**
     * Dangerous, if somebody builds a value derived from indvar and then uses that for addressing we would not notice.
     * But normally those should be folded into the accesses
     */
    void use_as_src_node(
        const std::string& container,
        const data_flow::AccessNode& node,
        const data_flow::Memlet& edge,
        const Block& block
    ) override {
        auto current = get_current_loop();
        if (current && edge.is_src_pointed_to_read()) {
            auto cand_it = fuse_candidates_.find(current->loop->element_id());
            if (cand_it != fuse_candidates_.end()) {
                auto& cand = *cand_it->second;
                auto arg_it = cand.args.find(container);
                if (arg_it != cand.args.end()) {
                    auto& fusion_arg = arg_it->second;
                    if (fusion_arg.subset.has_value() &&
                        !symbolic::vectors_of_expressions_match(fusion_arg.subset.value(), edge.subset())) {
                        fusion_arg.not_understood = true;
                    } else if (!fusion_arg.subset.has_value() && !fusion_arg.not_understood) {
                        fusion_arg.subset = edge.subset();
                    }
                }
            }
        }
    }
    void use_as_symbol_write(
        const symbolic::Symbol& container, const ControlFlowNode* node, const Element* user, SymbolWriteLocation loc
    ) override {}
};

FusionLoopCandidate* MapFusionByDomainPass::State::get_next_level_map_stack(FusionLoopCandidate& current) {
    auto& children = loop_analysis->children(current.loop);
    if (children.empty()) {
        return nullptr;
    }

    auto* next = children.at(0);
    return fuse_candidates.at(next->element_id()).get();
}

FusionLoopCandidate* MapFusionByDomainPass::State::get_parent(FusionLoopCandidate& current) {
    auto* parent = loop_analysis->parent_loop(current.loop);
    if (!parent) {
        return nullptr;
    }
    auto it = fuse_candidates.find(parent->element_id());
    if (it != fuse_candidates.end()) {
        return it->second.get();
    } else {
        return nullptr;
    }
}

bool MapFusionByDomainPass::run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    auto loop_ana = std::make_unique<analysis::LoopAnalysis>(builder.subject());
    loop_ana->run(analysis_manager);

    State state(builder, analysis_manager, std::move(loop_ana));

    auto& assumption_analysis = analysis_manager.get<analysis::AssumptionsAnalysis>();
    auto& arguments_analysis = analysis_manager.get<analysis::ArgumentsAnalysis>();

    for (auto* control_flow_node : state.loop_analysis->loops()) {
        if (auto* map = dynamic_cast<Map*>(control_flow_node)) {
            auto& indvar = map->indvar();
            auto& assumpts = assumption_analysis.get(map->root(), true);
            auto* indvar_boundaries = find_indvar_boundaries(indvar, assumpts);

            if (indvar_boundaries && !indvar_boundaries->tight_lower_bound().is_null() &&
                !indvar_boundaries->tight_upper_bound().is_null() && !indvar_boundaries->map().is_null()) {
                auto& args = arguments_analysis.arguments(analysis_manager, *map);
                auto cand = std::make_unique<FusionLoopCandidate>(map, indvar_boundaries);
                for (auto [name, arg] : args) {
                    cand->args.emplace(name, arg);
                }
                state.fuse_candidates[control_flow_node->element_id()] = std::move(cand);
            }
        }
    }

    LoopIndirectAccessFinder indirect_access_finder(*state.loop_analysis, state.fuse_candidates);
    indirect_access_finder.dispatch(builder.subject().root());

    const std::string* dir = nullptr;
    if (dump_infos) {
        dir = builder.subject().metadata_if_exists("output_dir");
        if (dir) {
            state.loop_analysis->dump_to_file(std::filesystem::path(*dir) / "loop_infos.pre-fusion.json");
        }
    }

    MapFusionHandler handler(state);

    NeighboringPatternVisitor v(handler);
    v.dispatch(builder.subject().root());

    if (dir) {
        state.loop_analysis->dump_to_file(std::filesystem::path(*dir) / "loop_infos.post-fusion.json");
    }

    return state.fused_count;
}

const symbolic::Assumption* MapFusionByDomainPass::
    find_indvar_boundaries(const symbolic::Symbol& indvar, const symbolic::Assumptions& assumptions) {
    auto it = assumptions.find(indvar);
    if (it != assumptions.end()) {
        return &it->second;
    }

    return nullptr;
}

MapFusionHandler::MapFusionHandler(MapFusionByDomainPass::State& state) : state_(state) {}

PatternHandler::MatchResult MapFusionHandler::fuse_contents(
    ControlFlowNode* first_top,
    FusionLoopCandidate* first_current,
    FusionLoopCandidate* second_innermost,
    const symbolic::ExpressionMapping& indvar_mapping,
    Sequence& target_root,
    bool can_remove_original
) {
    Sequence* append_root = nullptr;
    if (target_root.size() == 0) {
        // target seq is empty, so we can just append to it
        append_root = &target_root;
    } else {
        // there currently is no way to prepend-copy with replace, so add to new sequence,
        // replace on it, then flatten it into the existing
        append_root = &state_.builder.add_sequence_before(target_root, target_root.at(0).first, {}, {});
    }

    std::optional<std::unordered_map<const ControlFlowNode*, const ControlFlowNode*>> copy_mapping;
    if (can_remove_original) {
        state_.builder.move_children(first_current->loop->root(), *append_root);
    } else {
        deepcopy::StructuredSDFGDeepCopy copier(state_.builder, *append_root, first_current->loop->root());
        copy_mapping = copier.insert();
    }

    update_fused_seq(*append_root, indvar_mapping);

    if (append_root != &target_root) { // need to fixup / flatten the copied sequence into the target sequence
        state_.builder.move_children(*append_root, target_root, 0);
        state_.builder.remove_from_parent(*append_root);
        append_root = nullptr;
    }

    update_candidate_state(first_top, first_current, second_innermost, indvar_mapping);

    auto& first_children = state_.loop_analysis->children(first_current->loop);
    bool keep_visiting_second = !state_.loop_analysis->children(second_innermost->loop).empty() ||
                                !first_children.empty();
    auto& prev_local_info = state_.loop_analysis->loop_info_local(first_current->loop);
    if (can_remove_original) {
        for (auto& child : first_children) {
            state_.loop_analysis->moved_loop(child, second_innermost->loop, true);
        }
        state_.loop_analysis->added_local_contents(
            second_innermost->loop, prev_local_info.contains_side_effects, prev_local_info.contains_non_perfectly_nested
        );
    } else {
        for (auto& child : first_children) {
            state_.loop_analysis->copied_loop(
                child,
                second_innermost->loop,
                const_cast<structured_control_flow::ControlFlowNode*>(copy_mapping->at(child)),
                true
            );
        }
        state_.loop_analysis->added_local_contents(
            second_innermost->loop, prev_local_info.contains_side_effects, prev_local_info.contains_non_perfectly_nested
        );
    }

    bool removed_first = false;
    if (can_remove_original) {
        state_.loop_analysis->removed_loop(first_top);
        state_.builder.remove_from_parent(*first_top);
        removed_first = true;
    }

    state_.fused_count++;

    // if there are further loops inside the now fused body, visit those as well
    return {.removed_first = removed_first, .visit_second_body = keep_visiting_second};
}

PatternHandler::MatchResult MapFusionHandler::match(Map& first, Map& second, bool no_uses_between) {
    auto first_it = state_.fuse_candidates.find(first.element_id());
    if (first_it == state_.fuse_candidates.end()) {
        return {};
    }
    FusionLoopCandidate* first_current = nullptr;
    FusionLoopCandidate* first_next = first_it->second.get();

    auto second_it = state_.fuse_candidates.find(second.element_id());
    if (second_it == state_.fuse_candidates.end()) {
        return {};
    }
    FusionLoopCandidate* second_current = nullptr;
    FusionLoopCandidate* second_next = second_it->second.get();

    SymEngine::map_basic_basic indvar_mapping;
    bool level_match;
    int current_level = -1;
    int last_matched_level = -1;
    auto second_loop_info = state_.loop_analysis->loop_info(&second);
    auto first_map_stack_depth = state_.loop_analysis->loop_info(&first).map_stack_depth;
    auto max_map_stack_depth = std::min(first_map_stack_depth, second_loop_info.map_stack_depth) - 1;
    bool uneven = first_map_stack_depth != second_loop_info.map_stack_depth;

    do {
        ++current_level;
        indvar_mapping[first_next->loop->indvar()] = second_next->loop->indvar();

        level_match = this->loop_match(*first_next, *second_next, indvar_mapping);
        if (level_match) {
            auto res = this->check_ins_outs(*first_next, *second_next, indvar_mapping);
            level_match = res.no_conflicts;
            if (uneven && !res.overlap) { // heuristic: do not fuse if there is no memory shared between them
                return {};
            }
        }
        bool go_deeper = false;
        if (level_match) {
            last_matched_level = current_level;
            first_current = first_next;
            second_current = second_next;
            if (current_level < max_map_stack_depth) {
                go_deeper = true;
            }
        }
        if (go_deeper) {
            first_next = state_.get_next_level_map_stack(*first_next);
            second_next = state_.get_next_level_map_stack(*second_next);
        } else {
            first_next = nullptr;
            second_next = nullptr;
        }
    } while (first_next && second_next);

    if (last_matched_level >= 0) {
        DEBUG_PRINTLN(
            "Fusing map stack (" << last_matched_level + 1 << " lvls): #" << first.element_id() << " | #"
                                 << first_current->loop->element_id() << ", #" << second.element_id() << " | #"
                                 << second_current->loop->element_id()
        );

        auto& target_root = second_current->loop->root();
        return fuse_contents(&first, first_current, second_current, indvar_mapping, target_root, no_uses_between);
    }

    return {};
}

bool MapFusionHandler::
    loop_match(FusionLoopCandidate& first, FusionLoopCandidate& second, SymEngine::map_basic_basic& canonical_indvars) {
    // if (first.incompatible || second.incompatible) {
    //     return false;
    // }

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

data_flow::Subset updated_subset(const data_flow::Subset& subset, const symbolic::ExpressionMapping& canonical_indvars) {
    std::vector<symbolic::Expression> updated_subset(subset.size());
    for (auto i = 0; i < subset.size(); i++) {
        updated_subset[i] = symbolic::subs(subset[i], canonical_indvars);
    }
    return std::move(updated_subset);
}

void MapFusionHandler::update_candidate_state(
    ControlFlowNode* first_top,
    FusionLoopCandidate* first_current,
    FusionLoopCandidate* second_current,
    const symbolic::ExpressionMapping& canonical_indvars
) {
    auto terminate_at = state_.loop_analysis->parent_loop(first_top);
    do {
        auto& second_args = second_current->args;
        for (auto& [name, arg] : first_current->args) {
            if (first_current->loop->indvar()->get_name() == name) {
                // skip the induction variable, we already know those match and they would not be useful to track for
                // the next levels up
                continue;
            }
            auto it = second_args.find(name);
            if (it != second_args.end()) {
                auto& second_arg = it->second;
                if (second_arg.subset.has_value() && arg.subset.has_value() &&
                    !symbolic::vectors_of_expressions_match(
                        second_arg.subset.value(), arg.subset.value(), canonical_indvars
                    )) {
                    second_arg.not_understood = true;
                } else if (!second_arg.subset.has_value() && arg.subset.has_value()) {
                    second_arg.subset = updated_subset(arg.subset.value(), canonical_indvars);
                }
                second_arg.not_understood |= arg.not_understood;
                second_arg.arg.merge(arg.arg);
            } else {
                auto [it, fresh] = second_args.emplace(name, arg.arg);
                if (arg.subset.has_value()) {
                    it->second.subset = updated_subset(arg.subset.value(), canonical_indvars);
                    it->second.not_understood = arg.not_understood;
                }
            }
        }
        first_current = state_.get_parent(*first_current);
        second_current = state_.get_parent(*second_current);
    } while (first_current && first_current->loop != terminate_at);
}

MapFusionHandler::InOutCheckResult MapFusionHandler::check_ins_outs(
    const FusionLoopCandidate& first_candidate,
    const FusionLoopCandidate& second_candidate,
    symbolic::ExpressionMapping& canonical_indvars
) {
    auto& first_args = first_candidate.args;
    auto& second_args = second_candidate.args;

    bool overlap = false;

    for (auto& [name, prod_meta] : first_args) {
        auto cons_it = second_args.find(name);
        if (cons_it != second_args.end()) {
            auto& cons_meta = cons_it->second;
            if (prod_meta.arg.is_input &&
                cons_meta.arg.is_output /* && (!prod_meta.saw_access_locally() || cons_meta.saw_access_locally())*/) {
                // possibly consumer influencing producer in unsafe ways
                // if producer does not write the arg itself, this level should still be fine, the conflict would be
                // deeper down
                return {false, overlap};
            } else if (prod_meta.arg.is_output && cons_meta.arg.is_input) {
                overlap = true;
                if (prod_meta.not_understood || cons_meta.not_understood) {
                    return {false, overlap};
                }

                if (prod_meta.subset.has_value() && cons_meta.subset.has_value()) {
                    if (!symbolic::vectors_of_expressions_match(
                            prod_meta.subset.value(), cons_meta.subset.value(), canonical_indvars
                        )) {
                        return {false, overlap};
                    }
                }
            }
        }
    }

    return {true, overlap};
}

void MapFusionHandler::update_fused_seq(Sequence& sequence, const symbolic::ExpressionMapping& replacements) {
    sequence.replace(replacements);
}

bool FusionArg::saw_access_locally() const { return not_understood || subset.has_value(); }

void FusionLoopCandidate::non_indvar_writes() { this->incompatible = true; }

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

            auto result = handler_.match(*first, *second, true);

            if (!result.removed_first) {
                dispatch(child_node);
            }
            if (result.visit_second_body) {
                auto* second_updated_child = result.second_root_replacement ? result.second_root_replacement : second;
                dispatch(*second_updated_child);
            }
            if (result.removed_first) {
                // do not increment i, we can use at as next firs
                continue;
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
