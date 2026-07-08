#include "sdfg/passes/map_fusion_by_domain_pass.h"

#include "sdfg/analysis/assumptions_analysis.h"
#include "sdfg/analysis/base_user_visitor.h"
#include "sdfg/data_flow/library_nodes/stdlib/malloc.h"
#include "sdfg/deepcopy/structured_sdfg_deep_copy.h"
#include "sdfg/structured_sdfg.h"
#include "sdfg/symbolic/utils.h"
#include "sdfg/visitor/structured_sdfg_visitor.h"
#include "sdfg/visualizer/dot_visualizer.h"
#include "symengine/subs.h"

namespace sdfg::passes {

static const symbolic::Symbol lower_indvar_placeholder = symbolic::symbol("__lower_it");

class LoopIndirectAccessFinder : public analysis::BaseUserVisitor {
    const StructuredSDFG& sdfg_;
    analysis::LoopAnalysis& loop_analysis_;
    std::unordered_map<analysis::ElementId, std::unique_ptr<FusionLoopCandidate>>& fuse_candidates_;
    struct LoopEntry {
        ControlFlowNode* loop;
        analysis::LocalLoopInfo::LoopType type;
        FusionLoopCandidate& fusion_candidate;
        std::unordered_set<std::string> indvars;
    };
    std::deque<LoopEntry> loop_stack_;

    LoopEntry* get_current_loop() {
        if (loop_stack_.empty()) {
            return nullptr;
        }
        return &loop_stack_.back();
    }

    /**
     * Merge a newly observed access (its `subset` and `not_understood` flag) into the FusionArg `into`.
     * If `into` already tracks a different subset, we can no longer describe the access with a single
     * subset and mark it not_understood. Returns true if `into` was modified.
     */
    static bool merge_fusion_arg_props_into(
        FusionArg& into,
        const std::optional<data_flow::Subset>& subset,
        bool not_understood,
        const symbolic::ExpressionMapping& lower_indvars
    ) {
        bool updated = false;
        if (into.subset.has_value() && subset.has_value() &&
            !symbolic::vectors_of_expressions_match(into.subset.value(), subset.value(), lower_indvars)) {
            if (!into.not_understood) {
                into.not_understood = true;
                updated = true;
            }
        } else if (!into.subset.has_value() && subset.has_value() && !into.not_understood) {
            into.subset = subset.value();
            updated = true;
        }
        if (not_understood && !into.not_understood) {
            into.not_understood = true;
            updated = true;
        }
        return updated;
    }

    static bool merge_fusion_arg_props_into(
        FusionArg& into, const FusionArg& from, const symbolic::ExpressionMapping& lower_indvars
    ) {
        return merge_fusion_arg_props_into(into, from.subset, from.not_understood, lower_indvars);
    }

public:
    LoopIndirectAccessFinder(
        const StructuredSDFG& sdfg,
        analysis::LoopAnalysis& loops,
        std::unordered_map<analysis::ElementId, std::unique_ptr<FusionLoopCandidate>>& fuse_candidates
    )
        : sdfg_(sdfg), loop_analysis_(loops), fuse_candidates_(fuse_candidates) {}

    bool visit(sdfg::structured_control_flow::For& node) override {
        auto cand_it = fuse_candidates_.find(node.element_id());
        bool is_relevant_loop = cand_it != fuse_candidates_.end();
        if (is_relevant_loop) {
            loop_stack_.emplace_back(&node, analysis::LocalLoopInfo::LoopType::For, *cand_it->second.get());
            loop_stack_.back().indvars.emplace(node.indvar()->get_name());
        }
        auto res = ActualStructuredSDFGVisitor::visit(node);

        if (is_relevant_loop) {
            loop_stack_.pop_back();
        }

        return res;
    }

    bool visit(sdfg::structured_control_flow::While& node) override {
        // far from being supported as fuse candidates, so do the normal stuff
        auto res = ActualStructuredSDFGVisitor::visit(node);
        return res;
    }

    bool visit(sdfg::structured_control_flow::Map& node) override {
        auto cand_it = fuse_candidates_.find(node.element_id());
        bool is_relevant_loop = cand_it != fuse_candidates_.end();
        if (is_relevant_loop) {
            loop_stack_.emplace_back(&node, analysis::LocalLoopInfo::LoopType::Map, *cand_it->second.get());
            loop_stack_.back().indvars.emplace(node.indvar()->get_name());
        }
        auto res = ActualStructuredSDFGVisitor::visit(node);
        if (is_relevant_loop) {
            loop_stack_.pop_back();
        }
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

    void found_indirect_arg_access(
        const std::string& container, const data_flow::Memlet& edge, LoopEntry* current, bool is_write
    ) {
        auto& cand = current->fusion_candidate;
        auto arg_it = cand.args.find(container);
        if (arg_it != cand.args.end()) {
            auto& fusion_arg = arg_it->second;
            merge_fusion_arg_props_into(fusion_arg, edge.subset(), false, symbolic::ExpressionMapping());
            // propagate_arg_up(current, container, fusion_arg); // may include lower level indvars, in which case
            // it will block everything
        }
    }

    /**
     * Push an access change observed at the innermost loop (back of `loop_stack_`) up through its
     * enclosing loops, merging it into every ancestor that still tracks `arg`. Stops once the root
     * (front of the stack) is reached or a loop no longer contains the arg.
     */
    void propagate_arg_up(LoopEntry* start, const std::string& arg, FusionArg& changes) {
        if (loop_stack_.size() < 2) {
            return; // no enclosing loops above the current one
        }

        symbolic::ExpressionMapping lower_indvars;
        for (auto& indvar : start->indvars) {
            lower_indvars[symbolic::symbol(indvar)] = lower_indvar_placeholder;
        }

        // Start at the direct parent of the current (innermost) loop and walk toward the root.
        for (auto it = loop_stack_.end() - 2;; --it) {
            auto cand_it = fuse_candidates_.find(it->loop->element_id());
            if (cand_it == fuse_candidates_.end()) {
                break; // loop is not a tracked candidate, fusing will never cross that boundary anyway
            }
            auto& args = cand_it->second->args;
            auto arg_it = args.find(arg);
            if (arg_it == args.end()) {
                break; // arg does not reach this loop -> nothing higher up needs it either
            }

            merge_fusion_arg_props_into(arg_it->second, changes, lower_indvars);

            if (it == loop_stack_.begin()) {
                break; // reached the root of the stack
            }
        }
    }

    void use_as_dst_node(
        const std::string& container,
        const data_flow::AccessNode& node,
        const data_flow::Memlet& edge,
        const Block& block
    ) override {
        auto current = get_current_loop();
        if (current && edge.is_dst_pointed_to_write()) {
            found_indirect_arg_access(container, edge, current, true);
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
        if (current && (edge.is_src_address_leak() || edge.is_src_pointed_to_address_leak(sdfg_.type(container)))) {
            current->fusion_candidate.aliasing_encountered();
        } else if (current && edge.is_src_pointed_to_read()) {
            found_indirect_arg_access(container, edge, current, false);
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

    LoopIndirectAccessFinder indirect_access_finder(builder.subject(), *state.loop_analysis, state.fuse_candidates);
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
    auto first_elem_id = first_current->loop->element_id();

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

    auto first_children = state_.loop_analysis->children(first_current->loop);
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

    // static auto count = 0;
    // std::filesystem::path dir = state_.builder.subject().metadata("output_dir");
    // visualizer::DotVisualizer::writeToFile(
    //     state_.builder.subject(),
    //     dir /
    //         ("map_fusion_by_domain_pass_dump_" + std::to_string(count++) + "_" + std::to_string(first_elem_id) +
    //         ".dot")
    // );

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
    int current_level = -1;
    int last_matched_level = -1;
    auto first_info = state_.loop_analysis->loop_info(&first);
    auto second_info = state_.loop_analysis->loop_info(&second);

    // Skip if both have side effects
    if (first_info.has_side_effects && second_info.has_side_effects) {
        return {};
    }

    auto first_max_stack_depth = first_info.map_stack_depth - 1;
    auto second_max_stack_depth = second_info.map_stack_depth - 1;
    bool more_first = true;
    bool more_second = true;
    bool fusing_option = true;
    bool data_fusible;

    // descend the map stacks down. Last level on which everything matches is the one we can fuse
    // if one map stack ends, but the other one keeps going, we still need to descend the one that keeps going.
    // at this point, these are no longer candidates for fusing, but we can only fuse the last level we found,
    // if we find no subset conflicts. Because if not all subsets of the same structures match,
    // we cannot guarantee correctness

    do {
        ++current_level;
        bool uneven = !more_first || !more_second;
        if (fusing_option) {
            auto insertion = indvar_mapping.insert({first_next->loop->indvar(), second_next->loop->indvar()});
            assert(insertion.second);
            fusing_option = this->loop_match(*first_next, *second_next, indvar_mapping);
            if (!fusing_option) {
                indvar_mapping.erase(insertion.first);
            }
        }
        auto res = this->check_ins_outs(*first_next, *second_next, indvar_mapping);
        if (!res.no_conflicts) {
            // will occur on data-dependencies (from consumer to producer) or on subset mismatches
            fusing_option = false;
        }
        if (first_max_stack_depth != second_max_stack_depth && !res.overlap) { // heuristic: do not fuse if there is no
                                                                               // memory shared between uneven
                                                                               // candidates
            return {};
        }
        if (res.subset_mismatch) { // If subsets mismatch on any level, we cannot guarantee correctness without much
                                   // more
            // extensive checks, like done by the existing map_fusion
            // Future work: let the previous map_fusion check if it can prove non-conflicting on a more granular level
            return {};
        }

        if (fusing_option) {
            last_matched_level = current_level;
            first_current = first_next;
            second_current = second_next;
        }
        more_first = current_level < first_max_stack_depth;
        more_second = current_level < second_max_stack_depth;
        if (more_first) {
            first_next = state_.get_next_level_map_stack(*first_next);
        }
        if (more_second) {
            second_next = state_.get_next_level_map_stack(*second_next);
        }
        fusing_option &= more_first && more_second;
    } while (more_first || more_second);

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
    check_no_overlap(const Map& map, const Map& second, const std::unordered_set<std::string>& skipped_containers) {
    auto& first_cand = *state_.fuse_candidates.at(map.element_id());
    auto& second_cand = *state_.fuse_candidates.at(second.element_id());
    for (auto& arg : first_cand.args) {
        if (skipped_containers.contains(arg.first)) {
            return false;
        }
    }
    return true;
}

bool MapFusionHandler::
    loop_match(FusionLoopCandidate& first, FusionLoopCandidate& second, SymEngine::map_basic_basic& canonical_indvars) {
    if (first.incompatible || second.incompatible) {
        return false;
    }

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

void MapFusionHandler::update_child_candidate_states(FusionLoopCandidate* top, const symbolic::ExpressionMapping& replace) {
    auto& info = state_.loop_analysis->loop_info_local(top->loop);
    auto& candidates = state_.fuse_candidates;

    auto& by_id = state_.loop_analysis->loops_in_pre_order();
    for (auto i = info.loop_id; i <= info.last_child_id; ++i) {
        auto* child_loop = by_id.at(i);
        auto cand_it = candidates.find(child_loop->element_id());
        if (cand_it != candidates.end()) {
            auto& child_cand = *cand_it->second;
            child_cand.replace(replace);
        }
    }
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
    update_child_candidate_states(first_current, canonical_indvars);

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
            if (prod_meta.arg.is_input && cons_meta.arg.is_output) {
                // there could be conflicts here. So for now, abort.
                // Future Work: if both were to strictly match indvars (or never match other iterations),
                // it would never be a conflict
                return {false, overlap};
            } else if (prod_meta.arg.is_output && cons_meta.arg.is_input) {
                overlap = true;
                if (prod_meta.not_understood || cons_meta.not_understood) {
                    return {false, overlap, true};
                }

                if (prod_meta.subset.has_value() && cons_meta.subset.has_value()) {
                    if (!symbolic::vectors_of_expressions_match(
                            prod_meta.subset.value(), cons_meta.subset.value(), canonical_indvars
                        )) {
                        return {false, overlap, true};
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

void FusionLoopCandidate::aliasing_encountered() { this->incompatible = true; }

void FusionLoopCandidate::replace(const symbolic::ExpressionMapping& mapping) {
    for (auto& [name, arg] : args) {
        if (arg.subset.has_value()) {
            arg.subset = updated_subset(arg.subset.value(), mapping);
        }
    }
}

NeighboringPatternVisitor::NeighboringPatternVisitor(PatternHandler& handler) : handler_(handler) {}

bool NeighboringPatternVisitor::visit(sdfg::structured_control_flow::Sequence& node) {
    if (node.size() < 2) { // impossible to find a match, just descend into it
        return ActualStructuredSDFGVisitor::visit(node);
    }

    // Iterate over sequence looking for consecutive (Map, StructuredLoop) pairs
    size_t i = 0;
    structured_control_flow::ControlFlowNode* override_last = nullptr;
    while (i < node.size()) {
        auto& child_node = node.at(i).first;
        auto* first = dyn_cast<structured_control_flow::Map*>(&child_node);
        if (!first) {
            i++;
            dispatch(child_node);
            continue;
        }
        if (first->root().size() == 0) {
            i++;
            continue;
        }

        Map* second = nullptr;

        if (i + 1 < node.size()) {
            second = dyn_cast<structured_control_flow::Map*>(&node.at(i + 1).first);
            if (second) {
                if (second->root().size() == 0) {
                    i++;
                    continue;
                }

                auto result = handler_.match(*first, *second, true);

                if (!result.removed_first) {
                    dispatch(child_node);
                }
                if (result.visit_second_body) {
                    auto* second_updated_child = result.second_root_replacement ? result.second_root_replacement
                                                                                : second;
                    dispatch(*second_updated_child);
                }
                if (result.removed_first) {
                    // do not increment i, we can use at as next firs
                    continue;
                }
            } else if (i + 2 < node.size()) {
                auto* mid_block = dyn_cast<structured_control_flow::Block*>(&node.at(i + 1).first);
                bool skippable = false;
                std::unordered_set<std::string> skipped_containers;
                if (mid_block) {
                    if (mid_block->dataflow().nodes().empty()) {
                        skippable = true;
                    } else if (mid_block->is_a_library_node<stdlib::MallocNode>()) {
                        for (auto& data_flow_node : mid_block->dataflow().nodes()) {
                            if (auto* container = dynamic_cast<data_flow::AccessNode*>(&data_flow_node)) {
                                skipped_containers.emplace(container->data());
                            }
                        }
                        skippable = true;
                    }
                }
                if (skippable) {
                    second = dyn_cast<structured_control_flow::Map*>(&node.at(i + 2).first);
                    if (second) {
                        if (second->root().size() == 0) {
                            i += 2;
                            continue;
                        }

                        if (!handler_.check_no_overlap(*first, *second, skipped_containers)) {
                            i += 2;
                            continue;
                        }

                        auto result = handler_.match(*first, *second, true);

                        if (!result.removed_first) {
                            dispatch(child_node);
                        }
                        if (result.visit_second_body) {
                            auto* second_updated_child = result.second_root_replacement ? result.second_root_replacement
                                                                                        : second;
                            dispatch(*second_updated_child);
                        }
                        if (result.removed_first) {
                            i += 1; // skip the block, retry with second as next first
                            continue;
                        }
                        // we visited [first, skipped, second] successfully, without shifting indices, move to second as
                        // new first
                        i += 2;
                        continue;
                    } else {
                        // we know i+1 is worthless, so skip it
                        i += 2;
                        continue;
                    }
                }
            }
        } else {
            dispatch(child_node);
        }
        i++;
    }

    return true;
}

} // namespace sdfg::passes
