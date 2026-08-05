#include "sdfg/passes/loop_fusion/loop_fusion_pass.h"

#include "../../../../sdfg/include/sdfg/symbolic/assumptions.h"
#include "sdfg/analysis/assumptions_analysis.h"
#include "sdfg/analysis/base_user_visitor.h"
#include "sdfg/data_flow/library_nodes/stdlib/malloc.h"
#include "sdfg/deepcopy/structured_sdfg_deep_copy.h"
#include "sdfg/structured_sdfg.h"
#include "sdfg/symbolic/utils.h"
#include "sdfg/visitor/structured_sdfg_visitor.h"
#include "sdfg/visualizer/dot_visualizer.h"
#include "symengine/subs.h"

namespace sdfg::passes::loop_fusion {

static const symbolic::Symbol lower_indvar_placeholder = symbolic::symbol("__lower_it");

static inline constexpr bool DUMP_ASSUMPTIONS = false;
static inline constexpr bool DUMP_LOOP_INFOS = false;
static inline constexpr bool DUMP_GRAPHS = false;

static bool vectors_of_expressions_match(
    const std::vector<symbolic::Expression>& a,
    const std::vector<symbolic::Expression>& b,
    const symbolic::ExpressionMapping* replacements
) {
    if (replacements) {
        return symbolic::vectors_of_expressions_match(a, b, *replacements);
    } else {
        return symbolic::vectors_of_expressions_match(a, b);
    }
}

class LoopIndirectAccessFinder : public analysis::BaseUserVisitor {
    const StructuredSDFG& sdfg_;
    analysis::LoopAnalysis& loop_analysis_;
    std::unordered_map<analysis::ElementId, std::unique_ptr<FusionLoopCandidate>>& fuse_candidates_;
    struct LoopEntry {
        ControlFlowNode* loop;
        analysis::LocalLoopInfo::LoopType type;
        FusionLoopCandidate& fusion_candidate;
        symbolic::Expression indvar_placeholder; // SymEngine::Function(tight_lower_bound, tight_upper_bound, step,
                                                 // loop-level)
        std::unordered_set<std::string> indvars;
    };
    std::deque<LoopEntry> loop_stack_;

    LoopEntry* get_current_loop() {
        if (loop_stack_.empty()) {
            return nullptr;
        }
        return &loop_stack_.back();
    }

    static bool merge_fusion_arg_props_into(
        FusionArg& into,
        const std::optional<data_flow::Subset>& subset,
        bool not_understood,
        bool local_access,
        const symbolic::ExpressionMapping* lower_indvars = nullptr
    ) {
        bool updated = false;
        auto& target_access = local_access ? into.local_access : into.nested_access;

        if (target_access.common_subset.has_value() && subset.has_value() &&
            !vectors_of_expressions_match(target_access.common_subset.value(), subset.value(), lower_indvars)) {
            if (!target_access.subsets_conflict) {
                target_access.subsets_conflict = true;
                updated = true;
            }
        } else if (!target_access.common_subset.has_value() && subset.has_value() && !target_access.subsets_conflict) {
            target_access.common_subset = subset.value();
            updated = true;
        }
        if (not_understood && !target_access.subsets_conflict) {
            target_access.subsets_conflict = true;
            updated = true;
        }
        return updated;
    }

public:
    LoopIndirectAccessFinder(
        const StructuredSDFG& sdfg,
        analysis::LoopAnalysis& loops,
        std::unordered_map<analysis::ElementId, std::unique_ptr<FusionLoopCandidate>>& fuse_candidates
    )
        : sdfg_(sdfg), loop_analysis_(loops), fuse_candidates_(fuse_candidates) {}

    bool visit(sdfg::structured_control_flow::While& node) override {
        // far from being supported as fuse candidates, so do the normal stuff
        auto res = ActualStructuredSDFGVisitor::visit(node);
        return res;
    }

    static symbolic::Expression get_indvar_placeholder(FusionLoopCandidate& candidate, size_t level) {
        auto* indvar_bounds = candidate.indvar_boundaries;
        symbolic::Expression stride = symbolic::integer(1);
        if (symbolic::null_safe_eq(symbolic::sub(indvar_bounds->map(), indvar_bounds->symbol()), stride)) {
            return SymEngine::function_symbol(
                "indvar",
                {indvar_bounds->tight_lower_bound(), indvar_bounds->tight_upper_bound(), stride, symbolic::integer(level)
                }
            );
        } else {
            return {};
        }
    }

    bool handleStructuredLoop(sdfg::structured_control_flow::StructuredLoop& node) override {
        auto cand_it = fuse_candidates_.find(node.element_id());
        bool is_relevant_loop = cand_it != fuse_candidates_.end();
        if (is_relevant_loop) {
            auto type = is_a(node.type_id(), ElementType::Map) ? analysis::LocalLoopInfo::LoopType::Map
                                                               : analysis::LocalLoopInfo::LoopType::For;
            auto& candidate = *cand_it->second.get();
            loop_stack_.emplace_back(&node, type, candidate, get_indvar_placeholder(candidate, loop_stack_.size() - 1));
            loop_stack_.back().indvars.emplace(node.indvar()->get_name());
        }
        auto res = BaseUserVisitor::handleStructuredLoop(node);
        if (is_relevant_loop) {
            auto size = loop_stack_.size();
            if (size > 1) {
                auto& parent = loop_stack_.at(size - 2);
                propagate_indirect_accesses_up(loop_stack_.back(), parent);
            }
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

    static void found_indirect_arg_access(
        const std::string& container,
        const data_flow::Memlet& edge,
        const Block& block,
        LoopEntry* current,
        bool is_write
    ) {
        auto& cand = current->fusion_candidate;
        auto arg_it = cand.args.find(container);
        if (arg_it != cand.args.end()) {
            auto& fusion_arg = arg_it->second;
            fusion_arg.local_access.merge_into(const_cast<Block*>(&block), edge.subset(), false, is_write);
            std::optional<data_flow::Subset> generalized_subset_holder;
            const data_flow::Subset* generalized_subset = &edge.subset();
            if (!current->indvar_placeholder.is_null()) {
                generalized_subset_holder = symbolic::
                    substitute(*generalized_subset, {{cand.indvar_boundaries->symbol(), current->indvar_placeholder}});
                generalized_subset = &generalized_subset_holder.value();
            }

            fusion_arg.nested_access.merge_into(const_cast<Block*>(&block), *generalized_subset, false, is_write);
        }
    }

    static void propagate_indirect_accesses_up(LoopEntry& current, LoopEntry& parent) {
        auto& parent_cand = parent.fusion_candidate;
        std::optional<symbolic::ExpressionMapping> indvar_mapping;
        if (current.fusion_candidate.is_by_domain_candidate) {
            auto& indvar_bounds = current.fusion_candidate.indvar_boundaries;
            symbolic::Expression stride = symbolic::integer(1);
            if (symbolic::null_safe_eq(symbolic::sub(indvar_bounds->map(), indvar_bounds->symbol()), stride)) {
                indvar_mapping = symbolic::ExpressionMapping();
                indvar_mapping->emplace(
                    SymEngine::rcp_static_cast<const SymEngine::Basic>(indvar_bounds->symbol()),
                    current.indvar_placeholder
                );
            }
        }
        for (auto& [container, meta] : current.fusion_candidate.args) {
            auto arg_it = parent_cand.args.find(container);
            if (arg_it != parent_cand.args.end()) {
                auto& parent_arg = arg_it->second;
                parent_arg.nested_access
                    .merge_into(meta.nested_access, indvar_mapping.has_value() ? &*indvar_mapping : nullptr);
                parent_arg.nested_access
                    .merge_into(meta.local_access, indvar_mapping.has_value() ? &*indvar_mapping : nullptr);
            }
        }

        parent.fusion_candidate.nested_incompatible |= current.fusion_candidate.incompatible |
                                                       current.fusion_candidate.nested_incompatible;
    }


    void use_as_dst_node(
        const std::string& container,
        const data_flow::AccessNode& node,
        const data_flow::Memlet& edge,
        const Block& block
    ) override {
        auto current = get_current_loop();
        if (current && edge.is_dst_pointed_to_write()) {
            found_indirect_arg_access(container, edge, block, current, true);
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
            found_indirect_arg_access(container, edge, block, current, false);
        }
    }
    void use_as_symbol_write(
        const symbolic::Symbol& container, const ControlFlowNode* node, const Element* user, SymbolWriteLocation loc
    ) override {}
};

FusionLoopCandidate* LoopFusionPass::State::get_next_level_map_stack(FusionLoopCandidate& current) {
    auto& children = loop_analysis->children(current.loop);
    if (children.empty()) {
        return nullptr;
    }

    auto* next = children.at(0);
    return fuse_candidates.at(next->element_id()).get();
}

FusionLoopCandidate* LoopFusionPass::State::get_parent(FusionLoopCandidate& current) {
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

uint32_t LoopFusionPass::State::total_fused_count() const { return fused_by_domain_count + fused_by_access_count; }

std::ostream& operator<<(std::ostream& os, const symbolic::Expression& expr) {
    if (!expr.is_null()) {
        os << expr->__str__();
    } else {
        os << "null";
    }
    return os;
}

std::ostream& operator<<(std::ostream& os, const symbolic::Symbol& sym) {
    if (sym.is_null()) {
        os << "null";
    } else {
        os << sym->get_name();
    }
    return os;
}

std::ostream& operator<<(std::ostream& os, const symbolic::Assumption& assump) {
    os << "\t" << "const: " << (assump.constant() ? "true" : "false") << std::endl;
    os << "\t" << "map: " << assump.map() << std::endl;
    os << "\t" << "lower_bounds: " << assump.lower_bounds() << std::endl;
    os << "\t" << "upper_bounds: " << assump.upper_bounds() << std::endl;
    os << "\ttight_lower: " << assump.tight_lower_bound() << std::endl;
    os << "\ttight_upper: " << assump.tight_upper_bound() << std::endl;
    os << "\t" << "constraints: " << assump.constraints() << std::endl;
    return os;
}

std::ostream& operator<<(std::ostream& os, const symbolic::Assumptions& ass) {
    for (auto& [sym, as] : ass) {
        os << "\t" << sym << ":" << std::endl << as << std::endl;
    }
    return os;
}

LoopFusionPass::LoopFusionPass(const LoopFusionConfig& config) : config_(config) {}

LoopFusionPass::LoopFusionPass() = default;

bool LoopFusionPass::run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    auto loop_ana = std::make_unique<analysis::LoopAnalysis>(builder.subject());
    loop_ana->run(analysis_manager);

    static uint32_t run = 0;
    DEBUG_PRINTLN("LoopFusion pass #" << run);

    State state(builder, analysis_manager, std::move(loop_ana));
    state.run = run;
    run++;

    auto& assumption_analysis = analysis_manager.get<analysis::AssumptionsAnalysis>();
    auto& arguments_analysis = analysis_manager.get<analysis::ArgumentsAnalysis>();

    for (auto* control_flow_node : state.loop_analysis->loops()) {
        if (auto* loop = dyn_cast<StructuredLoop*>(control_flow_node)) {
            auto& indvar = loop->indvar();
            auto& assumpts = assumption_analysis.get(loop->root(), true);
            auto* indvar_boundaries = find_indvar_boundaries(indvar, assumpts);

            std::unique_ptr<FusionLoopCandidate> cand;

            bool tight = indvar_boundaries && !indvar_boundaries->tight_lower_bound().is_null() &&
                         !indvar_boundaries->tight_upper_bound().is_null() && !indvar_boundaries->map().is_null();
            bool is_map = is_a(loop->type_id(), ElementType::Map);
            cand = std::make_unique<FusionLoopCandidate>(loop, indvar_boundaries, assumpts, is_map, tight);
            auto& args = arguments_analysis.arguments(analysis_manager, *loop);
            for (auto [name, arg] : args) {
                cand->args.emplace(name, arg);
            }
            state.fuse_candidates[control_flow_node->element_id()] = std::move(cand);
        }
    }

    LoopIndirectAccessFinder indirect_access_finder(builder.subject(), *state.loop_analysis, state.fuse_candidates);
    indirect_access_finder.dispatch(builder.subject().root());

    const std::string* dir = nullptr;
    if (DUMP_LOOP_INFOS) {
        dir = builder.subject().metadata_if_exists("output_dir");
        if (dir) {
            state.loop_analysis->dump_to_file(std::filesystem::path(*dir) / "loop_infos.pre-fusion.json");
        }
    }

    LoopFusionHandler handler(config_, state);

    NeighboringPatternVisitor v(handler);
    v.dispatch(builder.subject().root());

    if (dir) {
        state.loop_analysis->dump_to_file(std::filesystem::path(*dir) / "loop_infos.post-fusion.json");
    }

    return state.total_fused_count();
}

const symbolic::Assumption* LoopFusionPass::
    find_indvar_boundaries(const symbolic::Symbol& indvar, const symbolic::Assumptions& assumptions) {
    auto it = assumptions.find(indvar);
    if (it != assumptions.end()) {
        return &it->second;
    }

    return nullptr;
}

LoopFusionHandler::LoopFusionHandler(const LoopFusionConfig& config, LoopFusionPass::State& state)
    : config_(config), state_(state), LoopFusionByAccessWorker(config.allow_init_hoist) {}

PatternHandler::MatchResult LoopFusionHandler::fuse_contents(
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
        append_root = &state_.builder.add_sequence_before(target_root, target_root.at(0), {});
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

    if constexpr (DUMP_GRAPHS) {
        auto dir = state_.builder.subject().metadata_if_exists("output_dir");
        if (dir) {
            std::filesystem::path pdir = *dir;
            visualizer::DotVisualizer::writeToFile(
                state_.builder.subject(),
                pdir / ("map_fusion_by_domain_pass_" + std::to_string(state_.run) + "_dump_" +
                        std::to_string(state_.fused_by_domain_count) + "_" +
                        std::to_string(second_innermost->loop->element_id()) + ".dot")
            );
        }
    }

    state_.fused_by_domain_count++;

    // if there are further loops inside the now fused body, visit those as well
    return {.removed_first = removed_first, .visit_second_body = keep_visiting_second};
}

analysis::LoopAnalysis& LoopFusionHandler::get_loop_analysis() { return *state_.loop_analysis; }

FusionLoopCandidate* LoopFusionHandler::get_fuse_candidate(StructuredLoop& loop) {
    return state_.fuse_candidates.at(loop.element_id()).get();
}

builder::StructuredSDFGBuilder& LoopFusionHandler::builder() { return state_.builder; }

void LoopFusionHandler::update_copied_leaf_contents_from_first_to_second(
    const Plan& plan, FusionLoopCandidate* first_current, FusionLoopCandidate* second_current
) {
    auto first_top = &plan.first;


    auto& fusion_regs = plan.fusion_candidates_;

    std::unordered_map<std::string, const loop_fusion::FusionRegCandidate*> cand_map;
    for (const auto& cand : fusion_regs) {
        cand_map[cand.container] = &cand;
    }

    update_candidate_args_up(first_top, first_current, second_current, [&](auto& name, auto& source_arg, auto& target_args) {
        auto cand_it = cand_map.find(name);
        if (cand_it != cand_map.end() && cand_it->second->integrated_rle) {
            // was RLEd, no longer exists
        } else {
            auto it = target_args.find(name);
            if (it != target_args.end()) {
                auto& second_arg = it->second;
                second_arg.local_access.merge_into(source_arg.local_access);
                second_arg.nested_access.merge_into(source_arg.nested_access);
                second_arg.arg.merge(source_arg.arg);
            } else {
                auto [it, fresh] = target_args.emplace(name, source_arg); // copy over
            }
        }
    });
}

PatternHandler::MatchResult LoopFusionHandler::match(StructuredLoop& first, StructuredLoop& second, bool no_uses_between) {
    auto first_it = state_.fuse_candidates.find(first.element_id());
    if (first_it == state_.fuse_candidates.end()) {
        return {};
    }
    FusionLoopCandidate* first_current = nullptr;
    FusionLoopCandidate* first_top = first_it->second.get();
    FusionLoopCandidate* first_next = first_top;

    auto second_it = state_.fuse_candidates.find(second.element_id());
    if (second_it == state_.fuse_candidates.end()) {
        return {};
    }
    FusionLoopCandidate* second_current = nullptr;
    FusionLoopCandidate* second_top = second_it->second.get();
    FusionLoopCandidate* second_next = second_top;

    SymEngine::map_basic_basic indvar_mapping;
    int current_level = -1;
    int last_matched_level = -1;
    auto first_info = state_.loop_analysis->loop_info(&first);
    auto second_info = state_.loop_analysis->loop_info(&second);

    // Skip if both have side effects
    if (first_info.has_side_effects && second_info.has_side_effects) {
        return {};
    }

    int32_t first_max_stack_depth = first_info.map_stack_depth - 1;
    int32_t second_max_stack_depth = second_info.map_stack_depth - 1;
    bool more_first = true;
    bool more_second = true;
    bool fusing_option = first_next->is_by_domain_candidate && second_next->is_by_domain_candidate;
    bool domains_match = true;
    bool both_map = first_next->is_map && second_next->is_map;
    bool no_overlap_candidate = false;

    // descend the map stacks down. Last level on which everything matches is the one we can fuse.
    // In case there are any further maps nested inside either one of the candidates, we then need to run verification
    // that there are no subset conflicts in those nested loops that prevent us from fusing the parents

    // descend evenly through candidates for fusion by domain.
    do {
        ++current_level;

        if (fusing_option) {
            auto insertion = indvar_mapping.insert({first_next->loop->indvar(), second_next->loop->indvar()});
            assert(insertion.second);
            fusing_option = this->loop_match(*first_next, *second_next, indvar_mapping);
            if (!fusing_option) {
                domains_match = false;
                indvar_mapping.erase(insertion.first);
            }
        } else {
            domains_match = false;
        }
        auto res = this->check_ins_outs(*first_next, *second_next, indvar_mapping, true, !both_map);
        if (!res.no_conflicts) {
            // will occur on data-dependencies (from consumer to producer) or on subset mismatches
            fusing_option = false;
        }
        if (!res.overlap) {
            // No shared memory between the 2 loops. This only makes sense if the iteration domain matches perfectly
            if (first_max_stack_depth != second_max_stack_depth) {
                // loop stacks are uneven
                return {};
            } else {
                no_overlap_candidate = true;
            }
        }
        if (res.subset_mismatch) { // If subsets mismatch on any level, we cannot guarantee correctness without much
                                   // more checks, so fusion-by-domain is out
            break;
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
    } while (more_first && more_second);

    if (last_matched_level >= 0) {
        if (no_overlap_candidate) {
            if (!state_.loop_analysis->children(first_current->loop).empty() ||
                !state_.loop_analysis->children(second_current->loop).empty()) {
                // we only would want to fuse no-overlap cases, if ALL dimensions match.
                // this means there can be no loops nested inside the level we are fusing
                return {};
            }
        }
        // we found a match for fusion-by-domain. In case there are nested loops we still need to verify they don't
        // conflict as well
        auto nested_check = this->check_ins_outs(*first_current, *second_current, indvar_mapping, false, !both_map);

        if (!nested_check.no_conflicts) {
            DEBUG_PRINTLN(
                "Should not have discovered fusion conflicts this late:"
                << last_matched_level + 1 << " lvls): #" << first.element_id() << " | #"
                << first_current->loop->element_id() << ", #" << second.element_id() << " | #"
                << second_current->loop->element_id()
            );
            return {};
        }
        if (!nested_check.subset_mismatch && config_.map_fusion_by_domain) {
            DEBUG_PRINTLN(
                "Fusing loop stack by-domain (" << last_matched_level + 1 << " lvls): #" << first.element_id() << " | #"
                                                << first_current->loop->element_id() << " -> #" << second.element_id()
                                                << " | #" << second_current->loop->element_id()
            );

            auto& target_root = second_current->loop->root();
            return fuse_contents(&first, first_current, second_current, indvar_mapping, target_root, no_uses_between);
        }
    }

    if (config_.map_fusion_by_access) {
        // we did not find an absolute blocker for fusing, but simple fusion by domain also did not work out, so try the
        // fusion-by-access
        StructuredLoop *first_loop, *second_loop;
        if (last_matched_level >= 0) {
            first_loop = first_current->loop;
            second_loop = second_current->loop;
        } else {
            first_loop = &first;
            second_loop = &second;
        }
        bool leaf_loops = state_.loop_analysis->children(first_loop).empty() &&
                          state_.loop_analysis->children(second_loop).empty();

        return try_complex_fuse_producer_into_consumer(
            *first_top, *second_top, no_uses_between, domains_match && leaf_loops
        );
    } else {
        return {};
    }
}

PatternHandler::MatchResult LoopFusionHandler::try_complex_fuse_producer_into_consumer(
    FusionLoopCandidate& first, FusionLoopCandidate& second, bool no_uses_between, bool domains_match
) {
    auto outcome = try_fuse_by_access(first, second, domains_match);

    if (outcome.fused) {
        if constexpr (DUMP_GRAPHS) {
            auto dir = state_.builder.subject().metadata_if_exists("output_dir");
            if (dir) {
                std::filesystem::path pdir = *dir;
                visualizer::DotVisualizer::writeToFile(
                    state_.builder.subject(),
                    pdir / ("map_fusion_by_domain_pass_" + std::to_string(state_.run) + "_dump_" +
                            std::to_string(state_.fused_by_domain_count) + "_" +
                            std::to_string(second.loop->element_id()) + ".dot")
                );
            }
        }

        state_.fused_by_access_count++;
    }

    return outcome.pattern_result;
}

bool LoopFusionHandler::check_no_overlap(
    const StructuredLoop& map, const StructuredLoop& second, const std::unordered_set<std::string>& skipped_containers
) {
    auto& first_cand = *state_.fuse_candidates.at(map.element_id());
    auto& second_cand = *state_.fuse_candidates.at(second.element_id());
    for (auto& arg : first_cand.args) {
        if (skipped_containers.contains(arg.first)) {
            return false;
        }
    }
    return true;
}

bool LoopFusionHandler::
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

void LoopFusionHandler::update_moved_candidate_states(FusionLoopCandidate* top, const symbolic::ExpressionMapping& replace) {
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

void LoopFusionHandler::update_candidate_state(
    ControlFlowNode* first_top,
    FusionLoopCandidate* first_current,
    FusionLoopCandidate* second_current,
    const symbolic::ExpressionMapping& canonical_indvars
) {
    // merge metadata from first -> second
    update_moved_candidate_states(first_current, canonical_indvars);

    update_candidate_args_up(
        first_top,
        first_current,
        second_current,
        [&](const std::string& name, FusionArg& source_arg, std::unordered_map<std::string, FusionArg>& target_args) {
            // skip the induction variable, we already know those match and they would not be useful to track for
            // the next levels up
            if (first_current->loop->indvar()->get_name() != name) {
                auto it = target_args.find(name);
                if (it != target_args.end()) {
                    auto& second_arg = it->second;
                    second_arg.local_access.merge_into(source_arg.local_access);
                    second_arg.nested_access.merge_into(source_arg.nested_access);
                    second_arg.arg.merge(source_arg.arg);
                } else {
                    auto [it, fresh] = target_args.emplace(name, source_arg); // copy over
                }
            }
        }
    );
}

void LoopFusionHandler::update_candidate_args_up(
    ControlFlowNode* first_top,
    FusionLoopCandidate* first_current,
    FusionLoopCandidate* second_current,
    const std::function<
        void(const std::string& name, FusionArg& source_arg, std::unordered_map<std::string, FusionArg>& target_args)>&
        action
) {
    auto terminate_at = state_.loop_analysis->parent_loop(first_top);
    do {
        auto& second_args = second_current->args;
        for (auto& [name, arg] : first_current->args) {
            action(name, arg, second_args);
        }
        // assumptions depend on loop and outer scopes, so they remain identical for the loop that is used as basis
        // (that the other gets inlined into) for now we always inline into 2nd.

        first_current = state_.get_parent(*first_current);
        second_current = state_.get_parent(*second_current);
    } while (first_current && first_current->loop != terminate_at);
}

LoopFusionHandler::InOutCheckResult LoopFusionHandler::check_ins_outs(
    const FusionLoopCandidate& first_candidate,
    const FusionLoopCandidate& second_candidate,
    symbolic::ExpressionMapping& canonical_indvars,
    bool local_not_nested,
    bool only_no_overlap
) {
    auto& first_args = first_candidate.args;
    auto& second_args = second_candidate.args;

    return check_ins_outs(first_args, second_args, canonical_indvars, local_not_nested, only_no_overlap);
}

LoopFusionHandler::InOutCheckResult LoopFusionHandler::check_ins_outs(
    const std::unordered_map<std::string, FusionArg>& first_args,
    const std::unordered_map<std::string, FusionArg>& second_args,
    symbolic::ExpressionMapping& canonical_indvars,
    bool local_not_nested,
    bool only_no_overlap
) {
    bool overlap = false;
    bool no_conflicts = true;
    bool subset_mismatch = false;

    for (auto& [name, prod_meta] : first_args) {
        auto cons_it = second_args.find(name);
        if (cons_it != second_args.end()) {
            auto& cons_meta = cons_it->second;
            if (prod_meta.arg.is_input && cons_meta.arg.is_output) {
                // there could be conflicts here. So for now, abort.
                // Future Work: if both were to strictly match indvars (or never match other iterations),
                // it would never be a conflict
                overlap = true;
                no_conflicts = false;
                continue;
            } else if (prod_meta.arg.is_output && cons_meta.arg.is_input) {
                overlap = true;
                auto& prod_collected_accesses = local_not_nested ? prod_meta.local_access : prod_meta.nested_access;
                auto& cons_collected_accesses = local_not_nested ? cons_meta.local_access : cons_meta.nested_access;
                if (prod_collected_accesses.subset_conflicts_with(cons_collected_accesses, canonical_indvars)) {
                    subset_mismatch = true;
                    // conflict (between vars unproved, fuse-by-access might find a solution with conflicting subsets)
                    continue;
                }
            } else if (prod_meta.arg.is_ptr && cons_meta.arg.is_ptr && prod_meta.arg.is_explicit_input &&
                       cons_meta.arg.is_explicit_input) {
                overlap = true;
                continue;
            }
        }
    }

    if (only_no_overlap && overlap) {
        no_conflicts = false;
    }

    return {no_conflicts, overlap, subset_mismatch};
}

void LoopFusionHandler::update_fused_seq(Sequence& sequence, const symbolic::ExpressionMapping& replacements) {
    sequence.replace(replacements);
}

/**
 * Merge a newly observed access (its `subset` and `not_understood` flag) into the FusionArg `into`.
 * If `into` already tracks a different subset, we can no longer describe the access with a single
 * subset and mark it not_understood. Returns true if `into` was modified.
 */
bool FusionArgCommonAccesses::merge_into(
    Block* block,
    const std::optional<data_flow::Subset>& subset,
    bool not_understood,
    bool write_not_read,
    const symbolic::ExpressionMapping* lower_indvars
) {
    bool updated = merge_subset(subset, not_understood, lower_indvars);

    if (write_not_read) {
        updated |= wr_block.merge_into(block, false);
    } else {
        updated |= rd_block.merge_into(block, false);
    }

    return updated;
}

bool FusionArgCommonBlock::merge_into(Block* other_block, bool other_conflict) {
    bool updated = false;

    if (other_conflict && !this->block_conflict) {
        this->block_conflict = true;
        updated = true;
    } else if (!this->block_conflict && this->common_block && other_block) {
        if (this->common_block != other_block) {
            this->block_conflict = true;
            this->common_block = nullptr;
            updated = true;
        }
    } else if (!this->block_conflict && !this->common_block && other_block) {
        this->common_block = other_block;
        updated = true;
    }

    return updated;
}

bool FusionArgCommonBlock::merge_into(const FusionArgCommonBlock& other) {
    return merge_into(other.common_block, other.block_conflict);
}

bool FusionArgCommonAccesses::
    merge_into(const FusionArgCommonAccesses& other, const symbolic::ExpressionMapping* lower_indvars) {
    bool update = merge_subset(other.common_subset, other.subsets_conflict, lower_indvars);
    update |= wr_block.merge_into(other.wr_block);
    update |= rd_block.merge_into(other.rd_block);
    return update;
}

bool FusionArgCommonAccesses::subset_conflicts_with(
    const FusionArgCommonAccesses& other_common_acceses, symbolic::ExpressionMapping& canonical_indvars
) const {
    if (subsets_conflict || other_common_acceses.subsets_conflict) {
        return true;
    }

    if (common_subset.has_value() && other_common_acceses.common_subset.has_value()) {
        if (!symbolic::vectors_of_expressions_match(
                common_subset.value(), other_common_acceses.common_subset.value(), canonical_indvars
            )) {
            return true;
        }
    }

    return false;
}

bool FusionArgCommonAccesses::merge_subset(
    const std::optional<data_flow::Subset>& subset,
    bool not_understood,
    const symbolic::ExpressionMapping* lower_indvars
) {
    bool updated = false;

    std::optional<data_flow::Subset> mapped_subset_holder;
    const data_flow::Subset* mapped_subset = nullptr;
    if (subset.has_value()) {
        if (lower_indvars) {
            mapped_subset_holder = symbolic::substitute(subset.value(), *lower_indvars);
            mapped_subset = &mapped_subset_holder.value();
        } else {
            mapped_subset = &subset.value();
        }
    }

    if (common_subset.has_value() && mapped_subset &&
        !symbolic::vectors_of_expressions_match(common_subset.value(), *mapped_subset)) {
        if (!subsets_conflict) {
            subsets_conflict = true;
            updated = true;
        }
    } else if (!common_subset.has_value() && mapped_subset && !subsets_conflict) {
        common_subset = *mapped_subset;
        updated = true;
    }
    if (not_understood && !subsets_conflict) {
        subsets_conflict = true;
        updated = true;
    }
    return updated;
}

bool FusionArg::saw_access_locally() const {
    return local_access.subsets_conflict || local_access.common_subset.has_value();
}

void FusionLoopCandidate::non_indvar_writes() { this->incompatible = true; }

void FusionLoopCandidate::aliasing_encountered() { this->incompatible = true; }

void FusionLoopCandidate::replace(const symbolic::ExpressionMapping& mapping) {
    for (auto& [name, arg] : args) {
        if (arg.local_access.common_subset.has_value()) {
            arg.local_access.common_subset = updated_subset(arg.local_access.common_subset.value(), mapping);
        }
        if (arg.nested_access.common_subset.has_value()) {
            arg.nested_access.common_subset = updated_subset(arg.nested_access.common_subset.value(), mapping);
        }
    }
    symbolic::substitute(assumptions, mapping);

    if constexpr (DUMP_ASSUMPTIONS) {
        std::cout << "Updated #" << this->loop->element_id() << " to:" << std::endl;
        std::cout << this->assumptions << std::endl;
    }
}

NeighboringPatternVisitor::NeighboringPatternVisitor(PatternHandler& handler) : handler_(handler) {}

bool NeighboringPatternVisitor::visit(sdfg::structured_control_flow::Sequence& node) {
    if (node.size() < 2) { // impossible to find a match, just descend into it
        return ActualStructuredSDFGVisitor::visit(node);
    }

    // Iterate over sequence looking for consecutive (StructuredLoop, StructuredLoop) pairs
    size_t i = 0;
    structured_control_flow::ControlFlowNode* override_last = nullptr;
    while (i < node.size()) {
        auto& child_node = node.at(i);
        auto* first = dyn_cast<structured_control_flow::StructuredLoop*>(&child_node);
        if (!first) {
            i++;
            dispatch(child_node);
            continue;
        }
        if (first->root().size() == 0) {
            i++;
            continue;
        }

        StructuredLoop* second = nullptr;

        if (i + 1 < node.size()) {
            second = dyn_cast<structured_control_flow::StructuredLoop*>(&node.at(i + 1));
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
                auto* mid_block = dyn_cast<structured_control_flow::Block*>(&node.at(i + 1));
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
                    second = dyn_cast<structured_control_flow::StructuredLoop*>(&node.at(i + 2));
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

} // namespace sdfg::passes::loop_fusion
