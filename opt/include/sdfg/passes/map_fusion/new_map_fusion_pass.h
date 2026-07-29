#pragma once

#include "sdfg/analysis/arguments_analysis.h"
#include "sdfg/analysis/loop_analysis.h"
#include "sdfg/analysis/structured_data_flow_analysis.h"
#include "sdfg/passes/map_fusion/map_fusion_by_accesses.h"
#include "sdfg/passes/pass.h"
#include "sdfg/visitor/structured_sdfg_visitor.h"

namespace sdfg::passes::map_fusion {

class NeighboringPatternVisitor : public sdfg::visitor::ActualStructuredSDFGVisitor {
    map_fusion::PatternHandler& handler_;

public:
    NeighboringPatternVisitor(map_fusion::PatternHandler& handler);

    bool visit(sdfg::structured_control_flow::Sequence& node) override;
};

class NewMapFusionPass : public sdfg::passes::Pass {
    friend class MapFusionHandler;
    LoopFusionConfig config_;

public:
    struct State {
        builder::StructuredSDFGBuilder& builder;
        analysis::AnalysisManager& analysis_manager;
        std::unique_ptr<analysis::LoopAnalysis> loop_analysis;
        std::unordered_map<analysis::ElementId, std::unique_ptr<map_fusion::FusionLoopCandidate>> fuse_candidates;
        uint32_t fused_by_domain_count = 0;
        uint32_t fused_by_access_count = 0;

        map_fusion::FusionLoopCandidate* get_next_level_map_stack(map_fusion::FusionLoopCandidate& current);

        map_fusion::FusionLoopCandidate* get_parent(map_fusion::FusionLoopCandidate& current);

        uint32_t total_fused_count() const;
    };

    NewMapFusionPass(const LoopFusionConfig& config);
    NewMapFusionPass();

    std::string name() override { return "NewMapFusionPass"; }

    bool run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

protected:
    const symbolic::Assumption*
    find_indvar_boundaries(const symbolic::Symbol& indvar, const symbolic::Assumptions& assumptions);

    // std::unique_ptr<MapFusionState> create_initial_state(const structured_control_flow::ControlFlowNode& node)
    // override;
};

class MapFusionHandler : public map_fusion::PatternHandler, map_fusion::MapFusionByAccessWorker {
    NewMapFusionPass::State& state_;
    LoopFusionConfig config_;

public:
    MapFusionHandler(const LoopFusionConfig& config, NewMapFusionPass::State& state);

    map_fusion::PatternHandler::MatchResult match(StructuredLoop& first, StructuredLoop& second, bool no_uses_between)
        override;

    map_fusion::PatternHandler::MatchResult try_complex_fuse_producer_into_consumer(
        FusionLoopCandidate& first, FusionLoopCandidate& second, bool no_uses_between, bool domains_match
    );

    bool check_no_overlap(
        const StructuredLoop& map,
        const StructuredLoop& second,
        const std::unordered_set<map_fusion::RegId>& skipped_containers
    ) override;

    struct InOutCheckResult {
        bool no_conflicts;
        bool overlap = false;
        bool subset_mismatch = false;
    };

protected:
    InOutCheckResult check_ins_outs(
        const map_fusion::FusionLoopCandidate& first_candidate,
        const map_fusion::FusionLoopCandidate& second_candidate,
        symbolic::ExpressionMapping& canonical_indvars,
        bool local_not_nested = true,
        bool only_no_overlap = false
    );

    InOutCheckResult check_ins_outs(
        const std::unordered_map<map_fusion::RegId, map_fusion::FusionArg>& first_args,
        const std::unordered_map<map_fusion::RegId, map_fusion::FusionArg>& second_args,
        symbolic::ExpressionMapping& canonical_indvars,
        bool local_not_nested = true,
        bool only_no_overlap = false
    );

    void update_fused_seq(Sequence& sequence, const symbolic::ExpressionMapping& replacements);

    bool loop_match(
        map_fusion::FusionLoopCandidate& first,
        map_fusion::FusionLoopCandidate& second,
        SymEngine::map_basic_basic& canonical_indvars
    );

    void update_moved_candidate_states(map_fusion::FusionLoopCandidate* top, const symbolic::ExpressionMapping& replace);

    void update_candidate_state(
        ControlFlowNode* first_top,
        map_fusion::FusionLoopCandidate* first_current,
        map_fusion::FusionLoopCandidate* second_current,
        const symbolic::ExpressionMapping& canonical_indvars
    );

    /**
     * Helper that walks up from the innermost loops fused (first_current & second_current) until reaching top
     * @param first_top the outermost loop considered for fusing (first_current is this or a nested loop)
     * @param first_current
     * @param second_current
     * @param action is given the container, the source arg from the first loop and the target_args from the second loop
     */
    void update_candidate_args_up(
        ControlFlowNode* first_top,
        map_fusion::FusionLoopCandidate* first_current,
        map_fusion::FusionLoopCandidate* second_current,
        const std::function<void(
            const std::string& name,
            map_fusion::FusionArg& source_arg,
            std::unordered_map<std::string, map_fusion::FusionArg>& target_args
        )>& action
    );

    map_fusion::PatternHandler::MatchResult fuse_contents(
        ControlFlowNode* first_top,
        map_fusion::FusionLoopCandidate* first_innermost,
        map_fusion::FusionLoopCandidate* second_innermost,
        const symbolic::ExpressionMapping& indvar_mapping,
        Sequence& target_root,
        bool can_remove_original
    );

    analysis::LoopAnalysis& get_loop_analysis() override;

    map_fusion::FusionLoopCandidate* get_fuse_candidate(StructuredLoop& loop) override;

    builder::StructuredSDFGBuilder& builder() override;

    void update_copied_leaf_contents_from_first_to_second(
        const Plan& plan, FusionLoopCandidate* first_current, FusionLoopCandidate* second_current
    ) override;
};

} // namespace sdfg::passes::map_fusion
