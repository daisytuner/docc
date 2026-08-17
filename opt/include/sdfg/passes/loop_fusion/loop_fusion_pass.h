#pragma once

#include "sdfg/analysis/arguments_analysis.h"
#include "sdfg/analysis/loop_analysis.h"
#include "sdfg/analysis/structured_data_flow_analysis.h"
#include "sdfg/passes/loop_fusion/loop_fusion_by_accesses.h"
#include "sdfg/passes/pass.h"
#include "sdfg/visitor/structured_sdfg_visitor.h"

namespace sdfg::passes::loop_fusion {

class NeighboringPatternVisitor : public sdfg::visitor::ActualStructuredSDFGVisitor {
    loop_fusion::PatternHandler& handler_;

public:
    NeighboringPatternVisitor(loop_fusion::PatternHandler& handler);

    bool visit(sdfg::structured_control_flow::Sequence& node) override;
};

/**
 * @brief A pass that performs loop fusion on a StructuredSDFG. Though mostly between Maps at this point.
 *
 * Current impl. walks the SDFG in execution order with a sliding window (of 3). It can only fuse loops in the same
 * sequence and at most with 1 independent block in between them and it does not retry past fused loops other then using
 * the result of the last fusion as start of the next (which works for simple chains that always fuse forward)
 *
 * The pass builds a FusionLoopCandidate cache at the start and then maintains all the data (assumptions, indvars,
 * arguments) across modifications
 *
 * The core function is in LoopFusionHandler::match, which gets called for candidate pairs.
 * It then tries
 *  * fusion by-domain
 *      * can fuse intermediate levels of loops (that contain further, arbitrary loops)
 *      * checks iteration domain for exact equality for all levels that will be fused across (only supported for
 * "map-stacks", which are perfectly nested & parallel)
 *      * checks for overlapping indirect memory usages (can fuse if there is no overlap at all, just no conflicts)
 *      * checks the subsets on all overlapping usages to match exactly
 *      * further nested loops indvars are represented by their iteration bounds, stepsize and nesting level for those
 * exact matches
 *  * fusion by-access via MapFusionByAccessWorker
 *      * if the domain or subset checks fail, we proceed to the more complex case, that collects every access and
 * solves for all reads being covered by matching writes.
 *      * this can fuse only innermost loops with restrictive content, but can handle changes in iteration domain
 *      * when the domains are not equal, it needs to copy one loop and replaces reads with the direct results of the
 * productions, elliding the indirect memory accesses
 *      * when the domains are equal, it keeps indirect accesses in same as by-domain (so the fused loop is a full
 * replacement for both source loops), leaving the RedundantLoadElimination and cleanup of superfluous accesses for
 * other passes
 *      * by-access is currently restricted to fusing single-block producers into a larger consumer and specific
 * reduction patterns
 *
 * Either way needs to rely only on the data cached in the FusionLoopCandidate and LoopAnalysis as only those are kept
 * up-to-date. After every change, LoopAnalysis and cached data update... functions need to be called to maintain the
 * cached data.
 *
 * Eventually, this should become a Dataflow-based pass, that will fixpoint iterate, can fuse loops that are further
 * apart and can prioritize the order of fusions other then greedy, forward-only. It was designed such that the match()
 * can still be called in that case
 */
class LoopFusionPass : public sdfg::passes::Pass {
    friend class LoopFusionHandler;
    LoopFusionConfig config_;

public:
    struct State {
        builder::StructuredSDFGBuilder& builder;
        analysis::AnalysisManager& analysis_manager;
        std::unique_ptr<analysis::LoopAnalysis> loop_analysis;
        std::unordered_map<analysis::ElementId, std::unique_ptr<loop_fusion::FusionLoopCandidate>> fuse_candidates;
        uint32_t fused_by_domain_count = 0;
        uint32_t fused_by_access_count = 0;

        State(
            builder::StructuredSDFGBuilder& builder,
            analysis::AnalysisManager& analysis_manager,
            std::unique_ptr<analysis::LoopAnalysis> loop_analysis
        )
            : builder(builder), analysis_manager(analysis_manager), loop_analysis(std::move(loop_analysis)), run(0) {}

        loop_fusion::FusionLoopCandidate* get_next_level_map_stack(loop_fusion::FusionLoopCandidate& current);

        loop_fusion::FusionLoopCandidate* get_parent(loop_fusion::FusionLoopCandidate& current);

        uint32_t total_fused_count() const;

        uint32_t run;
    };

    LoopFusionPass(const LoopFusionConfig& config);
    LoopFusionPass();

    std::string name() override { return "LoopFusionPass"; }

    bool run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

protected:
    const symbolic::Assumption*
    find_indvar_boundaries(const symbolic::Symbol& indvar, const symbolic::Assumptions& assumptions);

    // std::unique_ptr<MapFusionState> create_initial_state(const structured_control_flow::ControlFlowNode& node)
    // override;
};

class LoopFusionHandler : public loop_fusion::PatternHandler, loop_fusion::LoopFusionByAccessWorker {
    LoopFusionPass::State& state_;
    LoopFusionConfig config_;

public:
    LoopFusionHandler(const LoopFusionConfig& config, LoopFusionPass::State& state);

    loop_fusion::PatternHandler::MatchResult match(StructuredLoop& first, StructuredLoop& second, bool no_uses_between)
        override;

    loop_fusion::PatternHandler::MatchResult try_complex_fuse_producer_into_consumer(
        FusionLoopCandidate& first, FusionLoopCandidate& second, bool no_uses_between, bool domains_match
    );

    bool check_no_overlap(
        const StructuredLoop& map,
        const StructuredLoop& second,
        const std::unordered_set<loop_fusion::RegId>& skipped_containers
    ) override;

    struct InOutCheckResult {
        bool no_conflicts;
        bool overlap = false;
        bool subset_mismatch = false;
    };

protected:
    InOutCheckResult check_ins_outs(
        const loop_fusion::FusionLoopCandidate& first_candidate,
        const loop_fusion::FusionLoopCandidate& second_candidate,
        symbolic::ExpressionMapping& canonical_indvars,
        bool local_not_nested = true,
        bool only_no_overlap = false
    );

    InOutCheckResult check_ins_outs(
        const std::unordered_map<loop_fusion::RegId, loop_fusion::FusionArg>& first_args,
        const std::unordered_map<loop_fusion::RegId, loop_fusion::FusionArg>& second_args,
        symbolic::ExpressionMapping& canonical_indvars,
        bool local_not_nested = true,
        bool only_no_overlap = false
    );

    void update_fused_seq(Sequence& sequence, const symbolic::ExpressionMapping& replacements);

    bool loop_match(
        loop_fusion::FusionLoopCandidate& first,
        loop_fusion::FusionLoopCandidate& second,
        SymEngine::map_basic_basic& canonical_indvars
    );

    void update_moved_candidate_states(loop_fusion::FusionLoopCandidate* top, const symbolic::ExpressionMapping& replace);

    void update_candidate_state(
        ControlFlowNode* first_top,
        loop_fusion::FusionLoopCandidate* first_current,
        loop_fusion::FusionLoopCandidate* second_current,
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
        loop_fusion::FusionLoopCandidate* first_current,
        loop_fusion::FusionLoopCandidate* second_current,
        const std::function<void(
            const std::string& name,
            loop_fusion::FusionArg& source_arg,
            std::unordered_map<std::string, loop_fusion::FusionArg>& target_args
        )>& action
    );

    loop_fusion::PatternHandler::MatchResult fuse_contents(
        ControlFlowNode* first_top,
        loop_fusion::FusionLoopCandidate* first_innermost,
        loop_fusion::FusionLoopCandidate* second_innermost,
        const symbolic::ExpressionMapping& indvar_mapping,
        Sequence& target_root,
        bool can_remove_original
    );

    analysis::LoopAnalysis& get_loop_analysis() override;

    loop_fusion::FusionLoopCandidate* get_fuse_candidate(StructuredLoop& loop) override;

    builder::StructuredSDFGBuilder& builder() override;

    void update_copied_leaf_contents_from_first_to_second(
        const Plan& plan, FusionLoopCandidate* first_current, FusionLoopCandidate* second_current
    ) override;
};

} // namespace sdfg::passes::loop_fusion
