#pragma once

#include "sdfg/analysis/arguments_analysis.h"
#include "sdfg/analysis/loop_analysis.h"
#include "sdfg/analysis/structured_data_flow_analysis.h"
#include "sdfg/passes/pass.h"
#include "sdfg/visitor/structured_sdfg_visitor.h"

namespace sdfg::passes {

struct FusionCandidate {};

struct FusionContainerRef {
    FusionCandidate* candidate;
};

typedef std::string RegId;

struct MapFusionExposed {
    /**
     * Anything not understood fully, like aliasing ptrs etc. will be collected in execution order
     */
    std::unordered_map<RegId, std::string> ineligible_containers;
    /**
     * Index of the variables involved in fusion_candidates to quickly match them against kill lists
     */
    std::unordered_map<RegId, FusionContainerRef> tracked_var_refs;
    std::unordered_map<analysis::ElementId, std::unique_ptr<FusionCandidate>> fusion_candidates;
};

/**
 * SDFG-global state shared between visitors, matchers, pass.
 */
class MapFusionState : public analysis::DataFlowState<MapFusionExposed> {
    bool ran_ = false;
    MapFusionExposed incoming_;
    MapFusionExposed forward_exposed_;

public:
    bool ran_at_least_once() const override { return ran_; }

    bool update(const MapFusionExposed& exposed) override;

    bool update_incoming(const MapFusionExposed& incoming) override;

    bool update_forward_exposed(const MapFusionExposed& forward_exposed) override;

    const MapFusionExposed& forward_exposed() const override { return forward_exposed_; }
};

struct FusionArgCommonAccesses {
    bool subsets_conflict = false;
    std::optional<data_flow::Subset> common_subset;

    bool merge_into(
        const std::optional<data_flow::Subset>& subset,
        bool not_understood,
        const symbolic::ExpressionMapping* lower_indvars = nullptr
    );

    bool merge_into(const FusionArgCommonAccesses& other, const symbolic::ExpressionMapping* lower_indvars = nullptr);

    bool conflicts_with(const FusionArgCommonAccesses& other_common_acceses, symbolic::ExpressionMapping& canonical_indvars)
        const;
};

/**
 * Represents one "Argument" per ArgumentAnalysis of a FusionCandidate. I.e. contains Live-Ins, Live-Outs of the loop
 * @property arg contains the ArgumentAnalysis metadata that identifies how the arg is used (read, written) inside the
 * loop
 * @property local_access for accesses that happen directly inside the candidate (not in further nested candidates)
 */
struct FusionArg {
    analysis::RegionArgument arg;
    FusionArgCommonAccesses local_access;
    FusionArgCommonAccesses nested_access;

    FusionArg(const analysis::RegionArgument& arg) : arg(arg) {}
    FusionArg(const FusionArg& arg) : arg(arg.arg), local_access(arg.local_access), nested_access(arg.nested_access) {}

    bool saw_access_locally() const;
};

/**
 * Represents pre-cached data for 1 Loop that fullfills the prerequisits for fusing with another candidate
 * Goal is to collect every harder to get data once for all involved loops.
 * Because we likely need to compare a candidate against many others, reusing the same data.
 * Also structured in a way, that the data can be merged & updated if fusing between candidates actually happens
 *
 * Also, since not every loop is eligible to be a candidate, this flattens the loop-tree.
 * Relevant data of nested loops that are not themselves candidates will need to be folded into its nearest parent
 * candidate
 */
struct FusionLoopCandidate {
    StructuredLoop* loop;
    const symbolic::Assumption* indvar_boundaries;
    std::unordered_map<RegId, FusionArg> args;
    bool incompatible = false;
    bool nested_incompatible = false;

    void non_indvar_writes();

    void aliasing_encountered();

    void replace(const symbolic::ExpressionMapping& mapping);
};


class PatternHandler {
public:
    struct MatchResult {
        bool removed_first = false;
        bool visit_second_body = false;
        ControlFlowNode* second_root_replacement = nullptr;
    };

    virtual MatchResult match(Map& map, Map& second, bool no_uses_between) = 0;

    virtual bool
    check_no_overlap(const Map& map, const Map& second, const std::unordered_set<RegId>& skipped_containers) = 0;
};

class NeighboringPatternVisitor : public sdfg::visitor::ActualStructuredSDFGVisitor {
    PatternHandler& handler_;

public:
    NeighboringPatternVisitor(PatternHandler& handler);

    bool visit(sdfg::structured_control_flow::Sequence& node) override;
};

class MapFusionByDomainPass : public sdfg::passes::Pass {
    friend class MapFusionHandler;
    static constexpr bool dump_infos = true;

public:
    struct State {
        builder::StructuredSDFGBuilder& builder;
        analysis::AnalysisManager& analysis_manager;
        std::unique_ptr<analysis::LoopAnalysis> loop_analysis;
        std::unordered_map<analysis::ElementId, std::unique_ptr<FusionLoopCandidate>> fuse_candidates;
        uint32_t fused_count = 0;

        FusionLoopCandidate* get_next_level_map_stack(FusionLoopCandidate& current);

        FusionLoopCandidate* get_parent(FusionLoopCandidate& current);
    };

    MapFusionByDomainPass() = default;

    std::string name() override { return "MapFusionByDomainPass"; }

    bool run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

protected:
    const symbolic::Assumption*
    find_indvar_boundaries(const symbolic::Symbol& indvar, const symbolic::Assumptions& assumptions);

    // std::unique_ptr<MapFusionState> create_initial_state(const structured_control_flow::ControlFlowNode& node)
    // override;
};

class MapFusionHandler : public PatternHandler {
    MapFusionByDomainPass::State& state_;

public:
    MapFusionHandler(MapFusionByDomainPass::State& state);

    PatternHandler::MatchResult match(Map& first, Map& second, bool no_uses_between) override;

    bool check_no_overlap(const Map& map, const Map& second, const std::unordered_set<RegId>& skipped_containers)
        override;

    struct InOutCheckResult {
        bool no_conflicts;
        bool overlap = false;
        bool subset_mismatch = false;
    };

protected:
    InOutCheckResult check_ins_outs(
        const FusionLoopCandidate& first_candidate,
        const FusionLoopCandidate& second_candidate,
        symbolic::ExpressionMapping& canonical_indvars,
        bool local_not_nested
    );

    InOutCheckResult check_ins_outs(
        const std::unordered_map<RegId, FusionArg>& first_args,
        const std::unordered_map<RegId, FusionArg>& second_args,
        symbolic::ExpressionMapping& canonical_indvars,
        bool local_not_nested
    );

    void update_fused_seq(Sequence& sequence, const symbolic::ExpressionMapping& replacements);

    bool loop_match(FusionLoopCandidate& first, FusionLoopCandidate& second, SymEngine::map_basic_basic& canonical_indvars);

    void update_child_candidate_states(FusionLoopCandidate* top, const symbolic::ExpressionMapping& replace);

    void update_candidate_state(
        ControlFlowNode* first_top,
        FusionLoopCandidate* first_current,
        FusionLoopCandidate* second_current,
        const symbolic::ExpressionMapping& canonical_indvars
    );

    PatternHandler::MatchResult fuse_contents(
        ControlFlowNode* first_top,
        FusionLoopCandidate* first_innermost,
        FusionLoopCandidate* second_innermost,
        const symbolic::ExpressionMapping& indvar_mapping,
        Sequence& target_root,
        bool can_remove_original
    );
};

} // namespace sdfg::passes
