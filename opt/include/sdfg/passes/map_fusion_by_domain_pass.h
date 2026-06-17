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

struct MapFusionExposed {
    /**
     * Anything not understood fully, like aliasing ptrs etc. will be collected in execution order
     */
    std::unordered_map<std::string, std::string> ineligible_containers;
    /**
     * Index of the variables involved in fusion_candidates to quickly match them against kill lists
     */
    std::unordered_map<std::string, FusionContainerRef> tracked_var_refs;
    std::unordered_map<analysis::ElementId, std::unique_ptr<FusionCandidate>> fusion_candidates;
};

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

struct FusionCandidatePair {
    FusionCandidate* first;
    FusionCandidate* second;
};

struct FusionLoopCandidate {
    StructuredLoop* loop;
    const symbolic::Assumption* indvar_boundaries;
    const std::map<std::string, analysis::RegionArgument>* args;
};


class PatternHandler {
public:
    struct MatchResult {
        bool removed_first = false;
        bool visit_second_body = false;
        ControlFlowNode* second_root_replacement = nullptr;
    };

    virtual MatchResult match(Map& map, Map& second, bool no_uses_between) = 0;
};

class NeighboringPatternVisitor : public sdfg::visitor::ActualStructuredSDFGVisitor {
    PatternHandler& handler_;

public:
    NeighboringPatternVisitor(PatternHandler& handler);

    bool visit(sdfg::structured_control_flow::Sequence& node) override;
};

class MapFusionByDomainPass : public sdfg::passes::Pass {
    std::vector<FusionCandidatePair> candidate_pairs_;

public:
    struct State {
        builder::StructuredSDFGBuilder& builder;
        analysis::AnalysisManager& analysis_manager;
        std::unique_ptr<analysis::LoopAnalysis> loop_analysis;
        std::unordered_map<analysis::ElementId, FusionLoopCandidate> fuse_candidates;
        uint32_t fused_count = 0;

        FusionLoopCandidate* get_next_level_map_stack(FusionLoopCandidate& current);
    };

    MapFusionByDomainPass() = default;

    std::string name() override { return "NewMapFusionPass"; }

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

protected:
    bool no_data_conflicts(const FusionLoopCandidate& first_candidate, const FusionLoopCandidate& second_candidate);

    void update_fused_seq(Sequence& sequence, const symbolic::ExpressionMapping& replacements);

    bool loop_match(FusionLoopCandidate& first, FusionLoopCandidate& second, SymEngine::map_basic_basic& canonical_indvars);

    PatternHandler::MatchResult fuse_contents(
        ControlFlowNode* first_top,
        FusionLoopCandidate* first_innermost,
        FusionLoopCandidate* second_innermost,
        SymEngine::map_basic_basic indvar_mapping,
        Sequence& target_root,
        bool can_remove_original
    );
};

} // namespace sdfg::passes
