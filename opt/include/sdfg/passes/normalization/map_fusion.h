#pragma once

#include "sdfg/analysis/loop_analysis.h"
#include "sdfg/analysis/structured_data_flow_analysis.h"
#include "sdfg/passes/pass.h"
#include "sdfg/visitor/structured_sdfg_visitor.h"

namespace sdfg {
namespace passes {
namespace normalization {

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

class NewMapFusionPass : public analysis::ForwardStructuredDataFlowAnalysis<MapFusionState>, Pass {
    std::vector<FusionCandidatePair> candidate_pairs_;

public:
    NewMapFusionPass() = default;

    bool run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

protected:
    std::unique_ptr<MapFusionState> create_initial_state(const structured_control_flow::ControlFlowNode& node) override;
};


// ------ legacy impl.

class MapFusion : public visitor::NonStoppingStructuredSDFGVisitor {
public:
    MapFusion(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager);

    static std::string name() { return "MapFusion"; };

    bool accept(structured_control_flow::Sequence& node) override;
};

typedef VisitorPass<MapFusion> MapFusionPass;

} // namespace normalization
} // namespace passes
} // namespace sdfg
