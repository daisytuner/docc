#pragma once

#include <unordered_map>

#include "sdfg/analysis/analysis.h"
#include "sdfg/analysis/scope_analysis.h"
#include "sdfg/analysis/users.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/structured_sdfg.h"
#include "sdfg/symbolic/assumptions.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {
namespace analysis {

class AssumptionsAnalysis : public Analysis {
public:
    std::string name() const override { return "AssumptionsAnalysis"; }

private:
    // Data structures to hold assumptions
    std::unordered_map<structured_control_flow::ControlFlowNode*, symbolic::Assumptions> assumptions_;
    std::unordered_map<structured_control_flow::ControlFlowNode*, symbolic::Assumptions> assumptions_with_trivial_;

    // Data structures for sparse storage (nodes without own assumptions reference outer assumptions)
    std::unordered_map<structured_control_flow::ControlFlowNode*, const symbolic::Assumptions*> ref_assumptions_;
    std::unordered_map<structured_control_flow::ControlFlowNode*, const symbolic::Assumptions*>
        ref_assumptions_with_trivial_;

    symbolic::SymbolSet parameters_;

    analysis::Users* users_analysis_;

    // When false (default), IfElse branch conditions are not refined into
    // per-branch assumption bounds / coupled constraints. The branch bodies
    // simply inherit the outer scope's assumptions. This shortcut keeps the
    // overall analysis pipeline cheap; analyses that need the precise
    // branch-refined bounds (`MemoryLayoutAnalysis`,
    // `LoopCarriedDependencyAnalysis`, detailed
    // `DataDependencyAnalysis`) construct their own instance with this
    // flag set to true rather than going through `AnalysisManager`.
    bool with_branch_conditions_ = false;

    void traverse(
        structured_control_flow::ControlFlowNode& current,
        const symbolic::Assumptions& outer_assumptions,
        const symbolic::Assumptions& outer_assumptions_with_trivial
    );

    void traverse_structured_loop(
        structured_control_flow::StructuredLoop* loop,
        const symbolic::Assumptions& outer_assumptions,
        const symbolic::Assumptions& outer_assumptions_with_trivial
    );

    void propagate(
        structured_control_flow::ControlFlowNode& node,
        const symbolic::Assumptions& node_assumptions,
        const symbolic::Assumptions& outer_assumptions,
        const symbolic::Assumptions& outer_assumptions_with_trivial
    );

    void propagate_ref(
        structured_control_flow::ControlFlowNode& node,
        const symbolic::Assumptions& outer_assumptions,
        const symbolic::Assumptions& outer_assumptions_with_trivial
    );

    void determine_parameters(analysis::AnalysisManager& analysis_manager);

public:
    AssumptionsAnalysis(StructuredSDFG& sdfg);

    // Opt-in constructor used by analyses that require branch-refined
    // assumption propagation (see `with_branch_conditions_`). Not invoked
    // by `AnalysisManager`, which always calls the single-arg constructor.
    AssumptionsAnalysis(StructuredSDFG& sdfg, bool with_branch_conditions);

    // Public so analyses that own a manually-constructed instance (DDA in
    // detailed mode, LCDA, MLA) can drive `run()` themselves. The shared
    // `AnalysisManager` cache uses the friend declaration on the base
    // `Analysis` class as before.
    void run(analysis::AnalysisManager& analysis_manager) override;

    const symbolic::Assumptions& get(structured_control_flow::ControlFlowNode& node, bool include_trivial_bounds = false);

    const symbolic::SymbolSet& parameters();

    bool is_parameter(const symbolic::Symbol& container);

    bool is_parameter(const std::string& container);
};

} // namespace analysis
} // namespace sdfg
