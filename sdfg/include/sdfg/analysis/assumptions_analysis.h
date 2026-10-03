#pragma once

#include <unordered_map>

#include "sdfg/analysis/analysis.h"
#include "sdfg/analysis/users.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/structured_sdfg.h"
#include "sdfg/symbolic/assumptions.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {
namespace analysis {

class AssumptionsAnalysis : public Analysis {
public:
    std::string name() const override {
        return "AssumptionsAnalysis";
    }

private:
    using Node = structured_control_flow::ControlFlowNode;

    std::unordered_map<Node*, symbolic::Assumptions> own_assumptions_;
    std::unordered_map<Node*, Node*> parent_scope_;
    std::unordered_map<Node*, Node*> scope_of_;

    std::unordered_map<Node*, symbolic::Assumptions> assumptions_;
    std::unordered_map<Node*, symbolic::Assumptions> assumptions_with_trivial_;
    std::unordered_map<Node*, symbolic::SymbolSet> constant_symbols_;
    std::unordered_map<Node*, symbolic::SymbolSet> constant_symbols_with_trivial_;

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

    void traverse(Node& current, Node& scope);

    void traverse_structured_loop(structured_control_flow::StructuredLoop* loop, Node& scope);

    void add_scope(Node& node, symbolic::Assumptions own, Node& parent);

    const symbolic::Assumptions& materialize(Node& scope, bool include_trivial_bounds);

    const symbolic::SymbolSet& materialize_constants(Node& scope, bool include_trivial_bounds);

    void determine_parameters(analysis::AnalysisManager& analysis_manager);

public:
    AssumptionsAnalysis(StructuredSDFG& sdfg);
    AssumptionsAnalysis(StructuredSDFG& sdfg, bool with_branch_conditions);

    void run(analysis::AnalysisManager& analysis_manager) override;

    const symbolic::Assumptions& get(structured_control_flow::ControlFlowNode& node, bool include_trivial_bounds = false);

    /// Symbols whose assumption at `node` is `constant()`; equals filtering `get(node, ...)`, without the scan.
    const symbolic::SymbolSet&
    constant_symbols(structured_control_flow::ControlFlowNode& node, bool include_trivial_bounds = false);

    const symbolic::SymbolSet& parameters();

    bool is_parameter(const symbolic::Symbol& container);

    bool is_parameter(const std::string& container);

    static symbolic::SymbolSet per_symbol_refined_symbols(const symbolic::Condition& cond);
};

} // namespace analysis
} // namespace sdfg
