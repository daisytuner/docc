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

protected:
    void run(analysis::AnalysisManager& analysis_manager) override;

public:
    AssumptionsAnalysis(StructuredSDFG& sdfg);

    const symbolic::Assumptions& get(structured_control_flow::ControlFlowNode& node, bool include_trivial_bounds = false);

    const symbolic::SymbolSet& parameters();

    bool is_parameter(const symbolic::Symbol& container);

    bool is_parameter(const std::string& container);
};

} // namespace analysis
} // namespace sdfg
