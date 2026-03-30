#pragma once

#include <vector>
#include "sdfg/analysis/analysis.h"
#include "sdfg/structured_control_flow/control_flow_node.h"

namespace sdfg {
namespace analysis {

class ScopeAnalysis : public Analysis {
private:
    std::unordered_map<const structured_control_flow::ControlFlowNode*, structured_control_flow::ControlFlowNode*>
        scope_tree_;

    void run(structured_control_flow::ControlFlowNode* current, structured_control_flow::ControlFlowNode* parent_scope);

protected:
    void run(analysis::AnalysisManager& analysis_manager) override;

public:
    ScopeAnalysis(StructuredSDFG& sdfg);

    std::string name() const override { return "ScopeAnalysis"; }

    const std::unordered_map<const structured_control_flow::ControlFlowNode*, structured_control_flow::ControlFlowNode*>&
    scope_tree() const;

    structured_control_flow::ControlFlowNode* parent_scope(const structured_control_flow::ControlFlowNode* scope) const;

    std::vector<structured_control_flow::ControlFlowNode*> ancestor_scopes(const structured_control_flow::ControlFlowNode*
                                                                               scope) const;
};

} // namespace analysis
} // namespace sdfg
