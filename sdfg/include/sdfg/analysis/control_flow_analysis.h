#pragma once

#include <memory>
#include <unordered_map>
#include <unordered_set>

#include "sdfg/analysis/analysis.h"
#include "sdfg/element.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/structured_sdfg.h"

namespace sdfg {
namespace analysis {

class ControlFlowAnalysis : public Analysis {
    friend class AnalysisManager;

private:
    graph::Graph graph_;

    // Temporary storage for loop nodes
    graph::Vertex last_loop_;

    std::unordered_map<graph::Vertex, structured_control_flow::ControlFlowNode*, boost::hash<graph::Vertex>> nodes_;
    std::unordered_map<structured_control_flow::ControlFlowNode*, structured_control_flow::ControlFlowNode*> dom_tree_;
    std::unordered_map<structured_control_flow::ControlFlowNode*, structured_control_flow::ControlFlowNode*> pdom_tree_;

    std::pair<graph::Vertex, graph::Vertex> traverse(structured_control_flow::ControlFlowNode& current);

public:
    ControlFlowAnalysis(StructuredSDFG& sdfg);

    std::string name() const override { return "ControlFlowAnalysis"; }

    void run(analysis::AnalysisManager& analysis_manager) override;

    std::unordered_set<structured_control_flow::ControlFlowNode*> exits() const;

    const std::unordered_map<structured_control_flow::ControlFlowNode*, structured_control_flow::ControlFlowNode*>&
    dom_tree() const;

    const std::unordered_map<structured_control_flow::ControlFlowNode*, structured_control_flow::ControlFlowNode*>&
    pdom_tree() const;

    bool dominates(structured_control_flow::ControlFlowNode& a, structured_control_flow::ControlFlowNode& b) const;

    bool post_dominates(structured_control_flow::ControlFlowNode& a, structured_control_flow::ControlFlowNode& b) const;
};

} // namespace analysis
} // namespace sdfg
