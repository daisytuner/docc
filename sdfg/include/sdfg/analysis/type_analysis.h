#pragma once

#include "sdfg/analysis/analysis.h"
#include "sdfg/structured_control_flow/control_flow_node.h"
#include "sdfg/structured_sdfg.h"

namespace sdfg {
namespace analysis {

class TypeAnalysis : public Analysis {
private:
    std::unordered_map<std::string, std::set<const sdfg::types::IType*>> type_map_;
    structured_control_flow::ControlFlowNode* node_;

protected:
    void run(analysis::AnalysisManager& analysis_manager) override;

public:
    TypeAnalysis(StructuredSDFG& sdfg);
    TypeAnalysis(StructuredSDFG& sdfg, structured_control_flow::ControlFlowNode* node, AnalysisManager& analysis_manager);

    std::string name() const override { return "TypeAnalysis"; }

    const sdfg::types::IType* get_outer_type(const std::string& container) const;
};

} // namespace analysis
} // namespace sdfg
