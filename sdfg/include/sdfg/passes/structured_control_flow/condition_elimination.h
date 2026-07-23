#pragma once

#include "sdfg/passes/pass.h"
#include "sdfg/visitor/structured_sdfg_visitor.h"

namespace sdfg {
namespace passes {

class ConditionElimination : public visitor::NonStoppingStructuredSDFGVisitor {
private:
    bool eliminate_condition(structured_control_flow::Sequence& root, structured_control_flow::IfElse& match_node);

public:
    ConditionElimination(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager);

    static std::string name() { return "ConditionElimination"; };

    bool accept(structured_control_flow::Sequence& node) override;
};

typedef VisitorPass<ConditionElimination> ConditionEliminationPass;

} // namespace passes
} // namespace sdfg
