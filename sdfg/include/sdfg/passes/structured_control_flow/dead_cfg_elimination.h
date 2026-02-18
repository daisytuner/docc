#pragma once

#include "sdfg/passes/pass.h"

namespace sdfg {
namespace structured_control_flow {
class StructuredLoop;
}

namespace passes {

class DeadCFGElimination : public Pass {
private:
    bool permissive_;

    bool is_dead(const structured_control_flow::ControlFlowNode& node);

    bool is_trivial(structured_control_flow::StructuredLoop* loop);

public:
    DeadCFGElimination();

    DeadCFGElimination(bool permissive);

    virtual std::string name() override;

    virtual bool run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;
};

} // namespace passes
} // namespace sdfg
