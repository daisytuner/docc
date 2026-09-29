#pragma once

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/passes/pass.h"

namespace sdfg {
namespace structured_control_flow {
class Map;
}

namespace passes {

class DeadCFGElimination : public Pass {
private:
    bool permissive_;

    bool is_dead(const structured_control_flow::ControlFlowNode& node);

    bool is_trivial(structured_control_flow::Map* loop);

    void update_loop_indvar_accesses(builder::StructuredSDFGBuilder& builder, structured_control_flow::Map* loop);

public:
    DeadCFGElimination();

    DeadCFGElimination(bool permissive);

    virtual std::string name() override;

    virtual bool run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;
};

} // namespace passes
} // namespace sdfg
