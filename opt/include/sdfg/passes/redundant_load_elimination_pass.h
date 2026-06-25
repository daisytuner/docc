#pragma once

#include "sdfg/analysis/arguments_analysis.h"
#include "sdfg/analysis/loop_analysis.h"
#include "sdfg/analysis/structured_data_flow_analysis.h"
#include "sdfg/passes/pass.h"
#include "sdfg/visitor/structured_sdfg_visitor.h"

namespace sdfg::passes {


class RedundantLoadEliminationPass : public sdfg::passes::Pass {
public:
    struct State {
        size_t optimized = 0;
    };

    std::string name() override { return "MemoryDependency"; }

    bool run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;
};

class RedundantLoadVisitor : public sdfg::visitor::ActualStructuredSDFGVisitor {
    builder::StructuredSDFGBuilder& builder_;
    RedundantLoadEliminationPass::State& state_;

public:
    RedundantLoadVisitor(builder::StructuredSDFGBuilder& builder, RedundantLoadEliminationPass::State& state);

    bool visit(sdfg::structured_control_flow::Block& block) override;
};

} // namespace sdfg::passes
