#pragma once

#include <sdfg/passes/pass.h>
#include <sdfg/structured_control_flow/structured_loop.h>

namespace sdfg {
namespace passes {
namespace normalization {

class PerfectLoopDistributionPass : public passes::Pass {
private:
    bool can_be_applied(
        builder::StructuredSDFGBuilder& builder,
        analysis::AnalysisManager& analysis_manager,
        structured_control_flow::StructuredLoop& loop
    );

    void apply(
        builder::StructuredSDFGBuilder& builder,
        analysis::AnalysisManager& analysis_manager,
        structured_control_flow::StructuredLoop& loop
    );

public:
    PerfectLoopDistributionPass();

    std::string name() override { return "PerfectLoopDistribution"; };

    virtual bool run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;
};

} // namespace normalization
} // namespace passes
} // namespace sdfg
