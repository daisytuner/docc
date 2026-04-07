#pragma once

#include "sdfg/passes/pass.h"
#include "sdfg/visitor/structured_sdfg_visitor.h"

namespace sdfg {
namespace passes {
namespace normalization {

class LoopNormalFormPass : public passes::Pass {
private:
    bool apply(
        builder::StructuredSDFGBuilder& builder,
        analysis::AnalysisManager& analysis_manager,
        structured_control_flow::StructuredLoop& loop
    );

public:
    LoopNormalFormPass();

    std::string name() override { return "LoopNormalForm"; };

    virtual bool run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;
};

} // namespace normalization
} // namespace passes
} // namespace sdfg
