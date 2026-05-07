#pragma once

#include "sdfg/passes/pass.h"

namespace sdfg {
namespace passes {

class For2MapPass : public Pass {
private:
    bool can_be_applied(
        builder::StructuredSDFGBuilder& builder,
        analysis::AnalysisManager& analysis_manager,
        structured_control_flow::For& for_stmt
    );

public:
    virtual std::string name() override;

    virtual bool run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;
};

} // namespace passes
} // namespace sdfg
