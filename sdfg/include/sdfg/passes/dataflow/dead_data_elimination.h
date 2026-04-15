#pragma once

#include "sdfg/passes/pass.h"

namespace sdfg {
namespace passes {

class DeadDataElimination : public Pass {
private:
    bool legacy_removals_;

public:
    DeadDataElimination();

    DeadDataElimination(bool legacy_removals);

    virtual std::string name() override;

    virtual bool run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;
};

} // namespace passes
} // namespace sdfg
