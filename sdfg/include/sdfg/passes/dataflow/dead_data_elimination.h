#pragma once

#include "sdfg/passes/pass.h"

namespace sdfg {
namespace passes {

class DeadDataElimination : public Pass {
private:
    bool permissive_;

public:
    DeadDataElimination();

    DeadDataElimination(bool permissive);

    virtual std::string name() override;

    virtual bool run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;
};

} // namespace passes
} // namespace sdfg
