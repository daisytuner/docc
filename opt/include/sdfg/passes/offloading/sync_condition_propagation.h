#pragma once

#include "sdfg/passes/pass.h"

namespace sdfg::passes {

class SyncConditionPropagation : public Pass {
public:
    SyncConditionPropagation();

    std::string name() override {
        return "SyncConditionPropagation";
    };

    bool run_pass(sdfg::builder::StructuredSDFGBuilder& builder, sdfg::analysis::AnalysisManager& analysis_manager)
        override;
};

} // namespace sdfg::passes
