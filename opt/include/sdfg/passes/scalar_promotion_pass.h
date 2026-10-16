#pragma once

#include "sdfg/passes/pass.h"

namespace sdfg::passes {

/**
 * Promotes the array element a Reduce accumulates into, e.g. `c[i]` in `for k: c[i] += a[i, k]`, to a scalar:
 * LocalStorage stages the element in a one-element register buffer around the loop, the buffer's copies become
 * plain assignments, and TrivialArrayElimination turns the buffer into the scalar the Reduce then accumulates into.
 */
class ScalarPromotionPass : public Pass {
public:
    std::string name() override {
        return "ScalarPromotion";
    }

    bool run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;
};

} // namespace sdfg::passes
