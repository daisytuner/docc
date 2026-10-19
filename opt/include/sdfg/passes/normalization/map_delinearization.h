#pragma once

#include "sdfg/passes/pass.h"

namespace sdfg {
namespace passes {
namespace normalization {

/**
 * @brief Splits flattened maps into nested maps along the divisions of their induction variables.
 *
 * Applies transformations::MapDelinearize wherever it can, so that accesses like `A[64*idiv(i, 256) + imod(i, 64)]`,
 * which reshapes and fusing nests of different shapes leave behind, become affine in the induction variables of a
 * nest again.
 */
class MapDelinearizationPass : public passes::Pass {
public:
    std::string name() override {
        return "MapDelinearization";
    }

    bool run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;
};

} // namespace normalization
} // namespace passes
} // namespace sdfg
