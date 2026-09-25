#pragma once

#include "sdfg/passes/pass.h"

namespace sdfg {
namespace tiles {

/**
 * @brief Re-discharge every TileCopyNode's ragged boundary guard.
 *
 * A copy node's boundary guard is set when LocalStorage runs, from the assumptions
 * available then. When a later pass — loop peeling, condition propagation — proves a
 * tile dimension fully covering, this pass drops the now-redundant guard dims (via
 * TileCopyNode::normalize_guard against the current AssumptionsAnalysis), so a node
 * copy inherits the same guard simplification the old map-nest copy got for free.
 * Safe to run repeatedly; only ever removes provably-redundant predicates.
 */
class TileGuardNormalization : public passes::Pass {
public:
    TileGuardNormalization() = default;

    std::string name() override {
        return "TileGuardNormalization";
    }

    bool run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;
};

} // namespace tiles
} // namespace sdfg
