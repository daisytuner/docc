#pragma once

#include <vector>
#include "sdfg/passes/pass.h"
#include "sdfg/structured_control_flow/map.h"

namespace sdfg {
namespace passes {

/**
 * @brief Applies TrivialLoopElimination to a set of maps.
 *
 * Eliminated maps are removed from the input vector in-place and replaced by the maps
 * that were direct children of their body (which are, in turn, also eliminated if trivial).
 */
class TrivialLoopEliminationPass : public Pass {
private:
    std::vector<structured_control_flow::Map*>& maps_;

public:
    TrivialLoopEliminationPass(std::vector<structured_control_flow::Map*>& maps);
    ~TrivialLoopEliminationPass() override = default;

    bool run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    std::string name() override {
        return "TrivialLoopEliminationPass";
    }
};

} // namespace passes
} // namespace sdfg
