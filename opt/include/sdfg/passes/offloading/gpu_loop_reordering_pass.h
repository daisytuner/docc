#pragma once

#include <vector>
#include "sdfg/passes/pass.h"
#include "sdfg/structured_control_flow/map.h"

namespace sdfg {
namespace passes {

/**
 * @brief Phased GPU loop reordering pass.
 *
 * Given a set of maps, applies GPULoopReordering in two phases:
 * 1. can_be_applied phase: collects all maps where reordering is applicable
 * 2. apply phase: applies reordering to all collected maps
 */
class GPULoopReorderingPass : public Pass {
private:
    const std::vector<structured_control_flow::Map*>& maps_;

public:
    GPULoopReorderingPass(const std::vector<structured_control_flow::Map*>& maps);
    ~GPULoopReorderingPass() override = default;

    bool run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    std::string name() override { return "GPULoopReorderingPass"; }
};

} // namespace passes
} // namespace sdfg
