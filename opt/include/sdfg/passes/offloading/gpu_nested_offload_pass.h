#pragma once

#include <vector>
#include "sdfg/passes/offloading/gpu_nested_parallelization_pass.h"
#include "sdfg/passes/pass.h"
#include "sdfg/structured_control_flow/structured_loop.h"

namespace sdfg {
namespace passes {

/**
 * @brief Depth-aware nested GPU offload pass.
 *
 * For each already grid-offloaded outer loop, inspects the maximum nesting depth
 * of its loop nest and offloads the nested loops via GPUOffloadNestedLoop:
 *   - depth 2:    X_BLOCK
 *   - depth 3:    X_BLOCK, then WARP
 *   - depth >= 4: Y_GRID, X_BLOCK, Y_BLOCK
 *
 * Grid/block sizes derive from each loop's iteration count: a known integer count
 * is used directly for grid dimensions and capped at the default for block
 * dimensions; a symbolic count falls back to the default.
 */
class GPUNestedOffloadPass : public Pass {
private:
    const std::vector<structured_control_flow::StructuredLoop*>& loops_;
    GPUTarget target_;

public:
    GPUNestedOffloadPass(const std::vector<structured_control_flow::StructuredLoop*>& loops, GPUTarget target);
    ~GPUNestedOffloadPass() override = default;

    bool run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    std::string name() override {
        return "GPUNestedOffloadPass";
    }
};

} // namespace passes
} // namespace sdfg
