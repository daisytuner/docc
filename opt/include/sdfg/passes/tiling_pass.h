#pragma once

#include <vector>
#include "sdfg/passes/pass.h"
#include "sdfg/structured_control_flow/structured_loop.h"

namespace sdfg {
namespace passes {

/**
 * @brief Phased tiling pass: collects applicable tilings, then applies them.
 *
 * Given a set of loops, performs LoopTiling on each in two phases:
 * 1. can_be_applied phase: collects all loops where tiling is applicable
 * 2. apply phase: applies the tiling to all collected loops
 *
 * After application, the input vector is updated in-place with the new outer loops.
 */
class TilingPass : public Pass {
private:
    std::vector<structured_control_flow::StructuredLoop*>& loops_;
    size_t tile_size_;

public:
    TilingPass(std::vector<structured_control_flow::StructuredLoop*>& loops, size_t tile_size);
    ~TilingPass() override = default;

    bool run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    std::string name() override {
        return "TilingPass";
    }
};

} // namespace passes
} // namespace sdfg
