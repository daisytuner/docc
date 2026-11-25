#pragma once

#include "sdfg/passes/pass.h"

namespace sdfg {
namespace passes {

/**
 * @brief Cleans up an SDFG after its loops are offloaded to a GPU.
 *
 * Minimizes host-device transfers, reuses device buffers, removes the dead data, control flow and references this
 * leaves, and drops the zero-initializations of matmuls lowered to MMA.
 */
class OffloadingCleanupPass : public Pass {
public:
    std::string name() override {
        return "OffloadingCleanupPass";
    }

    bool run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;
};

} // namespace passes
} // namespace sdfg
