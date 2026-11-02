#pragma once

#include <string>

#include "sdfg/passes/pass.h"

namespace sdfg {
namespace passes {

/**
 * @brief Applies RedundantZeroInitElimination to every loop annotated as zero init (ScheduleType_ZeroInit).
 *
 * Intended to run after the MMA lowering of matmul nests, so that the zero-init nests emitted by the
 * naive MatMul/GEMM expansion are removed where the following MMA nest makes them redundant.
 */
class RedundantZeroInitEliminationPass : public Pass {
public:
    ~RedundantZeroInitEliminationPass() override = default;

    bool run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    std::string name() override {
        return "RedundantZeroInitEliminationPass";
    }
};

} // namespace passes
} // namespace sdfg
