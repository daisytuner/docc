#pragma once

#include "sdfg/passes/pass.h"

namespace sdfg {
namespace passes {

/**
 * @brief Materialize compact GPU reduction partials and their SDFG accesses
 *
 * Uses ReductionBufferAnalysis to declare private/shared arrays, record schedule
 * buffer references, and pack body memlets while retaining original writeback
 * indices. Initialization, combination, and publication remain in GPU codegen.
 * Existing valid materializations are checked and left unchanged.
 */
class ReductionSharedMemoryDelinearization : public Pass {
public:
    /// @return "ReductionSharedMemoryDelinearization".
    std::string name() override;

    /// Validate ownership and placement, then materialize pending GPU reductions.
    /// Invalidates all analyses after rewriting the graph.
    /// @return true if new buffers/accesses were materialized; false if none were pending.
    /// @throws InvalidSDFGException for unsupported footprints or incompatible buffer declarations/ownership.
    bool run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;
};

} // namespace passes
} // namespace sdfg
