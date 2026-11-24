#pragma once

#include "sdfg/passes/pass.h"

namespace sdfg {
namespace passes {

/**
 * @brief Removes the replication of non-collective side effects in lanes maps.
 *
 * Every lane of an X_BLOCK map with lanes > 1 executes the whole body of its iteration, which is
 * required for wave-collective operations (MMA, cooperative tile copies, barriers) and harmless
 * for thread-private computations, but replicates every write to memory the lanes share (global,
 * shared or static device memory, atomics). This pass wraps each run of blocks with such writes in
 * `if (threadIdx.x % lanes == 0)`, so only the leader lane of each group performs them, and inserts
 * a wave barrier after the run when other lanes read the written memory afterwards.
 *
 * Blocks containing collective nodes or indexing by `threadIdx.x` (lane-distributed work) are left
 * unchanged. Runs after reduction partials are materialized, whose thread-private buffers stay
 * replicated. Already predicated blocks are skipped, so the pass is idempotent.
 */
class LanesMapPredication : public Pass {
public:
    /// @return "LanesMapPredication".
    std::string name() override;

    /// @return true if any block was predicated.
    /// @throws InvalidSDFGException if a predicated block also produces a thread-private value read
    ///         elsewhere in the map, or mixes collective nodes with lane-shared writes.
    bool run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;
};

} // namespace passes
} // namespace sdfg
