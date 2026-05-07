#pragma once

#include <unordered_map>
#include <vector>

#include "sdfg/analysis/analysis.h"
#include "sdfg/analysis/data_dependency_analysis.h"
#include "sdfg/analysis/users.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/structured_sdfg.h"
#include "sdfg/symbolic/maps.h"

namespace sdfg {
namespace analysis {

enum LoopCarriedDependency {
    LOOP_CARRIED_DEPENDENCY_READ_WRITE,
    LOOP_CARRIED_DEPENDENCY_WRITE_WRITE,
};

/**
 * @brief Extended loop-carried dependency information including distance vectors.
 *
 * Combines the dependency type (read-write or write-write) with the full
 * ISL delta set representing all possible iteration-distance vectors.
 */
struct LoopCarriedDependencyInfo {
    LoopCarriedDependency type;
    symbolic::maps::DependenceDeltas deltas;
};

/**
 * @brief Per-pair loop-carried dependency record.
 *
 * Captures a single (writer, reader) user pair that participates in a loop-carried
 * dependency, together with the iteration-distance deltas. For write-write
 * dependencies, both `writer` and `reader` are write users (the earlier and the
 * later write in program order, respectively).
 */
struct LoopCarriedDependencyPair {
    User* writer;
    User* reader;
    LoopCarriedDependency type;
    symbolic::maps::DependenceDeltas deltas;
};

/**
 * @brief Standalone loop-carried dependency analysis.
 *
 * Computes per-loop loop-carried (cross-iteration) dependencies by consuming
 * `DataDependencyAnalysis`'s per-loop boundary snapshots (upward-exposed reads
 * and escaping definitions) and pairing them with `symbolic::maps::dependence_deltas`.
 *
 * For a loop L with induction variable i_L:
 *   pairs(L) = { (W,R, RAW, Δ_L(W,R)) : W ∈ esc(L), R ∈ ue(L),
 *                                       cont(W) = cont(R), Δ ≠ ∅ }
 *            ∪ { (W₁,W₂, WAW, Δ_L(W₁,W₂)) : W₁,W₂ ∈ esc(L),
 *                                           cont(W₁) = cont(W₂), Δ ≠ ∅ }
 */
class LoopCarriedDependencyAnalysis : public Analysis {
    friend class AnalysisManager;

private:
    structured_control_flow::Sequence& node_;

    std::unordered_map<
        structured_control_flow::StructuredLoop*,
        std::unordered_map<std::string, LoopCarriedDependencyInfo>>
        dependencies_;

    std::unordered_map<structured_control_flow::StructuredLoop*, std::vector<LoopCarriedDependencyPair>> pairs_;

    // Owned, detailed `DataDependencyAnalysis` instance constructed manually
    // (not through the shared `AnalysisManager` cache). LCDA needs the
    // precise symbolic-subset boundary information; the manager-cached DDA
    // runs in conservative mode for performance.
    std::unique_ptr<DataDependencyAnalysis> detailed_dda_;
    DataDependencyAnalysis& detailed_dda();

    void analyze_loop(analysis::AnalysisManager& analysis_manager, structured_control_flow::StructuredLoop& loop);

public:
    LoopCarriedDependencyAnalysis(StructuredSDFG& sdfg);

    LoopCarriedDependencyAnalysis(StructuredSDFG& sdfg, structured_control_flow::Sequence& node);

    std::string name() const override { return "LoopCarriedDependencyAnalysis"; }

    void run(analysis::AnalysisManager& analysis_manager) override;

    /**
     * @brief Whether this analysis has results for the given loop (i.e. the loop
     * was visited and analyzed). Loops outside the analysis scope return false.
     */
    bool available(structured_control_flow::StructuredLoop& loop) const;

    /**
     * @brief Per-container summary of loop-carried dependencies for a loop.
     *
     * Returns a map from container name to the (kind, merged delta-set) for
     * each container that participates in any loop-carried dependency.
     */
    const std::unordered_map<std::string, LoopCarriedDependencyInfo>& dependencies(structured_control_flow::StructuredLoop&
                                                                                       loop) const;

    /**
     * @brief Per-pair list of loop-carried dependencies for a loop.
     */
    const std::vector<LoopCarriedDependencyPair>& pairs(structured_control_flow::StructuredLoop& loop) const;

    /**
     * @brief Filter `pairs(loop)` to those whose writer lies in `subtree_a` and
     * reader lies in `subtree_b` (or vice versa for WAW between two writes).
     *
     * Useful for transformations like LoopDistribute that need to ask whether
     * any cross-iteration dependency crosses a particular partition boundary.
     */
    std::vector<const LoopCarriedDependencyPair*> pairs_between(
        structured_control_flow::StructuredLoop& loop,
        const structured_control_flow::ControlFlowNode& subtree_a,
        const structured_control_flow::ControlFlowNode& subtree_b,
        analysis::AnalysisManager& analysis_manager
    ) const;

    /**
     * @brief True if any loop-carried dependency exists for the loop.
     */
    bool has_loop_carried(structured_control_flow::StructuredLoop& loop) const;

    /**
     * @brief True if any loop-carried RAW dependency exists for the loop.
     */
    bool has_loop_carried_raw(structured_control_flow::StructuredLoop& loop) const;
};

} // namespace analysis
} // namespace sdfg
