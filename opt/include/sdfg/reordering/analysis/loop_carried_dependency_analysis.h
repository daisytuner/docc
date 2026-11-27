#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "sdfg/analysis/analysis.h"
#include "sdfg/analysis/assumptions_analysis.h"
#include "sdfg/analysis/data_dependency_analysis.h"
#include "sdfg/analysis/users.h"
#include "sdfg/structured_control_flow/reduce.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/structured_sdfg.h"
#include "sdfg/symbolic/maps.h"

namespace sdfg {
namespace reordering {

enum LoopCarriedDependency {
    LOOP_CARRIED_DEPENDENCY_READ_WRITE,
    LOOP_CARRIED_DEPENDENCY_WRITE_WRITE,
    LOOP_CARRIED_DEPENDENCY_UNDEFINED,
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
    analysis::User* writer;
    analysis::User* reader;
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
 *
 * Results are computed lazily on the first query for a loop and cached until the analysis is
 * invalidated. The data dependency analysis runs once per outermost loop nest and serves all
 * loops in it; dependence distances and reductions are computed per queried loop.
 */
class LoopCarriedDependencyAnalysis : public analysis::Analysis {
    friend class analysis::AnalysisManager;

private:
    struct LoopResult {
        // False for loops that cannot be analyzed (non-monotonic, no boundary information).
        bool available = false;
        std::unordered_map<std::string, LoopCarriedDependencyInfo> dependencies;
        std::vector<LoopCarriedDependencyPair> pairs;
        // Recognized reductions: loop-carried read-write dependencies realized by a single
        // associative/commutative combine on a loop-invariant accumulator location.
        std::vector<structured_control_flow::ReductionInfo> reductions;
    };

    analysis::AnalysisManager* analysis_manager_ = nullptr;

    std::unordered_map<const structured_control_flow::StructuredLoop*, LoopResult> results_;

    // Detailed data dependency analyses, keyed by the outermost loop of a nest. Constructed
    // manually: the manager-cached DDA runs in a cheaper, conservative mode.
    std::unordered_map<const structured_control_flow::StructuredLoop*, std::unique_ptr<analysis::DataDependencyAnalysis>>
        nest_ddas_;

    // Branch-condition-aware assumptions (the manager-cached AA skips IfElse refinement, which
    // the ISL formulation needs to prove halo-style patterns non-loop-carried).
    std::unique_ptr<analysis::AssumptionsAnalysis> detailed_assumptions_;

    const LoopResult& result(structured_control_flow::StructuredLoop& loop);
    void compute(structured_control_flow::StructuredLoop& loop, LoopResult& result);
    void detect_reductions(structured_control_flow::StructuredLoop& loop, LoopResult& result);
    analysis::DataDependencyAnalysis& nest_dda(structured_control_flow::StructuredLoop& loop);
    analysis::AssumptionsAnalysis& detailed_assumptions();

public:
    LoopCarriedDependencyAnalysis(StructuredSDFG& sdfg);

    std::string name() const override {
        return "LoopCarriedDependencyAnalysis";
    }

    /// Only records the analysis manager; results are computed on demand per loop.
    void run(analysis::AnalysisManager& analysis_manager) override;

    /**
     * @brief Whether the loop can be analyzed (monotonic, with boundary information).
     */
    bool available(structured_control_flow::StructuredLoop& loop);

    /**
     * @brief Per-container summary of loop-carried dependencies for a loop.
     *
     * Returns a map from container name to the (kind, merged delta-set) for
     * each container that participates in any loop-carried dependency.
     */
    const std::unordered_map<std::string, LoopCarriedDependencyInfo>&
    dependencies(structured_control_flow::StructuredLoop& loop);

    /**
     * @brief Per-pair list of loop-carried dependencies for a loop.
     */
    const std::vector<LoopCarriedDependencyPair>& pairs(structured_control_flow::StructuredLoop& loop);

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
        const structured_control_flow::ControlFlowNode& subtree_b
    );

    /**
     * @brief True if any loop-carried dependency exists for the loop.
     */
    bool has_loop_carried(structured_control_flow::StructuredLoop& loop);

    /**
     * @brief True if any loop-carried RAW dependency exists for the loop.
     */
    bool has_loop_carried_raw(structured_control_flow::StructuredLoop& loop);

    /**
     * True if RAW or undefined loop-carried dependency exists. Either is a hazard for example for For2Map /
     * parallization
     */
    bool has_loop_carried_hazard(structured_control_flow::StructuredLoop& loop);

    /**
     * @brief Recognized reductions carried by the loop.
     *
     * Each entry pairs an associative/commutative operator with the accumulator
     * container it combines into. A container appears here iff it has a
     * loop-carried read-write dependency that is realized by a single combine
     * operator (`acc = acc OP x`) over a loop-invariant accumulator location.
     * Empty for loops with no recognized reduction.
     */
    const std::vector<structured_control_flow::ReductionInfo>& reductions(structured_control_flow::StructuredLoop& loop);

    /**
     * @brief True if the loop carries at least one recognized reduction.
     */
    bool has_reductions(structured_control_flow::StructuredLoop& loop);

    /**
     * @brief True if every loop-carried hazard (RAW or undefined) of the loop is
     * a recognized reduction, and at least one reduction exists.
     *
     * This is the precise condition under which a parallelizable `For` is a
     * reduction loop rather than a fully independent (Map) loop: the only
     * cross-iteration dependencies are reorderable accumulations.
     */
    bool is_reduction_only(structured_control_flow::StructuredLoop& loop);
};

} // namespace reordering
} // namespace sdfg
