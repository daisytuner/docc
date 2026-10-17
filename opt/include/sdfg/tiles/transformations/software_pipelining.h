#pragma once

#include <set>
#include <string>
#include <vector>

#include "sdfg/transformations/transformation.h"

namespace sdfg {
class Function;
namespace structured_control_flow {
class ControlFlowNode;
class StructuredLoop;
} // namespace structured_control_flow
namespace transformations {

/**
 * @brief Software-pipeline a sequential panel loop that cooperatively stages a
 *        shared-memory tile each iteration (the structure LocalStorage emits).
 *
 * Targets a sequential loop whose body is
 *   [ barrier; cooperative-copy(global -> shared); barrier; compute(reads shared) ]
 * and rewrites it into an @p stages -deep pipeline so the next panel's global
 * load (issued via `cp.async`) overlaps the current panel's compute:
 *
 *   prologue: async-copy panel 0 -> buf[0]; commit
 *   for p in [0, n):
 *       if p + stages-1 < n: async-copy panel p+stages-1 -> buf[(p+stages-1)%stages]; commit
 *       wait_prior(stages-1); barrier
 *       compute from buf[p % stages]
 *
 * The staged buffer gains a leading `[stages]` axis indexed by `p % stages`.
 * The staging copy's @ref tiles::TileCopyNode has its atom flipped to cp.async
 * (its plan offset biased per stage) and gains a PipelineCommitNode; a
 * PipelineWaitNode fences each panel's reads. Only fires inside a map whose tile
 * target supports asynchronous staging of the copied-in buffers
 * (@ref tiles::TileTarget::supports_async_staging; on GPUs the shared tier), on a
 * compile-time-constant panel count >= @p stages; buffers written by the compute
 * stay single-buffered.
 */
class SoftwarePipelining : public Transformation {
    structured_control_flow::StructuredLoop& loop_;
    size_t stages_;
    // When the loop stages two operands and shared memory can only hold a
    // double buffer of one of them at the target occupancy, pipeline just the
    // first (name-ordered) operand and leave the rest single-buffered +
    // synchronous. This keeps occupancy (fewer shared bytes) while still
    // overlapping the costlier operand's global load.
    bool single_operand_;
    // Register-staged mode: keep one shared buffer and split each staging copy into a
    // load into registers (issued for panel p+1 before compute p) and a store into the
    // buffer after compute p, fenced by barriers:
    //   prologue: copy panel 0; barrier
    //   for p: [p+1 < n: load p+1 -> regs]; compute p; barrier; [p+1 < n: regs -> buf]; barrier
    bool register_staged_;

public:
    explicit SoftwarePipelining(
        structured_control_flow::StructuredLoop& loop,
        size_t stages = 2,
        bool single_operand = false,
        bool register_staged = false
    );

    virtual std::string name() const override;

    /**
     * @brief The async-stageable buffers the copy-in nodes under @p root fill: the
     *        containers a pipeline would multi-buffer. Empty when @p root stages
     *        nothing (so it is not a pipelineable panel).
     *
     * A buffer qualifies when a cooperative @ref tiles::TileCopyNode copies into it
     * (its `_dst` pointer) and the enclosing map's tile target can asynchronously
     * multi-buffer its storage (@ref tiles::TileTarget::supports_async_staging; on
     * GPUs the shared tier). This is the single source of truth for "which buffers
     * get staged", shared by @ref can_be_applied, @ref apply, and search spaces.
     */
    static std::set<std::string>
    copied_in_buffers(const Function& function, structured_control_flow::ControlFlowNode& root);

    /**
     * @brief The candidate panel loops under @p root this transformation can target.
     *
     * A panel is the nearest loop enclosing a cooperative copy-in (a
     * @ref tiles::TileCopyNode that LocalStorage emits) that fills a buffer the
     * loop's enclosing tile target can asynchronously multi-buffer
     * (@ref tiles::TileTarget::supports_async_staging; on GPUs the shared tier).
     * Only sequential loops qualify: a copy-in sitting directly in a parallel Map
     * has no cross-iteration order to pipeline, so it nominates no panel. Returned
     * in first-appearance order. This is the single source of truth for "what can be
     * pipelined", so a search space can enumerate candidates without re-deriving the
     * scan.
     */
    static std::vector<structured_control_flow::StructuredLoop*>
    staging_panels(const Function& function, structured_control_flow::ControlFlowNode& root);

    virtual bool
    can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    virtual void apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    virtual void to_json(nlohmann::json& j) const override;

    static SoftwarePipelining from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& j);
};

} // namespace transformations
} // namespace sdfg
