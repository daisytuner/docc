#pragma once

#include "sdfg/transformations/transformation.h"

namespace sdfg {
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
 * The shared buffer gains a leading `[stages]` axis indexed by `p % stages`.
 * The synchronous copy tasklets become CpAsyncCopyNode + PipelineCommitNode; a
 * PipelineWaitNode fences each panel's reads. Only fires on CUDA (ROCm has no
 * portable cp.async), a compile-time-constant panel count >= @p stages, and a
 * genuinely block-cooperative shared tile — otherwise the extra buffer wastes
 * shared memory without overlap.
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
    // Widen each pipelined cp.async to a 16-byte (float4) transfer by striding
    // the cooperative-copy map by 4. Only sound when the copied run is
    // contiguous and 16-byte aligned; opt-in because clang cannot vectorize the
    // cp.async intrinsic (its width is whatever we emit).
    bool vectorize_;

public:
    explicit SoftwarePipelining(
        structured_control_flow::StructuredLoop& loop,
        size_t stages = 2,
        bool single_operand = false,
        bool vectorize = false
    );

    virtual std::string name() const override;

    virtual bool can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager)
        override;

    virtual void apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    virtual void to_json(nlohmann::json& j) const override;

    static SoftwarePipelining from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& j);
};

} // namespace transformations
} // namespace sdfg
