#pragma once

#include "sdfg/transformations/transformation.h"

namespace sdfg {
namespace transformations {

/**
 * @brief Widen every cooperative shared-staging copy in a subtree to the widest
 *        legal vector transfer, orthogonally to how the copy was produced.
 *
 * Runs after LocalStorage (which emits scalar copies) and, optionally, after
 * SoftwarePipelining (which turns them into minimal-width cp.async). For each
 * copy under a GPU coverage map it either:
 *   - lowers a scalar `assign` tasklet to a synchronous @ref tiles::VectorCopyNode
 *     (float4/float2), or
 *   - re-strides an existing @ref tiles::CpAsyncCopyNode / @ref tiles::VectorCopyNode
 *     to a wider transfer,
 * whenever the copied run is provably contiguous and aligned. Pipelined regions
 * have their @ref tiles::PipelineWaitNode `loads_per_group` recomputed from the
 * widened cp.async widths so the vmcnt fence stays correct.
 *
 * This is the single place vectorization happens; it is always safe to run (a
 * no-op when nothing widens) and composes with any copy representation.
 */
class TileVectorizer : public Transformation {
    structured_control_flow::StructuredLoop& loop_;

public:
    explicit TileVectorizer(structured_control_flow::StructuredLoop& loop);

    virtual std::string name() const override;

    virtual bool can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager)
        override;

    virtual void apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    virtual void to_json(nlohmann::json& j) const override;

    static TileVectorizer from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& j);
};

} // namespace transformations
} // namespace sdfg
