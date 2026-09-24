#pragma once

#include "sdfg/transformations/transformation.h"

namespace sdfg {
namespace transformations {

/**
 * @brief Widen every cooperative shared-staging copy in a subtree to the widest
 *        legal vector transfer.
 *
 * Runs after LocalStorage (which emits a scalar @ref tiles::TileCopyNode) and,
 * optionally, after SoftwarePipelining (which flips the node's atom to cp.async).
 * For each @ref tiles::TileCopyNode it widens the transfer — a ScalarSync copy
 * becomes VectorSync, a CpAsync copy keeps its atom — whenever the copied run is
 * provably contiguous and aligned (derived from the plan layouts). Pipelined
 * regions have their @ref tiles::PipelineWaitNode `loads_per_group` recomputed
 * from the widened cp.async widths so the vmcnt fence stays correct.
 *
 * This is the single place vectorization happens; it is always safe to run (a
 * no-op when nothing widens).
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
