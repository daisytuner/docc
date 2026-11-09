#pragma once

#include "sdfg/transformations/transformation.h"

namespace sdfg {
namespace transformations {

/**
 * @brief Mark per-thread vector copy-ins in a subtree as streaming (nontemporal) loads.
 *
 * Applies to each per-thread (@ref data_flow::ImplementationType_NONE) VectorSync copy-in
 * @ref tiles::TileCopyNode whose source footprint is provably read once per kernel launch:
 * folded with every enclosing in-kernel loop, it is an injective affine layout over a
 * launch-uniform origin. The node's dispatcher then loads the source through the language's
 * streaming copy (ROCm: `__builtin_nontemporal_load`), so the data does not evict reused lines.
 *
 * Whether that pays off depends on the kernel and its context (e.g. whether the source stays
 * cache-resident across launches), so it is opt-in rather than applied by codegen.
 */
class TileStreamingCopy : public Transformation {
    structured_control_flow::StructuredLoop& loop_;

public:
    explicit TileStreamingCopy(structured_control_flow::StructuredLoop& loop);

    virtual std::string name() const override;

    virtual bool
    can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    virtual void apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    virtual void to_json(nlohmann::json& j) const override;

    static TileStreamingCopy from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& j);
};

} // namespace transformations
} // namespace sdfg
