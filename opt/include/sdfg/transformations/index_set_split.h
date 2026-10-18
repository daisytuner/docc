#pragma once

#include <vector>

#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/transformations/transformation.h"

namespace sdfg {
namespace transformations {

/**
 * @brief Splits a loop where branches on its induction variable change, and folds the branches in each part.
 *
 * Branches that compare the induction variable with constants, such as the halves of a concatenation
 * `if (j < 32) ... else ...`, make the body differ between ranges of iterations. The loop is split with
 * LoopSplit into one loop per range, in order, and in each the branches whose conditions are constant over
 * its range fold away. Maps among the parts count from zero, as in loop normal form.
 *
 * @note The loop must step by 1 between constant bounds.
 */
class IndexSetSplit : public Transformation {
    structured_control_flow::StructuredLoop& loop_;
    std::vector<structured_control_flow::StructuredLoop*> loops_;

public:
    explicit IndexSetSplit(structured_control_flow::StructuredLoop& loop);

    std::string name() const override;

    bool can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    void apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    void to_json(nlohmann::json& j) const override;

    static IndexSetSplit from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& j);

    /// The loops the loop became, in order; empty before apply().
    const std::vector<structured_control_flow::StructuredLoop*>& loops() const;
};

} // namespace transformations
} // namespace sdfg
