#pragma once

#include <vector>

#include "sdfg/structured_control_flow/map.h"
#include "sdfg/transformations/transformation.h"

namespace sdfg {
namespace transformations {

/**
 * @brief Splits a map into nested maps along the divisions of its induction variable; the inverse of MapCollapse.
 *
 * Flattened loops, e.g. over the elements of a reshaped tensor or after fusing a nest into one, address data through
 * divisions and remainders of the induction variable by constants, such as `A[64*idiv(i, 256) + imod(i, 64)]`. Each
 * chain of divisions and remainders of `i` selects a range of its digits in a mixed radix: `(i mod high) div low`.
 * If the bounds of all those ranges divide one another and the extent of the map, the map becomes one map per digit,
 * outermost first, iterating in the same order, and each chain the weighted sum of its digits, which is affine.
 *
 * @note The map must be sequential, start at 0, step by 1 and have a constant extent.
 */
class MapDelinearize : public Transformation {
    structured_control_flow::Map& map_;
    std::vector<structured_control_flow::Map*> maps_;

public:
    explicit MapDelinearize(structured_control_flow::Map& map);

    std::string name() const override;

    bool can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    void apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    void to_json(nlohmann::json& j) const override;

    static MapDelinearize from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& j);

    /// The maps the map became, outermost first; empty before apply().
    const std::vector<structured_control_flow::Map*>& maps() const;
};

} // namespace transformations
} // namespace sdfg
