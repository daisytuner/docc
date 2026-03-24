#pragma once

#include "sdfg/structured_control_flow/map.h"
#include "sdfg/transformations/transformation.h"

namespace sdfg {
namespace transformations {

/**
 * @brief Collapse a perfectly-nested map nest into a target number of loops
 *
 * Given an outermost map of a perfectly-nested map chain and a target loop
 * count (1 or 2), this transformation collapses the nest:
 *
 * - `target_loops == 1`: all maps are collapsed into a single map.
 * - `target_loops == 2`: the nest is split into an outer half and an inner
 *   half, each collapsed separately.  For an odd number of maps the outer
 *   half includes the middle loop.
 *
 * @note Delegates to MapCollapse for each individual collapse step.
 */
class CollapseToDepth : public Transformation {
    structured_control_flow::Map& loop_;
    size_t target_loops_;
    bool applied_ = false;

    structured_control_flow::Map* outer_loop_ = nullptr;
    structured_control_flow::Map* inner_loop_ = nullptr;

public:
    /**
     * @brief Construct a collapse to depth transformation
     * @param loop The outermost map of the nest to collapse
     * @param target_loops Number of resulting loops (1 or 2)
     */
    CollapseToDepth(structured_control_flow::Map& loop, size_t target_loops);

    virtual std::string name() const override;

    virtual bool can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager)
        override;

    virtual void apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    virtual void to_json(nlohmann::json& j) const override;

    static CollapseToDepth from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& j);

    /**
     * @brief Get the outer collapsed loop (or the single loop when target_loops == 1)
     */
    structured_control_flow::Map* outer_loop();

    /**
     * @brief Get the inner collapsed loop (nullptr when target_loops == 1)
     */
    structured_control_flow::Map* inner_loop();
};

} // namespace transformations
} // namespace sdfg
