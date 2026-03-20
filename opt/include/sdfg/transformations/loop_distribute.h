#pragma once

#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/transformations/transformation.h"

namespace sdfg {
namespace transformations {

/**
 * @brief Loop distribution (loop fission) transformation
 *
 * This transformation splits a loop with multiple children into separate
 * loops, each containing one child. This can enable better optimization
 * opportunities by isolating different computations into separate loops.
 *
 * Loop distribution is applied incrementally, distributing the first child
 * into a separate loop each time it is applied.
 *
 * @note The loop must have at least 2 children
 * @note Transitions must not have assignments
 * @note The first child must not write to loop-local containers used by other children
 * @note Dependencies must be either child-local WAW or not used by the child
 */
class LoopDistribute : public Transformation {
    structured_control_flow::StructuredLoop& loop_;

public:
    /**
     * @brief Construct a loop distribute transformation
     * @param loop The loop to be distributed
     */
    LoopDistribute(structured_control_flow::StructuredLoop& loop);

    /**
     * @brief Get the name of this transformation
     * @return "LoopDistribute"
     */
    virtual std::string name() const override;

    /**
     * @brief Check if this transformation can be applied
     * @param builder The SDFG builder
     * @param analysis_manager The analysis manager
     * @return true if the transformation can be applied safely
     */
    virtual bool can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager)
        override;

    /**
     * @brief Apply the loop distribute transformation
     * @param builder The SDFG builder
     * @param analysis_manager The analysis manager
     */
    virtual void apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    /**
     * @brief Serialize this transformation to JSON
     * @param j JSON object to populate
     */
    virtual void to_json(nlohmann::json& j) const override;

    /**
     * @brief Deserialize a loop distribute transformation from JSON
     * @param builder The SDFG builder
     * @param j JSON description of the transformation
     * @return The deserialized transformation
     */
    static LoopDistribute from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& j);
};

} // namespace transformations
} // namespace sdfg
