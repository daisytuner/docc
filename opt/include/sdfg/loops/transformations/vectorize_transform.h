#pragma once

#include "sdfg/transformations/transformation.h"

namespace sdfg {
namespace loops {
namespace transformations {

/**
 * @brief VectorizeTransform transformation for StructuredLoop nodes
 *
 * Sets the `"vectorize"` property on the loop's existing schedule (preserving its
 * kind), which the loop codegen lowers to
 * `#pragma clang loop vectorize(enable) interleave(enable)`.
 *
 * @note Only applicable to Maps and Reduces with a sequential (non-parallel,
 *       non-offloaded) schedule that are not vectorized yet
 */
class VectorizeTransform : public ::sdfg::transformations::Transformation {
    structured_control_flow::StructuredLoop& loop_;

public:
    /**
     * @brief Construct a VectorizeTransform transformation
     * @param loop The loop to be vectorized
     */
    VectorizeTransform(structured_control_flow::StructuredLoop& loop);

    /**
     * @brief Get the name of this transformation
     * @return "VectorizeTransform"
     */
    virtual std::string name() const override;

    /**
     * @brief Check if this transformation can be applied
     * @param builder The SDFG builder
     * @param analysis_manager The analysis manager
     * @return true if the loop is a sequential, not yet vectorized Map or Reduce
     */
    virtual bool
    can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    /**
     * @brief Apply the VectorizeTransform transformation
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
     * @brief Deserialize a VectorizeTransform transformation from JSON
     * @param builder The SDFG builder
     * @param j JSON description of the transformation
     * @return The deserialized transformation
     */
    static VectorizeTransform from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& j);
};

} // namespace transformations
} // namespace loops
} // namespace sdfg
