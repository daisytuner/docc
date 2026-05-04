#pragma once

#include "sdfg/targets/vectorize/schedule.h"
#include "sdfg/transformations/transformation.h"

namespace sdfg {
namespace transformations {

/**
 * @brief VectorizeTransform transformation for Map nodes
 *
 * This transformation changes the schedule type of a Map from sequential
 * to vectorized execution (Vectorize). This enables the loop iterations
 * to be executed in a vectorized manner, potentially improving performance on
 * processors.
 *
 * @note Only applicable to Maps with sequential schedule type
 */
class VectorizeTransform : public Transformation {
    structured_control_flow::Map& map_;

public:
    /**
     * @brief Construct a VectorizeTransform transformation
     * @param map The map to be vectorized
     */
    VectorizeTransform(structured_control_flow::Map& map);

    /**
     * @brief Get the name of this transformation
     * @return "VectorizeTransform"
     */
    virtual std::string name() const override;

    /**
     * @brief Check if this transformation can be applied
     * @param builder The SDFG builder
     * @param analysis_manager The analysis manager
     * @return true if the map is sequential
     */
    virtual bool can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager)
        override;

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
} // namespace sdfg
