#pragma once

#include "sdfg/structured_control_flow/map.h"
#include "sdfg/transformations/transformation.h"

namespace sdfg {
namespace transformations {

/**
 * @brief Map collapse transformation
 *
 * This transformation collapses `count` tightly-nested maps into a single
 * flat map, increasing the iteration space and exposing more parallelism.
 *
 * @note The outermost map of the nest is passed as `loop`
 * @note `count` must be >= 2
 */
class MapCollapse : public Transformation {
    structured_control_flow::Map& loop_;
    size_t count_;
    bool applied_ = false;

    structured_control_flow::Map* collapsed_loop_ = nullptr;

public:
    /**
     * @brief Construct a map collapse transformation
     * @param loop The outermost map of the nest to collapse
     * @param count The number of maps to collapse (must be >= 2)
     */
    MapCollapse(structured_control_flow::Map& loop, size_t count);

    /**
     * @brief Get the name of this transformation
     * @return "MapCollapse"
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
     * @brief Apply the map collapse transformation
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
     * @brief Deserialize a map collapse transformation from JSON
     * @param builder The SDFG builder
     * @param j JSON description of the transformation
     * @return The deserialized transformation
     */
    static MapCollapse from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& j);

    /**
     * @brief Get the resulting collapsed map after the transformation has been applied
     */
    structured_control_flow::Map* collapsed_loop();
};

} // namespace transformations
} // namespace sdfg
