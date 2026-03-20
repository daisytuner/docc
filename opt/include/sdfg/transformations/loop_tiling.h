#pragma once

#include "sdfg/transformations/transformation.h"

namespace sdfg {
namespace transformations {

/**
 * @brief Loop tiling (blocking) transformation
 *
 * This transformation splits a loop into two nested loops: an outer loop that
 * iterates over tiles and an inner loop that iterates within each tile.
 * Loop tiling improves data locality by ensuring that data accessed within
 * a tile fits in cache.
 *
 * @note The loop must be contiguous (analyzed via LoopAnalysis)
 * @note The tile size must be greater than 1
 */
class LoopTiling : public Transformation {
    structured_control_flow::StructuredLoop& loop_;
    size_t tile_size_;
    bool applied_ = false;

    structured_control_flow::StructuredLoop* inner_loop_ = nullptr;
    structured_control_flow::StructuredLoop* outer_loop_ = nullptr;


public:
    /**
     * @brief Construct a loop tiling transformation
     * @param loop The loop to be tiled
     * @param tile_size The size of each tile (must be > 1)
     */
    LoopTiling(structured_control_flow::StructuredLoop& loop, size_t tile_size);

    /**
     * @brief Get the name of this transformation
     * @return "LoopTiling"
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
     * @brief Apply the loop tiling transformation
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
     * @brief Deserialize a loop tiling transformation from JSON
     * @param builder The SDFG builder
     * @param j JSON description of the transformation
     * @return The deserialized transformation
     */
    static LoopTiling from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& j);

    structured_control_flow::StructuredLoop* inner_loop();
    structured_control_flow::StructuredLoop* outer_loop();
};

} // namespace transformations
} // namespace sdfg
