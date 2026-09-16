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
protected:
    structured_control_flow::StructuredLoop& loop_;
    size_t tile_size_;
    bool simplify_bounds_ = false;
    bool applied_ = false;

    structured_control_flow::StructuredLoop* inner_loop_ = nullptr;
    structured_control_flow::StructuredLoop* outer_loop_ = nullptr;

    /**
     * @brief Tile a single loop into an outer tile loop and an inner element loop
     *
     * Splits @p loop in place: @p loop becomes the inner (element) loop and a new
     * outer loop iterating over tiles is created before it. Map/Reduce loops
     * preserve their schedule/reductions on the outer loop, while a tiled Map's
     * inner loop is converted to a sequential schedule to avoid repeated GPU
     * dimensions.
     *
     * @param builder The SDFG builder
     * @param loop The loop to tile (becomes the inner loop)
     * @param tile_size The size of each tile (must be > 1)
     * @param simplify_bounds Drop the redundant original bound on the inner loop when the tile
     *        evenly divides the (constant) trip count, yielding a clean constant-trip tile that
     *        unrolls/vectorizes. Off by default: keeping the guard preserves the loop shape later
     *        passes (e.g. cooperative-copy vectorization) rely on.
     * @return The newly created outer tile loop
     */
    static structured_control_flow::StructuredLoop& tile_loop(
        builder::StructuredSDFGBuilder& builder,
        structured_control_flow::StructuredLoop& loop,
        size_t tile_size,
        bool simplify_bounds = false
    );

public:
    /**
     * @brief Construct a loop tiling transformation
     * @param loop The loop to be tiled
     * @param tile_size The size of each tile (must be > 1)
     * @param simplify_bounds Drop the redundant inner bound for perfectly dividing tiles (off by
     *        default; see @ref tile_loop)
     */
    LoopTiling(structured_control_flow::StructuredLoop& loop, size_t tile_size, bool simplify_bounds = false);

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
