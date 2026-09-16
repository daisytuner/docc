#pragma once

#include "sdfg/transformations/loop_tiling.h"

namespace sdfg {
namespace transformations {

/**
 * @brief Multi-level (two-level) loop tiling transformation
 *
 * This transformation extends LoopTiling by applying a second level of tiling.
 * The result is three nested loops: an outer loop that iterates over outer tiles,
 * a middle loop that iterates over inner tiles, and an innermost loop that
 * iterates over individual elements.
 *
 * @note The outer tile size must be greater than 1
 * @note The inner tile size must be greater than 1 and less than the outer tile size
 * @note The outer tile size must be divisible by the inner tile size
 */
class MultiLevelTiling : public LoopTiling {
    size_t tile_size_2_;

    structured_control_flow::StructuredLoop* middle_loop_ = nullptr;

public:
    /**
     * @brief Construct a two-level loop tiling transformation
     * @param loop The loop to be tiled
     * @param tile_size The size of the outer tile (must be > 1)
     * @param tile_size_2 The size of the inner tile (must be > 1 and < tile_size)
     * @param simplify_bounds Drop the redundant inner bound for perfectly dividing tiles at both
     *        levels (off by default; see LoopTiling::tile_loop)
     */
    MultiLevelTiling(
        structured_control_flow::StructuredLoop& loop, size_t tile_size, size_t tile_size_2, bool simplify_bounds = false
    );

    std::string name() const override;

    bool can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    void apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    void to_json(nlohmann::json& j) const override;

    static MultiLevelTiling from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& j);

    structured_control_flow::StructuredLoop* middle_loop();
};

} // namespace transformations
} // namespace sdfg
