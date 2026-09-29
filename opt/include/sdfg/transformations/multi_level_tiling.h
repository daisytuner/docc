#pragma once

#include <cstddef>
#include <vector>

#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/transformations/loop_tiling.h"

namespace sdfg {
namespace transformations {

/**
 * @brief Multi-level loop tiling transformation
 *
 * This transformation extends LoopTiling by applying multiple levels of tiling. Therefore, at least two tile sizes are
 * expected. For n tile sizes, the result is n + 1 nested loops: n loops that iterate over the corresponding tile size,
 * and an innermost loop that iterates over individual elements.
 *
 * @note The outermost tile size must be greater than 1.
 * @note Every inner tile size must be greate than 1 and less than the tile size from one level outter.
 * @note Every tile size must be divisible by the tile size one level inner.
 */
class MultiLevelTiling : public LoopTiling {
    std::vector<size_t> additional_tile_sizes_;

    std::vector<structured_control_flow::StructuredLoop*> middle_loops_;

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

    /**
     * @brief Construct a multi-level loop tiling transformation
     * @param loop The loop to be tiled
     * @param tile_sizes The tile sizes (must be at least 2)
     * @param simplify_bounds Drop the redundant inner bound for perfectly dividing tiles at all levels (off by default;
     *        see LoopTiling::tile_loop)
     */
    MultiLevelTiling(
        structured_control_flow::StructuredLoop& loop,
        const std::vector<size_t>& tile_sizes,
        bool simplify_bounds = false
    );

    std::string name() const override;

    bool can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    void apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    void to_json(nlohmann::json& j) const override;

    static MultiLevelTiling from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& j);

    const std::vector<structured_control_flow::StructuredLoop*>& middle_loops();

    structured_control_flow::StructuredLoop* middle_loop(size_t idx);

    /// @deprecated Kept for backwards compatibility
    [[deprecated("Use middle_loop with index instead")]] structured_control_flow::StructuredLoop* middle_loop();
};

} // namespace transformations
} // namespace sdfg
