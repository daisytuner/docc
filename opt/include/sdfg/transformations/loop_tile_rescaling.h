#pragma once

#include "sdfg/transformations/transformation.h"

namespace sdfg {
namespace transformations {

/**
 * @brief Rescales the tile size of an already-tiled loop pair.
 *
 * After LoopTiling has been applied, the SDFG contains an outer tile-loop
 * (indvar `i_tile`, update `i_tile + old_tile_size`) and an inner point-loop
 * (indvar `i`, init `i_tile`, condition `i < i_tile + old_tile_size && ...`).
 * This transformation replaces `old_tile_size` with `new_tile_size` in both
 * the outer update expression and the inner loop condition, leaving everything
 * else intact.
 *
 * @note Precondition: the outer loop's update must equal `indvar + old_tile_size`
 *       and the inner loop's init must equal the outer indvar.
 * @note new_tile_size == old_tile_size is a no-op (can_be_applied returns true,
 *       apply does nothing).
 */
class LoopTileRescaling : public Transformation {
    structured_control_flow::StructuredLoop& outer_loop_;
    /// Located during can_be_applied; must not be used before that call.
    structured_control_flow::StructuredLoop* inner_loop_ = nullptr;
    size_t new_tile_size_;

    /// Derived during can_be_applied; used in apply.
    size_t old_tile_size_ = 0;

public:
    /**
     * @param outer_loop   The outer tile-loop produced by LoopTiling.
     * @param new_tile_size New tile size to apply. The single inner loop is
     *                      located automatically from the outer loop's body
     *                      during can_be_applied.
     */
    LoopTileRescaling(
        structured_control_flow::StructuredLoop& outer_loop,
        size_t new_tile_size
    );

    std::string name() const override;

    bool can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    void apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    void to_json(nlohmann::json& j) const override;

    static LoopTileRescaling from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& j);
};

} // namespace transformations
} // namespace sdfg
