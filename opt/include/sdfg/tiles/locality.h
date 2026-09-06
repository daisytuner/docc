#pragma once

#include <optional>
#include <utility>
#include <vector>

#include "sdfg/analysis/analysis.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/tiles/tile.h"

namespace sdfg {
namespace tiles {

/**
 * @brief How a tile relates to its enclosing parallel loop nest, and the storage
 *        space that relationship demands.
 *
 * A @ref LocalityPlan is the tile's parallel @ref TileAxis partition plus a few
 * loop-context flags; @ref required_space maps it to the memory @ref Space a
 * localizing transformation must stage the tile in (or `std::nullopt` when the
 * schedule cannot be safely localized). Build one with @ref Tile::placement, or
 * directly with @ref LocalityPlan::analyze.
 */
class LocalityPlan {
    std::vector<TileAxis> axes_; ///< enclosing parallel axes, innermost-first
    bool loop_is_outermost_ = false; ///< the localized loop is the outermost loop
    bool loop_has_scratchpad_ = false; ///< the localized loop itself is scratchpad-scheduled
    bool has_scratchpad_descendant_ = false; ///< a scratchpad-scheduled map lives inside the loop body
    /// The localized loop is a scratchpad map whose body has group-scheduled consumers
    /// of the tile: stage once per group into shared, reused by every sibling below.
    bool enclosing_cooperative_ = false;

public:
    LocalityPlan() = default; // an empty plan; fill via analyze()

    /// Construct a plan from known @p axes and loop-context flags (e.g. for tests or
    /// callers that already have the facts); @ref analyze derives the flags instead.
    LocalityPlan(
        std::vector<TileAxis> axes,
        bool loop_is_outermost = false,
        bool loop_has_scratchpad = false,
        bool has_scratchpad_descendant = false,
        bool enclosing_cooperative = false
    )
        : axes_(std::move(axes)), loop_is_outermost_(loop_is_outermost), loop_has_scratchpad_(loop_has_scratchpad),
          has_scratchpad_descendant_(has_scratchpad_descendant), enclosing_cooperative_(enclosing_cooperative) {}

    /// Analyze the placement of a tile with the given @p axes at @p loop
    /// (@ref TileAxis::enclosing produces the axes; @ref Tile::placement is the
    /// usual entry). Fills the loop-context flags from the surrounding nest.
    static LocalityPlan analyze(
        structured_control_flow::StructuredLoop& loop,
        const std::vector<TileAxis>& axes,
        analysis::AnalysisManager& analysis_manager
    );

    const std::vector<TileAxis>& axes() const { return axes_; }
    bool loop_is_outermost() const { return loop_is_outermost_; }
    bool loop_has_scratchpad() const { return loop_has_scratchpad_; }
    bool has_scratchpad_descendant() const { return has_scratchpad_descendant_; }
    bool enclosing_cooperative() const { return enclosing_cooperative_; }

    /// True when a scratchpad-scheduled axis encloses us (we are inside a device kernel).
    bool inside_scratchpad_scope() const;
    /// True when any cooperative scratchpad axis exists (threads share the tile on-chip).
    bool has_scratchpad_cooperative() const;
    /// True when a cooperative scratchpad axis at @p level exists.
    bool has_cooperative_at(Level level) const;
    /// True when a cooperative axis backed only by global memory exists (no private
    /// stack fits: a host-parallel shared operand).
    bool has_global_cooperative() const;
    /// The scratchpad per-thread axes (indvar in a tile base) — each owns a buffer slot.
    std::vector<TileAxis> private_axes() const;
    /// The scratchpad cooperative axes (the shared/copy parallelism axes).
    std::vector<TileAxis> cooperative_axes() const;

    /**
     * @brief The memory space this tile must be staged in, or `std::nullopt` when
     *        the schedule cannot be safely localized.
     *
     * - `Register` — thread-private / sequential buffer (also a CPU stack buffer).
     * - `Shared`   — block-shared GPU buffer.
     * - `Global`   — grid-wide GPU buffer.
     *
     * A cooperative CPU read tile replicates privately (Register); a cooperative
     * write is a reduction/race and declines. A cooperative GPU read tile follows
     * the coarsest cooperative level; warp-only cooperation is served by shuffles
     * (declines).
     */
    std::optional<Space> required_space(bool container_written) const;
};

/// First block-level GPU-offloaded loop in @p loop's body (a cooperative copy /
/// consumer axis for enclosing-scope staging), or nullptr.
structured_control_flow::StructuredLoop* find_block_scheduled_descendant(
    structured_control_flow::StructuredLoop& loop, analysis::AnalysisManager& analysis_manager
);

} // namespace tiles
} // namespace sdfg
