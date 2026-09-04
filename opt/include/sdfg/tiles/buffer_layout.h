#pragma once

#include "sdfg/symbolic/symbolic.h"
#include "sdfg/tiles/layout.h"
#include "sdfg/tiles/swizzle.h"

namespace sdfg {
namespace tiles {

/**
 * @file buffer_layout.h
 * @brief The destination side of a tiled copy — the packed local buffer — as a
 *        (possibly swizzled) @ref Layout over the coordinate `[slot ++ tile]`.
 *
 * The buffer is a dense staging area `[per-thread slot dims][per-slot tile block]`;
 * modelling its element offset lets the emitter address it from the algebra. The
 * four placements (@ref BufferKind) differ only in the per-slot block and an
 * optional swizzle — see the module README.
 */
enum class BufferKind {
    MultiDim, ///< dense `[slot dims ++ tile dims]`
    Padded, ///< `[slot dims ++ padded flat block]` (bank-conflict avoidance)
    Swizzle, ///< `[slot dims ++ natural flat block]`, inner index XOR-swizzled
    Linearized, ///< fully flat `[slot ++ tile]` (same offset as MultiDim)
};

/// Build the packed buffer as a @ref ComposedLayout over `[slot ++ tile]`;
/// `apply_coords(slot ++ tile)` is the scalar offset. @p inner_stride applies
/// only to @ref BufferKind::Padded (others use the natural `product(tile_sizes)`).
ComposedLayout buffer_layout(
    const symbolic::MultiExpression& slot_sizes,
    const symbolic::MultiExpression& tile_sizes,
    BufferKind kind,
    const symbolic::Expression& inner_stride
);

/// The packed local buffer as a value: its nested-array shape (@ref axes), the
/// matching per-axis element address (@ref subset), and the scalar-offset
/// @ref layout. The multidimensional type loses no information.
struct PackedBuffer {
    symbolic::MultiExpression slot_sizes; ///< per-thread slot dims (may be empty)
    symbolic::MultiExpression tile_sizes; ///< per-slot tile block dims
    BufferKind kind = BufferKind::MultiDim; ///< placement
    /// Cooperative-store span (per-warp thread count of the coop axis): pads the
    /// Padded inner stride congruent to it mod 32 so a warp's stores are
    /// bank-conflict-free. 0 falls back to the next coprime-with-32 (odd) stride.
    size_t coop_warp_span = 0;

    /// Total scalar slots = product(slot_sizes) * product(tile_sizes).
    symbolic::Expression total_size() const;
    /// One per-slot block = product(tile_sizes).
    symbolic::Expression tile_total_size() const;
    /// Padded per-slot block stride (Padded only; @ref tile_total_size otherwise).
    symbolic::Expression inner_stride() const;
    /// Row-major decomposition of a flat tile index into per-tile-dim indices.
    std::vector<symbolic::Expression> delinearize_tile(const symbolic::Expression& flat) const;

    /// Per-axis extents of the nested-array type, outermost first.
    symbolic::MultiExpression axes() const;
    /// Per-axis index tuple addressing one element (matches @ref axes).
    symbolic::MultiExpression
    subset(const symbolic::MultiExpression& slot_indices, const symbolic::MultiExpression& tile_indices) const;
    /// The scalar element offset as a (possibly swizzled) layout over `[slot ++ tile]`.
    ComposedLayout layout() const { return buffer_layout(slot_sizes, tile_sizes, kind, inner_stride()); }
};

} // namespace tiles
} // namespace sdfg
