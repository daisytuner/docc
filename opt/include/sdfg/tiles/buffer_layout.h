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
 * The buffer is a staging area `[per-thread slot dims][per-slot tile block]`. Two
 * independent choices shape it: the @ref Placement orders the tile axes (an affine
 * relabelling), and the @ref ConflictPolicy spreads accesses over memory banks
 * (padding or an XOR swizzle) — see the module README.
 */
enum class Placement {
    RowMajor, ///< `[slot ++ tile]`, last tile dim fastest
    ColMajor, ///< `[slot ++ reversed tile]`, first tile dim fastest
    Flat, ///< one flat axis over `[slot ++ tile]` (lane-contiguous staging)
};

enum class ConflictPolicy {
    None,
    PadRow, ///< append @ref PackedBuffer::pad elements to every stored innermost row
    PadSlot, ///< flatten each per-slot block and widen its stride by @ref PackedBuffer::pad elements
    Swizzle, ///< flatten each per-slot block and XOR its index with the slot index
};

/// The packed local buffer as a value: its nested-array shape (@ref axes), the
/// matching per-axis element address (@ref subset), and the scalar-offset
/// @ref layout. The multidimensional type loses no information.
struct PackedBuffer {
    symbolic::MultiExpression slot_sizes; ///< per-thread slot dims (may be empty)
    symbolic::MultiExpression tile_sizes; ///< per-slot tile block dims
    Placement placement = Placement::RowMajor;
    ConflictPolicy conflict = ConflictPolicy::None;
    size_t pad = 0; ///< padding elements of @ref ConflictPolicy::PadRow / @ref ConflictPolicy::PadSlot

    /// Allocated scalar slots, padding included.
    symbolic::Expression total_size() const;
    /// One logical per-slot block = product(tile_sizes).
    symbolic::Expression tile_total_size() const;
    /// Distance between consecutive slots, padding included.
    symbolic::Expression slot_stride() const;

    /// Per-axis extents of the nested-array type, outermost first.
    symbolic::MultiExpression axes() const;
    /// Per-axis index tuple addressing one element (matches @ref axes).
    symbolic::MultiExpression
    subset(const symbolic::MultiExpression& slot_indices, const symbolic::MultiExpression& tile_indices) const;
    /// The scalar element offset as a (possibly swizzled) layout over `[slot ++ tile]`.
    ComposedLayout layout() const;
};

} // namespace tiles
} // namespace sdfg
