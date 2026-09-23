#pragma once

#include <cstddef>
#include <vector>

#include "sdfg/symbolic/assumptions.h"
#include "sdfg/tiles/layout.h"
#include "sdfg/tiles/swizzle.h"

namespace sdfg {
namespace tiles {

/**
 * @file tiled_copy.h
 * @brief A tiled copy plan: move one tile between memory levels.
 *
 * A @ref TiledCopy is a pure *plan* (no SDFG state): `src` maps a tile coordinate
 * to the global element and `dst` to the local-buffer slot (with an optional XOR
 * `dst_swizzle` on the buffer offset). A @ref TileCopyNode carries one and its
 * backend dispatcher lowers the cooperative copy.
 */

/// The per-lane transfer primitive a tiled copy lowers to.
enum class CopyAtom {
    ScalarSync, ///< one element per lane, synchronous store
    VectorSync, ///< a contiguous 4/8/16-byte vector per lane, synchronous
    CpAsync, ///< a contiguous 4/8/16-byte async global->shared cp.async
};

/// Row-major delinearization of a flat index into per-dim coordinates (dim 0
/// slowest), matching the buffer's tile linearization so both plan sides agree.
symbolic::MultiExpression delinearize_rowmajor(const symbolic::Expression& flat, const symbolic::MultiExpression& sizes);

/// Which way the tile moves between the flat (global) side and the packed buffer.
enum class CopyDirection {
    In, ///< read the global tile into the local buffer (copy-in / stage)
    Out, ///< write the local buffer back to the global tile (copy-out / writeback)
};

struct TiledCopy {
    Layout src{symbolic::MultiExpression{}}; ///< tile-local coordinate -> global element
    Layout dst{symbolic::MultiExpression{}}; ///< tile-local coordinate -> local-buffer slot
    CopyAtom atom = CopyAtom::ScalarSync;
    Swizzle dst_swizzle; ///< XOR-swizzle on the buffer (dst) offset; identity = plain
};

/**
 * @brief The ragged-tile boundary predicate, stored per-dim.
 *
 * An over-approximated tile may address out-of-bounds global memory on ragged
 * blocks; the copy runs only where `base + tile_coord <= max` for every dim. Kept
 * as a list of un-discharged dims (not a baked `symbolic::Condition`) so a later
 * pass that proves a dim fully covering can re-discharge it — inheriting the guard
 * simplification loop peeling / condition propagation give the map-nest form.
 */
struct TileGuard {
    struct Dim {
        size_t axis; ///< which tile coordinate (index into @ref tile_sizes)
        symbolic::Expression base; ///< global base of the tile along this dim
        symbolic::Expression max; ///< inclusive global upper bound
    };
    symbolic::MultiExpression tile_sizes; ///< tile extents (the delinearization domain)
    std::vector<Dim> dims; ///< un-discharged dims; empty ⇒ always valid

    bool trivial() const { return dims.empty(); }

    /// Predicate over explicit per-mode tile coordinates (one per @ref tile_sizes).
    symbolic::Condition predicate(const symbolic::MultiExpression& coords) const;
    /// Predicate over the flat index, row-major delinearized into @ref tile_sizes.
    symbolic::Condition predicate(const symbolic::Expression& flat) const;

    void collect_symbols(symbolic::SymbolSet& set) const;
    void replace_symbols(const symbolic::ExpressionMapping& replacements);

    /// Drop dims provably full (`base + size - 1 <= max`) under the assumptions;
    /// returns true if any dim was discharged.
    bool discharge(const symbolic::SymbolSet& parameters, const symbolic::Assumptions& assumptions);
};


} // namespace tiles
} // namespace sdfg
