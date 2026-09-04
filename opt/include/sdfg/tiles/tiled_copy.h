#pragma once

#include <cstddef>
#include <optional>
#include <string>

#include "sdfg/tiles/layout.h"

namespace sdfg {
namespace tiles {

/**
 * @file tiled_copy.h
 * @brief A CuTe-style tiled copy: move one tile between memory levels under a
 *        (thread, value) partition, with a legality predicate over the algebra.
 *
 * A @ref TiledCopy is a pure *plan* (no SDFG state): `src` maps a tile coordinate
 * to the global element, `dst` to the local-buffer slot, and (`thread`, `value`)
 * partition the tile via `partition = concat(value, thread)`. @ref verify decides
 * legality on the host, catching two bug-classes that otherwise surface as GPU
 * faults: a `cp.async` width not in {4,8,16} and a lane-contiguous DMA whose dst is
 * not stride-1 across lanes (the ROCm `global_load_lds` scramble). See the README.
 */

/// The per-lane transfer primitive a tiled copy lowers to.
enum class CopyAtom {
    ScalarSync, ///< one element per lane, synchronous store
    VectorSync, ///< a contiguous 4/8/16-byte vector per lane, synchronous
    CpAsync, ///< a contiguous 4/8/16-byte async global->shared cp.async
    LaneContiguousDMA, ///< CDNA global->LDS DMA: 4-byte/lane, lanes contiguous in dst
};

struct TiledCopy {
    Layout src; ///< tile-local coordinate -> global element
    Layout dst; ///< tile-local coordinate -> local-buffer slot
    Layout thread; ///< thread index -> tile coordinate of that lane's slice base
    Layout value; ///< value index -> tile coordinate within a lane's slice
    CopyAtom atom = CopyAtom::ScalarSync;

    /// The combined (thread, value) partition of the tile: `concat(value, thread)`.
    Layout partition() const;

    /// Legality of the plan: `std::nullopt` when provably correct and the atom is
    /// legal, else a reason. Checks (1) src/dst span the same tile; (2) the
    /// partition covers it bijectively; (3) dst is injective; (4) atom width /
    /// contiguity / lane-contiguity. Pure — no SDFG, no device code.
    std::optional<std::string> verify(size_t elem_bytes) const;
};

/// Does @p dst place consecutive lanes on consecutive slots
/// (`dst(thread(l+1)) - dst(thread(l)) == 1`)? This is the lane-contiguity the
/// CDNA `global_load_lds` DMA requires, and the predicate that replaces a hard
/// buffer-kind fork: any satisfying dst is DMA-legal. Trivially true for 0/1 lanes.
bool is_lane_contiguous(const Layout& dst, const Layout& thread);

} // namespace tiles
} // namespace sdfg
