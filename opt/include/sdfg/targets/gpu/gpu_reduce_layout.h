#pragma once

#include <cstdint>
#include <vector>

#include "sdfg/symbolic/symbolic.h"

namespace sdfg {
namespace gpu {

/**
 * @brief Dense mixed-radix indexing for a strided reduction output footprint
 *
 * Maps reachable original addresses to contiguous partial-buffer slots. Dimensions
 * are stored in increasing source-stride order, with unit-count axes removed.
 * An empty dimension list represents a single output at base.
 */
struct ReductionLayout {
    /// One source-address axis, measured in scalar elements rather than bytes.
    struct Dimension {
        int64_t stride;
        int64_t count;
    };

    /// Original address for the all-zero coordinate.
    symbolic::Expression base;
    std::vector<Dimension> dimensions;
    /// Number of compact slots, equal to the product of axis counts.
    int64_t extent = 1;

    /// Sort and normalize @p axes relative to @p origin.
    /// @throws InvalidSDFGException for a null origin, nonpositive axes, overlapping
    ///         dimensions, or a slot count/source span that overflows int64_t.
    ReductionLayout(symbolic::Expression origin, std::vector<Dimension> axes);

    /// Return the original address for a compact slot.
    /// @pre @p slot is in [0, extent); no runtime bounds check is generated.
    symbolic::Expression unpack(symbolic::Expression slot) const;

    /// Return the compact slot for an original address, omitting gaps between axes.
    /// @pre @p index belongs to this footprint; no membership check is generated.
    symbolic::Expression pack(symbolic::Expression index) const;
};

} // namespace gpu
} // namespace sdfg
