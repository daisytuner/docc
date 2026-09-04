#pragma once

#include "sdfg/symbolic/symbolic.h"
#include "sdfg/tiles/layout.h"

namespace sdfg {
namespace tiles {

/**
 * @brief XOR-swizzle as a composition functor over a @ref Layout.
 *
 * XOR is nonlinear, so a swizzle is not a (shape, stride) layout; following
 * CuTe's `Swizzle<B, M, S>` it rides *on top* of a layout's offset. Applied
 * identically to writes and reads, it is a pure relabelling (bank spreading).
 *
 * `Swizzle{bits=B, base=M, shift=S}` XORs the `B`-bit field at position `M` with
 * the field at `M + |S|`:
 *     yyy = (o >> (M + |S|)) & ((1 << B) - 1);   o' = o XOR (yyy << M)
 * an involution; `bits == 0` is the identity.
 */
struct Swizzle {
    int bits = 0; ///< B: width of the swizzled field (0 = identity)
    int base = 0; ///< M: low bit position left untouched below the field
    int shift = 0; ///< S: distance between the two fields (sign = direction)

    bool is_identity() const { return bits == 0; }

    /// Permute a linear offset (a symbolic `bit_xor` of shifted/masked fields).
    symbolic::Expression apply(const symbolic::Expression& offset) const;

    bool operator==(const Swizzle& other) const {
        return bits == other.bits && base == other.base && shift == other.shift;
    }
};

/// A layout with a swizzle on its output offset (`S o L`) — the bank-conflict
/// `Swizzle` placement. `MultiDim`/`Linearized`/`Padded` use an identity swizzle.
struct ComposedLayout {
    Swizzle swizzle;
    Layout layout;

    /// `apply(i) = swizzle.apply(layout.apply(i))`.
    symbolic::Expression apply(const symbolic::Expression& index) const;

    /// `apply_coords(c) = swizzle.apply(layout.apply_coords(c))`.
    symbolic::Expression apply_coords(const symbolic::MultiExpression& coords) const;

    bool is_plain() const { return swizzle.is_identity(); }
};

} // namespace tiles
} // namespace sdfg
