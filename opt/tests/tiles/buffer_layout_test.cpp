#include "sdfg/tiles/buffer_layout.h"

#include <gtest/gtest.h>

#include <symengine/add.h>
#include <symengine/functions.h>
#include <symengine/integer.h>
#include <symengine/mul.h>
#include <symengine/pow.h>

#include <cstdlib>
#include <vector>

#include "sdfg/symbolic/symbolic.h"
#include "sdfg/tiles/layout.h"

using namespace sdfg;

namespace {

symbolic::MultiExpression ints(const std::vector<long>& xs) {
    symbolic::MultiExpression out;
    for (long x : xs) out.push_back(symbolic::integer(x));
    return out;
}

// Numerically evaluate a concrete-integer expression, including the non-folding
// integer intrinsics (`bit_xor`, `imod`, `idiv`, `iabs`) the swizzle relies on.
long eval(const symbolic::Expression& e) {
    using namespace SymEngine;
    if (is_a<Integer>(*e)) return static_cast<long>(rcp_static_cast<const Integer>(e)->as_int());
    if (is_a<Add>(*e)) {
        long s = 0;
        for (auto& a : e->get_args()) s += eval(a);
        return s;
    }
    if (is_a<Mul>(*e)) {
        long p = 1;
        for (auto& a : e->get_args()) p *= eval(a);
        return p;
    }
    if (is_a<Pow>(*e)) {
        auto args = e->get_args();
        long b = eval(args[0]), ex = eval(args[1]), r = 1;
        for (long i = 0; i < ex; ++i) r *= b;
        return r;
    }
    if (is_a<FunctionSymbol>(*e)) {
        auto name = rcp_static_cast<const FunctionSymbol>(e)->get_name();
        auto args = e->get_args();
        if (name == "bit_xor") return eval(args[0]) ^ eval(args[1]);
        if (name == "imod") return eval(args[0]) % eval(args[1]);
        if (name == "idiv") return eval(args[0]) / eval(args[1]);
        if (name == "iabs") return std::labs(eval(args[0]));
    }
    ADD_FAILURE() << "eval: unhandled expression " << e->__str__();
    return 0;
}

// Row-major linear offset from a buffer's own axes()/subset() — exactly how
// codegen resolves the nested-array memlet.
long packed_offset(const tiles::PackedBuffer& pb, const std::vector<long>& slot, const std::vector<long>& tile) {
    auto axes = pb.axes();
    auto subset = pb.subset(ints(slot), ints(tile));
    EXPECT_EQ(axes.size(), subset.size());
    long off = 0, stride = 1;
    for (int i = static_cast<int>(axes.size()) - 1; i >= 0; --i) {
        off += eval(subset[i]) * stride;
        stride *= eval(axes[i]);
    }
    return off;
}

// Decompose a row-major linear index into per-dim coordinates.
std::vector<long> delin(long lin, const std::vector<long>& sizes) {
    std::vector<long> c(sizes.size());
    for (int i = static_cast<int>(sizes.size()) - 1; i >= 0; --i) {
        c[i] = lin % sizes[i];
        lin /= sizes[i];
    }
    return c;
}

// The core invariant: the scalar-offset layout() agrees with the multidimensional
// axes()/subset() addressing over the whole [slot ++ tile] domain.
void expect_consistent(
    const std::vector<long>& slot_sizes,
    const std::vector<long>& tile_sizes,
    tiles::BufferKind kind,
    size_t coop_warp_span = 0
) {
    tiles::PackedBuffer pb{ints(slot_sizes), ints(tile_sizes), kind, coop_warp_span};
    auto composed = pb.layout();
    long ntile = 1;
    for (long t : tile_sizes) ntile *= t;
    long nslot = 1;
    for (long s : slot_sizes) nslot *= s;
    for (long si = 0; si < nslot; ++si) {
        for (long ti = 0; ti < ntile; ++ti) {
            auto slot = delin(si, slot_sizes);
            auto tile = delin(ti, tile_sizes);
            symbolic::MultiExpression coords;
            for (long v : slot) coords.push_back(symbolic::integer(v));
            for (long v : tile) coords.push_back(symbolic::integer(v));
            EXPECT_EQ(eval(composed.apply_coords(coords)), packed_offset(pb, slot, tile));
        }
    }
}

// The buffer-side view of the lane-contiguity predicate: walking the flat thread
// index (tile fastest within a slot), do consecutive lanes land on consecutive
// buffer offsets? Exactly what the CDNA global_load_lds DMA requires.
bool lane_contiguous_flat(const std::vector<long>& slot_sizes, const std::vector<long>& tile_sizes, tiles::BufferKind kind) {
    tiles::PackedBuffer pb{ints(slot_sizes), ints(tile_sizes), kind};
    auto composed = pb.layout();
    long ntile = 1;
    for (long t : tile_sizes) ntile *= t;
    long nslot = 1;
    for (long s : slot_sizes) nslot *= s;
    long prev = 0;
    for (long t = 0; t < ntile * nslot; ++t) {
        auto sc = delin(t / ntile, slot_sizes); // slot changes slowest
        auto tc = delin(t % ntile, tile_sizes); // tile fastest
        symbolic::MultiExpression coords;
        for (long v : sc) coords.push_back(symbolic::integer(v));
        for (long v : tc) coords.push_back(symbolic::integer(v));
        long off = eval(composed.apply_coords(coords));
        if (t > 0 && off - prev != 1) return false;
        prev = off;
    }
    return true;
}

} // namespace

// MultiDim: dense row-major [slot ++ tile], no swizzle.
TEST(BufferLayoutTest, MultiDim_NoSlots) { expect_consistent({}, {3, 4}, tiles::BufferKind::MultiDim); }
TEST(BufferLayoutTest, MultiDim_WithSlots) { expect_consistent({2}, {3, 4}, tiles::BufferKind::MultiDim); }
TEST(BufferLayoutTest, MultiDim_MultiSlot) { expect_consistent({2, 5}, {4}, tiles::BufferKind::MultiDim); }

// Linearized: fully flat, consistent offset/addressing.
TEST(BufferLayoutTest, Linearized_Consistent) { expect_consistent({2}, {3, 4}, tiles::BufferKind::Linearized); }

// Padded: the per-slot block is widened to inner_stride (coprime with 32).
TEST(BufferLayoutTest, Padded_NextOdd) { expect_consistent({8}, {16}, tiles::BufferKind::Padded); }
TEST(BufferLayoutTest, Padded_CoopWarpSpan) {
    // A cooperative-store span pads inner_stride congruent to it mod 32.
    expect_consistent({8}, {16}, tiles::BufferKind::Padded, /*coop_warp_span=*/4);
}

// Swizzle: natural power-of-two block, inner index XOR-ed by the slot.
TEST(BufferLayoutTest, Swizzle_PowerOfTwoBlock) { expect_consistent({2}, {8}, tiles::BufferKind::Swizzle); }
TEST(BufferLayoutTest, Swizzle_MultiDimTileBlock) {
    // tile {2,4} -> block 8 (power of two); slots {2} fit within the block.
    expect_consistent({2}, {2, 4}, tiles::BufferKind::Swizzle);
}

// ---- lane-contiguity: the predicate that replaces the MultiDim-vs-flat fork ----
// A padding/swizzle-free buffer (Linearized, or a plain MultiDim) is lane-contiguous
// under the natural flat thread order — DMA-legal. Padding and swizzling break it,
// which is exactly why the CDNA global_load_lds path demands the flat layout.
TEST(BufferLayoutTest, LaneContiguous_Linearized) {
    EXPECT_TRUE(lane_contiguous_flat({2}, {8}, tiles::BufferKind::Linearized));
}
TEST(BufferLayoutTest, LaneContiguous_MultiDimNoPad) {
    EXPECT_TRUE(lane_contiguous_flat({2}, {8}, tiles::BufferKind::MultiDim));
}
TEST(BufferLayoutTest, LaneContiguous_PaddedBreaks) {
    EXPECT_FALSE(lane_contiguous_flat({2}, {8}, tiles::BufferKind::Padded));
}
TEST(BufferLayoutTest, LaneContiguous_SwizzleBreaks) {
    EXPECT_FALSE(lane_contiguous_flat({2}, {8}, tiles::BufferKind::Swizzle));
}

// ---- axes() + subset() against hand-computed expectations ---------------------
TEST(BufferLayoutTest, Subset_MultiDim) {
    tiles::PackedBuffer pb{
        {symbolic::integer(2)}, {symbolic::integer(3), symbolic::integer(4)}, tiles::BufferKind::MultiDim
    };
    // axes = [slot ++ tile]; subset = [slot ++ tile indices] verbatim.
    auto axes = pb.axes();
    ASSERT_EQ(axes.size(), 3u);
    EXPECT_EQ(eval(axes[0]), 2);
    EXPECT_EQ(eval(axes[1]), 3);
    EXPECT_EQ(eval(axes[2]), 4);
    auto s = pb.subset({symbolic::integer(1)}, {symbolic::integer(2), symbolic::integer(3)});
    ASSERT_EQ(s.size(), 3u);
    EXPECT_EQ(eval(s[0]), 1);
    EXPECT_EQ(eval(s[1]), 2);
    EXPECT_EQ(eval(s[2]), 3);
}
TEST(BufferLayoutTest, Subset_Padded) {
    // coop=4, tile 16 -> inner_stride 36; subset = [slot, rowmajor(tile)].
    tiles::PackedBuffer pb{{symbolic::integer(8)}, {symbolic::integer(16)}, tiles::BufferKind::Padded, 4};
    auto axes = pb.axes();
    ASSERT_EQ(axes.size(), 2u);
    EXPECT_EQ(eval(axes[0]), 8);
    EXPECT_EQ(eval(axes[1]), 36); // padded inner stride
    auto s = pb.subset({symbolic::integer(5)}, {symbolic::integer(7)});
    ASSERT_EQ(s.size(), 2u);
    EXPECT_EQ(eval(s[0]), 5);
    EXPECT_EQ(eval(s[1]), 7);
}
TEST(BufferLayoutTest, Subset_Swizzle) {
    // subset inner = xor(rowmajor(tile), rowmajor(slot)).
    tiles::PackedBuffer pb{{symbolic::integer(2)}, {symbolic::integer(8)}, tiles::BufferKind::Swizzle};
    auto s = pb.subset({symbolic::integer(1)}, {symbolic::integer(5)});
    ASSERT_EQ(s.size(), 2u);
    EXPECT_EQ(eval(s[0]), 1);
    EXPECT_EQ(eval(s[1]), 5 ^ 1);
}
TEST(BufferLayoutTest, Subset_Linearized) {
    // A single flat axis; subset = rowmajor over [slot ++ tile].
    tiles::PackedBuffer pb{
        {symbolic::integer(2)}, {symbolic::integer(3), symbolic::integer(4)}, tiles::BufferKind::Linearized
    };
    auto axes = pb.axes();
    ASSERT_EQ(axes.size(), 1u);
    EXPECT_EQ(eval(axes[0]), 24);
    auto s = pb.subset({symbolic::integer(1)}, {symbolic::integer(2), symbolic::integer(3)});
    ASSERT_EQ(s.size(), 1u);
    EXPECT_EQ(eval(s[0]), 1 * 12 + 2 * 4 + 3);
}

// ---- scalar methods against hand-computed expectations ------------------------
TEST(BufferLayoutTest, ScalarMethods) {
    tiles::PackedBuffer pb{
        {symbolic::integer(2)}, {symbolic::integer(3), symbolic::integer(4)}, tiles::BufferKind::MultiDim
    };
    EXPECT_EQ(eval(pb.total_size()), 24);
    EXPECT_EQ(eval(pb.tile_total_size()), 12);
    auto d = pb.delinearize_tile(symbolic::integer(7)); // row-major over {3,4}: {1, 3}
    ASSERT_EQ(d.size(), 2u);
    EXPECT_EQ(eval(d[0]), 1);
    EXPECT_EQ(eval(d[1]), 3);
}
TEST(BufferLayoutTest, InnerStride) {
    // Even natural block -> next odd; coop span -> congruent mod 32, multiple of 4.
    EXPECT_EQ(eval(tiles::PackedBuffer({}, {symbolic::integer(16)}, tiles::BufferKind::Padded, 0).inner_stride()), 17);
    EXPECT_EQ(eval(tiles::PackedBuffer({}, {symbolic::integer(16)}, tiles::BufferKind::Padded, 4).inner_stride()), 36);
}
