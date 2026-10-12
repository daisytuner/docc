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
    for (long x : xs) {
        out.push_back(symbolic::integer(x));
    }
    return out;
}

// Numerically evaluate a concrete-integer expression, including the non-folding
// integer intrinsics (`bit_xor`, `imod`, `idiv`, `iabs`) the swizzle relies on.
long eval(const symbolic::Expression& e) {
    using namespace SymEngine;
    if (is_a<Integer>(*e)) {
        return static_cast<long>(rcp_static_cast<const Integer>(e)->as_int());
    }
    if (is_a<Add>(*e)) {
        long s = 0;
        for (auto& a : e->get_args()) {
            s += eval(a);
        }
        return s;
    }
    if (is_a<Mul>(*e)) {
        long p = 1;
        for (auto& a : e->get_args()) {
            p *= eval(a);
        }
        return p;
    }
    if (is_a<Pow>(*e)) {
        auto args = e->get_args();
        long b = eval(args[0]), ex = eval(args[1]), r = 1;
        for (long i = 0; i < ex; ++i) {
            r *= b;
        }
        return r;
    }
    if (is_a<FunctionSymbol>(*e)) {
        auto name = rcp_static_cast<const FunctionSymbol>(e)->get_name();
        auto args = e->get_args();
        if (name == "bit_xor") {
            return eval(args[0]) ^ eval(args[1]);
        }
        if (name == "imod") {
            return eval(args[0]) % eval(args[1]);
        }
        if (name == "idiv") {
            return eval(args[0]) / eval(args[1]);
        }
        if (name == "iabs") {
            return std::labs(eval(args[0]));
        }
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

tiles::PackedBuffer make(
    const std::vector<long>& slot_sizes,
    const std::vector<long>& tile_sizes,
    tiles::Placement placement = tiles::Placement::RowMajor,
    tiles::ConflictPolicy conflict = tiles::ConflictPolicy::None,
    size_t pad = 0
) {
    return tiles::PackedBuffer{ints(slot_sizes), ints(tile_sizes), placement, conflict, pad};
}

std::vector<long> longs(const symbolic::MultiExpression& xs) {
    std::vector<long> out;
    for (const auto& x : xs) {
        out.push_back(eval(x));
    }
    return out;
}

long product(const std::vector<long>& xs) {
    long p = 1;
    for (long x : xs) {
        p *= x;
    }
    return p;
}

// The core invariant: the scalar-offset layout() agrees with the multidimensional
// axes()/subset() addressing over the whole [slot ++ tile] domain.
void expect_consistent(const tiles::PackedBuffer& pb) {
    const auto slot_sizes = longs(pb.slot_sizes);
    const auto tile_sizes = longs(pb.tile_sizes);
    auto composed = pb.layout();
    for (long si = 0; si < product(slot_sizes); ++si) {
        for (long ti = 0; ti < product(tile_sizes); ++ti) {
            auto slot = delin(si, slot_sizes);
            auto tile = delin(ti, tile_sizes);
            symbolic::MultiExpression coords;
            for (long v : slot) {
                coords.push_back(symbolic::integer(v));
            }
            for (long v : tile) {
                coords.push_back(symbolic::integer(v));
            }
            EXPECT_EQ(eval(composed.apply_coords(coords)), packed_offset(pb, slot, tile));
        }
    }
}

// The buffer-side view of the lane-contiguity predicate: walking the flat thread
// index (tile fastest within a slot), do consecutive lanes land on consecutive
// buffer offsets? Exactly what the CDNA global_load_lds DMA requires.
bool lane_contiguous_flat(const tiles::PackedBuffer& pb) {
    const auto slot_sizes = longs(pb.slot_sizes);
    const auto tile_sizes = longs(pb.tile_sizes);
    const long ntile = product(tile_sizes);
    auto composed = pb.layout();
    long prev = 0;
    for (long t = 0; t < ntile * product(slot_sizes); ++t) {
        auto sc = delin(t / ntile, slot_sizes); // slot changes slowest
        auto tc = delin(t % ntile, tile_sizes); // tile fastest
        symbolic::MultiExpression coords;
        for (long v : sc) {
            coords.push_back(symbolic::integer(v));
        }
        for (long v : tc) {
            coords.push_back(symbolic::integer(v));
        }
        long off = eval(composed.apply_coords(coords));
        if (t > 0 && off - prev != 1) {
            return false;
        }
        prev = off;
    }
    return true;
}

} // namespace

using tiles::ConflictPolicy;
using tiles::Placement;

// RowMajor: dense row-major [slot ++ tile], no swizzle.
TEST(BufferLayoutTest, RowMajor_NoSlots) {
    expect_consistent(make({}, {3, 4}));
}
TEST(BufferLayoutTest, RowMajor_WithSlots) {
    expect_consistent(make({2}, {3, 4}));
}
TEST(BufferLayoutTest, RowMajor_MultiSlot) {
    expect_consistent(make({2, 5}, {4}));
}

// Row pad: each stored innermost row is widened, the logical tile is unchanged.
TEST(BufferLayoutTest, RowMajor_PadRow) {
    auto pb = make({2}, {3, 4}, Placement::RowMajor, ConflictPolicy::PadRow, 2);
    expect_consistent(pb);
    auto axes = pb.axes();
    ASSERT_EQ(axes.size(), 3u);
    EXPECT_EQ(eval(axes[2]), 6);
    EXPECT_EQ(eval(pb.total_size()), 2 * 3 * 6);
    EXPECT_EQ(eval(pb.tile_total_size()), 3 * 4);
    auto layout = pb.layout().layout;
    EXPECT_EQ(eval(layout.strides()[0]), 18); // slot = 3 rows of 6
    EXPECT_EQ(eval(layout.strides()[1]), 6); // padded row
    EXPECT_EQ(eval(layout.strides()[2]), 1);
    EXPECT_EQ(eval(layout.shape()[2]), 4); // logical row length
}

// Flat: one flat axis, consistent offset/addressing.
TEST(BufferLayoutTest, Flat_Consistent) {
    expect_consistent(make({2}, {3, 4}, Placement::Flat));
}

// PadSlot: each per-slot block is flattened and its stride widened.
TEST(BufferLayoutTest, PadSlot_Consistent) {
    expect_consistent(make({8}, {16}, Placement::RowMajor, ConflictPolicy::PadSlot, 1));
    expect_consistent(make({8}, {16}, Placement::RowMajor, ConflictPolicy::PadSlot, 20));
    expect_consistent(make({2}, {3, 4}, Placement::ColMajor, ConflictPolicy::PadSlot, 3));
}

// Swizzle: natural power-of-two block, inner index XOR-ed by the slot.
TEST(BufferLayoutTest, Swizzle_PowerOfTwoBlock) {
    expect_consistent(make({2}, {8}, Placement::RowMajor, ConflictPolicy::Swizzle));
}
TEST(BufferLayoutTest, Swizzle_MultiDimTileBlock) {
    // tile {2,4} -> block 8 (power of two); slots {2} fit within the block.
    expect_consistent(make({2}, {2, 4}, Placement::RowMajor, ConflictPolicy::Swizzle));
}

// ColMajor: dense, but the tile axes are stored column-major (reversed).
TEST(BufferLayoutTest, ColMajor_NoSlots) {
    expect_consistent(make({}, {3, 4}, Placement::ColMajor));
}
TEST(BufferLayoutTest, ColMajor_WithSlots) {
    expect_consistent(make({2}, {4, 8}, Placement::ColMajor));
}
// The reversed axes physically swap a 2D tile: axes() is [N, M] for a tile [M, N].
TEST(BufferLayoutTest, ColMajor_ReversesAxes) {
    auto pb = make({}, {3, 4}, Placement::ColMajor);
    auto axes = pb.axes();
    ASSERT_EQ(axes.size(), 2u);
    EXPECT_EQ(eval(axes[0]), 4); // N
    EXPECT_EQ(eval(axes[1]), 3); // M
    // Element (m=1, n=2) lands at physical [n=2][m=1] -> 2*3 + 1 = 7.
    EXPECT_EQ(eval(pb.layout().apply_coords({symbolic::integer(1), symbolic::integer(2)})), 7);
}
// Placement and conflict policy compose: a column-major tile pads its stored row (tile dim 0).
TEST(BufferLayoutTest, ColMajor_PadRowWidensDim0) {
    auto pb = make({}, {64, 32}, Placement::ColMajor, ConflictPolicy::PadRow, 4);
    expect_consistent(make({2}, {6, 4}, Placement::ColMajor, ConflictPolicy::PadRow, 4));
    auto axes = pb.axes();
    ASSERT_EQ(axes.size(), 2u);
    EXPECT_EQ(eval(axes[0]), 32);
    EXPECT_EQ(eval(axes[1]), 68);
    EXPECT_EQ(eval(pb.total_size()), 32 * 68);
    auto layout = pb.layout().layout;
    EXPECT_EQ(eval(layout.strides()[0]), 1);
    EXPECT_EQ(eval(layout.strides()[1]), 68);
    EXPECT_EQ(eval(pb.layout().apply_coords({symbolic::integer(5), symbolic::integer(2)})), 2 * 68 + 5);
}

// ---- lane-contiguity: the predicate that replaces the RowMajor-vs-flat fork ----
// A padding/swizzle-free buffer (Flat, or a plain RowMajor) is lane-contiguous
// under the natural flat thread order — DMA-legal. Padding and swizzling break it,
// which is exactly why the CDNA global_load_lds path demands the flat layout.
TEST(BufferLayoutTest, LaneContiguous_Flat) {
    EXPECT_TRUE(lane_contiguous_flat(make({2}, {8}, Placement::Flat)));
}
TEST(BufferLayoutTest, LaneContiguous_RowMajorNoPad) {
    EXPECT_TRUE(lane_contiguous_flat(make({2}, {8})));
}
TEST(BufferLayoutTest, LaneContiguous_PadSlotBreaks) {
    EXPECT_FALSE(lane_contiguous_flat(make({2}, {8}, Placement::RowMajor, ConflictPolicy::PadSlot, 1)));
}
TEST(BufferLayoutTest, LaneContiguous_SwizzleBreaks) {
    EXPECT_FALSE(lane_contiguous_flat(make({2}, {8}, Placement::RowMajor, ConflictPolicy::Swizzle)));
}

// ---- axes() + subset() against hand-computed expectations ---------------------
TEST(BufferLayoutTest, Subset_RowMajor) {
    auto pb = make({2}, {3, 4});
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
TEST(BufferLayoutTest, Subset_PadSlot) {
    // tile 16 padded by 20 -> slot stride 36; subset = [slot, rowmajor(tile)].
    auto pb = make({8}, {16}, Placement::RowMajor, ConflictPolicy::PadSlot, 20);
    auto axes = pb.axes();
    ASSERT_EQ(axes.size(), 2u);
    EXPECT_EQ(eval(axes[0]), 8);
    EXPECT_EQ(eval(axes[1]), 36);
    auto s = pb.subset({symbolic::integer(5)}, {symbolic::integer(7)});
    ASSERT_EQ(s.size(), 2u);
    EXPECT_EQ(eval(s[0]), 5);
    EXPECT_EQ(eval(s[1]), 7);
}
TEST(BufferLayoutTest, Subset_Swizzle) {
    // subset inner = xor(rowmajor(tile), rowmajor(slot)).
    auto pb = make({2}, {8}, Placement::RowMajor, ConflictPolicy::Swizzle);
    auto s = pb.subset({symbolic::integer(1)}, {symbolic::integer(5)});
    ASSERT_EQ(s.size(), 2u);
    EXPECT_EQ(eval(s[0]), 1);
    EXPECT_EQ(eval(s[1]), 5 ^ 1);
}
TEST(BufferLayoutTest, Subset_Flat) {
    // A single flat axis; subset = rowmajor over [slot ++ tile].
    auto pb = make({2}, {3, 4}, Placement::Flat);
    auto axes = pb.axes();
    ASSERT_EQ(axes.size(), 1u);
    EXPECT_EQ(eval(axes[0]), 24);
    auto s = pb.subset({symbolic::integer(1)}, {symbolic::integer(2), symbolic::integer(3)});
    ASSERT_EQ(s.size(), 1u);
    EXPECT_EQ(eval(s[0]), 1 * 12 + 2 * 4 + 3);
}

// ---- scalar methods against hand-computed expectations ------------------------
TEST(BufferLayoutTest, ScalarMethods) {
    auto pb = make({2}, {3, 4});
    EXPECT_EQ(eval(pb.total_size()), 24);
    EXPECT_EQ(eval(pb.tile_total_size()), 12);
    EXPECT_EQ(eval(pb.slot_stride()), 12);
    EXPECT_EQ(eval(make({2}, {16}, Placement::RowMajor, ConflictPolicy::PadSlot, 1).slot_stride()), 17);
}
