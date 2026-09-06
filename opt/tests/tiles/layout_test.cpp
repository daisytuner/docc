#include <gtest/gtest.h>

#include "sdfg/tiles/layout.h"
#include "sdfg/tiles/swizzle.h"

using namespace sdfg;
using namespace sdfg::tiles;
using sdfg::symbolic::Expression;
using sdfg::symbolic::integer;

namespace {

Expression E(long v) { return integer(v); }

// Provable equality to a concrete integer (folds div/mod/mul on integers).
bool eqi(const Expression& e, long v) { return symbolic::eq(symbolic::simplify(e), integer(v)); }

// Provable equality between two expressions.
bool eqe(const Expression& a, const Expression& b) {
    return symbolic::eq(symbolic::simplify(a), symbolic::simplify(b));
}

Layout L(std::vector<long> shape, std::vector<long> stride = {}) {
    symbolic::MultiExpression s, d;
    for (long x : shape) s.push_back(E(x));
    for (long x : stride) d.push_back(E(x));
    return Layout(s, d);
}

} // namespace

TEST(LayoutTest, DefaultStridesAreColexContiguous) {
    Layout a = L({4, 3}); // strides default to (1, 4)
    ASSERT_EQ(a.rank(), 2u);
    EXPECT_TRUE(eqi(a.stride()[0], 1));
    EXPECT_TRUE(eqi(a.stride()[1], 4));
    // A contiguous layout is the identity function on [0, size).
    for (long i = 0; i < 12; ++i) {
        EXPECT_TRUE(eqi(a.apply(E(i)), i)) << "i=" << i;
    }
}

TEST(LayoutTest, SizeAndCosize) {
    Layout a = L({4, 3}, {1, 4});
    EXPECT_TRUE(eqi(a.size(), 12));
    EXPECT_TRUE(eqi(a.cosize(), 12));
    // A padded row (stride 8 over 3 rows) spans further than it fills.
    Layout p = L({4, 3}, {1, 8});
    EXPECT_TRUE(eqi(p.size(), 12));
    EXPECT_TRUE(eqi(p.cosize(), 20)); // offset 0 + (4-1)*1 + (3-1)*8 + 1 = 20
}

TEST(LayoutTest, IsBijective) {
    EXPECT_TRUE(L({4, 3}, {1, 4}).is_bijective());
    EXPECT_TRUE(L({3, 4}, {4, 1}).is_bijective()); // transposed but dense
    EXPECT_FALSE(L({4, 3}, {1, 8}).is_bijective()); // gap
}

TEST(LayoutTest, CoalesceMergesAndIsIdempotent) {
    Layout a = L({4, 3}, {1, 4}); // contiguous -> single mode (12):(1)
    Layout c = coalesce(a);
    ASSERT_EQ(c.rank(), 1u);
    EXPECT_TRUE(eqi(c.shape()[0], 12));
    EXPECT_TRUE(eqi(c.stride()[0], 1));
    EXPECT_TRUE(coalesce(c) == c); // idempotent
    // Function preserved.
    for (long i = 0; i < 12; ++i) EXPECT_TRUE(eqe(c.apply(E(i)), a.apply(E(i))));
}

TEST(LayoutTest, ConcatAppendsModes) {
    // `4:(1) ++ 3:(4)` is the dense permutation of [0, 12).
    Layout a = L({4}, {1});
    Layout b = L({3}, {4});
    Layout full = concat(a, b);
    ASSERT_EQ(full.rank(), 2u);
    EXPECT_TRUE(eqi(full.shape()[0], 4));
    EXPECT_TRUE(eqi(full.shape()[1], 3));
    EXPECT_TRUE(eqi(full.stride()[1], 4));
    EXPECT_TRUE(full.is_bijective());
    EXPECT_TRUE(eqi(full.size(), 12));
}

TEST(LayoutTest, SwizzleIdentityIsNoOp) {
    Swizzle id{};
    EXPECT_TRUE(id.is_identity());
    for (long x = 0; x < 8; ++x) EXPECT_TRUE(eqi(id.apply(E(x)), x));
    ComposedLayout cl{Swizzle{}, L({4, 3}, {1, 4})};
    EXPECT_TRUE(cl.is_plain());
    for (long i = 0; i < 12; ++i) EXPECT_TRUE(eqi(cl.apply(E(i)), i));
}
