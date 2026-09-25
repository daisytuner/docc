#include <gtest/gtest.h>

#include "sdfg/tiles/layout.h"
#include "sdfg/tiles/swizzle.h"

using namespace sdfg;
using namespace sdfg::tiles;
using sdfg::symbolic::Expression;
using sdfg::symbolic::integer;

namespace {

Expression E(long v) {
    return integer(v);
}

// Provable equality to a concrete integer (folds div/mod/mul on integers).
bool eqi(const Expression& e, long v) {
    return symbolic::eq(symbolic::simplify(e), integer(v));
}

Layout L(std::vector<long> shape, std::vector<long> stride = {}) {
    symbolic::MultiExpression s, d;
    for (long x : shape) {
        s.push_back(E(x));
    }
    for (long x : stride) {
        d.push_back(E(x));
    }
    return Layout(s, d);
}

} // namespace

// Coordinate for a 2D layout.
symbolic::MultiExpression C(long a, long b) {
    return {E(a), E(b)};
}

TEST(LayoutTest, ResolveElementDotsCoords) {
    Layout a = L({4, 3}, {1, 4});
    ASSERT_EQ(a.dims(), 2);
    EXPECT_TRUE(eqi(a.strides()[0], 1));
    EXPECT_TRUE(eqi(a.strides()[1], 4));
    // resolve_element maps (i, j) to i + 4*j.
    for (long j = 0; j < 3; ++j) {
        for (long i = 0; i < 4; ++i) {
            EXPECT_TRUE(eqi(a.resolve_element(C(i, j)), i + 4 * j));
        }
    }
}

TEST(LayoutTest, TotalElements) {
    EXPECT_TRUE(eqi(L({4, 3}, {1, 4}).total_elements(), 12));
    EXPECT_TRUE(eqi(L({4, 3}, {1, 8}).total_elements(), 12)); // counts elements, not span
}

TEST(LayoutTest, SwizzleIdentityIsNoOp) {
    Swizzle id{};
    EXPECT_TRUE(id.is_identity());
    for (long x = 0; x < 8; ++x) {
        EXPECT_TRUE(eqi(id.apply(E(x)), x));
    }
    ComposedLayout cl{Swizzle{}, L({4, 3}, {1, 4})};
    EXPECT_TRUE(cl.is_plain());
    for (long j = 0; j < 3; ++j) {
        for (long i = 0; i < 4; ++i) {
            EXPECT_TRUE(eqi(cl.apply_coords(C(i, j)), i + 4 * j));
        }
    }
}
