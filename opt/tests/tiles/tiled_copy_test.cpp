#include <gtest/gtest.h>

#include "sdfg/tiles/layout.h"
#include "sdfg/tiles/tiled_copy.h"

using namespace sdfg;
using namespace sdfg::tiles;
using sdfg::symbolic::integer;

namespace {

Layout L(std::vector<long> shape, std::vector<long> stride = {}) {
    symbolic::MultiExpression s, d;
    for (long x : shape) s.push_back(integer(x));
    for (long x : stride) d.push_back(integer(x));
    return Layout(s, d);
}

} // namespace

// A dense scalar copy: 8-element tile, 4 lanes x 2 values, dense buffer.
TEST(TiledCopyTest, DenseScalarSyncValid) {
    TiledCopy c;
    c.src = L({8}, {1}); // global contiguous
    c.dst = L({8}, {1}); // dense buffer
    c.thread = L({4}, {2}); // lane t owns tile coords {2t, 2t+1}
    c.value = L({2}, {1}); // 2 contiguous values per lane
    c.atom = CopyAtom::ScalarSync;
    EXPECT_EQ(c.verify(/*elem_bytes*/ 4), std::nullopt);
}

// fp16 cp.async: a single half per lane is a 2-byte transfer -> illegal (the trap).
TEST(TiledCopyTest, Fp16CpAsyncScalarRejected) {
    TiledCopy c;
    c.src = L({8}, {1});
    c.dst = L({8}, {1});
    c.thread = L({8}, {1});
    c.value = L({1}, {1}); // one element per lane
    c.atom = CopyAtom::CpAsync;
    auto why = c.verify(/*fp16*/ 2);
    ASSERT_TRUE(why.has_value());
    EXPECT_NE(why->find("width 2"), std::string::npos);
}

// fp16 cp.async widened to 2 halves per lane = 4 bytes -> legal.
TEST(TiledCopyTest, Fp16CpAsyncHalf2Accepted) {
    TiledCopy c;
    c.src = L({8}, {1});
    c.dst = L({8}, {1});
    c.thread = L({4}, {2});
    c.value = L({2}, {1}); // half2: 2*2 = 4 bytes
    c.atom = CopyAtom::CpAsync;
    EXPECT_EQ(c.verify(/*fp16*/ 2), std::nullopt);
}

// Lane-contiguous DMA: consecutive lanes hit consecutive dst slots -> legal.
TEST(TiledCopyTest, LaneContiguousDmaValid) {
    TiledCopy c;
    c.src = L({8}, {1});
    c.dst = L({8}, {1}); // dense: dst(t) = t
    c.thread = L({8}, {1}); // lanes 0..7 -> tile coords 0..7 -> dst 0..7 (stride 1)
    c.value = L({1}, {1});
    c.atom = CopyAtom::LaneContiguousDMA;
    EXPECT_EQ(c.verify(/*fp32*/ 4), std::nullopt);
}

// Lane-contiguous DMA into a padded dst (lane stride 2) -> illegal (the ROCm scramble).
TEST(TiledCopyTest, LaneContiguousDmaPaddedDstRejected) {
    TiledCopy c;
    c.src = L({8}, {1});
    c.dst = L({8}, {2}); // padded: dst(t) = 2t, injective but not lane-contiguous
    c.thread = L({8}, {1});
    c.value = L({1}, {1});
    c.atom = CopyAtom::LaneContiguousDMA;
    auto why = c.verify(/*fp32*/ 4);
    ASSERT_TRUE(why.has_value());
    EXPECT_NE(why->find("consecutive in dst"), std::string::npos);
}

// A partition that skips/doubles elements is rejected.
TEST(TiledCopyTest, NonBijectivePartitionRejected) {
    TiledCopy c;
    c.src = L({8}, {1});
    c.dst = L({8}, {1});
    c.thread = L({4}, {1}); // lanes 0..3 overlap the value stride -> not a partition
    c.value = L({2}, {1});
    c.atom = CopyAtom::ScalarSync;
    auto why = c.verify(4);
    ASSERT_TRUE(why.has_value());
    EXPECT_NE(why->find("bijection"), std::string::npos);
}

// A destination that double-writes a slot is rejected.
TEST(TiledCopyTest, DoubleWriteDstRejected) {
    TiledCopy c;
    c.src = L({8}, {1});
    c.dst = L({8}, {0}); // every element -> slot 0
    c.thread = L({8}, {1});
    c.value = L({1}, {1});
    c.atom = CopyAtom::ScalarSync;
    auto why = c.verify(4);
    ASSERT_TRUE(why.has_value());
    EXPECT_NE(why->find("double-writes"), std::string::npos);
}

// The lane-contiguity predicate directly: a dense dst is contiguous, a padded one
// is not, and a degenerate 1-lane partition is contiguous by default.
TEST(TiledCopyTest, IsLaneContiguous_Predicate) {
    EXPECT_TRUE(is_lane_contiguous(/*dst*/ L({8}, {1}), /*thread*/ L({8}, {1})));
    EXPECT_FALSE(is_lane_contiguous(/*dst*/ L({8}, {2}), /*thread*/ L({8}, {1})));
    // Lanes stride the tile by 2, dst is dense -> lane l lands on dst 2l -> not unit.
    EXPECT_FALSE(is_lane_contiguous(/*dst*/ L({8}, {1}), /*thread*/ L({4}, {2})));
    EXPECT_TRUE(is_lane_contiguous(/*dst*/ L({8}, {5}), /*thread*/ L({1}, {1})));
}
