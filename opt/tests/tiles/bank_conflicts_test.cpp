#include "sdfg/tiles/bank_conflicts.h"

#include <gtest/gtest.h>

using namespace sdfg;

namespace {

// The CDNA MFMA A-operand read: lane l holds row l % rows and per_lane adjacent k values.
data_flow::SubgroupAccess fragment_read(int rows, int k, int lanes = 64) {
    data_flow::SubgroupAccess access;
    const int per_lane = rows * k / lanes;
    access.lanes.resize(lanes);
    for (int l = 0; l < lanes; ++l) {
        for (int e = 0; e < per_lane; ++e) {
            access.lanes[l].push_back({l % rows, per_lane * (l / rows) + e});
        }
    }
    return access;
}

} // namespace

// 64 lanes x 8 B fill four phases of 16 lanes; one conflict-free phase is one cycle.
TEST(BankConflictsTest, RowsSpreadByPadAreConflictFree) {
    const tiles::BankModel banks;
    const auto read = fragment_read(32, 8);
    // Unpadded 64-half rows (128 B): every row starts in bank 0 -> 16-way per phase.
    EXPECT_EQ(tiles::access_cycles(banks, 2, {64, 1}, read), 4u * 16u);
    // Padded by 4 halves (136 B rows): the 16 rows of a phase cover all 32 banks.
    EXPECT_EQ(tiles::access_cycles(banks, 2, {68, 1}, read), 4u);
}

// A lane whose elements are not adjacent issues one access per element.
TEST(BankConflictsTest, StridedLaneElementsSplitIntoScalarAccesses) {
    const tiles::BankModel banks;
    const auto read = fragment_read(32, 8);
    // Column-major storage of the same operand: k is strided, rows are adjacent.
    // 4 scalar accesses of 2 B, two phases of 32 lanes each reading 32 adjacent halves.
    EXPECT_EQ(tiles::access_cycles(banks, 2, {1, 40}, read), 4u * 2u);
}

// Lanes reading the same word share it (broadcast), costing no extra cycle.
TEST(BankConflictsTest, SameWordBroadcasts) {
    data_flow::SubgroupAccess access;
    access.lanes.assign(32, {{0}});
    EXPECT_EQ(tiles::access_cycles(tiles::BankModel{}, 4, {1}, access), 1u);
}

TEST(BankConflictsTest, SlotPad) {
    const tiles::BankModel banks;
    // No span: the next odd stride.
    EXPECT_EQ(tiles::slot_pad(banks, 16, 0), 1u);
    EXPECT_EQ(tiles::slot_pad(banks, 17, 0), 0u);
    // Span 4: a stride congruent to 4 mod 32, a multiple of 4 -> 36.
    EXPECT_EQ(tiles::slot_pad(banks, 16, 4), 20u);
}

// The solver picks the least-cycle pad; among equal cycles the aligned row, then the smaller pad.
TEST(BankConflictsTest, RowPadMinimizesCycles) {
    const tiles::BankModel banks;
    const auto read = fragment_read(32, 8);
    tiles::PackedBuffer buffer{{}, {symbolic::integer(32), symbolic::integer(64)}};
    auto cost = [&](const tiles::PackedBuffer& candidate) -> std::optional<size_t> {
        const int64_t row = 64 + static_cast<int64_t>(candidate.pad);
        return tiles::access_cycles(banks, 2, {row, 1}, read);
    };
    EXPECT_EQ(tiles::row_pad(banks, 2, buffer, 16, cost), 4u);

    // Every pad conflict-free: keep the row unpadded.
    auto flat_cost = [](const tiles::PackedBuffer&) -> std::optional<size_t> {
        return 4;
    };
    EXPECT_EQ(tiles::row_pad(banks, 2, buffer, 16, flat_cost), 0u);
}
