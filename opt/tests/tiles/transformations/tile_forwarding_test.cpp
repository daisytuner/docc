#include "sdfg/tiles/transformations/tile_forwarding.h"

#include <gtest/gtest.h>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/tiles/library_nodes/tile_copy_node.h"
#include "sdfg/tiles/tiled_copy.h"
#include "sdfg/types/array.h"
#include "sdfg/types/pointer.h"
#include "sdfg/types/scalar.h"

using namespace sdfg;

namespace {

types::Scalar element() {
    return types::Scalar(types::PrimitiveType::Float);
}
types::Pointer pointer() {
    return types::Pointer(element());
}
types::Array buffer() {
    return types::Array(element(), symbolic::integer(4));
}

struct Program {
    builder::StructuredSDFGBuilder builder{"sdfg", FunctionType_CPU};

    Program() {
        for (auto name : {"A", "B", "C", "D"}) {
            builder.add_container(name, pointer(), true);
        }
        for (auto name : {"L1", "L2"}) {
            builder.add_container(name, buffer());
        }
    }

    /// Copies the 4 elements of @p global from @p offset into or out of @p local.
    structured_control_flow::Block&
    copy(const std::string& global, int64_t offset, const std::string& local, tiles::CopyDirection direction) {
        auto& block = builder.add_block(builder.subject().root());
        tiles::TiledCopy plan;
        plan.src = tiles::Layout({symbolic::integer(4)}, {symbolic::integer(1)}, symbolic::integer(offset));
        plan.dst = tiles::Layout({symbolic::integer(4)}, {symbolic::integer(1)}, symbolic::zero());
        plan.atom = tiles::CopyAtom::ScalarSync;
        auto& node = builder.add_library_node<
            tiles::TileCopyNode>(block, DebugInfo(), data_flow::ImplementationType{"DUMMY"}, plan, direction, 16u);
        bool in = direction == tiles::CopyDirection::In;
        builder
            .add_computational_memlet(block, builder.add_access(block, in ? local : global), node, "_dst", {}, pointer());
        builder
            .add_computational_memlet(block, builder.add_access(block, in ? global : local), node, "_src", {}, pointer());
        return block;
    }

    /// dst[write] = src[read], with the subsets typed after the containers.
    structured_control_flow::Block& assign(
        const std::string& src,
        const symbolic::Expression& read,
        const std::string& dst,
        const symbolic::Expression& write
    ) {
        auto& block = builder.add_block(builder.subject().root());
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
        auto type = [&](const std::string& name) -> const types::IType& {
            return builder.subject().type(name);
        };
        builder.add_computational_memlet(block, builder.add_access(block, src), tasklet, "_in", {read}, type(src));
        builder.add_computational_memlet(block, tasklet, "_out", builder.add_access(block, dst), {write}, type(dst));
        return block;
    }
};

auto at(int64_t index) {
    return symbolic::integer(index);
}

} // namespace

TEST(TileForwardingTest, CopyInTakesTheTileAnEarlierCopyInLeft) {
    // L1 = A[4:8]; C[0] = L1[0]; L2 = A[4:8]; D[0] = L2[1]
    Program p;
    auto& earlier = p.copy("A", 4, "L1", tiles::CopyDirection::In);
    p.assign("L1", at(0), "C", at(0));
    auto& later = p.copy("A", 4, "L2", tiles::CopyDirection::In);
    p.assign("L2", at(1), "D", at(0));
    analysis::AnalysisManager analysis_manager(p.builder.subject());
    transformations::TileForwarding forwarding(earlier, later);
    ASSERT_TRUE(forwarding.can_be_applied(p.builder, analysis_manager));
    forwarding.apply(p.builder, analysis_manager);
    auto& root = p.builder.subject().root();
    ASSERT_EQ(root.size(), 3u);
    EXPECT_FALSE(p.builder.subject().exists("L2"));
    auto& use = static_cast<structured_control_flow::Block&>(root.at(2));
    bool reads_l1 = false;
    for (auto* node : use.dataflow().data_nodes()) {
        reads_l1 |= node->data() == "L1";
    }
    EXPECT_TRUE(reads_l1);
}

TEST(TileForwardingTest, CopyInTakesTheTileAnEarlierCopyOutLeft) {
    // L1[0] = C[0]; B[4:8] = L1; L2 = B[4:8]; D[0] = L2[0]
    Program p;
    p.assign("C", at(0), "L1", at(0));
    auto& earlier = p.copy("B", 4, "L1", tiles::CopyDirection::Out);
    auto& later = p.copy("B", 4, "L2", tiles::CopyDirection::In);
    p.assign("L2", at(0), "D", at(0));
    analysis::AnalysisManager analysis_manager(p.builder.subject());
    EXPECT_TRUE(transformations::TileForwarding(earlier, later).can_be_applied(p.builder, analysis_manager));
}

TEST(TileForwardingTest, WriteOfTheTileBetweenKeepsTheCopy) {
    // L1 = A[4:8]; A[5] = C[0]; L2 = A[4:8]: the tile changed.
    Program p;
    auto& earlier = p.copy("A", 4, "L1", tiles::CopyDirection::In);
    p.assign("C", at(0), "A", at(5));
    auto& later = p.copy("A", 4, "L2", tiles::CopyDirection::In);
    analysis::AnalysisManager analysis_manager(p.builder.subject());
    EXPECT_FALSE(transformations::TileForwarding(earlier, later).can_be_applied(p.builder, analysis_manager));
}

TEST(TileForwardingTest, WriteOutsideTheTileBetweenKeepsTheForwarding) {
    // L1 = A[4:8]; A[0] = C[0]; L2 = A[4:8]: the tile did not change.
    Program p;
    auto& earlier = p.copy("A", 4, "L1", tiles::CopyDirection::In);
    p.assign("C", at(0), "A", at(0));
    auto& later = p.copy("A", 4, "L2", tiles::CopyDirection::In);
    analysis::AnalysisManager analysis_manager(p.builder.subject());
    EXPECT_TRUE(transformations::TileForwarding(earlier, later).can_be_applied(p.builder, analysis_manager));
}

TEST(TileForwardingTest, WriteOfTheEarlierBufferBetweenKeepsTheCopy) {
    // L1 = A[4:8]; L1[0] = C[0]; L2 = A[4:8]: the buffer no longer holds the tile.
    Program p;
    auto& earlier = p.copy("A", 4, "L1", tiles::CopyDirection::In);
    p.assign("C", at(0), "L1", at(0));
    auto& later = p.copy("A", 4, "L2", tiles::CopyDirection::In);
    analysis::AnalysisManager analysis_manager(p.builder.subject());
    EXPECT_FALSE(transformations::TileForwarding(earlier, later).can_be_applied(p.builder, analysis_manager));
}

TEST(TileForwardingTest, OtherRegionKeepsTheCopy) {
    Program p;
    auto& earlier = p.copy("A", 4, "L1", tiles::CopyDirection::In);
    auto& later = p.copy("A", 8, "L2", tiles::CopyDirection::In);
    analysis::AnalysisManager analysis_manager(p.builder.subject());
    EXPECT_FALSE(transformations::TileForwarding(earlier, later).can_be_applied(p.builder, analysis_manager));
}

TEST(TileForwardingTest, LaterBufferWrittenAfterKeepsTheCopy) {
    // L1 = A[4:8]; L2 = A[4:8]; L2[0] = C[0]; D[0] = L1[0]: renamed, the write would change what L1 holds.
    Program p;
    auto& earlier = p.copy("A", 4, "L1", tiles::CopyDirection::In);
    auto& later = p.copy("A", 4, "L2", tiles::CopyDirection::In);
    p.assign("C", at(0), "L2", at(0));
    p.assign("L1", at(0), "D", at(0));
    analysis::AnalysisManager analysis_manager(p.builder.subject());
    EXPECT_FALSE(transformations::TileForwarding(earlier, later).can_be_applied(p.builder, analysis_manager));
}

TEST(TileForwardingTest, SerializationRoundTrip) {
    Program p;
    auto& earlier = p.copy("A", 4, "L1", tiles::CopyDirection::In);
    auto& later = p.copy("A", 4, "L2", tiles::CopyDirection::In);
    nlohmann::json j;
    transformations::TileForwarding(earlier, later).to_json(j);
    EXPECT_EQ(j["transformation_type"], "TileForwarding");
    nlohmann::json again;
    transformations::TileForwarding::from_json(p.builder, j).to_json(again);
    EXPECT_EQ(j, again);
}
