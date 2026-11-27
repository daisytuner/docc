#include "sdfg/code_motion/transformations/move_child.h"

#include <gtest/gtest.h>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_nodes/barrier_local_node.h"
#include "sdfg/data_flow/library_nodes/stdlib/free.h"
#include "sdfg/data_flow/library_nodes/stdlib/malloc.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/structured_control_flow/return.h"
#include "sdfg/tiles/library_nodes/tile_copy_node.h"
#include "sdfg/tiles/tiled_copy.h"
#include "sdfg/types/pointer.h"
#include "sdfg/types/scalar.h"

using namespace sdfg;
using code_motion::transformations::MoveChild;

namespace {

types::Pointer pointer() {
    return types::Pointer(types::Scalar(types::PrimitiveType::Double));
}

/// dst[c] = src[c] for c in [begin, end), as a map in @p parent counting with @p counter.
structured_control_flow::Map& copy(
    builder::StructuredSDFGBuilder& builder,
    structured_control_flow::Sequence& parent,
    const std::string& counter,
    const std::string& src,
    const std::string& dst,
    const symbolic::Expression& begin,
    const symbolic::Expression& end
) {
    auto c = symbolic::symbol(counter);
    auto& map = builder.add_map(
        parent,
        c,
        symbolic::Lt(c, end),
        begin,
        symbolic::add(c, symbolic::one()),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    auto& block = builder.add_block(map.root());
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, builder.add_access(block, src), tasklet, "_in", {c}, pointer());
    builder.add_computational_memlet(block, tasklet, "_out", builder.add_access(block, dst), {c}, pointer());
    return map;
}

void containers(builder::StructuredSDFGBuilder& builder) {
    types::Scalar index(types::PrimitiveType::Int64);
    builder.add_container("N", index, true);
    for (auto name : {"i", "j", "k"}) {
        builder.add_container(name, index);
    }
    for (auto name : {"A", "B", "C", "D"}) {
        builder.add_container(name, pointer(), true);
    }
}

} // namespace

TEST(MoveChildTest, IndependentChildMovesAhead) {
    // B = A; C = B; D = A: the last needs neither before it.
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    containers(builder);
    auto& root = builder.subject().root();
    auto N = symbolic::symbol("N");
    copy(builder, root, "i", "A", "B", symbolic::zero(), N);
    copy(builder, root, "j", "B", "C", symbolic::zero(), N);
    auto& last = copy(builder, root, "k", "A", "D", symbolic::zero(), N);
    analysis::AnalysisManager analysis_manager(builder.subject());
    MoveChild move(last, 0);
    ASSERT_TRUE(move.can_be_applied(builder, analysis_manager));
    move.apply(builder, analysis_manager);
    EXPECT_EQ(&root.at(0), &last);
}

TEST(MoveChildTest, ConsumerDoesNotPassItsProducer) {
    // B = A; C = B: C reads what the first writes.
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    containers(builder);
    auto& root = builder.subject().root();
    auto N = symbolic::symbol("N");
    copy(builder, root, "i", "A", "B", symbolic::zero(), N);
    auto& consumer = copy(builder, root, "j", "B", "C", symbolic::zero(), N);
    analysis::AnalysisManager analysis_manager(builder.subject());
    EXPECT_FALSE(MoveChild(consumer, 0).can_be_applied(builder, analysis_manager));
}

TEST(MoveChildTest, DisjointPartsOfAContainerMovePastEachOther) {
    // B[0:N] = A; C = B[N:2N]: the same N in both, so they access different halves of B.
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    containers(builder);
    auto& root = builder.subject().root();
    auto N = symbolic::symbol("N");
    copy(builder, root, "i", "A", "B", symbolic::zero(), N);
    auto& second = copy(builder, root, "j", "B", "C", N, symbolic::mul(symbolic::integer(2), N));
    analysis::AnalysisManager analysis_manager(builder.subject());
    EXPECT_TRUE(MoveChild(second, 0).can_be_applied(builder, analysis_manager));
}

TEST(MoveChildTest, OverlappingPartsOfAContainerStay) {
    // B[0:N] = A; C = B[N-1:2N]: both access B[N-1].
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    containers(builder);
    auto& root = builder.subject().root();
    auto N = symbolic::symbol("N");
    copy(builder, root, "i", "A", "B", symbolic::zero(), N);
    auto& second =
        copy(builder, root, "j", "B", "C", symbolic::sub(N, symbolic::one()), symbolic::mul(symbolic::integer(2), N));
    analysis::AnalysisManager analysis_manager(builder.subject());
    EXPECT_FALSE(MoveChild(second, 0).can_be_applied(builder, analysis_manager));
}

TEST(MoveChildTest, WithinAnIterationOnlyItsElementsCount) {
    // for t: B[t] = A[t]; C[t] = B[t + 1]: in one iteration, they access different elements of B.
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    containers(builder);
    builder.add_container("t", types::Scalar(types::PrimitiveType::Int64));
    auto t = symbolic::symbol("t");
    auto& loop = builder.add_for(
        builder.subject().root(),
        t,
        symbolic::Lt(t, symbolic::symbol("N")),
        symbolic::zero(),
        symbolic::add(t, symbolic::one())
    );
    auto assign = [&](const std::string& src, const symbolic::Expression& read, const std::string& dst) {
        auto& block = builder.add_block(loop.root());
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
        builder.add_computational_memlet(block, builder.add_access(block, src), tasklet, "_in", {read}, pointer());
        builder.add_computational_memlet(block, tasklet, "_out", builder.add_access(block, dst), {t}, pointer());
        return &block;
    };
    assign("A", t, "B");
    auto* second = assign("B", symbolic::add(t, symbolic::one()), "C");
    analysis::AnalysisManager analysis_manager(builder.subject());
    EXPECT_TRUE(MoveChild(*second, 0).can_be_applied(builder, analysis_manager));
}

TEST(MoveChildTest, LoopDoesNotPassTheAssignmentOfItsBound) {
    // M = N; B[0:M] = A: the loop reads the symbol M.
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    containers(builder);
    builder.add_container("M", types::Scalar(types::PrimitiveType::Int64));
    auto& root = builder.subject().root();
    builder.add_assignments(root, {{symbolic::symbol("M"), symbolic::symbol("N")}});
    auto& loop = copy(builder, root, "i", "A", "B", symbolic::zero(), symbolic::symbol("M"));
    analysis::AnalysisManager analysis_manager(builder.subject());
    EXPECT_FALSE(MoveChild(loop, 0).can_be_applied(builder, analysis_manager));
}

TEST(MoveChildTest, NothingPassesAReturn) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    containers(builder);
    auto& root = builder.subject().root();
    builder.add_return(root, "");
    auto& loop = copy(builder, root, "i", "A", "B", symbolic::zero(), symbolic::symbol("N"));
    analysis::AnalysisManager analysis_manager(builder.subject());
    EXPECT_FALSE(MoveChild(loop, 0).can_be_applied(builder, analysis_manager));
}

TEST(MoveChildTest, SerializationRoundTrip) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    containers(builder);
    auto& root = builder.subject().root();
    auto N = symbolic::symbol("N");
    copy(builder, root, "i", "A", "B", symbolic::zero(), N);
    auto& last = copy(builder, root, "k", "A", "D", symbolic::zero(), N);
    nlohmann::json j;
    MoveChild(last, 0).to_json(j);
    EXPECT_EQ(j["transformation_type"], "MoveChild");
    auto restored = MoveChild::from_json(builder, j);
    nlohmann::json again;
    restored.to_json(again);
    EXPECT_EQ(j, again);
}

namespace {

structured_control_flow::Block& free_block(builder::StructuredSDFGBuilder& builder, const std::string& container) {
    auto& block = builder.add_block(builder.subject().root());
    auto& node = builder.add_library_node<stdlib::FreeNode>(block, DebugInfo());
    builder.add_computational_memlet(block, builder.add_access(block, container), node, "_ptr", {}, pointer());
    return block;
}

structured_control_flow::Block& malloc_block(builder::StructuredSDFGBuilder& builder, const std::string& container) {
    auto& block = builder.add_block(builder.subject().root());
    auto& node = builder.add_library_node<stdlib::MallocNode>(block, DebugInfo(), symbolic::integer(64));
    builder.add_computational_memlet(block, node, "_ret", builder.add_access(block, container), {}, pointer());
    return block;
}

} // namespace

TEST(MoveChildTest, FreeDoesNotPassAUseOfItsPointer) {
    // C = B; free(B): the free invalidates what the copy reads.
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    containers(builder);
    copy(builder, builder.subject().root(), "i", "B", "C", symbolic::zero(), symbolic::symbol("N"));
    auto& release = free_block(builder, "B");
    analysis::AnalysisManager analysis_manager(builder.subject());
    EXPECT_FALSE(MoveChild(release, 0).can_be_applied(builder, analysis_manager));
}

TEST(MoveChildTest, AllocationsAndFreesPassUnrelatedSteps) {
    // B = A; C = malloc(); free(D): neither touches A or B.
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    containers(builder);
    copy(builder, builder.subject().root(), "i", "A", "B", symbolic::zero(), symbolic::symbol("N"));
    auto& allocation = malloc_block(builder, "C");
    auto& release = free_block(builder, "D");
    analysis::AnalysisManager analysis_manager(builder.subject());
    EXPECT_TRUE(MoveChild(allocation, 0).can_be_applied(builder, analysis_manager));
    EXPECT_TRUE(MoveChild(release, 0).can_be_applied(builder, analysis_manager));
}

TEST(MoveChildTest, UseDoesNotPassTheAllocationOfItsPointer) {
    // B = malloc(); B = A: the copy writes what the allocation defines.
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    containers(builder);
    malloc_block(builder, "B");
    auto& use = copy(builder, builder.subject().root(), "i", "A", "B", symbolic::zero(), symbolic::symbol("N"));
    analysis::AnalysisManager analysis_manager(builder.subject());
    EXPECT_FALSE(MoveChild(use, 0).can_be_applied(builder, analysis_manager));
}

TEST(MoveChildTest, TileCopyPassesUnrelatedSteps) {
    // B = A; D[0:4] = C[0:4] by a tile copy, whose only effect is the tile it writes.
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    containers(builder);
    copy(builder, builder.subject().root(), "i", "A", "B", symbolic::zero(), symbolic::symbol("N"));
    auto& block = builder.add_block(builder.subject().root());
    tiles::TiledCopy plan;
    plan.src = tiles::Layout({symbolic::integer(4)}, {symbolic::integer(1)}, symbolic::zero());
    plan.dst = tiles::Layout({symbolic::integer(4)}, {symbolic::integer(1)}, symbolic::zero());
    plan.atom = tiles::CopyAtom::ScalarSync;
    auto& node = builder.add_library_node<tiles::TileCopyNode>(
        block, DebugInfo(), data_flow::ImplementationType{"DUMMY"}, plan, tiles::CopyDirection::In, 32u
    );
    builder.add_computational_memlet(block, builder.add_access(block, "D"), node, "_dst", {}, pointer());
    builder.add_computational_memlet(block, builder.add_access(block, "C"), node, "_src", {}, pointer());
    analysis::AnalysisManager analysis_manager(builder.subject());
    EXPECT_TRUE(MoveChild(block, 0).can_be_applied(builder, analysis_manager));
}

TEST(MoveChildTest, NothingPassesABarrier) {
    // B = A; barrier: a barrier's effects are no accesses of memlets, so it stays where it is.
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    containers(builder);
    copy(builder, builder.subject().root(), "i", "A", "B", symbolic::zero(), symbolic::symbol("N"));
    auto& block = builder.add_block(builder.subject().root());
    builder.add_library_node<data_flow::BarrierLocalNode>(block, DebugInfo());
    analysis::AnalysisManager analysis_manager(builder.subject());
    EXPECT_FALSE(MoveChild(block, 0).can_be_applied(builder, analysis_manager));
}
