#include "sdfg/code_motion/transformations/hoist_child.h"

#include <gtest/gtest.h>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/structured_control_flow/for.h"
#include "sdfg/types/pointer.h"
#include "sdfg/types/scalar.h"

using namespace sdfg;
using code_motion::transformations::HoistChild;

namespace {

types::Scalar element() {
    return types::Scalar(types::PrimitiveType::Double);
}
types::Pointer pointer() {
    return types::Pointer(element());
}

struct Loop {
    builder::StructuredSDFGBuilder builder{"sdfg", FunctionType_CPU};
    structured_control_flow::For* loop = nullptr;

    /// for i in [0, end): an empty body.
    explicit Loop(symbolic::Expression end = symbolic::integer(8)) {
        types::Scalar index(types::PrimitiveType::Int64);
        builder.add_container("N", index, true);
        builder.add_container("i", index);
        for (auto name : {"A", "B", "C", "W"}) {
            builder.add_container(name, pointer(), true);
        }
        builder.add_container("w", element());
        auto i = symbolic::symbol("i");
        loop = &builder.add_for(
            builder.subject().root(), i, symbolic::Lt(i, end), symbolic::zero(), symbolic::add(i, symbolic::one())
        );
    }

    /// dst[write] = src[read] in the body, scalars without an index.
    structured_control_flow::Block& assign(
        const std::string& src,
        std::optional<symbolic::Expression> read,
        const std::string& dst,
        std::optional<symbolic::Expression> write
    ) {
        auto& block = builder.add_block(loop->root());
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
        auto subset = [](auto& index) {
            return index ? data_flow::Subset{*index} : data_flow::Subset{};
        };
        auto type = [](auto& index) -> const types::IType& {
            static auto p = pointer();
            static auto e = element();
            return index ? static_cast<const types::IType&>(p) : static_cast<const types::IType&>(e);
        };
        builder
            .add_computational_memlet(block, builder.add_access(block, src), tasklet, "_in", subset(read), type(read));
        builder
            .add_computational_memlet(block, tasklet, "_out", builder.add_access(block, dst), subset(write), type(write));
        return block;
    }
};

auto i() {
    return symbolic::symbol("i");
}

} // namespace

TEST(HoistChildTest, InvariantLoadMovesBeforeTheLoop) {
    // for i: w = W[0]; B[i] = w
    Loop l;
    auto& load = l.assign("W", symbolic::zero(), "w", std::nullopt);
    l.assign("w", std::nullopt, "B", i());
    analysis::AnalysisManager analysis_manager(l.builder.subject());
    HoistChild hoist(*l.loop, load);
    ASSERT_TRUE(hoist.can_be_applied(l.builder, analysis_manager));
    hoist.apply(l.builder, analysis_manager);
    auto& root = l.builder.subject().root();
    ASSERT_EQ(root.size(), 2u);
    EXPECT_EQ(&root.at(0), &load);
    EXPECT_EQ(l.loop->root().size(), 1u);
}

TEST(HoistChildTest, EarlierReaderKeepsTheChildInTheLoop) {
    // for i: B[i] = w; w = W[0]: the first iteration reads w before it is loaded.
    Loop l;
    l.assign("w", std::nullopt, "B", i());
    auto& load = l.assign("W", symbolic::zero(), "w", std::nullopt);
    analysis::AnalysisManager analysis_manager(l.builder.subject());
    EXPECT_FALSE(HoistChild(*l.loop, load).can_be_applied(l.builder, analysis_manager));
    EXPECT_FALSE(HoistChild(*l.loop, load, true).can_be_applied(l.builder, analysis_manager));
}

TEST(HoistChildTest, InputWrittenInAnotherIterationKeepsTheChild) {
    // for i: W[i] = A[i]; w = W[0]: the first iteration writes what every later one loads.
    Loop l;
    l.assign("A", i(), "W", i());
    auto& load = l.assign("W", symbolic::zero(), "w", std::nullopt);
    analysis::AnalysisManager analysis_manager(l.builder.subject());
    EXPECT_FALSE(HoistChild(*l.loop, load).can_be_applied(l.builder, analysis_manager));
}

TEST(HoistChildTest, DisjointWritesAcrossIterationsMoveAfterTheLoop) {
    // for i in [0, 8): B[i] = A[i]; B[8] = C[0]: no iteration writes B[8] otherwise.
    Loop l;
    l.assign("A", i(), "B", i());
    auto& last = l.assign("C", symbolic::zero(), "B", symbolic::integer(8));
    analysis::AnalysisManager analysis_manager(l.builder.subject());
    HoistChild hoist(*l.loop, last, true);
    ASSERT_TRUE(hoist.can_be_applied(l.builder, analysis_manager));
    hoist.apply(l.builder, analysis_manager);
    auto& root = l.builder.subject().root();
    ASSERT_EQ(root.size(), 2u);
    EXPECT_EQ(&root.at(1), &last);
}

TEST(HoistChildTest, OverlappingWritesAcrossIterationsKeepTheChild) {
    // for i in [0, 8): B[i] = A[i]; B[7] = C[0]: the last iteration overwrites B[7] of the child's earlier runs.
    Loop l;
    l.assign("A", i(), "B", i());
    auto& last = l.assign("C", symbolic::zero(), "B", symbolic::integer(7));
    analysis::AnalysisManager analysis_manager(l.builder.subject());
    EXPECT_FALSE(HoistChild(*l.loop, last, true).can_be_applied(l.builder, analysis_manager));
}

TEST(HoistChildTest, ChildUsingTheCounterStays) {
    Loop l;
    auto& load = l.assign("W", i(), "w", std::nullopt);
    analysis::AnalysisManager analysis_manager(l.builder.subject());
    EXPECT_FALSE(HoistChild(*l.loop, load).can_be_applied(l.builder, analysis_manager));
}

TEST(HoistChildTest, ChildReadingWhatItWritesStays) {
    // for i: W[0] = W[0]: running it again reads what it wrote.
    Loop l;
    auto& update = l.assign("W", symbolic::zero(), "W", symbolic::zero());
    analysis::AnalysisManager analysis_manager(l.builder.subject());
    EXPECT_FALSE(HoistChild(*l.loop, update).can_be_applied(l.builder, analysis_manager));
}

TEST(HoistChildTest, LoopThatMayNotRunKeepsTheChild) {
    // for i in [0, N): N may be 0.
    Loop l(symbolic::symbol("N"));
    auto& load = l.assign("W", symbolic::zero(), "w", std::nullopt);
    l.assign("w", std::nullopt, "B", i());
    analysis::AnalysisManager analysis_manager(l.builder.subject());
    EXPECT_FALSE(HoistChild(*l.loop, load).can_be_applied(l.builder, analysis_manager));
}

TEST(HoistChildTest, SerializationRoundTrip) {
    Loop l;
    auto& load = l.assign("W", symbolic::zero(), "w", std::nullopt);
    nlohmann::json j;
    HoistChild(*l.loop, load, true).to_json(j);
    EXPECT_EQ(j["transformation_type"], "HoistChild");
    nlohmann::json again;
    HoistChild::from_json(l.builder, j).to_json(again);
    EXPECT_EQ(j, again);
}
