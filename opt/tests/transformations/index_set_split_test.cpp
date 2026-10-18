#include "sdfg/transformations/index_set_split.h"

#include <gtest/gtest.h>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/structured_control_flow/if_else.h"
#include "sdfg/structured_control_flow/map.h"

using namespace sdfg;

namespace {

structured_control_flow::Map& add_map(
    builder::StructuredSDFGBuilder& builder,
    structured_control_flow::Sequence& scope,
    const std::string& name,
    int64_t extent
) {
    builder.add_container(name, types::Scalar(types::PrimitiveType::Int64));
    auto indvar = symbolic::symbol(name);
    return builder.add_map(
        scope,
        indvar,
        symbolic::Lt(indvar, symbolic::integer(extent)),
        symbolic::zero(),
        symbolic::add(indvar, symbolic::one()),
        structured_control_flow::ScheduleType_Sequential::create()
    );
}

/// B[write] = A[read] in @p scope.
void copy(
    builder::StructuredSDFGBuilder& builder,
    structured_control_flow::Sequence& scope,
    const symbolic::Expression& read,
    const symbolic::Expression& write
) {
    types::Scalar element(types::PrimitiveType::Float);
    types::Pointer pointer(element);
    auto& block = builder.add_block(scope);
    auto& a = builder.add_access(block, "A");
    auto& b = builder.add_access(block, "B");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, a, tasklet, "_in", {read}, pointer);
    builder.add_computational_memlet(block, tasklet, "_out", b, {write}, pointer);
}

void add_arrays(builder::StructuredSDFGBuilder& builder) {
    types::Scalar element(types::PrimitiveType::Float);
    types::Pointer pointer(element);
    builder.add_container("A", pointer, true);
    builder.add_container("B", pointer, true);
}

symbolic::Expression read_of(structured_control_flow::ControlFlowNode& node) {
    auto& block = dynamic_cast<structured_control_flow::Block&>(node);
    for (auto& memlet : block.dataflow().edges()) {
        if (auto access = dynamic_cast<data_flow::AccessNode*>(&memlet.src()); access && access->data() == "A") {
            return memlet.subset().at(0);
        }
    }
    return SymEngine::null;
}

int64_t bound_of(structured_control_flow::StructuredLoop& loop) {
    return SymEngine::down_cast<const SymEngine::Integer&>(*loop.canonical_bound()).as_int();
}

} // namespace

// rotate_half: B[j] = j < 32 ? A[j + 32] : A[j - 32].
TEST(IndexSetSplitTest, SplitsAConcatenationIntoItsHalves) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);
    add_arrays(builder);
    auto& map = add_map(builder, builder.subject().root(), "j", 64);
    auto j = map.indvar();
    auto& branches = builder.add_if_else(map.root());
    auto& low =
        builder
            .add_case(branches, symbolic::And(symbolic::Le(symbolic::zero(), j), symbolic::Lt(j, symbolic::integer(32))));
    copy(builder, low, symbolic::add(j, symbolic::integer(32)), j);
    auto& high = builder.add_case(
        branches, symbolic::And(symbolic::Le(symbolic::integer(32), j), symbolic::Lt(j, symbolic::integer(64)))
    );
    copy(builder, high, symbolic::sub(j, symbolic::integer(32)), j);

    analysis::AnalysisManager analysis_manager(builder.subject());
    transformations::IndexSetSplit split(map);
    ASSERT_TRUE(split.can_be_applied(builder, analysis_manager));
    split.apply(builder, analysis_manager);

    auto& loops = split.loops();
    ASSERT_EQ(loops.size(), 2);
    ASSERT_EQ(builder.subject().root().size(), 2);
    EXPECT_TRUE(symbolic::eq(loops[0]->init(), symbolic::zero()));
    EXPECT_EQ(bound_of(*loops[0]), 32);
    // The second half counts from zero as well.
    EXPECT_TRUE(symbolic::eq(loops[1]->init(), symbolic::zero()));
    EXPECT_EQ(bound_of(*loops[1]), 32);
    for (auto* loop : loops) {
        ASSERT_EQ(loop->root().size(), 1);
        ASSERT_NE(dynamic_cast<structured_control_flow::Block*>(&loop->root().at(0)), nullptr);
    }
    EXPECT_TRUE(symbolic::eq(read_of(loops[0]->root().at(0)), symbolic::add(loops[0]->indvar(), symbolic::integer(32))));
    EXPECT_TRUE(symbolic::eq(read_of(loops[1]->root().at(0)), loops[1]->indvar()));
}

TEST(IndexSetSplitTest, FoldsBranchesInsideInnerLoops) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);
    add_arrays(builder);
    auto& outer = add_map(builder, builder.subject().root(), "j", 64);
    auto j = outer.indvar();
    auto& inner = add_map(builder, outer.root(), "t", 128);
    auto t = inner.indvar();
    auto& branches = builder.add_if_else(inner.root());
    auto index = symbolic::add(symbolic::mul(symbolic::integer(128), j), t);
    copy(builder, builder.add_case(branches, symbolic::Lt(j, symbolic::integer(16))), index, index);

    analysis::AnalysisManager analysis_manager(builder.subject());
    transformations::IndexSetSplit split(outer);
    ASSERT_TRUE(split.can_be_applied(builder, analysis_manager));
    split.apply(builder, analysis_manager);

    auto& loops = split.loops();
    ASSERT_EQ(loops.size(), 2);
    auto& first = dynamic_cast<structured_control_flow::StructuredLoop&>(loops[0]->root().at(0));
    ASSERT_EQ(first.root().size(), 1);
    EXPECT_NE(dynamic_cast<structured_control_flow::Block*>(&first.root().at(0)), nullptr);
    auto& second = dynamic_cast<structured_control_flow::StructuredLoop&>(loops[1]->root().at(0));
    EXPECT_EQ(second.root().size(), 0);
}

TEST(IndexSetSplitTest, KeepsBranchesOnOtherSymbols) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);
    add_arrays(builder);
    builder.add_container("N", types::Scalar(types::PrimitiveType::Int64), true);
    auto& map = add_map(builder, builder.subject().root(), "j", 64);
    auto j = map.indvar();
    auto& branches = builder.add_if_else(map.root());
    copy(builder, builder.add_case(branches, symbolic::Lt(j, symbolic::symbol("N"))), j, j);

    analysis::AnalysisManager analysis_manager(builder.subject());
    transformations::IndexSetSplit split(map);
    EXPECT_FALSE(split.can_be_applied(builder, analysis_manager));
}

TEST(IndexSetSplitTest, RejectsBranchesConstantOverTheLoop) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);
    add_arrays(builder);
    auto& map = add_map(builder, builder.subject().root(), "j", 64);
    auto j = map.indvar();
    auto& branches = builder.add_if_else(map.root());
    copy(builder, builder.add_case(branches, symbolic::Lt(j, symbolic::integer(100))), j, j);

    analysis::AnalysisManager analysis_manager(builder.subject());
    transformations::IndexSetSplit split(map);
    EXPECT_FALSE(split.can_be_applied(builder, analysis_manager));
}
