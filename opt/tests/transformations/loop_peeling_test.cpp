#include "sdfg/transformations/loop_peeling.h"

#include <gtest/gtest.h>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/structured_control_flow/if_else.h"

using namespace sdfg;

/// Build for(i = M; i < M + 8 && i < N; i++) { A[i] = A[i] } with a compound condition.
static builder::StructuredSDFGBuilder make_compound_loop(structured_control_flow::For*& out_loop) {
    builder::StructuredSDFGBuilder builder("pb_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    types::Pointer opaque_desc;
    builder.add_container("A", opaque_desc, true);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("M", sym_desc, true);
    builder.add_container("i", sym_desc);

    auto indvar = symbolic::symbol("i");
    auto init = symbolic::symbol("M");
    auto canonical = symbolic::add(symbolic::symbol("M"), symbolic::integer(8));
    auto dynamic = symbolic::symbol("N");
    auto condition = symbolic::And(symbolic::Lt(indvar, canonical), symbolic::Lt(indvar, dynamic));
    auto update = symbolic::add(indvar, symbolic::integer(1));

    auto& loop = builder.add_for(root, indvar, condition, init, update);
    auto& body = loop.root();

    auto& block = builder.add_block(body);
    auto& A_in = builder.add_access(block, "A");
    auto& A_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, A_in, tasklet, "_in", {indvar}, desc);
    builder.add_computational_memlet(block, tasklet, "_out", A_out, {indvar}, desc);

    out_loop = &loop;
    return builder;
}

/// Build a perfectly nested pair of compound-condition loops:
///   for(i = P; i < P+4 && i < N; i++) for(j = Q; j < Q+4 && j < N; j++) { A[i] = A[i] }
static builder::StructuredSDFGBuilder make_nested_loops(structured_control_flow::For*& out_outer) {
    builder::StructuredSDFGBuilder builder("pb_nest", FunctionType_CPU);
    auto& root = builder.subject().root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    types::Pointer opaque_desc;
    builder.add_container("A", opaque_desc, true);
    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    for (auto name : {"N", "P", "Q", "i", "j"}) {
        builder.add_container(name, sym_desc);
    }

    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");
    auto& outer = builder.add_for(
        root,
        i,
        symbolic::
            And(symbolic::Lt(i, symbolic::add(symbolic::symbol("P"), symbolic::integer(4))),
                symbolic::Lt(i, symbolic::symbol("N"))),
        symbolic::symbol("P"),
        symbolic::add(i, symbolic::integer(1))
    );
    auto& inner = builder.add_for(
        outer.root(),
        j,
        symbolic::
            And(symbolic::Lt(j, symbolic::add(symbolic::symbol("Q"), symbolic::integer(4))),
                symbolic::Lt(j, symbolic::symbol("N"))),
        symbolic::symbol("Q"),
        symbolic::add(j, symbolic::integer(1))
    );
    auto& block = builder.add_block(inner.root());
    auto& a_in = builder.add_access(block, "A");
    auto& a_out = builder.add_access(block, "A");
    auto& t = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, a_in, t, "_in", {i}, desc);
    builder.add_computational_memlet(block, t, "_out", a_out, {i}, desc);

    out_outer = &outer;
    return builder;
}

TEST(LoopPeelingTest, HoistedFormLeavesCleanThenBranch) {
    structured_control_flow::For* orig = nullptr;
    auto builder = make_compound_loop(orig);

    auto sdfg = builder.move();
    builder::StructuredSDFGBuilder b(sdfg);
    analysis::AnalysisManager am(b.subject());

    transformations::LoopPeeling t(*orig); // default: hoisted then/else
    EXPECT_TRUE(t.can_be_applied(b, am));
    t.apply(b, am);

    auto& s = b.subject();
    ASSERT_EQ(s.root().size(), 1);
    auto* if_else = dyn_cast<structured_control_flow::IfElse*>(&s.root().at(0));
    ASSERT_TRUE(if_else != nullptr);
    ASSERT_EQ(if_else->size(), 2);

    // "then" branch: clean 0-based constant-trip loop with an UNGUARDED body.
    auto then_case = if_else->at(0);
    ASSERT_EQ(then_case.first.size(), 1);
    auto* then_for = dyn_cast<structured_control_flow::For*>(&then_case.first.at(0));
    ASSERT_TRUE(then_for != nullptr);
    EXPECT_TRUE(symbolic::eq(then_for->init(), symbolic::integer(0)));
    EXPECT_TRUE(symbolic::eq(then_for->condition(), symbolic::Lt(symbolic::symbol("i"), symbolic::integer(8))));
    ASSERT_EQ(then_for->root().size(), 1);
    EXPECT_TRUE(dyn_cast<structured_control_flow::Block*>(&then_for->root().at(0)) != nullptr);

    // "else" branch: original variable-trip remainder.
    auto else_case = if_else->at(1);
    auto* else_for = dyn_cast<structured_control_flow::For*>(&else_case.first.at(0));
    ASSERT_TRUE(else_for != nullptr);
    EXPECT_TRUE(symbolic::eq(else_for->init(), symbolic::symbol("M")));
}

TEST(LoopPeelingTest, PredicatedFormCollectsNestAndGuardsInnermost) {
    structured_control_flow::For* outer = nullptr;
    auto builder = make_nested_loops(outer);

    auto sdfg = builder.move();
    builder::StructuredSDFGBuilder b(sdfg);
    analysis::AnalysisManager am(b.subject());

    transformations::LoopPeeling t(*outer, /*predicate=*/true);
    EXPECT_TRUE(t.can_be_applied(b, am));
    t.apply(b, am);

    auto& s = b.subject();
    ASSERT_EQ(s.root().size(), 1);

    // Both loops of the nest are collected and rebuilt 0-based (i<4, j<4)...
    auto* holder = dyn_cast<structured_control_flow::Sequence*>(&s.root().at(0));
    ASSERT_TRUE(holder != nullptr);
    ASSERT_EQ(holder->size(), 1);
    auto* for_i = dyn_cast<structured_control_flow::For*>(&holder->at(0));
    ASSERT_TRUE(for_i != nullptr);
    EXPECT_TRUE(symbolic::eq(for_i->init(), symbolic::integer(0)));
    EXPECT_TRUE(symbolic::eq(for_i->condition(), symbolic::Lt(symbolic::symbol("i"), symbolic::integer(4))));
    ASSERT_EQ(for_i->root().size(), 1);
    auto* for_j = dyn_cast<structured_control_flow::For*>(&for_i->root().at(0));
    ASSERT_TRUE(for_j != nullptr);
    EXPECT_TRUE(symbolic::eq(for_j->init(), symbolic::integer(0)));
    EXPECT_TRUE(symbolic::eq(for_j->condition(), symbolic::Lt(symbolic::symbol("j"), symbolic::integer(4))));

    // ...with a single combined guard at the innermost body (no remainder branch).
    ASSERT_EQ(for_j->root().size(), 1);
    auto* if_else = dyn_cast<structured_control_flow::IfElse*>(&for_j->root().at(0));
    ASSERT_TRUE(if_else != nullptr);
    ASSERT_EQ(if_else->size(), 1);
    EXPECT_TRUE(dyn_cast<structured_control_flow::Block*>(&if_else->at(0).first.at(0)) != nullptr);
}

TEST(LoopPeelingTest, NotApplicableToSimpleLoop) {
    builder::StructuredSDFGBuilder builder("pb_simple", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Pointer opaque_desc;
    builder.add_container("A", opaque_desc, true);
    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);

    auto indvar = symbolic::symbol("i");
    auto condition = symbolic::Lt(indvar, symbolic::symbol("N"));
    auto& loop =
        builder.add_for(root, indvar, condition, symbolic::integer(0), symbolic::add(indvar, symbolic::integer(1)));
    builder.add_block(loop.root());

    auto sdfg_moved = builder.move();
    builder::StructuredSDFGBuilder b(sdfg_moved);
    analysis::AnalysisManager am(b.subject());

    transformations::LoopPeeling t(loop);
    EXPECT_FALSE(t.can_be_applied(b, am));
}
