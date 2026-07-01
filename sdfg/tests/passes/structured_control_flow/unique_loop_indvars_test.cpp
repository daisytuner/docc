#include "sdfg/passes/structured_control_flow/unique_loop_indvars.h"

#include <gtest/gtest.h>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/structured_control_flow/for.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/array.h"
#include "sdfg/types/pointer.h"
#include "sdfg/types/scalar.h"

using namespace sdfg;

TEST(UniqueLoopIndvarsTest, DistinctIndvars_NoChange) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& root = builder.subject().root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);

    auto bound = symbolic::symbol("N");
    auto init = symbolic::integer(0);

    auto i = symbolic::symbol("i");
    builder.add_for(root, i, symbolic::Lt(i, bound), init, symbolic::add(i, symbolic::integer(1)));

    auto j = symbolic::symbol("j");
    builder.add_for(root, j, symbolic::Lt(j, bound), init, symbolic::add(j, symbolic::integer(1)));

    // Run pass
    auto sdfg_opt = builder.move();
    builder::StructuredSDFGBuilder builder_opt(sdfg_opt);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::UniqueLoopIndvars pass;
    EXPECT_FALSE(pass.run(builder_opt, analysis_manager));

    auto& result = builder_opt.subject();
    auto* loop0 = dynamic_cast<const structured_control_flow::StructuredLoop*>(&result.root().at(0).first);
    auto* loop1 = dynamic_cast<const structured_control_flow::StructuredLoop*>(&result.root().at(1).first);
    ASSERT_NE(loop0, nullptr);
    ASSERT_NE(loop1, nullptr);
    EXPECT_TRUE(symbolic::eq(loop0->indvar(), symbolic::symbol("i")));
    EXPECT_TRUE(symbolic::eq(loop1->indvar(), symbolic::symbol("j")));
}

TEST(UniqueLoopIndvarsTest, SiblingLoops_SameIndvar_Renamed) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& root = builder.subject().root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);

    auto bound = symbolic::symbol("N");
    auto init = symbolic::integer(0);
    auto i = symbolic::symbol("i");

    builder.add_for(root, i, symbolic::Lt(i, bound), init, symbolic::add(i, symbolic::integer(1)));
    builder.add_for(root, i, symbolic::Lt(i, bound), init, symbolic::add(i, symbolic::integer(1)));

    // Run pass
    auto sdfg_opt = builder.move();
    builder::StructuredSDFGBuilder builder_opt(sdfg_opt);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::UniqueLoopIndvars pass;
    EXPECT_TRUE(pass.run(builder_opt, analysis_manager));

    auto& result = builder_opt.subject();
    auto* loop0 = dynamic_cast<const structured_control_flow::StructuredLoop*>(&result.root().at(0).first);
    auto* loop1 = dynamic_cast<const structured_control_flow::StructuredLoop*>(&result.root().at(1).first);
    ASSERT_NE(loop0, nullptr);
    ASSERT_NE(loop1, nullptr);

    // First loop keeps the original indvar, second is renamed to something unique.
    EXPECT_TRUE(symbolic::eq(loop0->indvar(), symbolic::symbol("i")));
    EXPECT_FALSE(symbolic::eq(loop1->indvar(), symbolic::symbol("i")));
    EXPECT_TRUE(result.exists(loop1->indvar()->get_name()));
}

TEST(UniqueLoopIndvarsTest, NestedLoops_SameIndvar_InnerRenamed) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& root = builder.subject().root();

    types::Scalar float_type(types::PrimitiveType::Float);
    types::Array array_type(float_type, {symbolic::symbol("M")});
    types::Pointer pointer_type(array_type);
    builder.add_container("A", pointer_type, true);
    builder.add_container("B", pointer_type);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("M", sym_desc, true);
    builder.add_container("i", sym_desc);

    auto init = symbolic::integer(0);
    auto i = symbolic::symbol("i");

    auto& outer =
        builder.add_for(root, i, symbolic::Lt(i, symbolic::symbol("N")), init, symbolic::add(i, symbolic::integer(1)));
    auto& inner = builder.add_for(
        outer.root(), i, symbolic::Lt(i, symbolic::symbol("M")), init, symbolic::add(i, symbolic::integer(1))
    );

    auto& block = builder.add_block(inner.root());
    auto& a = builder.add_access(block, "A");
    auto& b = builder.add_access(block, "B");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, a, tasklet, "_in", {i}, pointer_type);
    builder.add_computational_memlet(block, tasklet, "_out", b, {i}, pointer_type);

    // Run pass
    auto sdfg_opt = builder.move();
    builder::StructuredSDFGBuilder builder_opt(sdfg_opt);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::UniqueLoopIndvars pass;
    EXPECT_TRUE(pass.run(builder_opt, analysis_manager));

    auto& result = builder_opt.subject();
    auto* outer_res = dynamic_cast<const structured_control_flow::StructuredLoop*>(&result.root().at(0).first);
    ASSERT_NE(outer_res, nullptr);
    auto* inner_res = dynamic_cast<const structured_control_flow::StructuredLoop*>(&outer_res->root().at(0).first);
    ASSERT_NE(inner_res, nullptr);

    // Outer keeps the original indvar; the inner one is disambiguated.
    EXPECT_TRUE(symbolic::eq(outer_res->indvar(), symbolic::symbol("i")));
    EXPECT_FALSE(symbolic::eq(inner_res->indvar(), symbolic::symbol("i")));
    auto new_indvar = inner_res->indvar();
    EXPECT_TRUE(result.exists(new_indvar->get_name()));

    // Uses within the inner loop are rewritten to the new indvar.
    EXPECT_TRUE(symbolic::eq(inner_res->condition(), symbolic::Lt(new_indvar, symbolic::symbol("M"))));
}

TEST(UniqueLoopIndvarsTest, NestedMaps_SameIndvar_InnerRenamed) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& root = builder.subject().root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("M", sym_desc, true);
    builder.add_container("i", sym_desc);

    auto init = symbolic::integer(0);
    auto i = symbolic::symbol("i");

    auto& outer = builder.add_map(
        root,
        i,
        symbolic::Lt(i, symbolic::symbol("N")),
        init,
        symbolic::add(i, symbolic::integer(1)),
        ScheduleType_Sequential::create()
    );
    builder.add_map(
        outer.root(),
        i,
        symbolic::Lt(i, symbolic::symbol("M")),
        init,
        symbolic::add(i, symbolic::integer(1)),
        ScheduleType_Sequential::create()
    );

    // Run pass
    auto sdfg_opt = builder.move();
    builder::StructuredSDFGBuilder builder_opt(sdfg_opt);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::UniqueLoopIndvars pass;
    EXPECT_TRUE(pass.run(builder_opt, analysis_manager));

    auto& result = builder_opt.subject();
    auto* outer_res = dynamic_cast<const structured_control_flow::StructuredLoop*>(&result.root().at(0).first);
    ASSERT_NE(outer_res, nullptr);
    auto* inner_res = dynamic_cast<const structured_control_flow::StructuredLoop*>(&outer_res->root().at(0).first);
    ASSERT_NE(inner_res, nullptr);

    EXPECT_TRUE(symbolic::eq(outer_res->indvar(), symbolic::symbol("i")));
    EXPECT_FALSE(symbolic::eq(inner_res->indvar(), symbolic::symbol("i")));
    EXPECT_TRUE(result.exists(inner_res->indvar()->get_name()));
}

TEST(UniqueLoopIndvarsTest, Idempotent) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& root = builder.subject().root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("M", sym_desc, true);
    builder.add_container("i", sym_desc);

    auto init = symbolic::integer(0);
    auto i = symbolic::symbol("i");

    auto& outer =
        builder.add_for(root, i, symbolic::Lt(i, symbolic::symbol("N")), init, symbolic::add(i, symbolic::integer(1)));
    builder
        .add_for(outer.root(), i, symbolic::Lt(i, symbolic::symbol("M")), init, symbolic::add(i, symbolic::integer(1)));

    auto sdfg_opt = builder.move();
    builder::StructuredSDFGBuilder builder_opt(sdfg_opt);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());

    passes::UniqueLoopIndvars pass;
    EXPECT_TRUE(pass.run(builder_opt, analysis_manager));
    // Second run has nothing left to disambiguate.
    EXPECT_FALSE(pass.run(builder_opt, analysis_manager));
}

TEST(UniqueLoopIndvarsTest, IndvarReadAfterLoop_NotRenamed) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& root = builder.subject().root();

    types::Scalar float_type(types::PrimitiveType::Float);
    types::Pointer pointer_type(float_type);
    builder.add_container("A", pointer_type, true);
    builder.add_container("B", pointer_type, true);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);

    auto init = symbolic::integer(0);
    auto i = symbolic::symbol("i");
    auto cond = symbolic::Lt(i, symbolic::symbol("N"));
    auto upd = symbolic::add(i, symbolic::integer(1));

    // Helper: A[i] = A[i] inside the given sequence (reads the induction variable).
    auto add_copy = [&](structured_control_flow::Sequence& seq) {
        auto& block = builder.add_block(seq);
        auto& in = builder.add_access(block, "A");
        auto& out = builder.add_access(block, "B");
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
        builder.add_computational_memlet(block, in, tasklet, "_in", {i}, pointer_type);
        builder.add_computational_memlet(block, tasklet, "_out", out, {i}, pointer_type);
    };

    // First loop reserves the induction variable "i".
    auto& loop0 = builder.add_for(root, i, cond, init, upd);
    add_copy(loop0.root());

    // Second loop clashes on "i" and is genuinely read in its body.
    auto& loop1 = builder.add_for(root, i, cond, init, upd);
    add_copy(loop1.root());

    // Valid user of "i" AFTER the second loop: its final value escapes the loop,
    // so a local rename of loop1 would leave this read dangling.
    add_copy(root);

    // Run pass
    auto sdfg_opt = builder.move();
    builder::StructuredSDFGBuilder builder_opt(sdfg_opt);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::UniqueLoopIndvars pass;
    EXPECT_FALSE(pass.run(builder_opt, analysis_manager));

    auto& result = builder_opt.subject();
    auto* l0 = dynamic_cast<const structured_control_flow::StructuredLoop*>(&result.root().at(0).first);
    auto* l1 = dynamic_cast<const structured_control_flow::StructuredLoop*>(&result.root().at(1).first);
    ASSERT_NE(l0, nullptr);
    ASSERT_NE(l1, nullptr);

    // Neither loop is renamed: the clashing loop is left alone because its
    // induction variable is read after the loop.
    EXPECT_TRUE(symbolic::eq(l0->indvar(), symbolic::symbol("i")));
    EXPECT_TRUE(symbolic::eq(l1->indvar(), symbolic::symbol("i")));
}
