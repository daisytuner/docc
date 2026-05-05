#include "sdfg/analysis/flop_analysis.h"

#include <gtest/gtest.h>
#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_nodes/math/cmath/cmath_node.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/element.h"
#include "sdfg/function.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/pointer.h"
#include "sdfg/types/scalar.h"
#include "sdfg/types/type.h"

using namespace sdfg;

TEST(FlopAnalysis, Tasklet) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar desc(types::PrimitiveType::Float);
    builder.add_container("a", desc);
    builder.add_container("b", desc);
    builder.add_container("c", desc);

    // Add block with tasklet
    auto& block = builder.add_block(root);
    auto& a = builder.add_access(block, "a");
    auto& b = builder.add_access(block, "b");
    auto& c = builder.add_access(block, "c");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, {"_out"}, {"_in1", "_in2"});
    builder.add_computational_memlet(block, a, tasklet, "_in1", {});
    builder.add_computational_memlet(block, b, tasklet, "_in2", {});
    builder.add_computational_memlet(block, tasklet, "_out", c, {});

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::FlopAnalysis>();

    // Check
    EXPECT_TRUE(analysis.precise());
    symbolic::Expression flop;
    ASSERT_TRUE(analysis.contains(&block));
    flop = analysis.get(&block);
    ASSERT_FALSE(flop.is_null());
    EXPECT_TRUE(symbolic::eq(flop, symbolic::one()));
    ASSERT_TRUE(analysis.contains(&root));
    flop = analysis.get(&root);
    ASSERT_FALSE(flop.is_null());
    EXPECT_TRUE(symbolic::eq(flop, symbolic::one()));
}

TEST(FlopAnalysis, MultipleTasklets) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar desc(types::PrimitiveType::Float);
    builder.add_container("a", desc);
    builder.add_container("b", desc);
    builder.add_container("c", desc);
    builder.add_container("d", desc);
    builder.add_container("e", desc);
    builder.add_container("f", desc);

    // Add block with two tasklets
    auto& block = builder.add_block(root);
    auto& a = builder.add_access(block, "a");
    auto& b = builder.add_access(block, "b");
    auto& c = builder.add_access(block, "c");
    auto& d = builder.add_access(block, "d");
    auto& e = builder.add_access(block, "e");
    auto& f = builder.add_access(block, "f");
    auto& tasklet1 = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, {"_out"}, {"_in1", "_in2"});
    builder.add_computational_memlet(block, a, tasklet1, "_in1", {});
    builder.add_computational_memlet(block, b, tasklet1, "_in2", {});
    builder.add_computational_memlet(block, tasklet1, "_out", c, {});
    auto& tasklet2 = builder.add_tasklet(block, data_flow::TaskletCode::fp_fma, {"_out"}, {"_in1", "_in2", "_in3"});
    builder.add_computational_memlet(block, c, tasklet2, "_in1", {});
    builder.add_computational_memlet(block, d, tasklet2, "_in2", {});
    builder.add_computational_memlet(block, e, tasklet2, "_in3", {});
    builder.add_computational_memlet(block, tasklet2, "_out", f, {});

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::FlopAnalysis>();

    // Check
    EXPECT_TRUE(analysis.precise());
    symbolic::Expression flop;
    ASSERT_TRUE(analysis.contains(&block));
    flop = analysis.get(&block);
    ASSERT_FALSE(flop.is_null());
    EXPECT_TRUE(symbolic::eq(flop, symbolic::integer(3)));
    ASSERT_TRUE(analysis.contains(&root));
    flop = analysis.get(&root);
    ASSERT_FALSE(flop.is_null());
    EXPECT_TRUE(symbolic::eq(flop, symbolic::integer(3)));
}

TEST(FlopAnalysis, Loop) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar sym_desc(types::PrimitiveType::Int64);
    builder.add_container("i", sym_desc);
    builder.add_container("n", sym_desc, true);
    types::Scalar desc(types::PrimitiveType::Float);
    builder.add_container("a", desc, true);
    builder.add_container("b", desc, true);
    builder.add_container("c", desc, true);

    // Add loop
    auto i = symbolic::symbol("i");
    auto n = symbolic::symbol("n");
    auto& loop = builder.add_for(root, i, symbolic::Lt(i, n), symbolic::zero(), symbolic::add(i, symbolic::one()));

    // Add block with tasklet
    auto& block = builder.add_block(loop.root());
    auto& a = builder.add_access(block, "a");
    auto& b = builder.add_access(block, "b");
    auto& c = builder.add_access(block, "c");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, {"_out"}, {"_in1", "_in2"});
    builder.add_computational_memlet(block, a, tasklet, "_in1", {});
    builder.add_computational_memlet(block, b, tasklet, "_in2", {});
    builder.add_computational_memlet(block, tasklet, "_out", c, {});

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::FlopAnalysis>();

    // Check
    EXPECT_TRUE(analysis.precise());
    symbolic::Expression flop;
    ASSERT_TRUE(analysis.contains(&block));
    flop = analysis.get(&block);
    ASSERT_FALSE(flop.is_null());
    EXPECT_TRUE(symbolic::eq(flop, symbolic::one()));
    ASSERT_TRUE(analysis.contains(&loop.root()));
    flop = analysis.get(&loop.root());
    ASSERT_FALSE(flop.is_null());
    EXPECT_TRUE(symbolic::eq(flop, symbolic::one()));
    ASSERT_TRUE(analysis.contains(&loop));
    flop = analysis.get(&loop);
    ASSERT_FALSE(flop.is_null());
    EXPECT_TRUE(symbolic::eq(flop, symbolic::symbol("n")));
    ASSERT_TRUE(analysis.contains(&root));
    flop = analysis.get(&root);
    ASSERT_FALSE(flop.is_null());
    EXPECT_TRUE(symbolic::eq(flop, symbolic::symbol("n")));
}

TEST(FlopAnalysis, DISABLED_LoopNest) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar sym_desc(types::PrimitiveType::Int64);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);
    builder.add_container("n", sym_desc, true);
    builder.add_container("m", sym_desc, true);
    builder.add_container("k", sym_desc, true);
    types::Scalar desc(types::PrimitiveType::Float);
    builder.add_container("a", desc);
    builder.add_container("b", desc);
    builder.add_container("c", desc);
    builder.add_container("d", desc);

    // Add first loop
    auto i = symbolic::symbol("i");
    auto n = symbolic::symbol("n");
    auto& loop1 = builder.add_for(root, i, symbolic::Lt(i, n), symbolic::zero(), symbolic::add(i, symbolic::one()));

    // Add second loop, bound: j <= m && j < k, update: j = j + 2
    auto j = symbolic::symbol("j");
    auto bound2 = symbolic::And(symbolic::Le(j, symbolic::symbol("m")), symbolic::Lt(j, symbolic::symbol("k")));
    auto& loop2 = builder.add_for(loop1.root(), j, bound2, symbolic::one(), symbolic::add(j, symbolic::integer(2)));

    // Add block with tasklet
    auto& block = builder.add_block(loop2.root());
    auto& a = builder.add_access(block, "a");
    auto& b = builder.add_access(block, "b");
    auto& c = builder.add_access(block, "c");
    auto& d = builder.add_access(block, "d");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_fma, {"_out"}, {"_in1", "_in2", "_in3"});
    builder.add_computational_memlet(block, a, tasklet, "_in1", {});
    builder.add_computational_memlet(block, b, tasklet, "_in2", {});
    builder.add_computational_memlet(block, c, tasklet, "_in2", {});
    builder.add_computational_memlet(block, tasklet, "_out", d, {});

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::FlopAnalysis>();

    // Check
    EXPECT_FALSE(analysis.precise()); // because of update j = j + 2
    symbolic::Expression flop;
    ASSERT_TRUE(analysis.contains(&block));
    flop = analysis.get(&block);
    ASSERT_FALSE(flop.is_null());
    EXPECT_TRUE(symbolic::eq(flop, symbolic::integer(2)));
    ASSERT_TRUE(analysis.contains(&loop2.root()));
    flop = analysis.get(&loop2.root());
    ASSERT_FALSE(flop.is_null());
    EXPECT_TRUE(symbolic::eq(flop, symbolic::integer(2)));
    ASSERT_TRUE(analysis.contains(&loop2));
    flop = analysis.get(&loop2);
    ASSERT_FALSE(flop.is_null());
    EXPECT_TRUE(symbolic::eq(flop, symbolic::parse("idiv(min(m, -1 + k), 2) * 2")));
    ASSERT_TRUE(analysis.contains(&loop1.root()));
    flop = analysis.get(&loop1.root());
    ASSERT_FALSE(flop.is_null());
    EXPECT_TRUE(symbolic::eq(flop, symbolic::parse("idiv(min(m, -1 + k), 2) * 2")));
    ASSERT_TRUE(analysis.contains(&loop1));
    flop = analysis.get(&loop1);
    ASSERT_FALSE(flop.is_null());
    EXPECT_TRUE(symbolic::eq(flop, symbolic::parse("n * idiv(min(m, -1 + k), 2) * 2")));
    ASSERT_TRUE(analysis.contains(&root));
    flop = analysis.get(&root);
    ASSERT_FALSE(flop.is_null());
    EXPECT_TRUE(symbolic::eq(flop, symbolic::parse("n * idiv(min(m, -1 + k), 2) * 2")));
}

TEST(FlopAnalysis, Intrinsic) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar desc(types::PrimitiveType::Float);
    builder.add_container("a", desc);
    builder.add_container("b", desc);

    // Add block with library node
    auto& block = builder.add_block(root);
    auto& a = builder.add_access(block, "a");
    auto& b = builder.add_access(block, "b");
    auto& libnode = builder.add_library_node<
        math::cmath::CMathNode>(block, DebugInfo(), math::cmath::CMathFunction::sin, types::PrimitiveType::Float);
    builder.add_computational_memlet(block, a, libnode, "_in1", {}, desc);
    builder.add_computational_memlet(block, libnode, "_out", b, {}, desc);

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::FlopAnalysis>();

    // Check
    EXPECT_TRUE(analysis.precise());
    symbolic::Expression flop;
    ASSERT_TRUE(analysis.contains(&block));
    flop = analysis.get(&block);
    ASSERT_FALSE(flop.is_null());
    EXPECT_TRUE(symbolic::eq(flop, symbolic::one()));
    ASSERT_TRUE(analysis.contains(&root));
    flop = analysis.get(&root);
    ASSERT_FALSE(flop.is_null());
    EXPECT_TRUE(symbolic::eq(flop, symbolic::one()));
}

TEST(FlopAnalysis, IfElse) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar desc(types::PrimitiveType::Float);
    builder.add_container("a", desc);
    builder.add_container("b", desc);
    builder.add_container("c", desc);
    builder.add_container("d", desc);

    // Add IfElse
    auto& if_else = builder.add_if_else(root);
    auto& case1 = builder.add_case(if_else, symbolic::Eq(symbolic::symbol("a"), symbolic::zero()));
    auto& case2 = builder.add_case(if_else, symbolic::Ne(symbolic::symbol("a"), symbolic::zero()));

    // Add first block with tasklet
    auto& block1 = builder.add_block(case1);
    {
        auto& b = builder.add_access(block1, "b");
        auto& c = builder.add_access(block1, "c");
        auto& d = builder.add_access(block1, "d");
        auto& tasklet = builder.add_tasklet(block1, data_flow::TaskletCode::fp_add, {"_out"}, {"_in1", "_in2"});
        builder.add_computational_memlet(block1, b, tasklet, "_in1", {});
        builder.add_computational_memlet(block1, c, tasklet, "_in2", {});
        builder.add_computational_memlet(block1, tasklet, "_out", d, {});
    }

    // Add second block with tasklet
    auto& block2 = builder.add_block(case2);
    {
        auto& a = builder.add_access(block2, "a");
        auto& b = builder.add_access(block2, "b");
        auto& c = builder.add_access(block2, "c");
        auto& d = builder.add_access(block2, "d");
        auto& tasklet = builder.add_tasklet(block2, data_flow::TaskletCode::fp_fma, {"_out"}, {"_in1", "_in2", "_in3"});
        builder.add_computational_memlet(block2, a, tasklet, "_in1", {});
        builder.add_computational_memlet(block2, b, tasklet, "_in2", {});
        builder.add_computational_memlet(block2, c, tasklet, "_in3", {});
        builder.add_computational_memlet(block2, tasklet, "_out", d, {});
    }

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::FlopAnalysis>();

    // Check
    EXPECT_FALSE(analysis.precise());
    symbolic::Expression flop;
    ASSERT_TRUE(analysis.contains(&block1));
    flop = analysis.get(&block1);
    ASSERT_FALSE(flop.is_null());
    EXPECT_TRUE(symbolic::eq(flop, symbolic::one()));
    ASSERT_TRUE(analysis.contains(&block2));
    flop = analysis.get(&block2);
    ASSERT_FALSE(flop.is_null());
    EXPECT_TRUE(symbolic::eq(flop, symbolic::integer(2)));
    ASSERT_TRUE(analysis.contains(&if_else));
    flop = analysis.get(&if_else);
    ASSERT_FALSE(flop.is_null());
    EXPECT_TRUE(symbolic::eq(flop, symbolic::max(symbolic::one(), symbolic::integer(2))));
    ASSERT_TRUE(analysis.contains(&root));
    flop = analysis.get(&root);
    ASSERT_FALSE(flop.is_null());
    EXPECT_TRUE(symbolic::eq(flop, symbolic::max(symbolic::one(), symbolic::integer(2))));
}

TEST(FlopAnalysis, WhileLoop) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar desc(types::PrimitiveType::Float);
    builder.add_container("a", desc);
    builder.add_container("b", desc);
    builder.add_container("c", desc);

    // Add loop
    auto& loop = builder.add_while(root);

    // Add block with tasklet
    auto& block = builder.add_block(loop.root());
    auto& a = builder.add_access(block, "a");
    auto& b = builder.add_access(block, "b");
    auto& c = builder.add_access(block, "c");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, {"_out"}, {"_in1", "_in2"});
    builder.add_computational_memlet(block, a, tasklet, "_in1", {});
    builder.add_computational_memlet(block, b, tasklet, "_in2", {});
    builder.add_computational_memlet(block, tasklet, "_out", c, {});

    // Add break
    auto& break_node = builder.add_break(loop.root());

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::FlopAnalysis>();

    // Check
    EXPECT_FALSE(analysis.precise());
    symbolic::Expression flop;
    ASSERT_TRUE(analysis.contains(&block));
    flop = analysis.get(&block);
    ASSERT_FALSE(flop.is_null());
    EXPECT_TRUE(symbolic::eq(flop, symbolic::one()));
    ASSERT_TRUE(analysis.contains(&break_node));
    flop = analysis.get(&break_node);
    ASSERT_FALSE(flop.is_null());
    EXPECT_TRUE(symbolic::eq(flop, symbolic::zero()));
    ASSERT_TRUE(analysis.contains(&loop.root()));
    flop = analysis.get(&loop.root());
    ASSERT_FALSE(flop.is_null());
    EXPECT_TRUE(symbolic::eq(flop, symbolic::one()));
    ASSERT_TRUE(analysis.contains(&loop));
    flop = analysis.get(&loop);
    ASSERT_TRUE(flop.is_null());
    ASSERT_TRUE(analysis.contains(&root));
    flop = analysis.get(&root);
    ASSERT_TRUE(flop.is_null());
}

TEST(FlopAnalysis, LoopIndvarDependency) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar sym_desc(types::PrimitiveType::Int64);
    builder.add_container("i", sym_desc);
    builder.add_container("m", sym_desc);
    builder.add_container("j", sym_desc);
    builder.add_container("n", sym_desc);
    types::Scalar desc(types::PrimitiveType::Float);
    builder.add_container("a", desc);
    builder.add_container("b", desc);
    builder.add_container("c", desc);

    // Add first loop
    auto i = symbolic::symbol("i");
    auto n = symbolic::symbol("n");
    auto& loop1 = builder.add_for(root, i, symbolic::Lt(i, n), symbolic::zero(), symbolic::add(i, symbolic::one()));

    // Add second loop
    auto j = symbolic::symbol("j");
    auto m = symbolic::symbol("m");
    auto& loop2 = builder.add_for(loop1.root(), j, symbolic::Lt(j, m), i, symbolic::add(j, symbolic::one()));

    // Add block with tasklet
    auto& block = builder.add_block(loop2.root());
    auto& a = builder.add_access(block, "a");
    auto& b = builder.add_access(block, "b");
    auto& c = builder.add_access(block, "c");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, {"_out"}, {"_in1", "_in2"});
    builder.add_computational_memlet(block, a, tasklet, "_in1", {});
    builder.add_computational_memlet(block, b, tasklet, "_in2", {});
    builder.add_computational_memlet(block, tasklet, "_out", c, {});

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::FlopAnalysis>();

    // Check
    EXPECT_FALSE(analysis.precise());
    symbolic::Expression flop;
    ASSERT_TRUE(analysis.contains(&block));
    flop = analysis.get(&block);
    ASSERT_FALSE(flop.is_null());
    EXPECT_TRUE(symbolic::eq(flop, symbolic::one()));
    ASSERT_TRUE(analysis.contains(&loop2.root()));
    flop = analysis.get(&loop2.root());
    ASSERT_FALSE(flop.is_null());
    EXPECT_TRUE(symbolic::eq(flop, symbolic::one()));
    ASSERT_TRUE(analysis.contains(&loop2));
    flop = analysis.get(&loop2);
    ASSERT_FALSE(flop.is_null());
    EXPECT_TRUE(symbolic::eq(flop, symbolic::parse("m - idiv(n - 1, 2)")));
    ASSERT_TRUE(analysis.contains(&loop1.root()));
    flop = analysis.get(&loop1.root());
    ASSERT_FALSE(flop.is_null());
    EXPECT_TRUE(symbolic::eq(flop, symbolic::parse("m - idiv(n - 1, 2)")));
    ASSERT_TRUE(analysis.contains(&loop1));
    flop = analysis.get(&loop1);
    ASSERT_FALSE(flop.is_null());
    EXPECT_TRUE(symbolic::eq(flop, symbolic::parse("n * (m - idiv(n - 1, 2))")));
    ASSERT_TRUE(analysis.contains(&root));
    flop = analysis.get(&root);
    ASSERT_FALSE(flop.is_null());
    EXPECT_TRUE(symbolic::eq(flop, symbolic::parse("n * (m - idiv(n - 1, 2))")));
}

// Disable
TEST(FlopAnalysis, DISABLED_SPMV) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar sym_desc(types::PrimitiveType::Int64);
    types::Pointer sym_desc2(sym_desc);
    types::Scalar desc(types::PrimitiveType::Float);
    types::Pointer desc2(desc);
    builder.add_container("nrows", sym_desc, true);
    builder.add_container("Arow", sym_desc2, true);
    builder.add_container("Acol", sym_desc2, true);
    builder.add_container("Aval", desc2, true);
    builder.add_container("x", desc2, true);
    builder.add_container("y", desc2, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);
    builder.add_container("start", sym_desc);
    builder.add_container("end", sym_desc);
    builder.add_container("tmp", sym_desc);

    // Add first loop
    auto i = symbolic::symbol("i");
    auto n = symbolic::symbol("nrows");
    auto& loop1 = builder.add_for(root, i, symbolic::Lt(i, n), symbolic::zero(), symbolic::add(i, symbolic::one()));

    // Add block with start and end values for second loop
    auto& block1 = builder.add_block(loop1.root());
    auto& Arow = builder.add_access(block1, "Arow");
    auto& start1 = builder.add_access(block1, "start");
    auto& end1 = builder.add_access(block1, "end");
    auto& tasklet1 = builder.add_tasklet(block1, data_flow::TaskletCode::assign, {"_out"}, {"_in"});
    builder.add_computational_memlet(block1, Arow, tasklet1, "_in", {i});
    builder.add_computational_memlet(block1, tasklet1, "_out", start1, {});
    auto& tasklet2 = builder.add_tasklet(block1, data_flow::TaskletCode::assign, {"_out"}, {"_in"});
    builder.add_computational_memlet(block1, Arow, tasklet2, "_in", {symbolic::add(i, symbolic::one())});
    builder.add_computational_memlet(block1, tasklet2, "_out", end1, {});

    // Add second loop
    auto j = symbolic::symbol("j");
    auto& loop2 = builder.add_for(
        loop1.root(),
        j,
        symbolic::Lt(j, symbolic::symbol("end")),
        symbolic::symbol("start"),
        symbolic::add(j, symbolic::one())
    );

    // Add block that fills temporary value
    auto& block2 = builder.add_block(loop2.root());
    auto& Acol = builder.add_access(block2, "Acol");
    auto& tmp = builder.add_access(block2, "tmp");
    auto& tasklet3 = builder.add_tasklet(block2, data_flow::TaskletCode::assign, {"_out"}, {"_in"});
    builder.add_computational_memlet(block2, Acol, tasklet3, "_in", {j});
    builder.add_computational_memlet(block2, tasklet3, "_out", tmp, {});

    // Add block that does the computation
    auto& block3 = builder.add_block(loop2.root());
    auto& Aval = builder.add_access(block3, "Aval");
    auto& x = builder.add_access(block3, "x");
    auto& y1 = builder.add_access(block3, "y");
    auto& y2 = builder.add_access(block3, "y");
    auto& tasklet4 = builder.add_tasklet(block3, data_flow::TaskletCode::fp_fma, {"_out"}, {"_in1", "_in2", "_in3"});
    builder.add_computational_memlet(block3, Aval, tasklet4, "_in1", {j});
    builder.add_computational_memlet(block3, x, tasklet4, "_in2", {symbolic::symbol("tmp")});
    builder.add_computational_memlet(block3, y1, tasklet4, "_in3", {i});
    builder.add_computational_memlet(block3, tasklet4, "_out", y2, {i});

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::FlopAnalysis>();

    // Check
    EXPECT_FALSE(analysis.precise());
    symbolic::Expression flop;
    ASSERT_TRUE(analysis.contains(&block2));
    flop = analysis.get(&block2);
    ASSERT_FALSE(flop.is_null());
    EXPECT_TRUE(symbolic::eq(flop, symbolic::zero()));
    ASSERT_TRUE(analysis.contains(&block3));
    flop = analysis.get(&block3);
    ASSERT_FALSE(flop.is_null());
    EXPECT_TRUE(symbolic::eq(flop, symbolic::integer(2)));
    ASSERT_TRUE(analysis.contains(&loop2.root()));
    flop = analysis.get(&loop2.root());
    ASSERT_FALSE(flop.is_null());
    EXPECT_TRUE(symbolic::eq(flop, symbolic::integer(2)));
    ASSERT_TRUE(analysis.contains(&loop2));
    flop = analysis.get(&loop2);
    ASSERT_FALSE(flop.is_null());
    EXPECT_TRUE(symbolic::eq(
        flop, symbolic::mul(symbolic::integer(2), symbolic::sub(symbolic::symbol("end"), symbolic::symbol("start")))
    ));
    ASSERT_TRUE(analysis.contains(&block1));
    flop = analysis.get(&block1);
    ASSERT_FALSE(flop.is_null());
    EXPECT_TRUE(symbolic::eq(flop, symbolic::zero()));
    ASSERT_TRUE(analysis.contains(&loop1.root()));
    flop = analysis.get(&loop1.root());
    ASSERT_FALSE(flop.is_null());
    auto expected_flop_loop2 = symbolic::
        mul(symbolic::integer(2),
            symbolic::
                add(symbolic::
                        min(symbolic::sub(symbolic::dynamic_sizeof(symbolic::symbol("Aval")), symbolic::one()),
                            symbolic::sub(symbolic::dynamic_sizeof(symbolic::symbol("Acol")), symbolic::one())),
                    symbolic::one()));
    EXPECT_TRUE(symbolic::eq(flop, expected_flop_loop2));
    ASSERT_TRUE(analysis.contains(&loop1));
    flop = analysis.get(&loop1);
    ASSERT_FALSE(flop.is_null());
    EXPECT_TRUE(symbolic::eq(flop, symbolic::mul(expected_flop_loop2, symbolic::symbol("nrows"))));
    ASSERT_TRUE(analysis.contains(&root));
    flop = analysis.get(&root);
    ASSERT_FALSE(flop.is_null());
    EXPECT_TRUE(symbolic::eq(flop, symbolic::mul(expected_flop_loop2, symbolic::symbol("nrows"))));
}

TEST(FlopAnalysis, DISABLED_NestedParameters) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar sym_desc(types::PrimitiveType::Int64);
    builder.add_container("n", sym_desc, true);
    builder.add_container("nLocal", sym_desc);
    builder.add_container("m", sym_desc, true);
    builder.add_container("mLocal", sym_desc);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Array desc(base_desc, symbolic::integer(42));
    types::Pointer desc2(desc);
    builder.add_container("A", desc2, true);

    // nLocal = n
    auto& block1 = builder.add_block(root);
    auto& n = builder.add_access(block1, "n");
    auto& nLocal = builder.add_access(block1, "nLocal");
    auto& tasklet1 = builder.add_tasklet(block1, data_flow::TaskletCode::assign, {"_out"}, {"_in"});
    builder.add_computational_memlet(block1, n, tasklet1, "_in", {});
    builder.add_computational_memlet(block1, tasklet1, "_out", nLocal, {});

    // for (i = 0; i < nLocal; i = 1 + i) { ... }
    auto i = symbolic::symbol("i");
    auto& loop1 = builder.add_for(
        root, i, symbolic::Lt(i, symbolic::symbol("nLocal")), symbolic::zero(), symbolic::add(i, symbolic::one())
    );

    // mLocal = m
    auto& block2 = builder.add_block(loop1.root());
    auto& m = builder.add_access(block2, "m");
    auto& mLocal = builder.add_access(block2, "mLocal");
    auto& tasklet2 = builder.add_tasklet(block2, data_flow::TaskletCode::assign, {"_out"}, {"_in"});
    builder.add_computational_memlet(block2, m, tasklet2, "_in", {});
    builder.add_computational_memlet(block2, tasklet2, "_out", mLocal, {});

    // for (j = 0; j < mLocal; j = 1 + j) { ... }
    auto j = symbolic::symbol("j");
    auto& loop2 = builder.add_for(
        loop1.root(), j, symbolic::Lt(j, symbolic::symbol("mLocal")), symbolic::zero(), symbolic::add(j, symbolic::one())
    );

    // A[i][j] *= 2
    auto& block3 = builder.add_block(loop2.root());
    auto& two = builder.add_constant(block3, "2.0", base_desc);
    auto& A1 = builder.add_access(block3, "A");
    auto& A2 = builder.add_access(block3, "A");
    auto& tasklet3 = builder.add_tasklet(block3, data_flow::TaskletCode::fp_mul, {"_out"}, {"_in1", "_in2"});
    builder.add_computational_memlet(block3, two, tasklet3, "_in1", {});
    builder.add_computational_memlet(block3, A1, tasklet3, "_in2", {i, j});
    builder.add_computational_memlet(block3, tasklet3, "_out", A2, {i, j});

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::FlopAnalysis>();

    // Check
    EXPECT_FALSE(analysis.precise());
    symbolic::Expression flop;
    ASSERT_TRUE(analysis.contains(&block3));
    flop = analysis.get(&block3);
    ASSERT_FALSE(flop.is_null());
    EXPECT_TRUE(symbolic::eq(flop, symbolic::one()));
    ASSERT_TRUE(analysis.contains(&loop2.root()));
    flop = analysis.get(&loop2.root());
    ASSERT_FALSE(flop.is_null());
    EXPECT_TRUE(symbolic::eq(flop, symbolic::one()));
    ASSERT_TRUE(analysis.contains(&loop2));
    flop = analysis.get(&loop2);
    ASSERT_FALSE(flop.is_null());
    EXPECT_TRUE(symbolic::eq(flop, symbolic::symbol("mLocal")));
    ASSERT_TRUE(analysis.contains(&block2));
    flop = analysis.get(&block2);
    ASSERT_FALSE(flop.is_null());
    EXPECT_TRUE(symbolic::eq(flop, symbolic::zero()));
    ASSERT_TRUE(analysis.contains(&loop1.root()));
    flop = analysis.get(&loop1.root());
    ASSERT_FALSE(flop.is_null());
    EXPECT_TRUE(symbolic::eq(flop, symbolic::dynamic_sizeof(symbolic::symbol("A"))));
    ASSERT_TRUE(analysis.contains(&loop1));
    flop = analysis.get(&loop1);
    ASSERT_FALSE(flop.is_null());
    EXPECT_TRUE(symbolic::
                    eq(flop, symbolic::mul(symbolic::symbol("nLocal"), symbolic::dynamic_sizeof(symbolic::symbol("A"))))
    );
    ASSERT_TRUE(analysis.contains(&block1));
    flop = analysis.get(&block1);
    ASSERT_FALSE(flop.is_null());
    EXPECT_TRUE(symbolic::eq(flop, symbolic::zero()));
    ASSERT_TRUE(analysis.contains(&root));
    flop = analysis.get(&root);
    ASSERT_FALSE(flop.is_null());
    EXPECT_TRUE(symbolic::eq(flop, symbolic::mul(symbolic::integer(42), symbolic::dynamic_sizeof(symbolic::symbol("A"))))
    );
}
