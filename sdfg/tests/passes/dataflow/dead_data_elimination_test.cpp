#include "sdfg/passes/dataflow/dead_data_elimination.h"

#include <gtest/gtest.h>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_nodes/call_node.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/element.h"
#include "sdfg/function.h"
#include "sdfg/types/function.h"
#include "sdfg/types/scalar.h"
#include "sdfg/types/type.h"

using namespace sdfg;

TEST(DeadDataEliminationTest, Unused) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::UInt32);
    builder.add_container("i", desc);
    auto sdfg = builder.move();

    // Apply pass
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::DeadDataElimination pass;
    bool applied = true;
    do {
        applied = pass.run(builder_opt, analysis_manager);
    } while (applied);
    sdfg = builder_opt.move();

    // Check result
    EXPECT_EQ(sdfg->root().size(), 0);
    EXPECT_EQ(sdfg->containers().size(), 0);
}

TEST(DeadDataEliminationTest, WriteWithoutRead_Transition) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::UInt32);
    builder.add_container("i", desc);
    auto sym1 = symbolic::symbol("i");

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root, {{sym1, symbolic::integer(0)}});

    auto sdfg = builder.move();

    // Apply pass
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::DeadDataElimination pass;
    bool applied = true;
    do {
        applied = pass.run(builder_opt, analysis_manager);
    } while (applied);
    sdfg = builder_opt.move();

    // Check result
    EXPECT_EQ(sdfg->root().size(), 1);
    auto child1 = sdfg->root().at(0);
    EXPECT_EQ(child1.second.assignments().size(), 0);

    EXPECT_EQ(sdfg->containers().size(), 0);
}

TEST(DeadDataEliminationTest, WriteWithoutRead_Dataflow) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::UInt32);
    builder.add_container("j", desc);

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);

    auto& output_node = builder.add_access(block, "j");
    auto& zero_node = builder.add_constant(block, "0", desc);
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, zero_node, tasklet, "_in", {});
    builder.add_computational_memlet(block, tasklet, "_out", output_node, {});

    auto sdfg = builder.move();

    // Apply pass
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::DeadDataElimination pass;
    bool applied = true;
    do {
        applied = pass.run(builder_opt, analysis_manager);
    } while (applied);
    sdfg = builder_opt.move();

    // Check result
    EXPECT_EQ(sdfg->root().size(), 1);
    auto& block1 = static_cast<const structured_control_flow::Block&>(sdfg->root().at(0).first);
    EXPECT_EQ(block1.dataflow().nodes().size(), 0);
    EXPECT_EQ(block1.dataflow().edges().size(), 0);

    EXPECT_EQ(sdfg->containers().size(), 0);
}

TEST(DeadDataEliminationTest, WriteAfterWrite_For) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::UInt32);
    builder.add_container("i", desc);
    auto sym = symbolic::symbol("i");

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root, {{sym, symbolic::integer(0)}});
    auto& loop = builder.add_for(
        root,
        sym,
        symbolic::Lt(sym, symbolic::integer(10)),
        symbolic::integer(0),
        symbolic::add(sym, symbolic::integer(1))
    );

    auto sdfg = builder.move();

    // Apply pass
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::DeadDataElimination pass;
    bool applied = true;
    do {
        applied = pass.run(builder_opt, analysis_manager);
    } while (applied);
    sdfg = builder_opt.move();

    // Check result
    EXPECT_EQ(sdfg->root().size(), 2);
    auto child0 = sdfg->root().at(0);
    EXPECT_EQ(child0.second.assignments().size(), 0);

    EXPECT_EQ(sdfg->containers().size(), 1);
}

TEST(DeadDataEliminationTest, WriteAfterWrite_WhileBody) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::UInt32);
    builder.add_container("i", desc);
    builder.add_container("N", desc, true);
    auto sym = symbolic::symbol("i");
    auto symN = symbolic::symbol("N");

    auto& root = builder.subject().root();
    auto& before = builder.add_block(root, {{sym, symbolic::integer(0)}});

    auto& loop = builder.add_while(root);
    auto& block1 = builder.add_block(loop.root(), {{sym, symbolic::integer(0)}});
    auto& block2 = builder.add_block(loop.root(), {{symN, sym}});

    auto sdfg = builder.move();

    // Apply pass
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::DeadDataElimination pass;
    bool applied = true;
    do {
        applied = pass.run(builder_opt, analysis_manager);
    } while (applied);
    sdfg = builder_opt.move();

    // Check result
    auto child0 = sdfg->root().at(0);
    EXPECT_EQ(child0.second.assignments().size(), 0);

    auto& child1 = static_cast<structured_control_flow::While&>(sdfg->root().at(1).first);
    auto& body = child1.root();
    EXPECT_EQ(body.at(0).second.assignments().size(), 1);
    EXPECT_EQ(body.at(1).second.assignments().size(), 1);

    EXPECT_EQ(sdfg->containers().size(), 2);
}

TEST(DeadDataEliminationTest, WriteAfterWrite_ClosedBranches) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::UInt32);
    builder.add_container("i", desc);
    builder.add_container("N", desc, true);
    auto sym = symbolic::symbol("i");
    auto symN = symbolic::symbol("N");

    auto& root = builder.subject().root();

    auto& before = builder.add_block(root, {{sym, symbolic::integer(0)}});

    auto& if_else = builder.add_if_else(root);
    auto& case1 = builder.add_case(if_else, symbolic::__true__());
    auto& case2 = builder.add_case(if_else, symbolic::__false__());
    auto& block1 = builder.add_block(case1, {{sym, symbolic::integer(0)}});
    auto& block2 = builder.add_block(case2, {{sym, symbolic::integer(1)}});

    auto& after = builder.add_block(root, {{symN, sym}});

    auto sdfg = builder.move();

    // Apply pass
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::DeadDataElimination pass;
    bool applied = true;
    do {
        applied = pass.run(builder_opt, analysis_manager);
    } while (applied);
    sdfg = builder_opt.move();

    // Check result
    auto child0 = sdfg->root().at(0);
    EXPECT_EQ(child0.second.assignments().size(), 0);

    auto child2 = sdfg->root().at(2);
    EXPECT_EQ(child2.second.assignments().size(), 1);

    EXPECT_EQ(sdfg->containers().size(), 2);
}

TEST(DeadDataEliminationTest, WriteAfterWrite_OpenBranches) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::UInt32);
    builder.add_container("i", desc);
    builder.add_container("N", desc, true);
    auto sym = symbolic::symbol("i");
    auto symN = symbolic::symbol("N");

    auto& root = builder.subject().root();

    auto& before = builder.add_block(root, {{sym, symbolic::integer(0)}});

    auto& if_else = builder.add_if_else(root);
    auto& case1 = builder.add_case(if_else, symbolic::__true__());
    auto& case2 = builder.add_case(if_else, symbolic::__false__());
    auto& block1 = builder.add_block(case1, {{sym, symbolic::integer(0)}});
    auto& block2 = builder.add_block(case2, control_flow::Assignments{});

    auto& after = builder.add_block(root, {{symN, sym}});

    auto sdfg = builder.move();

    // Apply pass
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::DeadDataElimination pass;
    bool applied = true;
    do {
        applied = pass.run(builder_opt, analysis_manager);
    } while (applied);
    sdfg = builder_opt.move();

    // Check result
    auto child0 = sdfg->root().at(0);
    EXPECT_EQ(child0.second.assignments().size(), 1);

    auto child2 = sdfg->root().at(2);
    EXPECT_EQ(child2.second.assignments().size(), 1);

    EXPECT_EQ(sdfg->containers().size(), 2);
}

TEST(DeadDataEliminationTest, WriteAfterWrite_IncompleteBranches) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::UInt32);
    builder.add_container("i", desc);
    builder.add_container("N", desc, true);
    auto sym = symbolic::symbol("i");
    auto symN = symbolic::symbol("N");

    auto& root = builder.subject().root();

    auto& before = builder.add_block(root, {{sym, symbolic::integer(0)}});

    auto& if_else = builder.add_if_else(root);
    auto& case1 = builder.add_case(if_else, symbolic::Lt(symN, symbolic::integer(10)));
    auto& block1 = builder.add_block(case1, {{sym, symbolic::integer(0)}});

    auto& after = builder.add_block(root, {{symN, sym}});

    auto sdfg = builder.move();

    // Apply pass
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::DeadDataElimination pass;
    bool applied = true;
    do {
        applied = pass.run(builder_opt, analysis_manager);
    } while (applied);
    sdfg = builder_opt.move();

    // Check result
    auto child0 = sdfg->root().at(0);
    EXPECT_EQ(child0.second.assignments().size(), 1);

    auto child2 = sdfg->root().at(2);
    EXPECT_EQ(child2.second.assignments().size(), 1);

    EXPECT_EQ(sdfg->containers().size(), 2);
}

TEST(DeadDataEliminationTest, WriteAfterWrite_ContinueBreak_NoReads) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::UInt32);
    builder.add_container("i", desc);
    builder.add_container("N", desc, true);
    auto sym = symbolic::symbol("i");
    auto symN = symbolic::symbol("N");

    auto& root = builder.subject().root();
    auto& loop = builder.add_while(root);
    auto& body = loop.root();

    auto& before = builder.add_block(body, {{sym, symbolic::integer(0)}});

    auto& if_else = builder.add_if_else(body);
    auto& case1 = builder.add_case(if_else, symbolic::Lt(sym, symbolic::integer(10)));
    auto& block1 = builder.add_block(case1, {{sym, symbolic::integer(10)}});
    auto& cont1 = builder.add_continue(case1);
    auto& case2 = builder.add_case(if_else, symbolic::Ge(sym, symbolic::integer(10)));
    auto& block2 = builder.add_block(case2, {{sym, symbolic::integer(0)}});
    auto& break1 = builder.add_break(case2);

    auto sdfg = builder.move();

    // Apply pass
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::DeadDataElimination pass;
    bool applied = true;
    do {
        applied = pass.run(builder_opt, analysis_manager);
    } while (applied);
    sdfg = builder_opt.move();

    // Check result
    auto& new_loop = static_cast<structured_control_flow::While&>(sdfg->root().at(0).first);
    auto& new_body = new_loop.root();

    auto child1 = new_body.at(0);
    EXPECT_EQ(child1.second.assignments().size(), 1);

    auto& child2 = static_cast<structured_control_flow::IfElse&>(new_body.at(1).first);
    auto case1_1 = child2.at(0).first.at(0);
    EXPECT_EQ(case1_1.second.assignments().size(), 0);

    auto case2_1 = child2.at(1).first.at(0);
    EXPECT_EQ(case2_1.second.assignments().size(), 0);

    EXPECT_EQ(sdfg->containers().size(), 2);
}

TEST(DeadDataEliminationTest, WriteAfterWrite_ContinueBreak_Read) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::UInt32);
    builder.add_container("i", desc);
    builder.add_container("N", desc, true);
    auto sym = symbolic::symbol("i");
    auto symN = symbolic::symbol("N");

    auto& root = builder.subject().root();
    auto& loop = builder.add_while(root);
    auto& body = loop.root();

    auto& before = builder.add_block(body, {{sym, symbolic::integer(0)}});

    auto& if_else = builder.add_if_else(body);
    auto& case1 = builder.add_case(if_else, symbolic::Lt(sym, symbolic::integer(10)));
    auto& block1 = builder.add_block(case1, {{sym, symbolic::integer(10)}});
    auto& cont1 = builder.add_continue(case1);
    auto& case2 = builder.add_case(if_else, symbolic::Ge(sym, symbolic::integer(10)));
    auto& block2 = builder.add_block(case2, {{sym, symbolic::integer(0)}});
    auto& break1 = builder.add_break(case2);

    auto& after2 = builder.add_block(root, {{symN, sym}});

    auto sdfg = builder.move();

    // Apply pass
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::DeadDataElimination pass;
    bool applied = true;
    do {
        applied = pass.run(builder_opt, analysis_manager);
    } while (applied);
    sdfg = builder_opt.move();

    // Check result
    auto& new_loop = static_cast<structured_control_flow::While&>(sdfg->root().at(0).first);
    auto& new_body = new_loop.root();

    auto child1 = new_body.at(0);
    EXPECT_EQ(child1.second.assignments().size(), 1);

    auto& child2 = static_cast<structured_control_flow::IfElse&>(new_body.at(1).first);

    // Over-approximation. Can be zero when analysis becomes more precise.
    auto case1_1 = child2.at(0).first.at(0);
    EXPECT_EQ(case1_1.second.assignments().size(), 1);

    auto case2_1 = child2.at(1).first.at(0);
    EXPECT_EQ(case2_1.second.assignments().size(), 1);

    auto child3 = sdfg->root().at(1);
    EXPECT_EQ(child3.second.assignments().size(), 1);

    EXPECT_EQ(sdfg->containers().size(), 2);
}

TEST(DeadDataEliminationTest, WriteAfterWrite_ContinueBreak_OpenRead) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::UInt32);
    builder.add_container("i", desc);
    builder.add_container("N", desc, true);
    auto sym = symbolic::symbol("i");
    auto symN = symbolic::symbol("N");

    auto& root = builder.subject().root();
    auto& loop = builder.add_while(root);
    auto& body = loop.root();

    auto& if_else = builder.add_if_else(body);
    auto& case1 = builder.add_case(if_else, symbolic::Lt(sym, symbolic::integer(10)));
    auto& block1 = builder.add_block(case1, {{sym, symbolic::integer(10)}});
    auto& cont1 = builder.add_continue(case1);
    auto& case2 = builder.add_case(if_else, symbolic::Ge(sym, symbolic::integer(10)));
    auto& block2 = builder.add_block(case2, {{sym, symbolic::integer(0)}});
    auto& break1 = builder.add_break(case2);

    auto sdfg = builder.move();

    // Apply pass
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::DeadDataElimination pass;
    bool applied = true;
    do {
        applied = pass.run(builder_opt, analysis_manager);
    } while (applied);
    sdfg = builder_opt.move();

    // Check result
    auto& new_loop = static_cast<structured_control_flow::While&>(sdfg->root().at(0).first);
    auto& new_body = new_loop.root();

    auto& child1 = static_cast<structured_control_flow::IfElse&>(new_body.at(0).first);

    auto case1_1 = child1.at(0).first.at(0);
    EXPECT_EQ(case1_1.second.assignments().size(), 1);

    // Over-approximation. Can be zero when analysis becomes more precise.
    auto case2_1 = child1.at(1).first.at(0);
    EXPECT_EQ(case2_1.second.assignments().size(), 1);

    EXPECT_EQ(sdfg->containers().size(), 2);
}

TEST(DeadDataEliminationTest, DanglingRead) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar desc(types::PrimitiveType::Int32);
    builder.add_container("a", desc, true);
    builder.add_container("b", desc);
    builder.add_container("c", desc);

    types::Scalar void_type(types::PrimitiveType::Void);
    types::Function my_custom_exit_type(void_type);
    builder.add_container("my_custom_exit", my_custom_exit_type, false, true);

    auto a = symbolic::symbol("a");

    auto& if_else_1 = builder.add_if_else(root);
    auto& if_else_1_case_1 = builder.add_case(if_else_1, symbolic::Lt(a, symbolic::zero()));
    auto& if_else_1_case_2 = builder.add_case(if_else_1, symbolic::Not(symbolic::Lt(a, symbolic::zero())));

    auto& block1 = builder.add_block(if_else_1_case_1);
    std::vector<std::string> empty;
    auto& libnode = builder.add_library_node<data_flow::CallNode>(block1, DebugInfo(), "my_custom_exit", empty, empty);

    auto& if_else_2 = builder.add_if_else(if_else_1_case_1);
    auto& if_else_2_case_1 = builder.add_case(if_else_2, symbolic::Lt(symbolic::symbol("b"), symbolic::integer(10)));
    auto& if_else_2_case_2 =
        builder.add_case(if_else_2, symbolic::Not(symbolic::Lt(symbolic::symbol("b"), symbolic::integer(10))));

    auto& block2 = builder.add_block(if_else_2_case_1);
    {
        auto& a = builder.add_access(block2, "a");
        auto& c = builder.add_access(block2, "c");
        auto& tasklet = builder.add_tasklet(block2, data_flow::TaskletCode::assign, "_out", {"_in"});
        builder.add_computational_memlet(block2, a, tasklet, "_in", {});
        builder.add_computational_memlet(block2, tasklet, "_out", c, {});
    }

    auto& block3 = builder.add_block(if_else_2_case_2);
    {
        auto& a = builder.add_access(block3, "a");
        auto& c = builder.add_access(block3, "c");
        auto& ten = builder.add_constant(block3, "10", desc);
        auto& tasklet = builder.add_tasklet(block3, data_flow::TaskletCode::int_sub, "_out", {"_in1", "_in2"});
        builder.add_computational_memlet(block3, ten, tasklet, "_in1", {});
        builder.add_computational_memlet(block3, a, tasklet, "_in2", {});
        builder.add_computational_memlet(block3, tasklet, "_out", c, {});
    }

    auto& block4 = builder.add_block(if_else_1_case_2, {{symbolic::symbol("b"), a}});

    auto& if_else_3 = builder.add_if_else(if_else_1_case_2);
    auto& if_else_3_case_1 = builder.add_case(if_else_3, symbolic::Lt(a, symbolic::integer(10)));
    auto& if_else_3_case_2 = builder.add_case(if_else_3, symbolic::Not(symbolic::Lt(a, symbolic::integer(10))));

    auto& block5 = builder.add_block(if_else_3_case_1);
    {
        auto& a = builder.add_access(block5, "a");
        auto& c = builder.add_access(block5, "c");
        auto& tasklet = builder.add_tasklet(block5, data_flow::TaskletCode::assign, "_out", {"_in"});
        builder.add_computational_memlet(block5, a, tasklet, "_in", {});
        builder.add_computational_memlet(block5, tasklet, "_out", c, {});
    }

    auto& block6 = builder.add_block(if_else_3_case_2);
    {
        auto& a = builder.add_access(block6, "a");
        auto& c = builder.add_access(block6, "c");
        auto& ten = builder.add_constant(block6, "10", desc);
        auto& tasklet = builder.add_tasklet(block6, data_flow::TaskletCode::int_sub, "_out", {"_in1", "_in2"});
        builder.add_computational_memlet(block6, ten, tasklet, "_in1", {});
        builder.add_computational_memlet(block6, a, tasklet, "_in2", {});
        builder.add_computational_memlet(block6, tasklet, "_out", c, {});
    }

    builder.add_return(root, "c");

    analysis::AnalysisManager analysis_manager(sdfg);
    passes::DeadDataElimination pass;
    bool applied = true;
    do {
        applied = pass.run(builder, analysis_manager);
    } while (applied);

    // Check that assignment was eliminated
    EXPECT_EQ(if_else_1_case_2.size(), 2);
    EXPECT_TRUE(if_else_1_case_2.at(0).second.empty());

    // Check that container is still there for dangling read
    EXPECT_TRUE(sdfg.exists("b"));
}
