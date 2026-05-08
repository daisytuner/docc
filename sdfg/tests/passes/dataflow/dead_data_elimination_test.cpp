#include "sdfg/passes/dataflow/dead_data_elimination.h"

#include <gtest/gtest.h>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_nodes/call_node.h"
#include "sdfg/data_flow/library_nodes/invoke_node.h"
#include "sdfg/data_flow/library_nodes/stdlib/free.h"
#include "sdfg/data_flow/library_nodes/stdlib/malloc.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/element.h"
#include "sdfg/function.h"
#include "sdfg/passes/structured_control_flow/dead_cfg_elimination.h"
#include "sdfg/types/function.h"
#include "sdfg/types/scalar.h"
#include "sdfg/types/type.h"
#include "sdfg/visualizer/dot_visualizer.h"
#include "sdfg_debug_dump.h"

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

TEST(DeadDataEliminationTest, Does_not_remove_some_libNode_outputs) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::Int64);
    types::Pointer desc_ptr(desc);
    builder.add_container("a", desc, true);
    builder.add_container("b", desc, true);
    builder.add_container("result1", desc);
    builder.add_container("result2", desc);


    auto& root = builder.subject().root();
    EXPECT_EQ(root.element_id(), 0);

    auto& block = builder.add_block(root);

    auto& in_a_node = builder.add_access(block, "a");
    auto& in_b_node = builder.add_access(block, "b");
    auto& res1_node = builder.add_access(block, "result1");
    auto& res2_node = builder.add_access(block, "result2");
    auto function_type = types::Function(types::Scalar(types::Void));
    function_type.add_param(desc);
    function_type.add_param(desc);

    builder.add_external("blackbox_function", function_type, LinkageType_External);
    auto& black_fun = builder.add_library_node<data_flow::CallNode>(
        block,
        DebugInfo(),
        "blackbox_function",
        std::vector<std::string>{"res1", "res2"},
        std::vector<std::string>{"a", "b"}
    );

    builder.add_computational_memlet(block, in_a_node, black_fun, "a", {}, desc);
    builder.add_computational_memlet(block, in_b_node, black_fun, "b", {}, desc);
    builder.add_computational_memlet(block, black_fun, "res1", res1_node, {}, desc);
    builder.add_computational_memlet(block, black_fun, "res2", res2_node, {}, desc);

    builder.add_return(root, "result2");

    dump_sdfg(builder.subject(), "0-before");

    builder.subject().validate();

    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::DeadDataElimination pass;
    bool applied = true;
    do {
        applied = pass.run(builder, analysis_manager);
    } while (applied);

    dump_sdfg(builder.subject(), "1-after");

    builder.subject().validate();

    EXPECT_EQ(block.dataflow().nodes().size(), 5);
    EXPECT_EQ(block.dataflow().edges().size(), 4);
    EXPECT_EQ(black_fun.outputs().size(), 2);

    auto& dflow = block.dataflow();
    EXPECT_EQ(dflow.out_degree(black_fun), 2);
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

TEST(DeadDataEliminationTest, OwnedHeapMemoryIsRemoved) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar t_int_desc(types::PrimitiveType::UInt32);
    types::Scalar t_fp_desc(types::PrimitiveType::Float);
    types::Pointer t_fp_ptr(t_fp_desc);

    builder.add_container("out", t_fp_ptr, true);
    builder.add_container("N", t_int_desc, true);
    builder.add_container("tmp", t_fp_ptr);

    auto& root = builder.subject().root();
    auto& malloc_block = builder.add_block(root);
    auto sym_n = symbolic::symbol("N");
    auto& malloc_node = builder.add_library_node<stdlib::MallocNode>(malloc_block, DebugInfo(), sym_n);
    auto& malloc_access = builder.add_access(malloc_block, "tmp");
    builder.add_computational_memlet(malloc_block, malloc_node, "_ret", malloc_access, {}, t_fp_ptr);


    builder.add_container("i", t_int_desc);
    auto sym_i = symbolic::symbol("i");

    auto& loop = builder.add_map(
        root,
        sym_i,
        symbolic::Lt(sym_i, sym_n),
        symbolic::integer(0),
        symbolic::add(sym_i, symbolic::integer(1)),
        ScheduleType_Sequential::create()
    );
    auto& body = loop.root();

    auto& body_block = builder.add_block(body);
    auto& some_fp_const = builder.add_constant(body_block, "1.0", t_fp_ptr);
    auto& tmp_write = builder.add_access(body_block, "tmp");
    auto& out_write = builder.add_access(body_block, "out");
    auto& i_input = builder.add_access(body_block, "i");
    auto& tasklet = builder.add_tasklet(body_block, data_flow::TaskletCode::fp_add, "out", {"a", "b"});
    builder.add_computational_memlet(body_block, some_fp_const, tasklet, "a", {}, t_fp_ptr);
    builder.add_computational_memlet(body_block, i_input, tasklet, "b", {}, t_fp_ptr);
    builder.add_computational_memlet(body_block, tasklet, "out", tmp_write, {sym_i}, t_fp_ptr);
    auto& tasklet2 = builder.add_tasklet(body_block, data_flow::TaskletCode::fp_add, "out", {"a", "b"});
    builder.add_computational_memlet(body_block, some_fp_const, tasklet2, "a", {}, t_fp_ptr);
    builder.add_computational_memlet(body_block, i_input, tasklet2, "b", {}, t_fp_ptr);
    builder.add_computational_memlet(body_block, tasklet2, "out", out_write, {sym_i}, t_fp_ptr);

    auto sdfg = builder.move();

    dump_sdfg(*sdfg, "0-before");

    // Apply pass
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::DeadDataElimination pass;
    bool applied = pass.run(builder_opt, analysis_manager);
    passes::DeadCFGElimination().run(builder_opt, analysis_manager); // to remove the empty block left over

    sdfg = builder_opt.move();

    dump_sdfg(*sdfg, "1-after");

    EXPECT_TRUE(applied);

    // Check result
    EXPECT_FALSE(sdfg->exists("tmp"));
    EXPECT_TRUE(sdfg->exists("out"));
    EXPECT_TRUE(sdfg->exists("i"));
    EXPECT_EQ(sdfg->root().size(), 1);
}

TEST(DeadDataEliminationTest, OwnedHeapMemoryWithFree) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar t_int_desc(types::PrimitiveType::UInt32);
    types::Scalar t_fp_desc(types::PrimitiveType::Float);
    types::Pointer t_fp_ptr(t_fp_desc);

    builder.add_container("out", t_fp_ptr, true);
    builder.add_container("N", t_int_desc, true);
    builder.add_container("tmp", t_fp_ptr);

    auto& root = builder.subject().root();
    auto& malloc_block = builder.add_block(root);
    auto sym_n = symbolic::symbol("N");
    auto& malloc_node = builder.add_library_node<stdlib::MallocNode>(malloc_block, DebugInfo(), sym_n);
    auto& malloc_access = builder.add_access(malloc_block, "tmp");
    builder.add_computational_memlet(malloc_block, malloc_node, "_ret", malloc_access, {}, t_fp_ptr);


    builder.add_container("i", t_int_desc);
    auto sym_i = symbolic::symbol("i");

    auto& loop = builder.add_map(
        root,
        sym_i,
        symbolic::Lt(sym_i, sym_n),
        symbolic::integer(0),
        symbolic::add(sym_i, symbolic::integer(1)),
        ScheduleType_Sequential::create()
    );
    auto& body = loop.root();

    auto& body_block = builder.add_block(body);
    auto& some_fp_const = builder.add_constant(body_block, "1.0", t_fp_ptr);
    auto& tmp_write = builder.add_access(body_block, "tmp");
    auto& out_write = builder.add_access(body_block, "out");
    auto& i_input = builder.add_access(body_block, "i");
    auto& tasklet = builder.add_tasklet(body_block, data_flow::TaskletCode::fp_add, "out", {"a", "b"});
    builder.add_computational_memlet(body_block, some_fp_const, tasklet, "a", {}, t_fp_ptr);
    builder.add_computational_memlet(body_block, i_input, tasklet, "b", {}, t_fp_ptr);
    builder.add_computational_memlet(body_block, tasklet, "out", tmp_write, {sym_i}, t_fp_ptr);
    auto& tasklet2 = builder.add_tasklet(body_block, data_flow::TaskletCode::fp_add, "out", {"a", "b"});
    builder.add_computational_memlet(body_block, some_fp_const, tasklet2, "a", {}, t_fp_ptr);
    builder.add_computational_memlet(body_block, i_input, tasklet2, "b", {}, t_fp_ptr);
    builder.add_computational_memlet(body_block, tasklet2, "out", out_write, {sym_i}, t_fp_ptr);

    auto& free_block = builder.add_block(root);
    auto& tmp_read = builder.add_access(free_block, "tmp");
    auto& free_node = builder.add_library_node<stdlib::FreeNode>(free_block, DebugInfo());
    builder.add_computational_memlet(free_block, tmp_read, free_node, "_ptr", {}, t_fp_ptr);

    auto sdfg = builder.move();

    dump_sdfg(*sdfg, "0-before");

    // Apply pass
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::DeadDataElimination pass;
    bool applied = pass.run(builder_opt, analysis_manager);
    passes::DeadCFGElimination().run(builder_opt, analysis_manager); // to remove the empty block left over

    sdfg = builder_opt.move();

    dump_sdfg(*sdfg, "1-after");

    EXPECT_TRUE(applied);

    // Check result
    EXPECT_FALSE(sdfg->exists("tmp"));
    EXPECT_TRUE(sdfg->exists("out"));
    EXPECT_TRUE(sdfg->exists("i"));
    EXPECT_EQ(sdfg->root().size(), 1);
}

TEST(DeadDataEliminationTest, MultiWrittenOwnedHeapMemoryIsRemoved) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar t_int_desc(types::PrimitiveType::UInt32);
    types::Scalar t_fp_desc(types::PrimitiveType::Float);
    types::Pointer t_fp_ptr(t_fp_desc);

    builder.add_container("out", t_fp_ptr, true);
    builder.add_container("N", t_int_desc, true);
    builder.add_container("tmp", t_fp_ptr);

    auto& root = builder.subject().root();
    auto& malloc_block = builder.add_block(root);
    auto sym_n = symbolic::symbol("N");
    auto& malloc_node = builder.add_library_node<stdlib::MallocNode>(malloc_block, DebugInfo(), sym_n);
    auto& malloc_access = builder.add_access(malloc_block, "tmp");
    builder.add_computational_memlet(malloc_block, malloc_node, "_ret", malloc_access, {}, t_fp_ptr);


    builder.add_container("i", t_int_desc);
    auto sym_i = symbolic::symbol("i");

    auto& loop = builder.add_map(
        root,
        sym_i,
        symbolic::Lt(sym_i, sym_n),
        symbolic::integer(0),
        symbolic::add(sym_i, symbolic::integer(1)),
        ScheduleType_Sequential::create()
    );
    auto& body = loop.root();

    auto& body_block = builder.add_block(body);
    auto& some_fp_const = builder.add_constant(body_block, "1.0", t_fp_ptr);
    auto& tmp_write = builder.add_access(body_block, "tmp");
    auto& out_write = builder.add_access(body_block, "out");
    auto& i_input = builder.add_access(body_block, "i");
    auto& tasklet = builder.add_tasklet(body_block, data_flow::TaskletCode::fp_add, "out", {"a", "b"});
    builder.add_computational_memlet(body_block, some_fp_const, tasklet, "a", {}, t_fp_ptr);
    builder.add_computational_memlet(body_block, i_input, tasklet, "b", {}, t_fp_ptr);
    builder.add_computational_memlet(body_block, tasklet, "out", tmp_write, {sym_i}, t_fp_ptr);
    auto& tasklet2 = builder.add_tasklet(body_block, data_flow::TaskletCode::fp_add, "out", {"a", "b"});
    builder.add_computational_memlet(body_block, some_fp_const, tasklet2, "a", {}, t_fp_ptr);
    builder.add_computational_memlet(body_block, i_input, tasklet2, "b", {}, t_fp_ptr);
    builder.add_computational_memlet(body_block, tasklet2, "out", out_write, {sym_i}, t_fp_ptr);

    auto sdfg = builder.move();

    dump_sdfg(*sdfg, "0-before");

    // Apply pass
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::DeadDataElimination pass;
    bool applied = pass.run(builder_opt, analysis_manager);
    passes::DeadCFGElimination().run(builder_opt, analysis_manager); // to remove the empty block left over

    sdfg = builder_opt.move();

    dump_sdfg(*sdfg, "1-after");

    // Check result
    EXPECT_FALSE(sdfg->exists("tmp"));
    EXPECT_TRUE(sdfg->exists("out"));
    EXPECT_TRUE(sdfg->exists("i"));
    EXPECT_EQ(sdfg->root().size(), 1);
}

TEST(DeadDataEliminationTest, OwnedHeapMemoryPtrEscapedViaReturn) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar t_int_desc(types::PrimitiveType::UInt32);
    types::Scalar t_fp_desc(types::PrimitiveType::Float);
    types::Pointer t_fp_ptr(t_fp_desc);

    builder.add_container("out", t_fp_ptr, true);
    builder.add_container("N", t_int_desc, true);
    builder.add_container("tmp", t_fp_ptr);

    auto& root = builder.subject().root();
    auto& malloc_block = builder.add_block(root);
    auto sym_n = symbolic::symbol("N");
    auto& malloc_node = builder.add_library_node<stdlib::MallocNode>(malloc_block, DebugInfo(), sym_n);
    auto& malloc_access = builder.add_access(malloc_block, "tmp");
    builder.add_computational_memlet(malloc_block, malloc_node, "_ret", malloc_access, {}, t_fp_ptr);


    builder.add_container("i", t_int_desc);
    auto sym_i = symbolic::symbol("i");

    auto& loop = builder.add_map(
        root,
        sym_i,
        symbolic::Lt(sym_i, sym_n),
        symbolic::integer(0),
        symbolic::add(sym_i, symbolic::integer(1)),
        ScheduleType_Sequential::create()
    );
    auto& body = loop.root();

    auto& body_block = builder.add_block(body);
    auto& some_fp_const = builder.add_constant(body_block, "1.0", t_fp_ptr);
    auto& tmp_write = builder.add_access(body_block, "tmp");
    auto& out_write = builder.add_access(body_block, "out");
    auto& i_input = builder.add_access(body_block, "i");
    auto& tasklet = builder.add_tasklet(body_block, data_flow::TaskletCode::fp_add, "out", {"a", "b"});
    builder.add_computational_memlet(body_block, some_fp_const, tasklet, "a", {}, t_fp_ptr);
    builder.add_computational_memlet(body_block, i_input, tasklet, "b", {}, t_fp_ptr);
    builder.add_computational_memlet(body_block, tasklet, "out", tmp_write, {sym_i}, t_fp_ptr);
    auto& tasklet2 = builder.add_tasklet(body_block, data_flow::TaskletCode::fp_add, "out", {"a", "b"});
    builder.add_computational_memlet(body_block, some_fp_const, tasklet2, "a", {}, t_fp_ptr);
    builder.add_computational_memlet(body_block, i_input, tasklet2, "b", {}, t_fp_ptr);
    builder.add_computational_memlet(body_block, tasklet2, "out", out_write, {sym_i}, t_fp_ptr);

    //
    builder.add_return(root, "tmp");

    auto sdfg = builder.move();

    dump_sdfg(*sdfg, "0-before");

    // Apply pass
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::DeadDataElimination pass;
    bool applied = pass.run(builder_opt, analysis_manager);
    passes::DeadCFGElimination().run(builder_opt, analysis_manager); // to remove the empty block left over

    sdfg = builder_opt.move();

    dump_sdfg(*sdfg, "1-after");

    // Check result
    EXPECT_TRUE(sdfg->exists("tmp"));
    EXPECT_TRUE(sdfg->exists("out"));
    EXPECT_TRUE(sdfg->exists("i"));
    EXPECT_EQ(sdfg->root().size(), 3);
}

TEST(DeadDataEliminationTest, OwnedHeapMemorySubsetAddrEscapedViaReturn) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar t_int_desc(types::PrimitiveType::UInt32);
    types::Scalar t_fp_desc(types::PrimitiveType::Float);
    types::Pointer t_fp_ptr(t_fp_desc);

    builder.add_container("out", t_fp_ptr, true);
    builder.add_container("N", t_int_desc, true);
    builder.add_container("tmp", t_fp_ptr);

    auto& root = builder.subject().root();
    auto& malloc_block = builder.add_block(root);
    auto sym_n = symbolic::symbol("N");
    auto& malloc_node = builder.add_library_node<stdlib::MallocNode>(malloc_block, DebugInfo(), sym_n);
    auto& malloc_access = builder.add_access(malloc_block, "tmp");
    builder.add_computational_memlet(malloc_block, malloc_node, "_ret", malloc_access, {}, t_fp_ptr);


    builder.add_container("i", t_int_desc);
    auto sym_i = symbolic::symbol("i");

    auto& loop = builder.add_map(
        root,
        sym_i,
        symbolic::Lt(sym_i, sym_n),
        symbolic::integer(0),
        symbolic::add(sym_i, symbolic::integer(1)),
        ScheduleType_Sequential::create()
    );
    auto& body = loop.root();

    auto& body_block = builder.add_block(body);
    auto& some_fp_const = builder.add_constant(body_block, "1.0", t_fp_ptr);
    auto& tmp_write = builder.add_access(body_block, "tmp");
    auto& out_write = builder.add_access(body_block, "out");
    auto& i_input = builder.add_access(body_block, "i");
    auto& tasklet = builder.add_tasklet(body_block, data_flow::TaskletCode::fp_add, "out", {"a", "b"});
    builder.add_computational_memlet(body_block, some_fp_const, tasklet, "a", {}, t_fp_ptr);
    builder.add_computational_memlet(body_block, i_input, tasklet, "b", {}, t_fp_ptr);
    builder.add_computational_memlet(body_block, tasklet, "out", tmp_write, {sym_i}, t_fp_ptr);
    auto& tasklet2 = builder.add_tasklet(body_block, data_flow::TaskletCode::fp_add, "out", {"a", "b"});
    builder.add_computational_memlet(body_block, some_fp_const, tasklet2, "a", {}, t_fp_ptr);
    builder.add_computational_memlet(body_block, i_input, tasklet2, "b", {}, t_fp_ptr);
    builder.add_computational_memlet(body_block, tasklet2, "out", out_write, {sym_i}, t_fp_ptr);

    //
    auto& read_block = builder.add_block(root);
    builder.add_container("ptr_ret", t_fp_ptr);
    auto& tmp_read = builder.add_access(read_block, "tmp");
    auto& save_write = builder.add_access(read_block, "ptr_ret");
    builder.add_reference_memlet(read_block, tmp_read, save_write, {symbolic::integer(5)}, t_fp_ptr);

    builder.add_return(root, "ptr_ret");

    auto sdfg = builder.move();

    dump_sdfg(*sdfg, "0-before");

    // Apply pass
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::DeadDataElimination pass;
    bool applied = pass.run(builder_opt, analysis_manager);
    passes::DeadCFGElimination().run(builder_opt, analysis_manager); // to remove the empty block left over

    sdfg = builder_opt.move();

    dump_sdfg(*sdfg, "1-after");

    // Check result
    EXPECT_TRUE(sdfg->exists("tmp"));
    EXPECT_TRUE(sdfg->exists("out"));
    EXPECT_TRUE(sdfg->exists("i"));
    EXPECT_TRUE(sdfg->exists("ptr_ret"));
    EXPECT_EQ(sdfg->root().size(), 4);
}

TEST(DeadDataEliminationTest, OwnedHeapMemoryEscapesViaPtrWrite) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar t_int_desc(types::PrimitiveType::UInt32);
    types::Scalar t_fp_desc(types::PrimitiveType::Float);
    types::Pointer t_fp_ptr(t_fp_desc);
    types::Pointer t_fp_ptr_ptr(reinterpret_cast<types::IType&>(t_fp_ptr));

    builder.add_container("out", t_fp_ptr, true);
    builder.add_container("N", t_int_desc, true);
    builder.add_container("escape_hatch", t_fp_ptr_ptr, true);
    builder.add_container("tmp", t_fp_ptr);

    auto& root = builder.subject().root();
    auto& malloc_block = builder.add_block(root);
    auto sym_n = symbolic::symbol("N");
    auto& malloc_node = builder.add_library_node<stdlib::MallocNode>(malloc_block, DebugInfo(), sym_n);
    auto& malloc_access = builder.add_access(malloc_block, "tmp");
    builder.add_computational_memlet(malloc_block, malloc_node, "_ret", malloc_access, {}, t_fp_ptr);


    builder.add_container("i", t_int_desc);
    auto sym_i = symbolic::symbol("i");

    auto& loop = builder.add_map(
        root,
        sym_i,
        symbolic::Lt(sym_i, sym_n),
        symbolic::integer(0),
        symbolic::add(sym_i, symbolic::integer(1)),
        ScheduleType_Sequential::create()
    );
    auto& body = loop.root();

    auto& body_block = builder.add_block(body);
    auto& some_fp_const = builder.add_constant(body_block, "1.0", t_fp_ptr);
    auto& tmp_write = builder.add_access(body_block, "tmp");
    auto& out_write = builder.add_access(body_block, "out");
    auto& i_input = builder.add_access(body_block, "i");
    auto& tasklet = builder.add_tasklet(body_block, data_flow::TaskletCode::fp_add, "out", {"a", "b"});
    builder.add_computational_memlet(body_block, some_fp_const, tasklet, "a", {}, t_fp_ptr);
    builder.add_computational_memlet(body_block, i_input, tasklet, "b", {}, t_fp_ptr);
    builder.add_computational_memlet(body_block, tasklet, "out", tmp_write, {sym_i}, t_fp_ptr);
    auto& tasklet2 = builder.add_tasklet(body_block, data_flow::TaskletCode::fp_add, "out", {"a", "b"});
    builder.add_computational_memlet(body_block, some_fp_const, tasklet2, "a", {}, t_fp_ptr);
    builder.add_computational_memlet(body_block, i_input, tasklet2, "b", {}, t_fp_ptr);
    builder.add_computational_memlet(body_block, tasklet2, "out", out_write, {sym_i}, t_fp_ptr);

    //
    auto& read_block = builder.add_block(root);
    auto& escape_target = builder.add_access(read_block, "escape_hatch");
    auto& tmp_read = builder.add_access(read_block, "tmp");
    builder.add_dereference_memlet(read_block, tmp_read, escape_target, false, t_fp_ptr_ptr);

    auto sdfg = builder.move();

    dump_sdfg(*sdfg, "0-before");

    // Apply pass
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::DeadDataElimination pass;
    bool applied = pass.run(builder_opt, analysis_manager);
    passes::DeadCFGElimination().run(builder_opt, analysis_manager); // to remove the empty block left over

    sdfg = builder_opt.move();

    dump_sdfg(*sdfg, "1-after");

    // Check result
    EXPECT_TRUE(sdfg->exists("tmp"));
    EXPECT_TRUE(sdfg->exists("out"));
    EXPECT_TRUE(sdfg->exists("i"));
    EXPECT_EQ(sdfg->root().size(), 3);
}

TEST(DeadDataEliminationTest, OwnedHeapMemoryEscapesViaBlackbox) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar t_int_desc(types::PrimitiveType::UInt32);
    types::Scalar t_fp_desc(types::PrimitiveType::Float);
    types::Pointer t_fp_ptr(t_fp_desc);
    types::Pointer t_fp_ptr_ptr(reinterpret_cast<types::IType&>(t_fp_ptr));

    builder.add_container("out", t_fp_ptr, true);
    builder.add_container("N", t_int_desc, true);
    builder.add_container("tmp", t_fp_ptr);

    auto& root = builder.subject().root();
    auto& malloc_block = builder.add_block(root);
    auto sym_n = symbolic::symbol("N");
    auto& malloc_node = builder.add_library_node<stdlib::MallocNode>(malloc_block, DebugInfo(), sym_n);
    auto& malloc_access = builder.add_access(malloc_block, "tmp");
    builder.add_computational_memlet(malloc_block, malloc_node, "_ret", malloc_access, {}, t_fp_ptr);


    builder.add_container("i", t_int_desc);
    auto sym_i = symbolic::symbol("i");

    auto& loop = builder.add_map(
        root,
        sym_i,
        symbolic::Lt(sym_i, sym_n),
        symbolic::integer(0),
        symbolic::add(sym_i, symbolic::integer(1)),
        ScheduleType_Sequential::create()
    );
    auto& body = loop.root();

    auto& body_block = builder.add_block(body);
    auto& some_fp_const = builder.add_constant(body_block, "1.0", t_fp_ptr);
    auto& tmp_write = builder.add_access(body_block, "tmp");
    auto& out_write = builder.add_access(body_block, "out");
    auto& i_input = builder.add_access(body_block, "i");
    auto& tasklet = builder.add_tasklet(body_block, data_flow::TaskletCode::fp_add, "out", {"a", "b"});
    builder.add_computational_memlet(body_block, some_fp_const, tasklet, "a", {}, t_fp_ptr);
    builder.add_computational_memlet(body_block, i_input, tasklet, "b", {}, t_fp_ptr);
    builder.add_computational_memlet(body_block, tasklet, "out", tmp_write, {sym_i}, t_fp_ptr);
    auto& tasklet2 = builder.add_tasklet(body_block, data_flow::TaskletCode::fp_add, "out", {"a", "b"});
    builder.add_computational_memlet(body_block, some_fp_const, tasklet2, "a", {}, t_fp_ptr);
    builder.add_computational_memlet(body_block, i_input, tasklet2, "b", {}, t_fp_ptr);
    builder.add_computational_memlet(body_block, tasklet2, "out", out_write, {sym_i}, t_fp_ptr);

    //
    auto& read_block = builder.add_block(root);
    auto& tmp_read = builder.add_access(read_block, "tmp");
    std::vector<std::string> lib_node_outs;
    std::vector<std::string> lib_node_ins = {"ptr_in"};
    types::Scalar void_type(types::Void);
    types::Function func_type(void_type);
    func_type.add_param(t_fp_ptr_ptr);
    builder.add_external("blackbox_function", func_type, LinkageType_External);
    auto& blackbox =
        builder.add_library_node<data_flow::CallNode>(read_block, {}, "blackbox_function", lib_node_outs, lib_node_ins);
    builder.add_computational_memlet(read_block, tmp_read, blackbox, "ptr_in", {}, t_fp_ptr);

    auto sdfg = builder.move();

    dump_sdfg(*sdfg, "0-before");

    // Apply pass
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::DeadDataElimination pass;
    bool applied = pass.run(builder_opt, analysis_manager);
    passes::DeadCFGElimination().run(builder_opt, analysis_manager); // to remove the empty block left over

    sdfg = builder_opt.move();

    dump_sdfg(*sdfg, "1-after");

    // Check result
    EXPECT_TRUE(sdfg->exists("tmp"));
    EXPECT_TRUE(sdfg->exists("out"));
    EXPECT_TRUE(sdfg->exists("i"));
    EXPECT_EQ(sdfg->root().size(), 3);
}

TEST(DeadDataEliminationTest, OwnedHeapMemoryEscapesViaTasklet) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar t_int_desc(types::PrimitiveType::UInt32);
    types::Scalar t_int64_desc(types::PrimitiveType::UInt64);
    types::Scalar t_fp_desc(types::PrimitiveType::Float);
    types::Pointer t_fp_ptr(t_fp_desc);

    builder.add_container("out", t_fp_ptr, true);
    builder.add_container("N", t_int_desc, true);
    builder.add_container("tmp", t_fp_ptr);

    auto& root = builder.subject().root();
    auto& malloc_block = builder.add_block(root);
    auto sym_n = symbolic::symbol("N");
    auto& malloc_node = builder.add_library_node<stdlib::MallocNode>(malloc_block, DebugInfo(), sym_n);
    auto& malloc_access = builder.add_access(malloc_block, "tmp");
    builder.add_computational_memlet(malloc_block, malloc_node, "_ret", malloc_access, {}, t_fp_ptr);


    builder.add_container("i", t_int_desc);
    auto sym_i = symbolic::symbol("i");

    auto& loop = builder.add_map(
        root,
        sym_i,
        symbolic::Lt(sym_i, sym_n),
        symbolic::integer(0),
        symbolic::add(sym_i, symbolic::integer(1)),
        ScheduleType_Sequential::create()
    );
    auto& body = loop.root();

    auto& body_block = builder.add_block(body);
    auto& some_fp_const = builder.add_constant(body_block, "1.0", t_fp_ptr);
    auto& tmp_write = builder.add_access(body_block, "tmp");
    auto& out_write = builder.add_access(body_block, "out");
    auto& i_input = builder.add_access(body_block, "i");
    auto& tasklet = builder.add_tasklet(body_block, data_flow::TaskletCode::fp_add, "out", {"a", "b"});
    builder.add_computational_memlet(body_block, some_fp_const, tasklet, "a", {}, t_fp_ptr);
    builder.add_computational_memlet(body_block, i_input, tasklet, "b", {}, t_fp_ptr);
    builder.add_computational_memlet(body_block, tasklet, "out", tmp_write, {sym_i}, t_fp_ptr);
    auto& tasklet2 = builder.add_tasklet(body_block, data_flow::TaskletCode::fp_add, "out", {"a", "b"});
    builder.add_computational_memlet(body_block, some_fp_const, tasklet2, "a", {}, t_fp_ptr);
    builder.add_computational_memlet(body_block, i_input, tasklet2, "b", {}, t_fp_ptr);
    builder.add_computational_memlet(body_block, tasklet2, "out", out_write, {sym_i}, t_fp_ptr);

    //
    builder.add_container("scalar_res", t_int64_desc);
    auto& read_block = builder.add_block(root);
    auto& escape_target = builder.add_access(read_block, "scalar_res");
    auto& tmp_read = builder.add_access(read_block, "tmp");
    auto& comp_tasklet = builder.add_tasklet(read_block, data_flow::TaskletCode::int_add, "res", {"a", "b"});
    auto& int_const = builder.add_constant(read_block, "5", t_int64_desc);
    builder.add_computational_memlet(read_block, tmp_read, comp_tasklet, "a", {}, t_int64_desc);
    builder.add_computational_memlet(read_block, int_const, comp_tasklet, "b", {}, t_int64_desc);
    builder.add_computational_memlet(read_block, comp_tasklet, "res", escape_target, {}, t_int64_desc);

    builder.add_return(root, "scalar_res");

    auto sdfg = builder.move();

    dump_sdfg(*sdfg, "0-before");

    // Apply pass
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::DeadDataElimination pass;
    bool applied = pass.run(builder_opt, analysis_manager);
    passes::DeadCFGElimination().run(builder_opt, analysis_manager); // to remove the empty block left over

    sdfg = builder_opt.move();

    dump_sdfg(*sdfg, "1-after");

    // Check result
    EXPECT_TRUE(sdfg->exists("tmp"));
    EXPECT_TRUE(sdfg->exists("out"));
    EXPECT_TRUE(sdfg->exists("i"));
    EXPECT_TRUE(sdfg->exists("scalar_res"));
    EXPECT_EQ(sdfg->root().size(), 4);
}

TEST(DeadDataEliminationTest, OwnedHeapMemoryEscapesViaSymbol) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar t_int_desc(types::PrimitiveType::UInt32);
    types::Scalar t_int64_desc(types::PrimitiveType::UInt64);
    types::Scalar t_fp_desc(types::PrimitiveType::Float);
    types::Pointer t_fp_ptr(t_fp_desc);

    builder.add_container("out", t_fp_ptr, true);
    builder.add_container("N", t_int_desc, true);
    builder.add_container("tmp", t_fp_ptr);

    auto& root = builder.subject().root();
    auto& malloc_block = builder.add_block(root);
    auto sym_n = symbolic::symbol("N");
    auto& malloc_node = builder.add_library_node<stdlib::MallocNode>(malloc_block, DebugInfo(), sym_n);
    auto& malloc_access = builder.add_access(malloc_block, "tmp");
    builder.add_computational_memlet(malloc_block, malloc_node, "_ret", malloc_access, {}, t_fp_ptr);


    builder.add_container("i", t_int_desc);
    auto sym_i = symbolic::symbol("i");

    auto& loop = builder.add_map(
        root,
        sym_i,
        symbolic::Lt(sym_i, sym_n),
        symbolic::integer(0),
        symbolic::add(sym_i, symbolic::integer(1)),
        ScheduleType_Sequential::create()
    );
    auto& body = loop.root();

    auto& body_block = builder.add_block(body);
    auto& some_fp_const = builder.add_constant(body_block, "1.0", t_fp_ptr);
    auto& tmp_write = builder.add_access(body_block, "tmp");
    auto& out_write = builder.add_access(body_block, "out");
    auto& i_input = builder.add_access(body_block, "i");
    auto& tasklet = builder.add_tasklet(body_block, data_flow::TaskletCode::fp_add, "out", {"a", "b"});
    builder.add_computational_memlet(body_block, some_fp_const, tasklet, "a", {}, t_fp_ptr);
    builder.add_computational_memlet(body_block, i_input, tasklet, "b", {}, t_fp_ptr);
    builder.add_computational_memlet(body_block, tasklet, "out", tmp_write, {sym_i}, t_fp_ptr);
    auto& tasklet2 = builder.add_tasklet(body_block, data_flow::TaskletCode::fp_add, "out", {"a", "b"});
    builder.add_computational_memlet(body_block, some_fp_const, tasklet2, "a", {}, t_fp_ptr);
    builder.add_computational_memlet(body_block, i_input, tasklet2, "b", {}, t_fp_ptr);
    builder.add_computational_memlet(body_block, tasklet2, "out", out_write, {sym_i}, t_fp_ptr);

    //
    builder.add_container("scalar_res", t_int64_desc);
    builder.add_container("it", t_fp_ptr);
    auto it_symbol = symbolic::symbol("it");
    auto tmp_symbol = symbolic::symbol("tmp");
    auto& use_map = builder.add_map(
        root,
        it_symbol,
        symbolic::Lt(it_symbol, symbolic::add(tmp_symbol, symbolic::symbol("N"))),
        tmp_symbol,
        symbolic::add(tmp_symbol, symbolic::integer(1)),
        ScheduleType_Sequential::create(),
        {{symbolic::symbol("scalar_res"), symbolic::integer(0)}}
    );

    auto& use_map_body = use_map.root();
    auto& read_block = builder.add_block(use_map_body);
    auto& sum_node_rd = builder.add_access(read_block, "scalar_res");
    auto& sum_node_wr = builder.add_access(read_block, "scalar_res");
    auto& inc_1 = builder.add_constant(read_block, "1", t_int64_desc);
    auto& comp_tasklet = builder.add_tasklet(read_block, data_flow::TaskletCode::int_add, "res", {"a", "b"});
    builder.add_computational_memlet(read_block, sum_node_rd, comp_tasklet, "a", {}, t_int64_desc);
    builder.add_computational_memlet(read_block, inc_1, comp_tasklet, "b", {}, t_int64_desc);
    builder.add_computational_memlet(read_block, comp_tasklet, "res", sum_node_wr, {}, t_int64_desc);

    builder.add_return(root, "scalar_res");

    auto sdfg = builder.move();

    dump_sdfg(*sdfg, "0-before");

    // Apply pass
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::DeadDataElimination pass;
    bool applied = pass.run(builder_opt, analysis_manager);
    passes::DeadCFGElimination().run(builder_opt, analysis_manager); // to remove the empty block left over

    sdfg = builder_opt.move();

    dump_sdfg(*sdfg, "1-after");

    // Check result
    EXPECT_TRUE(sdfg->exists("tmp"));
    EXPECT_TRUE(sdfg->exists("out"));
    EXPECT_TRUE(sdfg->exists("i"));
    EXPECT_TRUE(sdfg->exists("scalar_res"));
    EXPECT_EQ(sdfg->root().size(), 4);
}

TEST(DeadDataEliminationTest, OwnedHeapMemoryElementRead) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar t_int_desc(types::PrimitiveType::UInt32);
    types::Scalar t_int64_desc(types::PrimitiveType::UInt64);
    types::Scalar t_fp_desc(types::PrimitiveType::Float);
    types::Pointer t_fp_ptr(t_fp_desc);

    builder.add_container("out", t_fp_ptr, true);
    builder.add_container("N", t_int_desc, true);
    builder.add_container("tmp", t_fp_ptr);

    auto& root = builder.subject().root();
    auto& malloc_block = builder.add_block(root);
    auto sym_n = symbolic::symbol("N");
    auto& malloc_node = builder.add_library_node<stdlib::MallocNode>(malloc_block, DebugInfo(), sym_n);
    auto& malloc_access = builder.add_access(malloc_block, "tmp");
    builder.add_computational_memlet(malloc_block, malloc_node, "_ret", malloc_access, {}, t_fp_ptr);


    builder.add_container("i", t_int_desc);
    auto sym_i = symbolic::symbol("i");

    auto& loop = builder.add_map(
        root,
        sym_i,
        symbolic::Lt(sym_i, sym_n),
        symbolic::integer(0),
        symbolic::add(sym_i, symbolic::integer(1)),
        ScheduleType_Sequential::create()
    );
    auto& body = loop.root();

    auto& body_block = builder.add_block(body);
    auto& some_fp_const = builder.add_constant(body_block, "1.0", t_fp_ptr);
    auto& tmp_write = builder.add_access(body_block, "tmp");
    auto& out_write = builder.add_access(body_block, "out");
    auto& i_input = builder.add_access(body_block, "i");
    auto& tasklet = builder.add_tasklet(body_block, data_flow::TaskletCode::fp_add, "out", {"a", "b"});
    builder.add_computational_memlet(body_block, some_fp_const, tasklet, "a", {}, t_fp_ptr);
    builder.add_computational_memlet(body_block, i_input, tasklet, "b", {}, t_fp_ptr);
    builder.add_computational_memlet(body_block, tasklet, "out", tmp_write, {sym_i}, t_fp_ptr);
    auto& tasklet2 = builder.add_tasklet(body_block, data_flow::TaskletCode::fp_add, "out", {"a", "b"});
    builder.add_computational_memlet(body_block, some_fp_const, tasklet2, "a", {}, t_fp_ptr);
    builder.add_computational_memlet(body_block, i_input, tasklet2, "b", {}, t_fp_ptr);
    builder.add_computational_memlet(body_block, tasklet2, "out", out_write, {sym_i}, t_fp_ptr);

    //
    builder.add_container("scalar_res", t_fp_desc);
    auto& read_block = builder.add_block(root);
    auto& escape_target = builder.add_access(read_block, "scalar_res");
    auto& tmp_read = builder.add_access(read_block, "tmp");
    auto& comp_tasklet = builder.add_tasklet(read_block, data_flow::TaskletCode::assign, "res", {"in"});
    builder.add_computational_memlet(read_block, tmp_read, comp_tasklet, "in", {symbolic::integer(5)}, t_fp_ptr);
    builder.add_computational_memlet(read_block, comp_tasklet, "res", escape_target, {}, t_int64_desc);

    builder.add_return(root, "scalar_res");

    auto sdfg = builder.move();

    dump_sdfg(*sdfg, "0-before");

    // Apply pass
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::DeadDataElimination pass;
    bool applied = pass.run(builder_opt, analysis_manager);
    passes::DeadCFGElimination().run(builder_opt, analysis_manager); // to remove the empty block left over

    sdfg = builder_opt.move();

    dump_sdfg(*sdfg, "1-after");

    // Check result
    EXPECT_TRUE(sdfg->exists("tmp"));
    EXPECT_TRUE(sdfg->exists("out"));
    EXPECT_TRUE(sdfg->exists("i"));
    EXPECT_TRUE(sdfg->exists("scalar_res"));
    EXPECT_EQ(sdfg->root().size(), 4);
}
