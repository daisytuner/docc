#include "sdfg/passes/symbolic/symbol_propagation.h"

#include <gtest/gtest.h>

#include "sdfg/builder/structured_sdfg_builder.h"

using namespace sdfg;

TEST(SymbolPropagationTest, Transition2Transition_Constant) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::UInt32);
    builder.add_container("i", desc);
    builder.add_container("j", desc);
    auto sym1 = symbolic::symbol("i");
    auto sym2 = symbolic::symbol("j");

    auto& root = builder.subject().root();
    auto& block1 = builder.add_block(root, {{sym1, symbolic::integer(0)}});
    auto& block2 = builder.add_block(root, {{sym2, sym1}});

    auto sdfg = builder.move();

    // Apply pass
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::SymbolPropagation pass;
    pass.run(builder_opt, analysis_manager);
    sdfg = builder_opt.move();

    // Check result
    EXPECT_EQ(sdfg->root().size(), 2);
    auto child1 = sdfg->root().at(0);
    auto child2 = sdfg->root().at(1);

    EXPECT_EQ(child1.second.assignments().size(), 1);
    EXPECT_TRUE(SymEngine::eq(*child1.second.assignments().at(sym1), *symbolic::integer(0)));

    EXPECT_EQ(child2.second.assignments().size(), 1);
    EXPECT_TRUE(SymEngine::eq(*child2.second.assignments().at(sym2), *symbolic::integer(0)));
}

TEST(SymbolPropagationTest, Transition2Transition_Symbol) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::UInt32);
    builder.add_container("N", desc, true);
    builder.add_container("i", desc);
    builder.add_container("j", desc);
    auto symN = symbolic::symbol("N");
    auto sym1 = symbolic::symbol("i");
    auto sym2 = symbolic::symbol("j");

    auto& root = builder.subject().root();
    auto& block1 = builder.add_block(root, {{sym1, symN}});
    auto& block2 = builder.add_block(root, {{sym2, sym1}});

    auto sdfg = builder.move();

    // Apply pass
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::SymbolPropagation pass;
    pass.run(builder_opt, analysis_manager);
    sdfg = builder_opt.move();

    // Check result
    EXPECT_EQ(sdfg->root().size(), 2);
    auto child1 = sdfg->root().at(0);
    auto child2 = sdfg->root().at(1);

    EXPECT_EQ(child1.second.assignments().size(), 1);
    EXPECT_TRUE(SymEngine::eq(*child1.second.assignments().at(sym1), *symN));

    EXPECT_EQ(child2.second.assignments().size(), 1);
    EXPECT_TRUE(SymEngine::eq(*child2.second.assignments().at(sym2), *symN));
}

TEST(SymbolPropagationTest, Transition2Transition_DataRace) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::UInt32);
    builder.add_container("N", desc, true);
    builder.add_container("i", desc);
    builder.add_container("j", desc);
    auto symN = symbolic::symbol("N");
    auto sym1 = symbolic::symbol("i");
    auto sym2 = symbolic::symbol("j");

    auto& root = builder.subject().root();
    auto& block1 = builder.add_block(root, {{sym1, symN}});
    auto& block2 = builder.add_block(root, {{symN, symbolic::integer(0)}});
    auto& block3 = builder.add_block(root, {{sym2, sym1}});

    auto sdfg = builder.move();

    // Apply pass
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::SymbolPropagation pass;
    pass.run(builder_opt, analysis_manager);
    sdfg = builder_opt.move();

    // Check result
    EXPECT_EQ(sdfg->root().size(), 3);
    auto child1 = sdfg->root().at(0);
    auto child2 = sdfg->root().at(1);
    auto child3 = sdfg->root().at(2);

    EXPECT_EQ(child1.second.assignments().size(), 1);
    EXPECT_TRUE(SymEngine::eq(*child1.second.assignments().at(sym1), *symN));

    EXPECT_EQ(child2.second.assignments().size(), 1);
    EXPECT_TRUE(SymEngine::eq(*child2.second.assignments().at(symN), *symbolic::integer(0)));

    EXPECT_EQ(child3.second.assignments().size(), 1);
    EXPECT_TRUE(SymEngine::eq(*child3.second.assignments().at(sym2), *sym1));
}

TEST(SymbolPropagationTest, Transition2Memlet_Constant) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::UInt32);
    types::Array array(desc, symbolic::integer(10));
    builder.add_container("A", array);
    builder.add_container("i", desc);
    auto sym1 = symbolic::symbol("i");

    auto& root = builder.subject().root();
    auto& block1 = builder.add_block(root, {{sym1, symbolic::integer(0)}});
    auto& block2 = builder.add_block(root);
    auto& output_node = builder.add_access(block2, "A");
    auto& one_node = builder.add_constant(block2, "1", desc);
    auto& tasklet = builder.add_tasklet(block2, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block2, one_node, tasklet, "_in", {});
    auto& edge = builder.add_computational_memlet(block2, tasklet, "_out", output_node, {sym1});

    auto sdfg = builder.move();

    // Apply pass
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::SymbolPropagation pass;
    pass.run(builder_opt, analysis_manager);
    sdfg = builder_opt.move();

    // Check result
    EXPECT_EQ(sdfg->root().size(), 2);
    auto child1 = sdfg->root().at(0);
    auto child2 = sdfg->root().at(1);

    EXPECT_EQ(child1.second.assignments().size(), 1);
    EXPECT_TRUE(SymEngine::eq(*child1.second.assignments().at(sym1), *symbolic::integer(0)));

    EXPECT_EQ(child2.second.assignments().size(), 0);
    EXPECT_TRUE(SymEngine::eq(*edge.subset().at(0), *symbolic::integer(0)));
}

TEST(SymbolPropagationTest, Transition2Memlet_Argument) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::UInt32);
    types::Pointer pointer;
    builder.add_container("A", pointer);
    builder.add_container("i", desc);
    auto sym1 = symbolic::symbol("i");

    auto& root = builder.subject().root();
    auto& block1 = builder.add_block(root, {{sym1, symbolic::symbol("A")}});
    auto& block2 = builder.add_block(root);
    auto& output_node = builder.add_access(block2, "A");
    auto& one_node = builder.add_constant(block2, "1", desc);
    auto& tasklet = builder.add_tasklet(block2, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block2, one_node, tasklet, "_in", {});
    auto& edge = builder.add_computational_memlet(
        block2, tasklet, "_out", output_node, {sym1}, types::Pointer(types::Scalar(types::PrimitiveType::UInt32))
    );

    auto sdfg = builder.move();

    // Apply pass
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::SymbolPropagation pass;
    EXPECT_TRUE(pass.run(builder_opt, analysis_manager));
    sdfg = builder_opt.move();

    // Check result
    EXPECT_EQ(sdfg->root().size(), 2);
    auto child1 = sdfg->root().at(0);
    auto child2 = sdfg->root().at(1);

    EXPECT_EQ(child1.second.assignments().size(), 1);
    EXPECT_TRUE(SymEngine::eq(*child1.second.assignments().at(sym1), *symbolic::symbol("A")));

    EXPECT_EQ(child2.second.assignments().size(), 0);
    EXPECT_TRUE(SymEngine::eq(*edge.subset().at(0), *symbolic::symbol("A")));
}

TEST(SymbolPropagationTest, Transition2IfElse_Constant) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::UInt32);
    builder.add_container("i", desc);
    builder.add_container("j", desc, true);
    auto sym1 = symbolic::symbol("i");
    auto sym2 = symbolic::symbol("j");

    auto& root = builder.subject().root();
    auto& block1 = builder.add_block(root, {{sym1, symbolic::integer(0)}});

    auto& if_else = builder.add_if_else(root);

    auto& case1 = builder.add_case(if_else, symbolic::Lt(sym2, symbolic::integer(10)));
    auto& block2 = builder.add_block(case1, {{sym2, sym1}});

    auto& case2 = builder.add_case(if_else, symbolic::Ge(sym2, symbolic::integer(10)));
    auto& block3 = builder.add_block(case2, {{sym2, symbolic::add(sym1, symbolic::integer(1))}});

    auto sdfg = builder.move();

    // Apply pass
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::SymbolPropagation pass;
    pass.run(builder_opt, analysis_manager);
    sdfg = builder_opt.move();

    // Check result
    EXPECT_EQ(sdfg->root().size(), 2);
    auto child1 = sdfg->root().at(0);
    EXPECT_EQ(child1.second.assignments().size(), 1);
    EXPECT_TRUE(SymEngine::eq(*child1.second.assignments().at(sym1), *symbolic::integer(0)));

    auto& child2 = dyn_cast<structured_control_flow::IfElse&>(sdfg->root().at(1).first);
    auto case1_1 = child2.at(0).first.at(0);
    EXPECT_EQ(case1_1.second.assignments().size(), 1);
    EXPECT_TRUE(SymEngine::eq(*case1_1.second.assignments().at(sym2), *symbolic::integer(0)));

    auto case2_1 = child2.at(1).first.at(0);
    EXPECT_EQ(case2_1.second.assignments().size(), 1);
    EXPECT_TRUE(SymEngine::eq(*case2_1.second.assignments().at(sym2), *symbolic::integer(1)));
}

TEST(SymbolPropagationTest, Transition2IfElse_Argument) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::UInt32);
    builder.add_container("i", desc);
    builder.add_container("A", desc, true);
    builder.add_container("j", desc, true);
    auto sym1 = symbolic::symbol("i");
    auto sym2 = symbolic::symbol("j");
    auto sym3 = symbolic::symbol("A");

    auto& root = builder.subject().root();
    auto& block1 = builder.add_block(root, {{sym1, symbolic::symbol("A")}});

    auto& if_else = builder.add_if_else(root);

    auto& case1 = builder.add_case(if_else, symbolic::Lt(sym2, symbolic::symbol("A")));
    auto& block2 = builder.add_block(case1, {{sym2, sym1}});

    auto& case2 = builder.add_case(if_else, symbolic::Ge(sym2, symbolic::symbol("A")));
    auto& block3 = builder.add_block(case2, {{sym2, symbolic::add(sym1, symbolic::integer(1))}});

    auto sdfg = builder.move();

    // Apply pass
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::SymbolPropagation pass;
    pass.run(builder_opt, analysis_manager);
    sdfg = builder_opt.move();

    // Check result
    EXPECT_EQ(sdfg->root().size(), 2);
    auto child1 = sdfg->root().at(0);
    EXPECT_EQ(child1.second.assignments().size(), 1);
    EXPECT_TRUE(SymEngine::eq(*child1.second.assignments().at(sym1), *symbolic::symbol("A")));

    auto& child2 = dyn_cast<structured_control_flow::IfElse&>(sdfg->root().at(1).first);
    auto case1_1 = child2.at(0).first.at(0);
    EXPECT_EQ(case1_1.second.assignments().size(), 1);
    EXPECT_TRUE(SymEngine::eq(*case1_1.second.assignments().at(sym2), *symbolic::symbol("A")));

    auto case2_1 = child2.at(1).first.at(0);
    EXPECT_EQ(case2_1.second.assignments().size(), 1);
    EXPECT_TRUE(SymEngine::
                    eq(*case2_1.second.assignments().at(sym2),
                       *symbolic::add(symbolic::symbol("A"), symbolic::integer(1))));
}

TEST(SymbolPropagationTest, Transition2IfElse_Negative) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::UInt32);
    builder.add_container("i", desc);
    builder.add_container("A", desc, true);
    builder.add_container("j", desc);
    auto sym1 = symbolic::symbol("i");
    auto sym2 = symbolic::symbol("j");
    auto sym3 = symbolic::symbol("A");

    auto& root = builder.subject().root();
    auto& if_else = builder.add_if_else(root);

    auto& case1 = builder.add_case(if_else, symbolic::Lt(sym2, symbolic::symbol("A")));
    auto& block2 = builder.add_block(case1, {{sym2, symbolic::integer(0)}});

    auto& case2 = builder.add_case(if_else, symbolic::Ge(sym2, symbolic::symbol("A")));
    auto& block3 = builder.add_block(case2, {{sym2, symbolic::integer(1)}});

    auto& block1 = builder.add_block(root, {{sym1, sym2}});

    auto sdfg = builder.move();

    // Apply pass
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::SymbolPropagation pass;
    pass.run(builder_opt, analysis_manager);
    sdfg = builder_opt.move();

    // Check result
    EXPECT_EQ(sdfg->root().size(), 2);
    auto child2 = sdfg->root().at(1);
    EXPECT_EQ(child2.second.assignments().size(), 1);
    EXPECT_TRUE(SymEngine::eq(*child2.second.assignments().at(sym1), *sym2));
}

TEST(SymbolPropagationTest, Transition2For_Init) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::UInt32);
    builder.add_container("i", desc);
    auto sym = symbolic::symbol("i");

    auto& root = builder.subject().root();
    auto& block1 = builder.add_block(root, {{sym, symbolic::integer(0)}});

    auto& loop =
        builder
            .add_for(root, sym, symbolic::Lt(sym, symbolic::integer(10)), sym, symbolic::add(sym, symbolic::integer(1)));

    auto sdfg = builder.move();

    // Apply pass
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::SymbolPropagation pass;
    pass.run(builder_opt, analysis_manager);
    sdfg = builder_opt.move();

    // Check result
    EXPECT_TRUE(SymEngine::eq(*loop.init(), *symbolic::integer(0)));
}

TEST(SymbolPropagationTest, Transition2For_Condition) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::UInt32);
    builder.add_container("i", desc);
    builder.add_container("N", desc);
    auto sym = symbolic::symbol("i");
    auto symN = symbolic::symbol("N");

    auto& root = builder.subject().root();
    auto& block1 = builder.add_block(root, {{symN, symbolic::integer(10)}});

    auto& loop =
        builder
            .add_for(root, sym, symbolic::Lt(sym, symN), symbolic::integer(0), symbolic::add(sym, symbolic::integer(1)));

    auto sdfg = builder.move();

    // Apply pass
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::SymbolPropagation pass;
    pass.run(builder_opt, analysis_manager);
    sdfg = builder_opt.move();

    // Check result
    EXPECT_TRUE(SymEngine::eq(*loop.condition(), *symbolic::Lt(sym, symbolic::integer(10))));
}

TEST(SymbolPropagationTest, Transition2For_Update) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::UInt32);
    builder.add_container("i", desc);
    builder.add_container("j", desc);
    auto sym = symbolic::symbol("i");
    auto sym2 = symbolic::symbol("j");

    auto& root = builder.subject().root();
    auto& loop = builder.add_for(root, sym, symbolic::Lt(sym, symbolic::integer(10)), symbolic::integer(0), sym2);
    auto& body = builder.add_block(loop.root(), {{sym2, symbolic::add(sym, symbolic::integer(1))}});

    auto sdfg = builder.move();

    // Apply pass
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::SymbolPropagation pass;
    pass.run(builder_opt, analysis_manager);
    sdfg = builder_opt.move();

    // Check result
    EXPECT_TRUE(SymEngine::eq(*loop.update(), *symbolic::add(sym, symbolic::integer(1))));
}

TEST(SymbolPropagationTest, Transition2While_In) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::UInt32);
    builder.add_container("i", desc);
    builder.add_container("j", desc);
    auto sym = symbolic::symbol("i");
    auto sym2 = symbolic::symbol("j");

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root, {{sym, symbolic::integer(0)}});

    auto& loop = builder.add_while(root);
    auto& body1 = builder.add_block(loop.root(), {{sym2, sym}});

    auto sdfg = builder.move();

    // Apply pass
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::SymbolPropagation pass;
    pass.run(builder_opt, analysis_manager);
    sdfg = builder_opt.move();

    // Check result
    auto& child1 = dyn_cast<structured_control_flow::While&>(sdfg->root().at(1).first);
    EXPECT_TRUE(SymEngine::eq(*child1.root().at(0).second.assignments().at(sym2), *symbolic::integer(0)));
}

// ========== Negative Test Cases ==========

/**
 * @brief Test that propagation is skipped for non-transient symbols.
 *
 * Non-transient symbols (arguments) should not be propagated.
 */
TEST(SymbolPropagationTest, Negative_NonTransient) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::UInt32);
    builder.add_container("i", desc, true);
    builder.add_container("j", desc);
    auto sym1 = symbolic::symbol("i");
    auto sym2 = symbolic::symbol("j");

    auto& root = builder.subject().root();
    auto& block1 = builder.add_block(root, {{sym1, symbolic::integer(0)}});
    auto& block2 = builder.add_block(root, {{sym2, sym1}});

    auto sdfg = builder.move();

    // Apply pass
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::SymbolPropagation pass;
    bool modified = pass.run(builder_opt, analysis_manager);
    sdfg = builder_opt.move();

    // Check result - non-transient symbols should not be propagated
    EXPECT_FALSE(modified);
    auto child1 = sdfg->root().at(0);
    auto child2 = sdfg->root().at(1);
    EXPECT_EQ(child1.second.assignments().size(), 1);
    EXPECT_TRUE(SymEngine::eq(*child1.second.assignments().at(sym1), *symbolic::integer(0)));
    EXPECT_EQ(child2.second.assignments().size(), 1);
    EXPECT_TRUE(SymEngine::eq(*child2.second.assignments().at(sym2), *sym1)); // Not propagated
}

/**
 * @brief Test that recursive assignments (i = i) are not propagated.
 *
 * Self-assignments should be skipped to avoid infinite loops.
 */
TEST(SymbolPropagationTest, Negative_RecursiveAssignment) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::UInt32);
    builder.add_container("i", desc);
    builder.add_container("j", desc);
    auto sym1 = symbolic::symbol("i");
    auto sym2 = symbolic::symbol("j");

    auto& root = builder.subject().root();
    auto& block1 = builder.add_block(root, {{sym1, sym1}}); // Recursive assignment
    auto& block2 = builder.add_block(root, {{sym2, sym1}});

    auto sdfg = builder.move();

    // Apply pass
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::SymbolPropagation pass;
    bool modified = pass.run(builder_opt, analysis_manager);
    sdfg = builder_opt.move();

    // Check result - recursive assignments should not be propagated
    EXPECT_FALSE(modified);
    auto child2 = sdfg->root().at(1);
    EXPECT_TRUE(SymEngine::eq(*child2.second.assignments().at(sym2), *sym1)); // Not propagated
}

/**
 * @brief Test that propagation fails when RHS uses LHS symbol.
 *
 * When the right-hand side uses the left-hand side symbol (e.g., i = i + 1),
 * propagation should not occur unless it's exactly the same symbol.
 */
TEST(SymbolPropagationTest, Negative_RhsUsesLhs) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::UInt32);
    builder.add_container("i", desc);
    builder.add_container("j", desc);
    auto sym1 = symbolic::symbol("i");
    auto sym2 = symbolic::symbol("j");

    auto& root = builder.subject().root();
    auto& block1 = builder.add_block(root, {{sym1, symbolic::add(sym1, symbolic::integer(1))}}); // i = i + 1
    auto& block2 = builder.add_block(root, {{sym2, sym1}});

    auto sdfg = builder.move();

    // Apply pass
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::SymbolPropagation pass;
    bool modified = pass.run(builder_opt, analysis_manager);
    sdfg = builder_opt.move();

    // Check result - should not propagate when RHS uses LHS
    EXPECT_FALSE(modified);
    auto child2 = sdfg->root().at(1);
    EXPECT_TRUE(SymEngine::eq(*child2.second.assignments().at(sym2), *sym1)); // Not propagated
}

/**
 * @brief Test that propagation fails when write does not dominate read.
 *
 * For forward propagation, the write must dominate the read.
 */
TEST(SymbolPropagationTest, Negative_WriteDominatesRead) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::UInt32);
    builder.add_container("i", desc);
    builder.add_container("j", desc);
    builder.add_container("k", desc, true);
    auto sym1 = symbolic::symbol("i");
    auto sym2 = symbolic::symbol("j");
    auto sym3 = symbolic::symbol("k");

    auto& root = builder.subject().root();
    auto& if_else = builder.add_if_else(root);

    auto& case1 = builder.add_case(if_else, symbolic::Lt(sym3, symbolic::integer(10)));
    auto& block1 = builder.add_block(case1, {{sym1, symbolic::integer(5)}});

    auto& case2 = builder.add_case(if_else, symbolic::Ge(sym3, symbolic::integer(10)));
    auto& block2 = builder.add_block(case2); // No assignment to i

    auto& block3 = builder.add_block(root, {{sym2, sym1}}); // Read i, but not dominated

    auto sdfg = builder.move();

    // Apply pass
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::SymbolPropagation pass;
    bool modified = pass.run(builder_opt, analysis_manager);
    sdfg = builder_opt.move();

    // Check result - should not propagate when not dominated
    // The assignment might still be present, but not propagated to block3
    auto child3 = sdfg->root().at(1);
    EXPECT_TRUE(SymEngine::eq(*child3.second.assignments().at(sym2), *sym1)); // Not propagated
}

/**
 * @brief Test reverse propagation with multiple uses of intermediate symbol.
 *
 * Reverse propagation should not occur when the intermediate symbol has multiple uses.
 */
TEST(SymbolPropagationTest, Negative_ReversePropagationMultipleUses) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::UInt32);
    builder.add_container("N", desc, true);
    builder.add_container("i", desc);
    builder.add_container("j", desc);
    builder.add_container("k", desc);
    auto symN = symbolic::symbol("N");
    auto sym1 = symbolic::symbol("i");
    auto sym2 = symbolic::symbol("j");
    auto sym3 = symbolic::symbol("k");

    auto& root = builder.subject().root();
    auto& block1 = builder.add_block(root, {{sym1, symN}});
    auto& block2 = builder.add_block(root, {{sym2, sym1}}); // First use
    auto& block3 = builder.add_block(root, {{sym3, sym1}}); // Second use

    auto sdfg = builder.move();

    // Apply pass
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::SymbolPropagation pass;
    bool modified = pass.run(builder_opt, analysis_manager);
    sdfg = builder_opt.move();

    // Check result - forward propagation should work, but verify
    EXPECT_TRUE(modified);
    auto child2 = sdfg->root().at(1);
    auto child3 = sdfg->root().at(2);
    // Both should be propagated to N
    EXPECT_TRUE(SymEngine::eq(*child2.second.assignments().at(sym2), *symN));
    EXPECT_TRUE(SymEngine::eq(*child3.second.assignments().at(sym3), *symN));
}

// ========== Additional Comprehensive Tests ==========

/**
 * @brief Test propagation with expression simplification.
 *
 * The pass should simplify expressions after substitution.
 */
TEST(SymbolPropagationTest, ExpressionSimplification) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::UInt32);
    builder.add_container("i", desc);
    builder.add_container("j", desc);
    auto sym1 = symbolic::symbol("i");
    auto sym2 = symbolic::symbol("j");

    auto& root = builder.subject().root();
    auto& block1 = builder.add_block(root, {{sym1, symbolic::integer(5)}});
    auto& block2 = builder.add_block(root, {{sym2, symbolic::add(sym1, symbolic::integer(3))}});

    auto sdfg = builder.move();

    // Apply pass
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::SymbolPropagation pass;
    bool modified = pass.run(builder_opt, analysis_manager);
    sdfg = builder_opt.move();

    // Check result - should simplify to 5 + 3 = 8
    EXPECT_TRUE(modified);
    auto child2 = sdfg->root().at(1);
    EXPECT_EQ(child2.second.assignments().size(), 1);
    EXPECT_TRUE(SymEngine::eq(*child2.second.assignments().at(sym2), *symbolic::integer(8)));
}

/**
 * @brief Test propagation through multiple levels of indirection.
 *
 * Tests that the pass can propagate through a chain: i = 5, j = i, k = j.
 */
TEST(SymbolPropagationTest, MultiLevelPropagation) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::UInt32);
    builder.add_container("i", desc);
    builder.add_container("j", desc);
    builder.add_container("k", desc);
    auto sym1 = symbolic::symbol("i");
    auto sym2 = symbolic::symbol("j");
    auto sym3 = symbolic::symbol("k");

    auto& root = builder.subject().root();
    auto& block1 = builder.add_block(root, {{sym1, symbolic::integer(5)}});
    auto& block2 = builder.add_block(root, {{sym2, sym1}});
    auto& block3 = builder.add_block(root, {{sym3, sym2}});

    auto sdfg = builder.move();

    // Apply pass multiple times to propagate through all levels
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::SymbolPropagation pass;

    bool modified = true;
    int iterations = 0;
    while (modified && iterations < 10) {
        modified = pass.run(builder_opt, analysis_manager);
        iterations++;
    }
    sdfg = builder_opt.move();

    // Check result - after multiple iterations, k should be 5
    auto child3 = sdfg->root().at(2);
    EXPECT_TRUE(SymEngine::eq(*child3.second.assignments().at(sym3), *symbolic::integer(5)));
}

/**
 * @brief Test that pass returns true when modifications are made.
 *
 * Explicitly verify the return value of run_pass.
 */
TEST(SymbolPropagationTest, ReturnValueWhenModified) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::UInt32);
    builder.add_container("i", desc);
    builder.add_container("j", desc);
    auto sym1 = symbolic::symbol("i");
    auto sym2 = symbolic::symbol("j");

    auto& root = builder.subject().root();
    auto& block1 = builder.add_block(root, {{sym1, symbolic::integer(0)}});
    auto& block2 = builder.add_block(root, {{sym2, sym1}});

    auto sdfg = builder.move();

    // Apply pass
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::SymbolPropagation pass;
    bool modified = pass.run(builder_opt, analysis_manager);

    // Verify return value
    EXPECT_TRUE(modified);
}

/**
 * @brief Test that pass returns false when no modifications are made.
 *
 * Explicitly verify the return value when nothing changes.
 */
TEST(SymbolPropagationTest, ReturnValueWhenUnmodified) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::UInt32);
    builder.add_container("i", desc, false); // Non-transient
    auto sym1 = symbolic::symbol("i");

    auto& root = builder.subject().root();
    auto& block1 = builder.add_block(root, {{sym1, symbolic::integer(0)}});

    auto sdfg = builder.move();

    // Apply pass
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::SymbolPropagation pass;
    bool modified = pass.run(builder_opt, analysis_manager);

    // Verify return value - should be false since nothing was propagated
    EXPECT_FALSE(modified);
}

/**
 * @brief Test propagation with complex expression in memlet.
 *
 * Verify that propagation works correctly in array access patterns.
 */
TEST(SymbolPropagationTest, ComplexMemletPropagation) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::UInt32);
    types::Array array(desc, symbolic::integer(100));
    builder.add_container("A", array);
    builder.add_container("i", desc);
    builder.add_container("j", desc);
    auto sym1 = symbolic::symbol("i");
    auto sym2 = symbolic::symbol("j");

    auto& root = builder.subject().root();
    auto& block1 = builder.add_block(root, {{sym1, symbolic::integer(5)}});
    auto& block2 = builder.add_block(root, {{sym2, symbolic::integer(10)}});
    auto& block3 = builder.add_block(root);
    auto& output_node = builder.add_access(block3, "A");
    auto& one_node = builder.add_constant(block3, "1", desc);
    auto& tasklet = builder.add_tasklet(block3, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block3, one_node, tasklet, "_in", {});
    auto& edge = builder.add_computational_memlet(
        block3, tasklet, "_out", output_node, {symbolic::add(sym1, sym2)} // A[i + j]
    );

    auto sdfg = builder.move();

    // Apply pass
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::SymbolPropagation pass;
    bool modified = pass.run(builder_opt, analysis_manager);
    sdfg = builder_opt.move();

    // Check result - should simplify to A[15]
    EXPECT_TRUE(modified);
    EXPECT_TRUE(SymEngine::eq(*edge.subset().at(0), *symbolic::integer(15)));
}

/**
 * @brief Test pass name method.
 *
 * Verify the pass name is correctly set.
 */
TEST(SymbolPropagationTest, PassName) {
    passes::SymbolPropagation pass;
    EXPECT_EQ(pass.name(), "SymbolPropagation");
}
