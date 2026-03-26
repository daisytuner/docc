#include "sdfg/passes/symbolic/symbol_promotion.h"

#include <gtest/gtest.h>


#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg_debug_dump.h"

using namespace sdfg;

TEST(SymbolPromotionTest, Assign_Signed) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::Int32);
    builder.add_container("i", desc);
    builder.add_container("j", desc);
    auto sym1 = symbolic::symbol("i");
    auto sym2 = symbolic::symbol("j");

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);
    auto& input_node = builder.add_access(block, "i");
    auto& output_node = builder.add_access(block, "j");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, input_node, tasklet, "_in", {});
    builder.add_computational_memlet(block, tasklet, "_out", output_node, {});

    dump_sdfg(builder.subject(), "0-before");

    auto sdfg = builder.move();

    // Apply pass
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::SymbolPromotion s2spass;
    EXPECT_TRUE(s2spass.run(builder_opt, analysis_manager));

    dump_sdfg(builder_opt.subject(), "1-after");
    sdfg = builder_opt.move();

    // Check result
    EXPECT_EQ(sdfg->root().size(), 2);
    auto child1 = sdfg->root().at(0);
    auto block1 = dynamic_cast<const structured_control_flow::Block*>(&child1.first);
    auto child2 = sdfg->root().at(1);
    auto block2 = dynamic_cast<const structured_control_flow::Block*>(&child2.first);

    EXPECT_NE(block1, nullptr);
    EXPECT_NE(block2, nullptr);

    EXPECT_EQ(block1->dataflow().nodes().size(), 0);
    EXPECT_EQ(child1.second.assignments().size(), 1);
    EXPECT_TRUE(SymEngine::eq(*child1.second.assignments().at(sym2), *sym1));

    EXPECT_EQ(block2->dataflow().nodes().size(), 0);
    EXPECT_EQ(child2.second.assignments().size(), 0);
}

TEST(SymbolPromotionTest, Assign_Signed_Constant) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::Int32);
    builder.add_container("i", desc);
    auto sym = symbolic::symbol("i");

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);
    auto& zero_node = builder.add_constant(block, "0", desc);
    auto& output_node = builder.add_access(block, "i");

    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});

    builder.add_computational_memlet(block, zero_node, tasklet, "_in", {});
    builder.add_computational_memlet(block, tasklet, "_out", output_node, {});

    auto sdfg = builder.move();

    // Apply pass
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::SymbolPromotion s2spass;
    EXPECT_TRUE(s2spass.run(builder_opt, analysis_manager));
    sdfg = builder_opt.move();

    // Check result
    EXPECT_EQ(sdfg->root().size(), 2);
    auto child1 = sdfg->root().at(0);
    auto block1 = dynamic_cast<const structured_control_flow::Block*>(&child1.first);
    auto child2 = sdfg->root().at(1);
    auto block2 = dynamic_cast<const structured_control_flow::Block*>(&child2.first);

    EXPECT_NE(block1, nullptr);
    EXPECT_NE(block2, nullptr);

    EXPECT_EQ(block1->dataflow().nodes().size(), 0);
    EXPECT_EQ(child1.second.assignments().size(), 1);
    EXPECT_TRUE(SymEngine::eq(*child1.second.assignments().at(sym), *symbolic::integer(0)));

    EXPECT_EQ(block2->dataflow().nodes().size(), 0);
    EXPECT_EQ(child2.second.assignments().size(), 0);
}

TEST(SymbolPromotionTest, Assign_Unsigned) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::UInt32);
    builder.add_container("i", desc);
    auto sym = symbolic::symbol("i");

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);
    auto& zero_node = builder.add_constant(block, "0", desc);
    auto& output_node = builder.add_access(block, "i");

    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});

    builder.add_computational_memlet(block, zero_node, tasklet, "_in", {});
    builder.add_computational_memlet(block, tasklet, "_out", output_node, {});

    auto sdfg = builder.move();

    // Apply pass
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::SymbolPromotion s2spass;
    EXPECT_FALSE(s2spass.run(builder_opt, analysis_manager));
}

TEST(SymbolPromotionTest, Assign_Unsigned_Constant) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar udesc(types::PrimitiveType::UInt32);
    types::Scalar desc(types::PrimitiveType::Int32);
    builder.add_container("i", desc);
    auto sym = symbolic::symbol("i");

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);
    auto& zero_node = builder.add_constant(block, "0", udesc);
    auto& output_node = builder.add_access(block, "i");

    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});

    builder.add_computational_memlet(block, zero_node, tasklet, "_in", {}, udesc);
    builder.add_computational_memlet(block, tasklet, "_out", output_node, {}, udesc);

    // Apply pass
    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::SymbolPromotion s2spass;
    EXPECT_TRUE(s2spass.run(builder, analysis_manager));

    auto& trans = builder.subject().root().at(0).second;
    EXPECT_EQ(trans.assignments().size(), 1);
    EXPECT_TRUE(symbolic::eq(trans.assignments().at(symbolic::symbol("i")), symbolic::zero()));
}

TEST(SymbolPromotionTest, Assign_Unsigned_Cast_Input) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::Int32);
    types::Scalar udesc(types::PrimitiveType::UInt32);
    builder.add_container("i", desc);
    auto sym = symbolic::symbol("i");

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);
    auto& zero_node = builder.add_constant(block, "0", desc);
    auto& output_node = builder.add_access(block, "i");

    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});

    builder.add_memlet(block, zero_node, "void", tasklet, "_in", {}, udesc, {});
    builder.add_computational_memlet(block, tasklet, "_out", output_node, {});

    auto sdfg = builder.move();

    // Apply pass
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::SymbolPromotion s2spass;
    EXPECT_FALSE(s2spass.run(builder_opt, analysis_manager));
}

TEST(SymbolPromotionTest, Assign_Unsigned_Cast_Output) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::Int32);
    types::Scalar udesc(types::PrimitiveType::UInt32);
    builder.add_container("i", desc);
    auto sym = symbolic::symbol("i");

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);
    auto& zero_node = builder.add_constant(block, "0", desc);
    auto& output_node = builder.add_access(block, "i");

    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});

    builder.add_computational_memlet(block, zero_node, tasklet, "_in", {});
    builder.add_memlet(block, tasklet, "_out", output_node, "void", {}, udesc, {});

    auto sdfg = builder.move();

    // Apply pass
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::SymbolPromotion s2spass;
    EXPECT_FALSE(s2spass.run(builder_opt, analysis_manager));
}

TEST(SymbolPromotionTest, Assign_Unsigned_Tasklet) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::Int32);
    types::Scalar udesc(types::PrimitiveType::UInt32);
    builder.add_container("i", desc);
    auto sym = symbolic::symbol("i");

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);
    auto& one_node = builder.add_constant(block, "1", desc);
    auto& output_node = builder.add_access(block, "i");

    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::int_udiv, "_out", {"_in1", "_in2"});

    builder.add_memlet(block, one_node, "void", tasklet, "_in1", {}, udesc, {});
    builder.add_memlet(block, one_node, "void", tasklet, "_in2", {}, udesc, {});
    builder.add_memlet(block, tasklet, "_out", output_node, "void", {}, udesc, {});

    auto sdfg = builder.move();

    // Apply pass
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::SymbolPromotion s2spass;
    EXPECT_FALSE(s2spass.run(builder_opt, analysis_manager));
}

TEST(SymbolPromotionTest, Assign_FP) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::UInt32);
    types::Scalar fdesc(types::PrimitiveType::Float);
    builder.add_container("i", fdesc);

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);
    auto& zero_node = builder.add_constant(block, "0", desc);
    auto& output_node = builder.add_access(block, "i");

    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});

    builder.add_computational_memlet(block, zero_node, tasklet, "_in", {});
    builder.add_computational_memlet(block, tasklet, "_out", output_node, {});

    auto sdfg = builder.move();

    // Apply pass
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::SymbolPromotion s2spass;
    EXPECT_FALSE(s2spass.run(builder_opt, analysis_manager));
}


TEST(SymbolPromotionTest, Add_Signed) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::Int32);
    builder.add_container("i", desc);
    builder.add_container("j", desc);
    auto sym1 = symbolic::symbol("i");
    auto sym2 = symbolic::symbol("j");

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);
    auto& input_node1 = builder.add_access(block, "i");
    auto& input_node2 = builder.add_access(block, "j");
    auto& output_node = builder.add_access(block, "j");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::int_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, input_node1, tasklet, "_in1", {});
    builder.add_computational_memlet(block, input_node2, tasklet, "_in2", {});
    builder.add_computational_memlet(block, tasklet, "_out", output_node, {});

    auto sdfg = builder.move();

    // Apply pass
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::SymbolPromotion s2spass;
    s2spass.run(builder_opt, analysis_manager);
    sdfg = builder_opt.move();

    // Check result
    EXPECT_EQ(sdfg->root().size(), 2);
    auto child1 = sdfg->root().at(0);
    auto block1 = dynamic_cast<const structured_control_flow::Block*>(&child1.first);
    auto child2 = sdfg->root().at(1);
    auto block2 = dynamic_cast<const structured_control_flow::Block*>(&child2.first);

    EXPECT_NE(block1, nullptr);
    EXPECT_NE(block2, nullptr);

    EXPECT_EQ(block1->dataflow().nodes().size(), 0);
    EXPECT_EQ(child1.second.assignments().size(), 1);
    EXPECT_TRUE(SymEngine::eq(*child1.second.assignments().at(sym2), *symbolic::add(sym1, sym2)));

    EXPECT_EQ(block2->dataflow().nodes().size(), 0);
}

TEST(SymbolPromotionTest, Add_Signed_Constant) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::Int32);
    builder.add_container("i", desc);
    auto sym = symbolic::symbol("i");

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);
    auto& output_node = builder.add_access(block, "i");
    auto& zero_node = builder.add_constant(block, "0", desc);
    auto& one_node = builder.add_constant(block, "1", desc);
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::int_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, zero_node, tasklet, "_in1", {});
    builder.add_computational_memlet(block, one_node, tasklet, "_in2", {});
    builder.add_computational_memlet(block, tasklet, "_out", output_node, {});

    auto sdfg = builder.move();

    // Apply pass
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::SymbolPromotion s2spass;
    s2spass.run(builder_opt, analysis_manager);
    sdfg = builder_opt.move();

    // Check result
    EXPECT_EQ(sdfg->root().size(), 2);
    auto child1 = sdfg->root().at(0);
    auto block1 = dynamic_cast<const structured_control_flow::Block*>(&child1.first);
    auto child2 = sdfg->root().at(1);
    auto block2 = dynamic_cast<const structured_control_flow::Block*>(&child2.first);

    EXPECT_NE(block1, nullptr);
    EXPECT_NE(block2, nullptr);

    EXPECT_EQ(block1->dataflow().nodes().size(), 0);
    EXPECT_EQ(child1.second.assignments().size(), 1);
    EXPECT_TRUE(SymEngine::eq(*child1.second.assignments().at(sym), *symbolic::integer(1)));

    EXPECT_EQ(block2->dataflow().nodes().size(), 0);
    EXPECT_EQ(child2.second.assignments().size(), 0);
}

TEST(SymbolPromotionTest, Sub_Signed) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::Int32);
    builder.add_container("i", desc);
    builder.add_container("j", desc);
    auto sym1 = symbolic::symbol("i");
    auto sym2 = symbolic::symbol("j");

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);
    auto& input_node1 = builder.add_access(block, "i");
    auto& input_node2 = builder.add_access(block, "j");
    auto& output_node = builder.add_access(block, "j");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::int_sub, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, input_node1, tasklet, "_in1", {});
    builder.add_computational_memlet(block, input_node2, tasklet, "_in2", {});
    builder.add_computational_memlet(block, tasklet, "_out", output_node, {});

    auto sdfg = builder.move();

    // Apply pass
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::SymbolPromotion s2spass;
    s2spass.run(builder_opt, analysis_manager);
    sdfg = builder_opt.move();

    // Check result
    EXPECT_EQ(sdfg->root().size(), 2);
    auto child1 = sdfg->root().at(0);
    auto block1 = dynamic_cast<const structured_control_flow::Block*>(&child1.first);
    auto child2 = sdfg->root().at(1);
    auto block2 = dynamic_cast<const structured_control_flow::Block*>(&child2.first);

    EXPECT_NE(block1, nullptr);
    EXPECT_NE(block2, nullptr);

    EXPECT_EQ(block1->dataflow().nodes().size(), 0);
    EXPECT_EQ(child1.second.assignments().size(), 1);
    EXPECT_TRUE(SymEngine::eq(*child1.second.assignments().at(sym2), *symbolic::sub(sym1, sym2)));

    EXPECT_EQ(block2->dataflow().nodes().size(), 0);
}

TEST(SymbolPromotionTest, Sub_Signed_Constant) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::Int32);
    builder.add_container("i", desc);
    auto sym = symbolic::symbol("i");

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);
    auto& output_node = builder.add_access(block, "i");
    auto& zero_node = builder.add_constant(block, "0", desc);
    auto& one_node = builder.add_constant(block, "1", desc);
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::int_sub, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, zero_node, tasklet, "_in1", {});
    builder.add_computational_memlet(block, one_node, tasklet, "_in2", {});
    builder.add_computational_memlet(block, tasklet, "_out", output_node, {});

    auto sdfg = builder.move();

    // Apply pass
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::SymbolPromotion s2spass;
    s2spass.run(builder_opt, analysis_manager);
    sdfg = builder_opt.move();

    // Check result
    EXPECT_EQ(sdfg->root().size(), 2);
    auto child1 = sdfg->root().at(0);
    auto block1 = dynamic_cast<const structured_control_flow::Block*>(&child1.first);
    auto child2 = sdfg->root().at(1);
    auto block2 = dynamic_cast<const structured_control_flow::Block*>(&child2.first);

    EXPECT_NE(block1, nullptr);
    EXPECT_NE(block2, nullptr);

    EXPECT_EQ(block1->dataflow().nodes().size(), 0);
    EXPECT_EQ(child1.second.assignments().size(), 1);
    EXPECT_TRUE(SymEngine::eq(*child1.second.assignments().at(sym), *symbolic::integer(-1)));

    EXPECT_EQ(block2->dataflow().nodes().size(), 0);
    EXPECT_EQ(child2.second.assignments().size(), 0);
}

TEST(SymbolPromotionTest, Mul_Signed) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::Int32);
    builder.add_container("i", desc);
    builder.add_container("j", desc);
    auto sym1 = symbolic::symbol("i");
    auto sym2 = symbolic::symbol("j");

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);
    auto& input_node1 = builder.add_access(block, "i");
    auto& input_node2 = builder.add_access(block, "j");
    auto& output_node = builder.add_access(block, "j");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::int_mul, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, input_node1, tasklet, "_in1", {});
    builder.add_computational_memlet(block, input_node2, tasklet, "_in2", {});
    builder.add_computational_memlet(block, tasklet, "_out", output_node, {});

    auto sdfg = builder.move();

    // Apply pass
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::SymbolPromotion s2spass;
    s2spass.run(builder_opt, analysis_manager);
    sdfg = builder_opt.move();

    // Check result
    EXPECT_EQ(sdfg->root().size(), 2);
    auto child1 = sdfg->root().at(0);
    auto block1 = dynamic_cast<const structured_control_flow::Block*>(&child1.first);
    auto child2 = sdfg->root().at(1);
    auto block2 = dynamic_cast<const structured_control_flow::Block*>(&child2.first);

    EXPECT_NE(block1, nullptr);
    EXPECT_NE(block2, nullptr);

    EXPECT_EQ(block1->dataflow().nodes().size(), 0);
    EXPECT_EQ(child1.second.assignments().size(), 1);
    EXPECT_TRUE(SymEngine::eq(*child1.second.assignments().at(sym2), *symbolic::mul(sym1, sym2)));

    EXPECT_EQ(block2->dataflow().nodes().size(), 0);
}

TEST(SymbolPromotionTest, Mul_Signed_Constant) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::Int32);
    builder.add_container("i", desc);
    auto sym = symbolic::symbol("i");

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);
    auto& output_node = builder.add_access(block, "i");
    auto& zero_node = builder.add_constant(block, "0", desc);
    auto& one_node = builder.add_constant(block, "1", desc);
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::int_mul, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, zero_node, tasklet, "_in1", {});
    builder.add_computational_memlet(block, one_node, tasklet, "_in2", {});
    builder.add_computational_memlet(block, tasklet, "_out", output_node, {});

    auto sdfg = builder.move();

    // Apply pass
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::SymbolPromotion s2spass;
    s2spass.run(builder_opt, analysis_manager);
    sdfg = builder_opt.move();

    // Check result
    EXPECT_EQ(sdfg->root().size(), 2);
    auto child1 = sdfg->root().at(0);
    auto block1 = dynamic_cast<const structured_control_flow::Block*>(&child1.first);
    auto child2 = sdfg->root().at(1);
    auto block2 = dynamic_cast<const structured_control_flow::Block*>(&child2.first);

    EXPECT_NE(block1, nullptr);
    EXPECT_NE(block2, nullptr);

    EXPECT_EQ(block1->dataflow().nodes().size(), 0);
    EXPECT_EQ(child1.second.assignments().size(), 1);
    EXPECT_TRUE(SymEngine::eq(*child1.second.assignments().at(sym), *symbolic::integer(0)));

    EXPECT_EQ(block2->dataflow().nodes().size(), 0);
    EXPECT_EQ(child2.second.assignments().size(), 0);
}

TEST(SymbolPromotionTest, SHL_Constant) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::Int32);
    builder.add_container("i", desc);
    builder.add_container("j", desc);
    auto sym = symbolic::symbol("i");
    auto sym2 = symbolic::symbol("j");

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);
    auto& output_node = builder.add_access(block, "i");
    auto& input_node = builder.add_access(block, "j");
    auto& one_node = builder.add_constant(block, "1", desc);
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::int_shl, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, input_node, tasklet, "_in1", {});
    builder.add_computational_memlet(block, one_node, tasklet, "_in2", {});
    builder.add_computational_memlet(block, tasklet, "_out", output_node, {});

    auto sdfg = builder.move();

    // Apply pass
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::SymbolPromotion s2spass;
    s2spass.run(builder_opt, analysis_manager);
    sdfg = builder_opt.move();

    // Check result
    EXPECT_EQ(sdfg->root().size(), 2);
    auto child1 = sdfg->root().at(0);
    auto block1 = dynamic_cast<const structured_control_flow::Block*>(&child1.first);
    auto child2 = sdfg->root().at(1);
    auto block2 = dynamic_cast<const structured_control_flow::Block*>(&child2.first);

    EXPECT_NE(block1, nullptr);
    EXPECT_NE(block2, nullptr);

    auto rhs = symbolic::mul(sym2, symbolic::integer(2));
    EXPECT_EQ(block1->dataflow().nodes().size(), 0);
    EXPECT_EQ(child1.second.assignments().size(), 1);
    EXPECT_TRUE(SymEngine::eq(*child1.second.assignments().at(sym), *rhs));

    EXPECT_EQ(block2->dataflow().nodes().size(), 0);
    EXPECT_EQ(child2.second.assignments().size(), 0);
}

TEST(SymbolPromotionTest, SHL_NonConstant) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::Int32);
    builder.add_container("i", desc);
    builder.add_container("j", desc);
    builder.add_container("k", desc);

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);
    auto& output_node = builder.add_access(block, "i");
    auto& input_node = builder.add_access(block, "j");
    auto& input2_node = builder.add_access(block, "k");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::int_shl, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, input_node, tasklet, "_in1", {});
    builder.add_computational_memlet(block, input2_node, tasklet, "_in2", {});
    builder.add_computational_memlet(block, tasklet, "_out", output_node, {});

    auto sdfg = builder.move();

    // Apply pass
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::SymbolPromotion s2spass;
    EXPECT_FALSE(s2spass.run(builder_opt, analysis_manager));
}

TEST(SymbolPromotionTest, Div_Signed) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::Int32);
    builder.add_container("i", desc);
    builder.add_container("j", desc);
    builder.add_container("k", desc);
    auto sym_i = symbolic::symbol("i");
    auto sym_j = symbolic::symbol("j");
    auto sym_k = symbolic::symbol("k");

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);
    auto& input_node1 = builder.add_access(block, "j");
    auto& input_node2 = builder.add_access(block, "k");
    auto& output_node = builder.add_access(block, "i");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::int_sdiv, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, input_node1, tasklet, "_in1", {});
    builder.add_computational_memlet(block, input_node2, tasklet, "_in2", {});
    builder.add_computational_memlet(block, tasklet, "_out", output_node, {});

    auto sdfg = builder.move();

    // Apply pass
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::SymbolPromotion s2spass;
    EXPECT_TRUE(s2spass.run(builder_opt, analysis_manager));
    sdfg = builder_opt.move();

    // Check result - verify expression is j / k
    EXPECT_EQ(sdfg->root().size(), 2);
    auto child1 = sdfg->root().at(0);
    auto block1 = dynamic_cast<const structured_control_flow::Block*>(&child1.first);
    EXPECT_NE(block1, nullptr);
    EXPECT_EQ(block1->dataflow().nodes().size(), 0);
    EXPECT_EQ(child1.second.assignments().size(), 1);

    auto expected = symbolic::div(sym_j, sym_k);
    EXPECT_TRUE(SymEngine::eq(*child1.second.assignments().at(sym_i), *expected));
}

TEST(SymbolPromotionTest, Div_Signed_Constant) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::Int32);
    builder.add_container("i", desc);
    auto sym = symbolic::symbol("i");

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);
    auto& output_node = builder.add_access(block, "i");
    auto& eight_node = builder.add_constant(block, "8", desc);
    auto& two_node = builder.add_constant(block, "2", desc);
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::int_sdiv, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, eight_node, tasklet, "_in1", {});
    builder.add_computational_memlet(block, two_node, tasklet, "_in2", {});
    builder.add_computational_memlet(block, tasklet, "_out", output_node, {});

    auto sdfg = builder.move();

    // Apply pass
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::SymbolPromotion s2spass;
    EXPECT_TRUE(s2spass.run(builder_opt, analysis_manager));
    sdfg = builder_opt.move();

    // Check result - 8 / 2 = 4
    EXPECT_EQ(sdfg->root().size(), 2);
    auto child1 = sdfg->root().at(0);
    EXPECT_EQ(child1.second.assignments().size(), 1);
    EXPECT_TRUE(SymEngine::eq(*child1.second.assignments().at(sym), *symbolic::integer(4)));
}

TEST(SymbolPromotionTest, Rem_Signed) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::Int32);
    builder.add_container("i", desc);
    builder.add_container("j", desc);
    builder.add_container("k", desc);
    auto sym_i = symbolic::symbol("i");
    auto sym_j = symbolic::symbol("j");
    auto sym_k = symbolic::symbol("k");

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);
    auto& input_node1 = builder.add_access(block, "j");
    auto& input_node2 = builder.add_access(block, "k");
    auto& output_node = builder.add_access(block, "i");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::int_srem, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, input_node1, tasklet, "_in1", {});
    builder.add_computational_memlet(block, input_node2, tasklet, "_in2", {});
    builder.add_computational_memlet(block, tasklet, "_out", output_node, {});

    auto sdfg = builder.move();

    // Apply pass
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::SymbolPromotion s2spass;
    EXPECT_TRUE(s2spass.run(builder_opt, analysis_manager));
    sdfg = builder_opt.move();

    // Check result - verify expression is j % k
    EXPECT_EQ(sdfg->root().size(), 2);
    auto child1 = sdfg->root().at(0);
    auto block1 = dynamic_cast<const structured_control_flow::Block*>(&child1.first);
    EXPECT_NE(block1, nullptr);
    EXPECT_EQ(block1->dataflow().nodes().size(), 0);
    EXPECT_EQ(child1.second.assignments().size(), 1);

    auto expected = symbolic::mod(sym_j, sym_k);
    EXPECT_TRUE(SymEngine::eq(*child1.second.assignments().at(sym_i), *expected));
}

TEST(SymbolPromotionTest, Rem_Signed_Constant) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::Int32);
    builder.add_container("i", desc);
    auto sym = symbolic::symbol("i");

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);
    auto& output_node = builder.add_access(block, "i");
    auto& seven_node = builder.add_constant(block, "7", desc);
    auto& three_node = builder.add_constant(block, "3", desc);
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::int_srem, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, seven_node, tasklet, "_in1", {});
    builder.add_computational_memlet(block, three_node, tasklet, "_in2", {});
    builder.add_computational_memlet(block, tasklet, "_out", output_node, {});

    auto sdfg = builder.move();

    // Apply pass
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::SymbolPromotion s2spass;
    EXPECT_TRUE(s2spass.run(builder_opt, analysis_manager));
    sdfg = builder_opt.move();

    // Check result - 7 % 3 = 1
    EXPECT_EQ(sdfg->root().size(), 2);
    auto child1 = sdfg->root().at(0);
    EXPECT_EQ(child1.second.assignments().size(), 1);
    EXPECT_TRUE(SymEngine::eq(*child1.second.assignments().at(sym), *symbolic::integer(1)));
}

TEST(SymbolPromotionTest, Min_Signed) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::Int32);
    builder.add_container("i", desc);
    builder.add_container("j", desc);
    builder.add_container("k", desc);
    auto sym_i = symbolic::symbol("i");
    auto sym_j = symbolic::symbol("j");
    auto sym_k = symbolic::symbol("k");

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);
    auto& input_node1 = builder.add_access(block, "j");
    auto& input_node2 = builder.add_access(block, "k");
    auto& output_node = builder.add_access(block, "i");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::int_smin, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, input_node1, tasklet, "_in1", {});
    builder.add_computational_memlet(block, input_node2, tasklet, "_in2", {});
    builder.add_computational_memlet(block, tasklet, "_out", output_node, {});

    auto sdfg = builder.move();

    // Apply pass
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::SymbolPromotion s2spass;
    EXPECT_TRUE(s2spass.run(builder_opt, analysis_manager));
    sdfg = builder_opt.move();

    // Check result - verify expression is min(j, k)
    EXPECT_EQ(sdfg->root().size(), 2);
    auto child1 = sdfg->root().at(0);
    auto block1 = dynamic_cast<const structured_control_flow::Block*>(&child1.first);
    EXPECT_NE(block1, nullptr);
    EXPECT_EQ(block1->dataflow().nodes().size(), 0);
    EXPECT_EQ(child1.second.assignments().size(), 1);

    auto expected = symbolic::min(sym_j, sym_k);
    EXPECT_TRUE(SymEngine::eq(*child1.second.assignments().at(sym_i), *expected));
}

TEST(SymbolPromotionTest, Min_Signed_Constant) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::Int32);
    builder.add_container("i", desc);
    auto sym = symbolic::symbol("i");

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);
    auto& output_node = builder.add_access(block, "i");
    auto& five_node = builder.add_constant(block, "5", desc);
    auto& three_node = builder.add_constant(block, "3", desc);
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::int_smin, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, five_node, tasklet, "_in1", {});
    builder.add_computational_memlet(block, three_node, tasklet, "_in2", {});
    builder.add_computational_memlet(block, tasklet, "_out", output_node, {});

    auto sdfg = builder.move();

    // Apply pass
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::SymbolPromotion s2spass;
    EXPECT_TRUE(s2spass.run(builder_opt, analysis_manager));
    sdfg = builder_opt.move();

    // Check result - min(5, 3) = 3
    EXPECT_EQ(sdfg->root().size(), 2);
    auto child1 = sdfg->root().at(0);
    EXPECT_EQ(child1.second.assignments().size(), 1);
    EXPECT_TRUE(SymEngine::eq(*child1.second.assignments().at(sym), *symbolic::integer(3)));
}

TEST(SymbolPromotionTest, Max_Signed) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::Int32);
    builder.add_container("i", desc);
    builder.add_container("j", desc);
    builder.add_container("k", desc);
    auto sym_i = symbolic::symbol("i");
    auto sym_j = symbolic::symbol("j");
    auto sym_k = symbolic::symbol("k");

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);
    auto& input_node1 = builder.add_access(block, "j");
    auto& input_node2 = builder.add_access(block, "k");
    auto& output_node = builder.add_access(block, "i");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::int_smax, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, input_node1, tasklet, "_in1", {});
    builder.add_computational_memlet(block, input_node2, tasklet, "_in2", {});
    builder.add_computational_memlet(block, tasklet, "_out", output_node, {});

    auto sdfg = builder.move();

    // Apply pass
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::SymbolPromotion s2spass;
    EXPECT_TRUE(s2spass.run(builder_opt, analysis_manager));
    sdfg = builder_opt.move();

    // Check result - verify expression is max(j, k)
    EXPECT_EQ(sdfg->root().size(), 2);
    auto child1 = sdfg->root().at(0);
    auto block1 = dynamic_cast<const structured_control_flow::Block*>(&child1.first);
    EXPECT_NE(block1, nullptr);
    EXPECT_EQ(block1->dataflow().nodes().size(), 0);
    EXPECT_EQ(child1.second.assignments().size(), 1);

    auto expected = symbolic::max(sym_j, sym_k);
    EXPECT_TRUE(SymEngine::eq(*child1.second.assignments().at(sym_i), *expected));
}

TEST(SymbolPromotionTest, Max_Signed_Constant) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::Int32);
    builder.add_container("i", desc);
    auto sym = symbolic::symbol("i");

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);
    auto& output_node = builder.add_access(block, "i");
    auto& five_node = builder.add_constant(block, "5", desc);
    auto& three_node = builder.add_constant(block, "3", desc);
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::int_smax, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, five_node, tasklet, "_in1", {});
    builder.add_computational_memlet(block, three_node, tasklet, "_in2", {});
    builder.add_computational_memlet(block, tasklet, "_out", output_node, {});

    auto sdfg = builder.move();

    // Apply pass
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::SymbolPromotion s2spass;
    EXPECT_TRUE(s2spass.run(builder_opt, analysis_manager));
    sdfg = builder_opt.move();

    // Check result - max(5, 3) = 5
    EXPECT_EQ(sdfg->root().size(), 2);
    auto child1 = sdfg->root().at(0);
    EXPECT_EQ(child1.second.assignments().size(), 1);
    EXPECT_TRUE(SymEngine::eq(*child1.second.assignments().at(sym), *symbolic::integer(5)));
}

TEST(SymbolPromotionTest, Abs_Signed) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::Int32);
    builder.add_container("i", desc);
    builder.add_container("j", desc);
    auto sym_i = symbolic::symbol("i");
    auto sym_j = symbolic::symbol("j");

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);
    auto& input_node = builder.add_access(block, "j");
    auto& output_node = builder.add_access(block, "i");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::int_abs, "_out", {"_in"});
    builder.add_computational_memlet(block, input_node, tasklet, "_in", {});
    builder.add_computational_memlet(block, tasklet, "_out", output_node, {});

    auto sdfg = builder.move();

    // Apply pass
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::SymbolPromotion s2spass;
    EXPECT_TRUE(s2spass.run(builder_opt, analysis_manager));
    sdfg = builder_opt.move();

    // Check result - verify expression is abs(j)
    EXPECT_EQ(sdfg->root().size(), 2);
    auto child1 = sdfg->root().at(0);
    auto block1 = dynamic_cast<const structured_control_flow::Block*>(&child1.first);
    EXPECT_NE(block1, nullptr);
    EXPECT_EQ(block1->dataflow().nodes().size(), 0);
    EXPECT_EQ(child1.second.assignments().size(), 1);

    auto expected = symbolic::abs(sym_j);
    EXPECT_TRUE(SymEngine::eq(*child1.second.assignments().at(sym_i), *expected));
}

TEST(SymbolPromotionTest, Abs_Signed_Constant) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::Int32);
    builder.add_container("i", desc);
    auto sym = symbolic::symbol("i");

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);
    auto& output_node = builder.add_access(block, "i");
    auto& neg_five_node = builder.add_constant(block, "-5", desc);
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::int_abs, "_out", {"_in"});
    builder.add_computational_memlet(block, neg_five_node, tasklet, "_in", {});
    builder.add_computational_memlet(block, tasklet, "_out", output_node, {});

    auto sdfg = builder.move();

    // Apply pass
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::SymbolPromotion s2spass;
    EXPECT_TRUE(s2spass.run(builder_opt, analysis_manager));
    sdfg = builder_opt.move();

    // Check result - abs(-5) = 5
    EXPECT_EQ(sdfg->root().size(), 2);
    auto child1 = sdfg->root().at(0);
    EXPECT_EQ(child1.second.assignments().size(), 1);
    EXPECT_TRUE(SymEngine::
                    eq(*child1.second.assignments().at(sym), *symbolic::simplify(symbolic::abs(symbolic::integer(-5))))
    );
}

TEST(SymbolPromotionTest, ASHR_Constant) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::Int32);
    builder.add_container("i", desc);
    builder.add_container("j", desc);
    auto sym_i = symbolic::symbol("i");
    auto sym_j = symbolic::symbol("j");

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);
    auto& output_node = builder.add_access(block, "i");
    auto& input_node = builder.add_access(block, "j");
    auto& two_node = builder.add_constant(block, "2", desc);
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::int_ashr, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, input_node, tasklet, "_in1", {});
    builder.add_computational_memlet(block, two_node, tasklet, "_in2", {});
    builder.add_computational_memlet(block, tasklet, "_out", output_node, {});

    auto sdfg = builder.move();

    // Apply pass
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::SymbolPromotion s2spass;
    EXPECT_TRUE(s2spass.run(builder_opt, analysis_manager));
    sdfg = builder_opt.move();

    // Check result - verify expression is j / 2^2 = j / 4
    EXPECT_EQ(sdfg->root().size(), 2);
    auto child1 = sdfg->root().at(0);
    auto block1 = dynamic_cast<const structured_control_flow::Block*>(&child1.first);
    EXPECT_NE(block1, nullptr);
    EXPECT_EQ(block1->dataflow().nodes().size(), 0);
    EXPECT_EQ(child1.second.assignments().size(), 1);

    auto expected = symbolic::div(sym_j, symbolic::integer(4));
    EXPECT_TRUE(SymEngine::eq(*child1.second.assignments().at(sym_i), *expected));
}

TEST(SymbolPromotionTest, ASHR_Constant_Literal) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::Int32);
    builder.add_container("i", desc);
    auto sym = symbolic::symbol("i");

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);
    auto& output_node = builder.add_access(block, "i");
    auto& eight_node = builder.add_constant(block, "8", desc);
    auto& two_node = builder.add_constant(block, "2", desc);
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::int_ashr, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, eight_node, tasklet, "_in1", {});
    builder.add_computational_memlet(block, two_node, tasklet, "_in2", {});
    builder.add_computational_memlet(block, tasklet, "_out", output_node, {});

    auto sdfg = builder.move();

    // Apply pass
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::SymbolPromotion s2spass;
    EXPECT_TRUE(s2spass.run(builder_opt, analysis_manager));
    sdfg = builder_opt.move();

    // Check result - 8 >> 2 = 2
    EXPECT_EQ(sdfg->root().size(), 2);
    auto child1 = sdfg->root().at(0);
    EXPECT_EQ(child1.second.assignments().size(), 1);
    EXPECT_TRUE(SymEngine::eq(*child1.second.assignments().at(sym), *symbolic::integer(2)));
}

TEST(SymbolPromotionTest, And_Bool) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::Bool);
    builder.add_container("i", desc);
    builder.add_container("j", desc);
    builder.add_container("k", desc);
    auto sym_i = symbolic::symbol("i");
    auto sym_j = symbolic::symbol("j");
    auto sym_k = symbolic::symbol("k");

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);
    auto& input_node1 = builder.add_access(block, "j");
    auto& input_node2 = builder.add_access(block, "k");
    auto& output_node = builder.add_access(block, "i");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::int_and, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, input_node1, tasklet, "_in1", {});
    builder.add_computational_memlet(block, input_node2, tasklet, "_in2", {});
    builder.add_computational_memlet(block, tasklet, "_out", output_node, {});

    auto sdfg = builder.move();

    // Apply pass
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::SymbolPromotion s2spass;
    EXPECT_TRUE(s2spass.run(builder_opt, analysis_manager));
    sdfg = builder_opt.move();

    // Check result - verify expression is (j == true) & (k == true)
    EXPECT_EQ(sdfg->root().size(), 2);
    auto child1 = sdfg->root().at(0);
    auto block1 = dynamic_cast<const structured_control_flow::Block*>(&child1.first);
    EXPECT_NE(block1, nullptr);
    EXPECT_EQ(block1->dataflow().nodes().size(), 0);
    EXPECT_EQ(child1.second.assignments().size(), 1);

    auto op1_is_true = symbolic::Eq(sym_j, symbolic::__true__());
    auto op2_is_true = symbolic::Eq(sym_k, symbolic::__true__());
    auto expected = symbolic::And(op1_is_true, op2_is_true);
    EXPECT_TRUE(SymEngine::eq(*child1.second.assignments().at(sym_i), *expected));
}

TEST(SymbolPromotionTest, Or_Bool) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::Bool);
    builder.add_container("i", desc);
    builder.add_container("j", desc);
    builder.add_container("k", desc);
    auto sym_i = symbolic::symbol("i");
    auto sym_j = symbolic::symbol("j");
    auto sym_k = symbolic::symbol("k");

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);
    auto& input_node1 = builder.add_access(block, "j");
    auto& input_node2 = builder.add_access(block, "k");
    auto& output_node = builder.add_access(block, "i");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::int_or, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, input_node1, tasklet, "_in1", {});
    builder.add_computational_memlet(block, input_node2, tasklet, "_in2", {});
    builder.add_computational_memlet(block, tasklet, "_out", output_node, {});

    auto sdfg = builder.move();

    // Apply pass
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::SymbolPromotion s2spass;
    EXPECT_TRUE(s2spass.run(builder_opt, analysis_manager));
    sdfg = builder_opt.move();

    // Check result - verify expression is (j == true) | (k == true)
    EXPECT_EQ(sdfg->root().size(), 2);
    auto child1 = sdfg->root().at(0);
    auto block1 = dynamic_cast<const structured_control_flow::Block*>(&child1.first);
    EXPECT_NE(block1, nullptr);
    EXPECT_EQ(block1->dataflow().nodes().size(), 0);
    EXPECT_EQ(child1.second.assignments().size(), 1);

    auto op1_is_true = symbolic::Eq(sym_j, symbolic::__true__());
    auto op2_is_true = symbolic::Eq(sym_k, symbolic::__true__());
    auto expected = symbolic::Or(op1_is_true, op2_is_true);
    EXPECT_TRUE(SymEngine::eq(*child1.second.assignments().at(sym_i), *expected));
}

TEST(SymbolPromotionTest, Xor_Bool) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::Bool);
    builder.add_container("i", desc);
    builder.add_container("j", desc);
    builder.add_container("k", desc);
    auto sym_i = symbolic::symbol("i");
    auto sym_j = symbolic::symbol("j");
    auto sym_k = symbolic::symbol("k");

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);
    auto& input_node1 = builder.add_access(block, "j");
    auto& input_node2 = builder.add_access(block, "k");
    auto& output_node = builder.add_access(block, "i");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::int_xor, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, input_node1, tasklet, "_in1", {});
    builder.add_computational_memlet(block, input_node2, tasklet, "_in2", {});
    builder.add_computational_memlet(block, tasklet, "_out", output_node, {});

    auto sdfg = builder.move();

    // Apply pass
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::SymbolPromotion s2spass;
    EXPECT_TRUE(s2spass.run(builder_opt, analysis_manager));
    sdfg = builder_opt.move();

    // Check result - verify expression is ((j == true) | (k == true)) & !((j == true) & (k == true))
    EXPECT_EQ(sdfg->root().size(), 2);
    auto child1 = sdfg->root().at(0);
    auto block1 = dynamic_cast<const structured_control_flow::Block*>(&child1.first);
    EXPECT_NE(block1, nullptr);
    EXPECT_EQ(block1->dataflow().nodes().size(), 0);
    EXPECT_EQ(child1.second.assignments().size(), 1);

    auto op1_is_true = symbolic::Eq(sym_j, symbolic::__true__());
    auto op2_is_true = symbolic::Eq(sym_k, symbolic::__true__());
    auto expected =
        symbolic::And(symbolic::Or(op1_is_true, op2_is_true), symbolic::Not(symbolic::And(op1_is_true, op2_is_true)));
    EXPECT_TRUE(SymEngine::eq(*child1.second.assignments().at(sym_i), *expected));
}

TEST(SymbolPromotionTest, ZExt_i32_to_i64) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc32(types::PrimitiveType::UInt32);
    types::Scalar desc64(types::PrimitiveType::UInt64);
    builder.add_container("i", desc64);
    builder.add_container("j", desc32);
    auto sym_i = symbolic::symbol("i");
    auto sym_j = symbolic::symbol("j");

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);
    auto& input_node = builder.add_access(block, "j");
    auto& output_node = builder.add_access(block, "i");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, input_node, tasklet, "_in", {});
    builder.add_computational_memlet(block, tasklet, "_out", output_node, {});

    auto sdfg = builder.move();

    // Apply pass
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::SymbolPromotion s2spass;
    EXPECT_TRUE(s2spass.run(builder_opt, analysis_manager));
    sdfg = builder_opt.move();

    // Check result - verify expression is zext_i64(j)
    EXPECT_EQ(sdfg->root().size(), 2);
    auto child1 = sdfg->root().at(0);
    auto block1 = dynamic_cast<const structured_control_flow::Block*>(&child1.first);
    EXPECT_NE(block1, nullptr);
    EXPECT_EQ(block1->dataflow().nodes().size(), 0);
    EXPECT_EQ(child1.second.assignments().size(), 1);

    auto expected = symbolic::zext_i64(SymEngine::rcp_static_cast<const SymEngine::Symbol>(sym_j));
    EXPECT_TRUE(SymEngine::eq(*child1.second.assignments().at(sym_i), *expected));
}

TEST(SymbolPromotionTest, Trunc_i64_to_i32) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc32(types::PrimitiveType::UInt32);
    types::Scalar desc64(types::PrimitiveType::UInt64);
    builder.add_container("i", desc32);
    builder.add_container("j", desc64);
    auto sym_i = symbolic::symbol("i");
    auto sym_j = symbolic::symbol("j");

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);
    auto& input_node = builder.add_access(block, "j");
    auto& output_node = builder.add_access(block, "i");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, input_node, tasklet, "_in", {});
    builder.add_computational_memlet(block, tasklet, "_out", output_node, {});

    auto sdfg = builder.move();

    // Apply pass
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::SymbolPromotion s2spass;
    EXPECT_TRUE(s2spass.run(builder_opt, analysis_manager));
    sdfg = builder_opt.move();

    // Check result - verify expression is trunc_i32(j)
    EXPECT_EQ(sdfg->root().size(), 2);
    auto child1 = sdfg->root().at(0);
    auto block1 = dynamic_cast<const structured_control_flow::Block*>(&child1.first);
    EXPECT_NE(block1, nullptr);
    EXPECT_EQ(block1->dataflow().nodes().size(), 0);
    EXPECT_EQ(child1.second.assignments().size(), 1);

    auto expected = symbolic::trunc_i32(SymEngine::rcp_static_cast<const SymEngine::Symbol>(sym_j));
    EXPECT_TRUE(SymEngine::eq(*child1.second.assignments().at(sym_i), *expected));
}

TEST(SymbolPromotionTest, Signed_Int_neg_1_xor) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::Int32);
    builder.add_container("i", desc);
    builder.add_container("j", desc);
    auto sym_i = symbolic::symbol("i");
    auto sym_j = symbolic::symbol("j");

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);
    auto& input_node = builder.add_access(block, "j");
    auto& output_node = builder.add_access(block, "i");
    auto& constant = builder.add_constant(block, "-1", desc);
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::int_xor, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, input_node, tasklet, "_in1", {});
    builder.add_computational_memlet(block, constant, tasklet, "_in2", {});
    builder.add_computational_memlet(block, tasklet, "_out", output_node, {});

    // Apply pass
    auto& sdfg = builder.subject();
    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::SymbolPromotion s2spass;
    EXPECT_TRUE(s2spass.run(builder, analysis_manager));

    // Check result - verify expression is - sym_j - 1
    EXPECT_EQ(sdfg.root().size(), 2);
    auto child1 = sdfg.root().at(0);
    auto block1 = dynamic_cast<const structured_control_flow::Block*>(&child1.first);
    EXPECT_NE(block1, nullptr);
    EXPECT_EQ(block1->dataflow().nodes().size(), 0);
    EXPECT_EQ(child1.second.assignments().size(), 1);

    auto expected = symbolic::add(symbolic::mul(symbolic::integer(-1), sym_j), symbolic::integer(-1));
    EXPECT_TRUE(SymEngine::eq(*child1.second.assignments().at(sym_i), *expected));
}
