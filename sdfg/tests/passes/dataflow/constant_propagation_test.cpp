#include <gtest/gtest.h>

#include <ostream>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/passes/dataflow/constant_propagation.h"

using namespace sdfg;

TEST(ConstantPropagationTest, Symbolic_Argument) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::Int32);
    builder.add_container("arg", desc, true);
    builder.add_container("i", desc);

    builder.add_block(builder.subject().root(), {{symbolic::symbol("i"), symbolic::symbol("arg")}});
    builder.add_block(builder.subject().root(), {{symbolic::symbol("i"), symbolic::symbol("arg")}});

    // Apply pass
    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::ConstantPropagation pass;
    EXPECT_TRUE(pass.run(builder, analysis_manager));

    // Check result
    auto& sdfg = builder.subject();

    auto& trans1 = sdfg.root().at(0).second;
    EXPECT_EQ(trans1.assignments().size(), 1);
    EXPECT_TRUE(symbolic::eq(trans1.assignments().at(symbolic::symbol("i")), symbolic::symbol("arg")));

    auto& trans2 = sdfg.root().at(1).second;
    EXPECT_EQ(trans2.assignments().size(), 0);
}

TEST(ConstantPropagationTest, Symbolic_Transient) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::Int32);
    builder.add_container("i", desc);
    builder.add_container("j", desc);

    builder.add_block(builder.subject().root(), {{symbolic::symbol("j"), symbolic::one()}});
    builder.add_block(builder.subject().root(), {{symbolic::symbol("i"), symbolic::symbol("j")}});
    builder.add_block(builder.subject().root(), {{symbolic::symbol("i"), symbolic::symbol("j")}});

    // Apply pass
    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::ConstantPropagation pass;
    EXPECT_TRUE(pass.run(builder, analysis_manager));

    // Check result
    auto& sdfg = builder.subject();

    auto& trans1 = sdfg.root().at(0).second;
    EXPECT_EQ(trans1.assignments().size(), 1);
    EXPECT_TRUE(symbolic::eq(trans1.assignments().at(symbolic::symbol("j")), symbolic::one()));

    auto& trans2 = sdfg.root().at(1).second;
    EXPECT_EQ(trans2.assignments().size(), 1);
    EXPECT_TRUE(symbolic::eq(trans2.assignments().at(symbolic::symbol("i")), symbolic::symbol("j")));

    auto& trans3 = sdfg.root().at(2).second;
    EXPECT_EQ(trans3.assignments().size(), 0);
}

TEST(ConstantPropagationTest, Dataflow_Argument) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Pointer opaque_ptr;
    types::Scalar desc(types::PrimitiveType::Int32);
    builder.add_container("p", opaque_ptr, true);
    builder.add_container("i", desc);

    auto& block1 = builder.add_block(builder.subject().root());
    {
        auto& in_node = builder.add_access(block1, "p");
        auto& out_node = builder.add_access(block1, "i");
        auto& tasklet = builder.add_tasklet(block1, data_flow::TaskletCode::assign, {"_out"}, {"_in"});
        builder.add_computational_memlet(block1, in_node, tasklet, "_in", {symbolic::zero()}, types::Pointer(desc));
        builder.add_computational_memlet(block1, tasklet, "_out", out_node, {}, desc);
    }

    auto& block2 = builder.add_block(builder.subject().root());
    {
        auto& in_node = builder.add_access(block2, "p");
        auto& out_node = builder.add_access(block2, "i");
        auto& tasklet = builder.add_tasklet(block2, data_flow::TaskletCode::assign, {"_out"}, {"_in"});
        builder.add_computational_memlet(block2, in_node, tasklet, "_in", {symbolic::zero()}, types::Pointer(desc));
        builder.add_computational_memlet(block2, tasklet, "_out", out_node, {}, desc);
    }

    // Apply pass
    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::ConstantPropagation pass;
    EXPECT_TRUE(pass.run(builder, analysis_manager));

    // Check result
    auto& sdfg = builder.subject();

    auto& child1 = dyn_cast<structured_control_flow::Block&>(sdfg.root().at(0).first);
    EXPECT_EQ(child1.dataflow().nodes().size(), 3);

    auto& child2 = dyn_cast<structured_control_flow::Block&>(sdfg.root().at(1).first);
    EXPECT_EQ(child2.dataflow().nodes().size(), 0);
}

TEST(ConstantPropagationTest, Dataflow_Transient) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Pointer opaque_ptr;
    types::Scalar desc(types::PrimitiveType::Int32);
    builder.add_container("p", opaque_ptr);
    builder.add_container("i", desc);

    auto& block0 = builder.add_block(builder.subject().root());
    {
        auto& zero_node = builder.add_constant(block0, "0", desc);
        auto& out_node = builder.add_access(block0, "p");
        auto& tasklet = builder.add_tasklet(block0, data_flow::TaskletCode::assign, {"_out"}, {"_in"});
        builder.add_computational_memlet(block0, zero_node, tasklet, "_in", {}, desc);
        builder.add_computational_memlet(block0, tasklet, "_out", out_node, {symbolic::zero()}, types::Pointer(desc));
    }

    auto& block1 = builder.add_block(builder.subject().root());
    {
        auto& in_node = builder.add_access(block1, "p");
        auto& out_node = builder.add_access(block1, "i");
        auto& tasklet = builder.add_tasklet(block1, data_flow::TaskletCode::assign, {"_out"}, {"_in"});
        builder.add_computational_memlet(block1, in_node, tasklet, "_in", {symbolic::zero()}, types::Pointer(desc));
        builder.add_computational_memlet(block1, tasklet, "_out", out_node, {}, desc);
    }

    auto& block2 = builder.add_block(builder.subject().root());
    {
        auto& in_node = builder.add_access(block2, "p");
        auto& out_node = builder.add_access(block2, "i");
        auto& tasklet = builder.add_tasklet(block2, data_flow::TaskletCode::assign, {"_out"}, {"_in"});
        builder.add_computational_memlet(block2, in_node, tasklet, "_in", {symbolic::zero()}, types::Pointer(desc));
        builder.add_computational_memlet(block2, tasklet, "_out", out_node, {}, desc);
    }

    // Apply pass
    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::ConstantPropagation pass;
    EXPECT_TRUE(pass.run(builder, analysis_manager));

    // Check result
    auto& sdfg = builder.subject();

    auto& child0 = dyn_cast<structured_control_flow::Block&>(sdfg.root().at(0).first);
    EXPECT_EQ(child0.dataflow().nodes().size(), 3);

    auto& child1 = dyn_cast<structured_control_flow::Block&>(sdfg.root().at(1).first);
    EXPECT_EQ(child1.dataflow().nodes().size(), 3);

    auto& child2 = dyn_cast<structured_control_flow::Block&>(sdfg.root().at(2).first);
    EXPECT_EQ(child2.dataflow().nodes().size(), 0);
}

TEST(ConstantPropagationTest, LoopInvariant_Symbolic) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::Int32);
    builder.add_container("i", desc);
    builder.add_container("j", desc);

    builder.add_block(builder.subject().root(), {{symbolic::symbol("j"), symbolic::one()}});
    builder.add_block(builder.subject().root(), {{symbolic::symbol("i"), symbolic::symbol("j")}});

    auto& loop_stmt = builder.add_while(builder.subject().root());
    builder.add_block(loop_stmt.root(), {{symbolic::symbol("i"), symbolic::symbol("j")}});

    // Apply pass
    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::ConstantPropagation pass;
    EXPECT_TRUE(pass.run(builder, analysis_manager));

    // Check result
    auto& sdfg = builder.subject();

    auto& trans1 = sdfg.root().at(0).second;
    EXPECT_EQ(trans1.assignments().size(), 1);
    EXPECT_TRUE(symbolic::eq(trans1.assignments().at(symbolic::symbol("j")), symbolic::one()));

    auto& trans2 = sdfg.root().at(1).second;
    EXPECT_EQ(trans2.assignments().size(), 1);
    EXPECT_TRUE(symbolic::eq(trans2.assignments().at(symbolic::symbol("i")), symbolic::symbol("j")));

    auto& trans3 = loop_stmt.root().at(0).second;
    EXPECT_EQ(trans3.assignments().size(), 0);
}

TEST(ConstantPropagationTest, Branchnvariant_Symbolic) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::Int32);
    builder.add_container("i", desc);
    builder.add_container("j", desc);

    builder.add_block(builder.subject().root(), {{symbolic::symbol("j"), symbolic::one()}});
    builder.add_block(builder.subject().root(), {{symbolic::symbol("i"), symbolic::symbol("j")}});

    auto& if_stmt = builder.add_if_else(builder.subject().root());
    auto& if_block = builder.add_case(if_stmt, symbolic::Eq(symbolic::symbol("i"), symbolic::zero()));
    builder.add_block(if_block, {{symbolic::symbol("i"), symbolic::symbol("j")}});

    // Apply pass
    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::ConstantPropagation pass;
    EXPECT_TRUE(pass.run(builder, analysis_manager));

    // Check result
    auto& sdfg = builder.subject();

    auto& trans1 = sdfg.root().at(0).second;
    EXPECT_EQ(trans1.assignments().size(), 1);
    EXPECT_TRUE(symbolic::eq(trans1.assignments().at(symbolic::symbol("j")), symbolic::one()));

    auto& trans2 = sdfg.root().at(1).second;
    EXPECT_EQ(trans2.assignments().size(), 1);
    EXPECT_TRUE(symbolic::eq(trans2.assignments().at(symbolic::symbol("i")), symbolic::symbol("j")));

    auto& trans3 = if_block.at(0).second;
    EXPECT_EQ(trans3.assignments().size(), 0);
}
