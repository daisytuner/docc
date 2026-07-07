#include "sdfg/passes/structured_control_flow/condition_elimination.h"

#include <gtest/gtest.h>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/symbolic/symbolic.h"

using namespace sdfg;

TEST(ConditionEliminationTest, Basic) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);

    types::Pointer opaque_desc;
    builder.add_container("A", opaque_desc, true);
    builder.add_container("B", opaque_desc, true);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);

    // Define ifelse
    auto& ifelse = builder.add_if_else(root);
    auto& if_body = builder.add_case(ifelse, symbolic::Lt(symbolic::zero(), symbolic::symbol("N")));

    // Define loop
    auto bound = symbolic::symbol("N");
    auto indvar = symbolic::symbol("i");
    auto init = symbolic::zero();
    auto condition = symbolic::Lt(indvar, bound);
    auto update = symbolic::add(indvar, symbolic::one());

    auto& loop = builder.add_for(if_body, indvar, condition, init, update);
    auto& body = loop.root();

    // Add computation
    auto& block = builder.add_block(body);
    auto& a = builder.add_access(block, "A");
    auto& b = builder.add_access(block, "B");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, a, tasklet, "_in", {indvar}, desc);
    builder.add_computational_memlet(block, tasklet, "_out", b, {indvar}, desc);

    // Analysis
    auto sdfg_opt = builder.move();
    builder::StructuredSDFGBuilder builder_opt(sdfg_opt);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::ConditionEliminationPass condition_elimination_pass;
    EXPECT_TRUE(condition_elimination_pass.run(builder_opt, analysis_manager));

    auto& new_sdfg = builder_opt.subject();

    // Check
    auto& root_new = new_sdfg.root();
    EXPECT_EQ(root_new.size(), 1);
    auto& seq_before = dyn_cast<structured_control_flow::Sequence&>(root_new.at(0).first);
    auto loop_new = dyn_cast<structured_control_flow::StructuredLoop*>(&seq_before.at(0).first);
    EXPECT_TRUE(loop_new != nullptr);
}

TEST(ConditionEliminationTest, Basic_WithAssignmentsAtLoop) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);

    types::Pointer opaque_desc;
    builder.add_container("A", opaque_desc, true);
    builder.add_container("B", opaque_desc, true);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);

    // Define ifelse
    auto& ifelse = builder.add_if_else(root);
    auto& if_body = builder.add_case(ifelse, symbolic::Lt(symbolic::zero(), symbolic::symbol("N")));

    // Define loop
    auto bound = symbolic::symbol("N");
    auto indvar = symbolic::symbol("i");
    auto init = symbolic::zero();
    auto condition = symbolic::Lt(indvar, bound);
    auto update = symbolic::add(indvar, symbolic::one());

    auto& loop = builder.add_for(if_body, indvar, condition, init, update, {{indvar, symbolic::zero()}});
    auto& body = loop.root();

    // Add computation
    auto& block = builder.add_block(body);
    auto& a = builder.add_access(block, "A");
    auto& b = builder.add_access(block, "B");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, a, tasklet, "_in", {indvar}, desc);
    builder.add_computational_memlet(block, tasklet, "_out", b, {indvar}, desc);

    // Analysis
    auto sdfg_opt = builder.move();
    builder::StructuredSDFGBuilder builder_opt(sdfg_opt);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::ConditionEliminationPass condition_elimination_pass;
    EXPECT_TRUE(condition_elimination_pass.run(builder_opt, analysis_manager));

    auto& new_sdfg = builder_opt.subject();

    // Check
    auto& root_new = new_sdfg.root();
    EXPECT_EQ(root_new.size(), 1);
    auto& seq_before = dyn_cast<structured_control_flow::Sequence&>(root_new.at(0).first);
    auto loop_new = dyn_cast<structured_control_flow::StructuredLoop*>(&seq_before.at(0).first);
    EXPECT_TRUE(loop_new != nullptr);

    auto& transition = seq_before.at(0).second;
    EXPECT_EQ(transition.assignments().size(), 1);
    EXPECT_TRUE(symbolic::eq(transition.assignments().at(indvar), symbolic::zero()));
}

TEST(ConditionEliminationTest, Basic_WithAssignmentsAtIfElse) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);

    types::Pointer opaque_desc;
    builder.add_container("A", opaque_desc, true);
    builder.add_container("B", opaque_desc, true);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);

    // Define ifelse
    auto& ifelse = builder.add_if_else(root, {{symbolic::symbol("N"), symbolic::zero()}});
    auto& if_body = builder.add_case(ifelse, symbolic::Lt(symbolic::zero(), symbolic::symbol("N")));

    // Define loop
    auto bound = symbolic::symbol("N");
    auto indvar = symbolic::symbol("i");
    auto init = symbolic::zero();
    auto condition = symbolic::Lt(indvar, bound);
    auto update = symbolic::add(indvar, symbolic::one());

    auto& loop = builder.add_for(if_body, indvar, condition, init, update);
    auto& body = loop.root();

    // Add computation
    auto& block = builder.add_block(body);
    auto& a = builder.add_access(block, "A");
    auto& b = builder.add_access(block, "B");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, a, tasklet, "_in", {indvar}, desc);
    builder.add_computational_memlet(block, tasklet, "_out", b, {indvar}, desc);

    // Analysis
    auto sdfg_opt = builder.move();
    builder::StructuredSDFGBuilder builder_opt(sdfg_opt);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::ConditionEliminationPass condition_elimination_pass;
    EXPECT_TRUE(condition_elimination_pass.run(builder_opt, analysis_manager));

    auto& new_sdfg = builder_opt.subject();

    // Check
    auto& root_new = new_sdfg.root();
    EXPECT_EQ(root_new.size(), 1);
    auto& seq_before = dyn_cast<structured_control_flow::Sequence&>(root_new.at(0).first);
    auto loop_new = dyn_cast<structured_control_flow::StructuredLoop*>(&seq_before.at(0).first);
    EXPECT_TRUE(loop_new != nullptr);

    auto& transition = root_new.at(0).second;
    EXPECT_EQ(transition.assignments().size(), 1);
    EXPECT_TRUE(symbolic::eq(transition.assignments().at(symbolic::symbol("N")), symbolic::zero()));
}

TEST(ConditionEliminationTest, IndvarReadAfterLoop) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);

    types::Pointer opaque_desc;
    builder.add_container("A", opaque_desc, true);
    builder.add_container("B", opaque_desc, true);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);

    // Define ifelse
    auto& ifelse = builder.add_if_else(root, {{symbolic::symbol("N"), symbolic::symbol("i")}});
    auto& if_body = builder.add_case(ifelse, symbolic::Lt(symbolic::zero(), symbolic::symbol("N")));

    // Define loop
    auto bound = symbolic::symbol("N");
    auto indvar = symbolic::symbol("i");
    auto init = symbolic::zero();
    auto condition = symbolic::Lt(indvar, bound);
    auto update = symbolic::add(indvar, symbolic::one());

    auto& loop = builder.add_for(if_body, indvar, condition, init, update);
    auto& body = loop.root();

    // Add computation
    auto& block = builder.add_block(body);
    auto& a = builder.add_access(block, "A");
    auto& b = builder.add_access(block, "B");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, a, tasklet, "_in", {indvar}, desc);
    builder.add_computational_memlet(block, tasklet, "_out", b, {indvar}, desc);


    // Analysis
    auto sdfg_opt = builder.move();
    builder::StructuredSDFGBuilder builder_opt(sdfg_opt);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::ConditionEliminationPass condition_elimination_pass;
    EXPECT_FALSE(condition_elimination_pass.run(builder_opt, analysis_manager));
}
