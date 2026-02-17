#include "sdfg/passes/scheduler/polly_scheduler.h"

#include <gtest/gtest.h>
#include "sdfg/passes/scheduler/loop_scheduling_pass.h"

using namespace sdfg;

TEST(PollySchedulerTest, SpatialProximity) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Array array_desc(base_desc, symbolic::integer(100));
    types::Pointer desc(array_desc);

    types::Pointer opaque_desc;
    builder.add_container("A", opaque_desc, true);
    builder.add_container("B", opaque_desc, true);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("M", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);

    // Define loop
    auto bound1 = symbolic::symbol("N");
    auto indvar1 = symbolic::symbol("i");
    auto init1 = symbolic::integer(0);
    auto condition1 = symbolic::Lt(indvar1, bound1);
    auto update1 = symbolic::add(indvar1, symbolic::integer(1));

    auto& loop1 = builder.add_for(root, indvar1, condition1, init1, update1);
    auto& body1 = loop1.root();

    auto bound2 = symbolic::symbol("M");
    auto indvar2 = symbolic::symbol("j");
    auto init2 = symbolic::integer(0);
    auto condition2 = symbolic::Lt(indvar2, bound2);
    auto update2 = symbolic::add(indvar2, symbolic::integer(1));

    auto& loop2 = builder.add_for(body1, indvar2, condition2, init2, update2);
    auto& body2 = loop2.root();

    // Add computation
    auto& block = builder.add_block(body2);
    auto& A_in = builder.add_access(block, "A");
    auto& B_out = builder.add_access(block, "B");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, A_in, tasklet, "_in", {indvar2, indvar1}, desc);
    builder.add_computational_memlet(block, tasklet, "_out", B_out, {indvar2, indvar1}, desc);

    // Apply
    analysis::AnalysisManager analysis_manager(builder.subject());

    passes::scheduler::register_polly_scheduler(false);

    passes::scheduler::LoopSchedulingPass loop_scheduling_pass({"polly"}, nullptr);
    loop_scheduling_pass.run(builder, analysis_manager);

    auto new_seq = dynamic_cast<structured_control_flow::Sequence*>(&builder.subject().root().at(0).first);
    ASSERT_TRUE(new_seq != nullptr);
    EXPECT_TRUE(new_seq->size() == 1);
    auto new_loop = dynamic_cast<structured_control_flow::Map*>(&new_seq->at(0).first);
    ASSERT_NE(new_loop, nullptr);
    EXPECT_TRUE(symbolic::eq(new_loop->indvar(), symbolic::symbol("c0")));

    auto new_inner_loop = dynamic_cast<structured_control_flow::Map*>(&new_loop->root().at(0).first);
    ASSERT_NE(new_inner_loop, nullptr);
    EXPECT_TRUE(symbolic::eq(new_inner_loop->indvar(), symbolic::symbol("c1")));

    auto& transition = new_inner_loop->root().at(0).second;
    EXPECT_EQ(transition.assignments().size(), 2);
    EXPECT_TRUE(symbolic::eq(transition.assignments().at(indvar1), symbolic::symbol("c1")));
    EXPECT_TRUE(symbolic::eq(transition.assignments().at(indvar2), symbolic::symbol("c0")));
}

TEST(PollySchedulerTest, OuterWhileWithSpatialProximity) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Array array_desc(base_desc, symbolic::integer(100));
    types::Pointer desc(array_desc);

    types::Pointer opaque_desc;
    builder.add_container("A", opaque_desc, true);
    builder.add_container("B", opaque_desc, true);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("M", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);

    auto& loop0 = builder.add_while(root);
    auto& body0 = loop0.root();

    // Define loop
    auto bound1 = symbolic::symbol("N");
    auto indvar1 = symbolic::symbol("i");
    auto init1 = symbolic::integer(0);
    auto condition1 = symbolic::Lt(indvar1, bound1);
    auto update1 = symbolic::add(indvar1, symbolic::integer(1));

    auto& loop1 = builder.add_for(body0, indvar1, condition1, init1, update1);
    auto& body1 = loop1.root();

    auto bound2 = symbolic::symbol("M");
    auto indvar2 = symbolic::symbol("j");
    auto init2 = symbolic::integer(0);
    auto condition2 = symbolic::Lt(indvar2, bound2);
    auto update2 = symbolic::add(indvar2, symbolic::integer(1));

    auto& loop2 = builder.add_for(body1, indvar2, condition2, init2, update2);
    auto& body2 = loop2.root();

    // Add computation
    auto& block = builder.add_block(body2);
    auto& A_in = builder.add_access(block, "A");
    auto& B_out = builder.add_access(block, "B");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, A_in, tasklet, "_in", {indvar2, indvar1}, desc);
    builder.add_computational_memlet(block, tasklet, "_out", B_out, {indvar2, indvar1}, desc);

    // Apply
    analysis::AnalysisManager analysis_manager(builder.subject());

    passes::scheduler::register_polly_scheduler(false);

    passes::scheduler::LoopSchedulingPass loop_scheduling_pass({"polly"}, nullptr);

    EXPECT_TRUE(loop_scheduling_pass.run(builder, analysis_manager));

    auto while_root = dynamic_cast<structured_control_flow::While*>(&builder.subject().root().at(0).first);
    ASSERT_NE(while_root, nullptr);

    auto new_seq = dynamic_cast<structured_control_flow::Sequence*>(&while_root->root().at(0).first);
    ASSERT_TRUE(new_seq != nullptr);
    EXPECT_TRUE(new_seq->size() == 1);
    auto new_loop = dynamic_cast<structured_control_flow::Map*>(&new_seq->at(0).first);
    ASSERT_NE(new_loop, nullptr);
    EXPECT_TRUE(symbolic::eq(new_loop->indvar(), symbolic::symbol("c0")));

    auto new_inner_loop = dynamic_cast<structured_control_flow::Map*>(&new_loop->root().at(0).first);
    ASSERT_NE(new_inner_loop, nullptr);
    EXPECT_TRUE(symbolic::eq(new_inner_loop->indvar(), symbolic::symbol("c1")));

    auto& transition = new_inner_loop->root().at(0).second;
    EXPECT_EQ(transition.assignments().size(), 2);
    EXPECT_TRUE(symbolic::eq(transition.assignments().at(indvar1), symbolic::symbol("c1")));
    EXPECT_TRUE(symbolic::eq(transition.assignments().at(indvar2), symbolic::symbol("c0")));
}
