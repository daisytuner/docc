#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include "sdfg/analysis/loop_analysis.h"

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_nodes/stdlib/malloc.h"
#include "sdfg/structured_control_flow/structured_loop.h"

using namespace sdfg;

TEST(LoopAnalysisInfoTest, SingleLoop) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    builder.add_container("i", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("N", types::Scalar(types::PrimitiveType::Int32));

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    auto indvar = symbolic::symbol("i");
    auto update = symbolic::add(indvar, symbolic::one());
    auto condition = symbolic::Lt(indvar, symbolic::symbol("N"));
    auto init = symbolic::zero();
    auto& loop = builder.add_for(root, indvar, condition, init, update);

    analysis::AnalysisManager manager(sdfg);
    auto& loop_analysis = manager.get<analysis::LoopAnalysis>();

    auto info = loop_analysis.loop_info(&loop);
    EXPECT_EQ(info.num_loops, 1);
    EXPECT_EQ(info.num_maps, 0);
    EXPECT_EQ(info.num_fors, 1);
    EXPECT_EQ(info.num_whiles, 0);
    EXPECT_EQ(info.max_depth, 1);
    EXPECT_TRUE(info.is_perfectly_nested);
    EXPECT_FALSE(info.is_perfectly_parallel);
    EXPECT_EQ(info.loop_level, 0);
    EXPECT_EQ(info.map_stack_depth, 0);
}

TEST(LoopAnalysisInfoTest, NestedLoops) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    builder.add_container("i", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("j", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("N", types::Scalar(types::PrimitiveType::Int32));

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    auto indvar_i = symbolic::symbol("i");
    auto update_i = symbolic::add(indvar_i, symbolic::one());
    auto condition_i = symbolic::Lt(indvar_i, symbolic::symbol("N"));
    auto init_i = symbolic::zero();
    auto& loop_i = builder.add_for(root, indvar_i, condition_i, init_i, update_i);

    auto indvar_j = symbolic::symbol("j");
    auto update_j = symbolic::add(indvar_j, symbolic::one());
    auto condition_j = symbolic::Lt(indvar_j, symbolic::symbol("N"));
    auto init_j = symbolic::zero();
    auto& loop_j = builder.add_for(loop_i.root(), indvar_j, condition_j, init_j, update_j);

    analysis::AnalysisManager manager(sdfg);
    auto& loop_analysis = manager.get<analysis::LoopAnalysis>();

    auto info_j = loop_analysis.loop_info(&loop_j);
    EXPECT_EQ(info_j.num_loops, 1);
    EXPECT_EQ(info_j.num_maps, 0);
    EXPECT_EQ(info_j.num_fors, 1);
    EXPECT_EQ(info_j.num_whiles, 0);
    EXPECT_EQ(info_j.max_depth, 1);
    EXPECT_TRUE(info_j.is_perfectly_nested);
    EXPECT_FALSE(info_j.is_perfectly_parallel);
    EXPECT_FALSE(info_j.is_elementwise);
    EXPECT_EQ(info_j.loop_level, 1);
    EXPECT_EQ(info_j.map_stack_depth, 0);

    auto info_i = loop_analysis.loop_info(&loop_i);
    EXPECT_EQ(info_i.num_loops, 2);
    EXPECT_EQ(info_i.num_maps, 0);
    EXPECT_EQ(info_i.num_fors, 2);
    EXPECT_EQ(info_i.num_whiles, 0);
    EXPECT_EQ(info_i.max_depth, 2);
    EXPECT_TRUE(info_i.is_perfectly_nested);
    EXPECT_FALSE(info_i.is_perfectly_parallel);
    EXPECT_FALSE(info_i.is_elementwise);
    EXPECT_EQ(info_i.loop_level, 0);
    EXPECT_EQ(info_i.map_stack_depth, 0);
}

TEST(LoopAnalysisInfoTest, NestedLoopsWithExtraStatement) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    builder.add_container("i", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("j", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("N", types::Scalar(types::PrimitiveType::Int32));

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    auto indvar_i = symbolic::symbol("i");
    auto update_i = symbolic::add(indvar_i, symbolic::one());
    auto condition_i = symbolic::Lt(indvar_i, symbolic::symbol("N"));
    auto init_i = symbolic::zero();
    auto& loop_i = builder.add_for(root, indvar_i, condition_i, init_i, update_i);

    builder.add_block(loop_i.root());

    auto indvar_j = symbolic::symbol("j");
    auto update_j = symbolic::add(indvar_j, symbolic::one());
    auto condition_j = symbolic::Lt(indvar_j, symbolic::symbol("N"));
    auto init_j = symbolic::zero();
    auto& loop_j = builder.add_for(loop_i.root(), indvar_j, condition_j, init_j, update_j);

    analysis::AnalysisManager manager(sdfg);
    auto& loop_analysis = manager.get<analysis::LoopAnalysis>();

    auto info_j = loop_analysis.loop_info(&loop_j);
    EXPECT_EQ(info_j.num_loops, 1);
    EXPECT_EQ(info_j.num_maps, 0);
    EXPECT_EQ(info_j.num_fors, 1);
    EXPECT_EQ(info_j.num_whiles, 0);
    EXPECT_EQ(info_j.max_depth, 1);
    EXPECT_TRUE(info_j.is_perfectly_nested);
    EXPECT_FALSE(info_j.is_perfectly_parallel);
    EXPECT_FALSE(info_j.is_elementwise);
    EXPECT_EQ(info_j.loop_level, 1);
    EXPECT_EQ(info_j.map_stack_depth, 0);

    auto info_i = loop_analysis.loop_info(&loop_i);
    EXPECT_EQ(info_i.num_loops, 2);
    EXPECT_EQ(info_i.num_maps, 0);
    EXPECT_EQ(info_i.num_fors, 2);
    EXPECT_EQ(info_i.num_whiles, 0);
    EXPECT_EQ(info_i.max_depth, 2);
    EXPECT_FALSE(info_i.is_perfectly_nested);
    EXPECT_FALSE(info_i.is_perfectly_parallel);
    EXPECT_FALSE(info_i.is_elementwise);
    EXPECT_EQ(info_i.loop_level, 0);
    EXPECT_EQ(info_i.map_stack_depth, 0);
}

TEST(LoopAnalysisInfoTest, NestedLoopsWithInnerSequence) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    builder.add_container("i", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("j", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("N", types::Scalar(types::PrimitiveType::Int32));

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    auto indvar_i = symbolic::symbol("i");
    auto update_i = symbolic::add(indvar_i, symbolic::one());
    auto condition_i = symbolic::Lt(indvar_i, symbolic::symbol("N"));
    auto init_i = symbolic::zero();
    auto& loop_i = builder.add_for(root, indvar_i, condition_i, init_i, update_i);

    auto indvar_j = symbolic::symbol("j");
    auto update_j = symbolic::add(indvar_j, symbolic::one());
    auto condition_j = symbolic::Lt(indvar_j, symbolic::symbol("N"));
    auto init_j = symbolic::zero();
    auto& loop_j = builder.add_for(loop_i.root(), indvar_j, condition_j, init_j, update_j);

    builder.add_block(loop_j.root());

    analysis::AnalysisManager manager(sdfg);
    auto& loop_analysis = manager.get<analysis::LoopAnalysis>();

    auto info_j = loop_analysis.loop_info(&loop_j);
    EXPECT_EQ(info_j.num_loops, 1);
    EXPECT_EQ(info_j.num_maps, 0);
    EXPECT_EQ(info_j.num_fors, 1);
    EXPECT_EQ(info_j.num_whiles, 0);
    EXPECT_EQ(info_j.max_depth, 1);
    EXPECT_TRUE(info_j.is_perfectly_nested);
    EXPECT_FALSE(info_j.is_perfectly_parallel);
    EXPECT_FALSE(info_j.is_elementwise);

    auto info_i = loop_analysis.loop_info(&loop_i);
    EXPECT_EQ(info_i.num_loops, 2);
    EXPECT_EQ(info_i.num_maps, 0);
    EXPECT_EQ(info_i.num_fors, 2);
    EXPECT_EQ(info_i.num_whiles, 0);
    EXPECT_EQ(info_i.max_depth, 2);
    EXPECT_TRUE(info_i.is_perfectly_nested);
    EXPECT_FALSE(info_i.is_perfectly_parallel);
    EXPECT_FALSE(info_i.is_elementwise);
}

TEST(LoopAnalysisInfoTest, PerfectlyParallel) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    builder.add_container("i", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("N", types::Scalar(types::PrimitiveType::Int32));

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    auto indvar = symbolic::symbol("i");
    auto update = symbolic::add(indvar, symbolic::one());
    auto condition = symbolic::Lt(indvar, symbolic::symbol("N"));
    auto init = symbolic::zero();
    auto schedule = structured_control_flow::ScheduleType_Sequential::create();
    auto& loop = builder.add_map(root, indvar, condition, init, update, schedule);

    analysis::AnalysisManager manager(sdfg);
    auto& loop_analysis = manager.get<analysis::LoopAnalysis>();

    auto info = loop_analysis.loop_info(&loop);
    EXPECT_EQ(info.num_loops, 1);
    EXPECT_EQ(info.num_maps, 1);
    EXPECT_EQ(info.num_fors, 0);
    EXPECT_EQ(info.num_whiles, 0);
    EXPECT_EQ(info.max_depth, 1);
    EXPECT_TRUE(info.is_perfectly_nested);
    EXPECT_TRUE(info.is_perfectly_parallel);
    EXPECT_TRUE(info.is_elementwise);
    EXPECT_EQ(info.loop_level, 0);
    EXPECT_EQ(info.map_stack_depth, 1);
}

TEST(LoopAnalysisInfoTest, ElementWise) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    builder.add_container("i", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("N", types::Scalar(types::PrimitiveType::Int32));

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    auto indvar = symbolic::symbol("i");
    auto update = symbolic::add(indvar, symbolic::one());
    auto condition = symbolic::Lt(indvar, symbolic::symbol("N"));
    auto init = symbolic::zero();
    auto schedule = structured_control_flow::ScheduleType_Sequential::create();
    auto& loop = builder.add_map(root, indvar, condition, init, update, schedule);

    analysis::AnalysisManager manager(sdfg);
    auto& loop_analysis = manager.get<analysis::LoopAnalysis>();

    auto info = loop_analysis.loop_info(&loop);
    EXPECT_TRUE(info.is_elementwise);
}

TEST(LoopAnalysisInfoTest, NotElementWise_NotContiguous) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    builder.add_container("i", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("N", types::Scalar(types::PrimitiveType::Int32));

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    auto indvar = symbolic::symbol("i");
    // i + 2 (stride 2, not contiguous)
    auto update = symbolic::add(indvar, symbolic::integer(2));
    auto condition = symbolic::Lt(indvar, symbolic::symbol("N"));
    auto init = symbolic::zero();
    auto schedule = structured_control_flow::ScheduleType_Sequential::create();
    auto& loop = builder.add_map(root, indvar, condition, init, update, schedule);

    analysis::AnalysisManager manager(sdfg);
    auto& loop_analysis = manager.get<analysis::LoopAnalysis>();

    auto info = loop_analysis.loop_info(&loop);
    EXPECT_FALSE(info.is_elementwise);
}

TEST(LoopAnalysisInfoTest, MixedLoopTypes) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    builder.add_container("i", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("j", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("N", types::Scalar(types::PrimitiveType::Int32));

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Outer For
    auto indvar_i = symbolic::symbol("i");
    auto update_i = symbolic::add(indvar_i, symbolic::one());
    auto condition_i = symbolic::Lt(indvar_i, symbolic::symbol("N"));
    auto init_i = symbolic::zero();
    auto& loop_for = builder.add_for(root, indvar_i, condition_i, init_i, update_i);

    // Inner Map
    auto indvar_j = symbolic::symbol("j");
    auto update_j = symbolic::add(indvar_j, symbolic::one());
    auto condition_j = symbolic::Lt(indvar_j, symbolic::symbol("N"));
    auto init_j = symbolic::zero();
    auto schedule = structured_control_flow::ScheduleType_Sequential::create();
    auto& loop_map = builder.add_map(loop_for.root(), indvar_j, condition_j, init_j, update_j, schedule);

    // Inner While
    auto& loop_while = builder.add_while(loop_map.root());

    analysis::AnalysisManager manager(sdfg);
    auto& loop_analysis = manager.get<analysis::LoopAnalysis>();

    auto info = loop_analysis.loop_info(&loop_for);
    EXPECT_EQ(info.num_loops, 3);
    EXPECT_EQ(info.num_fors, 1);
    EXPECT_EQ(info.num_maps, 1);
    EXPECT_EQ(info.num_whiles, 1);
    EXPECT_EQ(info.max_depth, 3);
    EXPECT_TRUE(info.is_perfectly_nested);
    EXPECT_FALSE(info.is_perfectly_parallel);
    EXPECT_FALSE(info.is_elementwise);
}

TEST(LoopAnalysisInfoTest, NotPerfectlyParallel_MapFor) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    builder.add_container("i", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("j", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("N", types::Scalar(types::PrimitiveType::Int32));

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Outer Map
    auto indvar_i = symbolic::symbol("i");
    auto update_i = symbolic::add(indvar_i, symbolic::one());
    auto condition_i = symbolic::Lt(indvar_i, symbolic::symbol("N"));
    auto init_i = symbolic::zero();
    auto schedule = structured_control_flow::ScheduleType_Sequential::create();
    auto& loop_map = builder.add_map(root, indvar_i, condition_i, init_i, update_i, schedule);

    // Inner For
    auto indvar_j = symbolic::symbol("j");
    auto update_j = symbolic::add(indvar_j, symbolic::one());
    auto condition_j = symbolic::Lt(indvar_j, symbolic::symbol("N"));
    auto init_j = symbolic::zero();
    auto& loop_for = builder.add_for(loop_map.root(), indvar_j, condition_j, init_j, update_j);

    analysis::AnalysisManager manager(sdfg);
    auto& loop_analysis = manager.get<analysis::LoopAnalysis>();

    auto info_map = loop_analysis.loop_info(&loop_map);
    EXPECT_EQ(info_map.num_loops, 2);
    EXPECT_EQ(info_map.num_maps, 1);
    EXPECT_EQ(info_map.num_fors, 1);
    EXPECT_FALSE(info_map.is_perfectly_parallel);
    EXPECT_FALSE(info_map.is_elementwise);
    EXPECT_EQ(info_map.map_stack_depth, 1);
    EXPECT_EQ(info_map.loop_level, 0);
    EXPECT_EQ(info_map.map_stack_depth, 1);
}

TEST(LoopAnalysisInfoTest, NotElementWise_NotPerfectlyNested) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    builder.add_container("i", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("j", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("N", types::Scalar(types::PrimitiveType::Int32));

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Outer Map
    auto indvar_i = symbolic::symbol("i");
    auto update_i = symbolic::add(indvar_i, symbolic::one());
    auto condition_i = symbolic::Lt(indvar_i, symbolic::symbol("N"));
    auto init_i = symbolic::zero();
    auto schedule = structured_control_flow::ScheduleType_Sequential::create();
    auto& loop_map_outer = builder.add_map(root, indvar_i, condition_i, init_i, update_i, schedule);

    // Extra statement
    builder.add_block(loop_map_outer.root());

    // Inner Map
    auto indvar_j = symbolic::symbol("j");
    auto update_j = symbolic::add(indvar_j, symbolic::one());
    auto condition_j = symbolic::Lt(indvar_j, symbolic::symbol("N"));
    auto init_j = symbolic::zero();
    auto& loop_map_inner = builder.add_map(loop_map_outer.root(), indvar_j, condition_j, init_j, update_j, schedule);

    analysis::AnalysisManager manager(sdfg);
    auto& loop_analysis = manager.get<analysis::LoopAnalysis>();

    auto info = loop_analysis.loop_info(&loop_map_outer);
    EXPECT_FALSE(info.is_perfectly_nested);
    EXPECT_FALSE(info.is_elementwise);
    EXPECT_EQ(info.loop_level, 0);
    EXPECT_EQ(info.map_stack_depth, 1);

    auto info_inner = loop_analysis.loop_info(&loop_map_inner);
    EXPECT_TRUE(info_inner.is_perfectly_nested);
    EXPECT_EQ(info_inner.loop_level, 1);
    EXPECT_EQ(info_inner.map_stack_depth, 1);
}

TEST(LoopAnalysisInfoTest, MapStack2_of_3) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    builder.add_container("i", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("j", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("k", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("N", types::Scalar(types::PrimitiveType::Int32));

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Outer Map
    auto indvar_i = symbolic::symbol("i");
    auto update_i = symbolic::add(indvar_i, symbolic::one());
    auto condition_i = symbolic::Lt(indvar_i, symbolic::symbol("N"));
    auto init_i = symbolic::zero();
    auto schedule = structured_control_flow::ScheduleType_Sequential::create();
    auto& loop_map_outer = builder.add_map(root, indvar_i, condition_i, init_i, update_i, schedule);

    // Middle Map
    auto indvar_j = symbolic::symbol("j");
    auto update_j = symbolic::add(indvar_j, symbolic::one());
    auto condition_j = symbolic::Lt(indvar_j, symbolic::symbol("N"));
    auto init_j = symbolic::zero();
    auto& loop_map_middle = builder.add_map(loop_map_outer.root(), indvar_j, condition_j, init_j, update_j, schedule);

    // Inner For
    auto indvar_k = symbolic::symbol("k");
    auto update_k = symbolic::add(indvar_k, symbolic::one());
    auto condition_k = symbolic::Lt(indvar_k, symbolic::symbol("N"));
    auto init_k = symbolic::zero();
    auto& loop_for_inner = builder.add_for(loop_map_middle.root(), indvar_k, condition_k, init_k, update_k);

    analysis::AnalysisManager manager(sdfg);
    auto& loop_analysis = manager.get<analysis::LoopAnalysis>();

    auto info_outer = loop_analysis.loop_info(&loop_map_outer);
    EXPECT_FALSE(info_outer.is_perfectly_parallel);
    EXPECT_TRUE(info_outer.is_perfectly_nested);
    EXPECT_FALSE(info_outer.is_elementwise);
    EXPECT_EQ(info_outer.loop_level, 0);
    EXPECT_EQ(info_outer.map_stack_depth, 2);

    auto info_middle = loop_analysis.loop_info(&loop_map_middle);
    EXPECT_FALSE(info_middle.is_perfectly_parallel);
    EXPECT_TRUE(info_middle.is_perfectly_nested);
    EXPECT_FALSE(info_middle.is_elementwise);
    EXPECT_EQ(info_middle.loop_level, 1);
    EXPECT_EQ(info_middle.map_stack_depth, 1);

    auto info_inner = loop_analysis.loop_info(&loop_for_inner);
    EXPECT_FALSE(info_inner.is_perfectly_parallel);
    EXPECT_TRUE(info_inner.is_perfectly_nested);
    EXPECT_FALSE(info_inner.is_elementwise);
    EXPECT_EQ(info_inner.loop_level, 2);
    EXPECT_EQ(info_inner.map_stack_depth, 0);
}

TEST(LoopAnalysisInfoTest, SideEffectsInherited) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    builder.add_container("i", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("j", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("k", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("N", types::Scalar(types::PrimitiveType::Int32));
    types::Scalar scalar_type(types::PrimitiveType::Int32);
    types::Pointer ptr_type(scalar_type);
    builder.add_container("ptr", ptr_type);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Outer Map
    auto indvar_i = symbolic::symbol("i");
    auto update_i = symbolic::add(indvar_i, symbolic::one());
    auto condition_i = symbolic::Lt(indvar_i, symbolic::symbol("N"));
    auto init_i = symbolic::zero();
    auto schedule = structured_control_flow::ScheduleType_Sequential::create();
    auto& loop_map_outer = builder.add_map(root, indvar_i, condition_i, init_i, update_i, schedule);

    // Middle Map
    auto indvar_j = symbolic::symbol("j");
    auto update_j = symbolic::add(indvar_j, symbolic::one());
    auto condition_j = symbolic::Lt(indvar_j, symbolic::symbol("N"));
    auto init_j = symbolic::zero();
    auto& loop_map_middle = builder.add_map(loop_map_outer.root(), indvar_j, condition_j, init_j, update_j, schedule);

    // Inner For
    auto indvar_k = symbolic::symbol("k");
    auto update_k = symbolic::add(indvar_k, symbolic::one());
    auto condition_k = symbolic::Lt(indvar_k, symbolic::symbol("N"));
    auto init_k = symbolic::zero();
    auto& loop_map_inner = builder.add_map(loop_map_middle.root(), indvar_k, condition_k, init_k, update_k, schedule);

    auto& inner_block = builder.add_block(loop_map_inner.root());
    auto& ptr_acc = builder.add_access(inner_block, "ptr");
    auto& ptr_malloc = builder.add_library_node<stdlib::MallocNode>(inner_block, {}, symbolic::symbol("N"));
    builder.add_computational_memlet(inner_block, ptr_malloc, "_ret", ptr_acc, {}, ptr_type);
    EXPECT_TRUE(ptr_malloc.side_effect());

    analysis::AnalysisManager manager(sdfg);
    auto& loop_analysis = manager.get<analysis::LoopAnalysis>();

    auto info_inner = loop_analysis.loop_info(&loop_map_inner);
    auto info2_inner = loop_analysis.loop_info_local(&loop_map_inner);
    EXPECT_TRUE(info2_inner.contains_side_effects);
    EXPECT_TRUE(info_inner.has_side_effects);

    auto info_middle = loop_analysis.loop_info(&loop_map_middle);
    auto info2_middle = loop_analysis.loop_info_local(&loop_map_middle);
    EXPECT_FALSE(info2_middle.contains_side_effects);
    EXPECT_TRUE(info_middle.has_side_effects);

    auto info_outer = loop_analysis.loop_info(&loop_map_outer);
    auto info2_outer = loop_analysis.loop_info_local(&loop_map_outer);
    EXPECT_FALSE(info2_outer.contains_side_effects);
    EXPECT_TRUE(info_outer.has_side_effects);
}


TEST(LoopAnalysisInfoTest, WhileLoop) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    auto& loop_while = builder.add_while(root);

    analysis::AnalysisManager manager(sdfg);
    auto& loop_analysis = manager.get<analysis::LoopAnalysis>();

    auto info = loop_analysis.loop_info(&loop_while);
    EXPECT_EQ(info.num_loops, 1);
    EXPECT_EQ(info.num_whiles, 1);
    EXPECT_FALSE(info.is_perfectly_parallel);
    EXPECT_FALSE(info.is_elementwise);
    EXPECT_EQ(info.loop_level, 0);
    EXPECT_EQ(info.map_stack_depth, 0);
}

TEST(LoopAnalysisInfoTest, LoopInfoSerialization) {
    sdfg::analysis::LoopInfo info;
    info.loopnest_index = 2;
    info.num_loops = 3;
    info.num_maps = 1;
    info.num_fors = 1;
    info.num_whiles = 1;
    info.max_depth = 3;
    info.is_perfectly_nested = true;
    info.is_perfectly_parallel = false;
    info.is_elementwise = true;
    info.has_side_effects = false;

    nlohmann::json j = analysis::loop_info_to_json(info);

    EXPECT_EQ(info.loopnest_index, j["loopnest_index"].get<int>());
    EXPECT_EQ(info.num_loops, j["num_loops"].get<int>());
    EXPECT_EQ(info.num_maps, j["num_maps"].get<int>());
    EXPECT_EQ(info.num_fors, j["num_fors"].get<int>());
    EXPECT_EQ(info.num_whiles, j["num_whiles"].get<int>());
    EXPECT_EQ(info.max_depth, j["max_depth"].get<int>());
    EXPECT_EQ(info.is_perfectly_nested, j["is_perfectly_nested"].get<bool>());
    EXPECT_EQ(info.is_perfectly_parallel, j["is_perfectly_parallel"].get<bool>());
    EXPECT_EQ(info.is_elementwise, j["is_elementwise"].get<bool>());
    EXPECT_EQ(info.has_side_effects, j["has_side_effects"].get<bool>());
}
