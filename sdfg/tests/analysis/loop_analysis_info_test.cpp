#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include "sdfg/analysis/loop_analysis.h"

#include "sdfg/builder/structured_sdfg_builder.h"
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

    auto info_i = loop_analysis.loop_info(&loop_i);
    EXPECT_EQ(info_i.num_loops, 2);
    EXPECT_EQ(info_i.num_maps, 0);
    EXPECT_EQ(info_i.num_fors, 2);
    EXPECT_EQ(info_i.num_whiles, 0);
    EXPECT_EQ(info_i.max_depth, 2);
    EXPECT_FALSE(info_i.is_perfectly_nested);
    EXPECT_FALSE(info_i.is_perfectly_parallel);
    EXPECT_FALSE(info_i.is_elementwise);
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

    auto info = loop_analysis.loop_info(&loop_map);
    EXPECT_EQ(info.num_loops, 2);
    EXPECT_EQ(info.num_maps, 1);
    EXPECT_EQ(info.num_fors, 1);
    EXPECT_FALSE(info.is_perfectly_parallel);
    EXPECT_FALSE(info.is_elementwise);
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
