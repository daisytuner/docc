#include <gtest/gtest.h>
#include <sdfg/transformations/recorder.h>
#include <sdfg/transformations/replayer.h>

#include <fstream>
#include <iostream>
#include <memory>

#include "sdfg/analysis/loop_analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/transformations/loop_interchange.h"
#include "sdfg/transformations/loop_tiling.h"
#include "sdfg/types/pointer.h"
#include "sdfg/types/type.h"

using namespace sdfg;

class RecorderLoopTilingTest : public ::testing::Test {
protected:
    std::unique_ptr<builder::StructuredSDFGBuilder> builder_;
    std::unique_ptr<analysis::AnalysisManager> analysis_manager_;
    structured_control_flow::For* loop_;

    void SetUp() override {
        builder_ = std::make_unique<builder::StructuredSDFGBuilder>("sdfg_test", FunctionType_CPU);
        auto& sdfg = builder_->subject();
        auto& root = sdfg.root();

        // Add containers
        types::Scalar base_desc(types::PrimitiveType::Float);
        types::Pointer desc(base_desc);
        builder_->add_container("A", desc, true);

        types::Scalar sym_desc(types::PrimitiveType::UInt64);
        builder_->add_container("N", sym_desc, true);
        builder_->add_container("i", sym_desc);

        // Define loop
        auto bound = symbolic::symbol("N");
        auto indvar = symbolic::symbol("i");
        auto init = symbolic::integer(0);
        auto condition = symbolic::Lt(indvar, bound);
        auto update = symbolic::add(indvar, symbolic::integer(1));

        loop_ = &builder_->add_for(root, indvar, condition, init, update);
        auto& body = loop_->root();

        // Add computation
        auto& block = builder_->add_block(body);
        auto& A_in = builder_->add_access(block, "A");
        auto& A_out = builder_->add_access(block, "A");
        auto& tasklet = builder_->add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
        builder_->add_computational_memlet(block, A_in, tasklet, "_in", {symbolic::symbol("i")});
        builder_->add_computational_memlet(block, tasklet, "_out", A_out, {symbolic::symbol("i")});

        analysis_manager_ = std::make_unique<analysis::AnalysisManager>(builder_->subject());
    }
    void TearDown() override {
        // Cleanup if necessary
    };
};

TEST_F(RecorderLoopTilingTest, Apply_LoopTiling) {
    transformations::Recorder recorder;

    EXPECT_TRUE(loop_ != nullptr);
    EXPECT_TRUE(analysis_manager_ != nullptr);
    EXPECT_NO_THROW(recorder.apply<transformations::LoopTiling>(*builder_, *analysis_manager_, true, *loop_, 32));
}

TEST_F(RecorderLoopTilingTest, Apply_InvalidTransformation) {
    transformations::Recorder recorder;

    EXPECT_THROW(
        recorder.apply<transformations::LoopTiling>(*builder_, *analysis_manager_, false, *loop_, 0),
        transformations::InvalidTransformationException
    );
}

TEST_F(RecorderLoopTilingTest, Save_SingleTransformation) {
    transformations::Recorder recorder;

    size_t loop_id = loop_->element_id();

    EXPECT_NO_THROW(recorder.apply<transformations::LoopTiling>(*builder_, *analysis_manager_, true, *loop_, 32));

    // Use temporary file to save the transformation
    std::filesystem::path tmp_file = std::filesystem::temp_directory_path() / "Save_SingleTransformation.json";
    EXPECT_NO_THROW(recorder.save(tmp_file));
    EXPECT_TRUE(std::filesystem::exists(tmp_file));

    std::ifstream file(tmp_file);
    nlohmann::json j = nlohmann::json::array();
    file >> j;
    file.close();

    std::cout << j.dump(4) << std::endl;

    EXPECT_TRUE(j.is_array());
    EXPECT_EQ(j.size(), 1);
    EXPECT_EQ(j[0]["transformation_type"], "LoopTiling");
    EXPECT_EQ(j[0]["parameters"]["tile_size"], 32);
    EXPECT_EQ(j[0]["subgraph"]["0"]["element_id"], loop_id);
}

TEST_F(RecorderLoopTilingTest, Replay_Transformations) {
    builder::StructuredSDFGBuilder replay_builder("sdfg_test", FunctionType_CPU);

    transformations::Replayer replayer;

    nlohmann::json j_array = nlohmann::json::array();
    nlohmann::json j;
    j["transformation_type"] = "LoopTiling";
    j["subgraph"] = {{"0", {{"element_id", 1}, {"type", "for"}}}};
    j["parameters"] = {{"tile_size", 0}};
    j_array.push_back(j);

    EXPECT_NO_THROW(replayer.replay(*builder_, *analysis_manager_, j_array));
}

class RecorderMultiTransformationTest : public ::testing::Test {
protected:
    std::unique_ptr<builder::StructuredSDFGBuilder> builder_;
    std::unique_ptr<analysis::AnalysisManager> analysis_manager_;

    void SetUp() override {
        builder_ = std::make_unique<builder::StructuredSDFGBuilder>("sdfg_test", FunctionType_CPU);

        auto& sdfg = builder_->subject();
        auto& root = sdfg.root();

        /**
         * for (i = 0; i < N; i++)
         *   for (j = 0; j < M; j++)
         *     A[i][j] = A[i][j] + 1;
         */

        // Add containers
        types::Scalar base_desc(types::PrimitiveType::Float);
        types::Array desc_1(base_desc, symbolic::symbol("M"));
        types::Pointer desc(desc_1);

        builder_->add_container("A", desc, true);

        types::Scalar sym_desc(types::PrimitiveType::UInt64);
        builder_->add_container("N", sym_desc, true);
        builder_->add_container("M", sym_desc, true);
        builder_->add_container("i", sym_desc);
        builder_->add_container("j", sym_desc);

        // Define loop 1
        auto bound1 = symbolic::symbol("N");
        auto indvar1 = symbolic::symbol("i");
        auto& loop_1 = builder_->add_map(
            root,
            indvar1,
            symbolic::Lt(indvar1, bound1),
            symbolic::integer(0),
            symbolic::add(indvar1, symbolic::integer(1)),
            structured_control_flow::ScheduleType_Sequential::create()
        );
        auto& body1 = loop_1.root();

        // Define loop 2
        auto bound2 = symbolic::symbol("M");
        auto indvar2 = symbolic::symbol("j");
        auto& loop_2 = builder_->add_map(
            body1,
            indvar2,
            symbolic::Lt(indvar2, bound2),
            symbolic::integer(0),
            symbolic::add(indvar2, symbolic::integer(1)),
            structured_control_flow::ScheduleType_Sequential::create()
        );
        auto& body2 = loop_2.root();

        // Add computation
        auto& block = builder_->add_block(body2);
        auto& A_in = builder_->add_access(block, "A");
        auto& A_out = builder_->add_access(block, "A");
        auto& one_node = builder_->add_constant(block, "1.0", base_desc);

        auto& tasklet = builder_->add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});

        builder_->add_computational_memlet(block, A_in, tasklet, "_in1", {symbolic::symbol("i"), symbolic::symbol("j")});
        builder_->add_computational_memlet(block, one_node, tasklet, "_in2", {});
        builder_
            ->add_computational_memlet(block, tasklet, "_out", A_out, {symbolic::symbol("i"), symbolic::symbol("j")});

        analysis_manager_ = std::make_unique<analysis::AnalysisManager>(builder_->subject());
    }
    void TearDown() override { analysis_manager_->invalidate<analysis::LoopAnalysis>(); };
};

TEST_F(RecorderMultiTransformationTest, Apply_LoopInterchange) {
    transformations::Recorder recorder;

    auto& loop_analysis = analysis_manager_->get<analysis::LoopAnalysis>();
    structured_control_flow::Map* loop_1_ = nullptr;
    structured_control_flow::Map* loop_2_ = nullptr;
    for (auto& loop : loop_analysis.loops()) {
        auto* structured_loop = dynamic_cast<structured_control_flow::Map*>(loop);
        if (structured_loop->indvar()->get_name() == "i") {
            loop_1_ = structured_loop;
        } else if (structured_loop->indvar()->get_name() == "j") {
            loop_2_ = structured_loop;
        }
    }
    EXPECT_TRUE((loop_1_ != nullptr && loop_2_ != nullptr));

    EXPECT_NO_THROW(recorder
                        .apply<transformations::LoopInterchange>(*builder_, *analysis_manager_, true, *loop_1_, *loop_2_)
    );
}

TEST_F(RecorderMultiTransformationTest, Apply_Transformations) {
    /**
     * for (i = 0; i < N; i = i + 32)
     *   for (j = 0; j < M; j = j + 16)
     *     for (i_tile = i; i_tile < N && i_tile < i + 32; i_tile++)
     *       for (j_tile = j; j_tile < M && j_tile < j + 16; j_tile++)
     *         A[i_tile][j_tile] = A[i_tile][j_tile] + 1;
     */

    transformations::Recorder recorder;

    auto& loop_analysis_1 = analysis_manager_->get<analysis::LoopAnalysis>();
    structured_control_flow::Map* loop_1_ = nullptr;
    for (auto& loop : loop_analysis_1.loops()) {
        auto* structured_loop = dynamic_cast<structured_control_flow::Map*>(loop);
        if (structured_loop->indvar()->get_name() == "i") {
            loop_1_ = structured_loop;
        }
    }
    EXPECT_TRUE(loop_1_ != nullptr);
    recorder.apply<transformations::LoopTiling>(*builder_, *analysis_manager_, true, *loop_1_, 32);

    analysis_manager_->invalidate_all();

    auto& loop_analysis_2 = analysis_manager_->get<analysis::LoopAnalysis>();
    structured_control_flow::Map* loop_2_ = nullptr;
    for (auto& loop : loop_analysis_2.loops()) {
        auto* structured_loop = dynamic_cast<structured_control_flow::Map*>(loop);
        if (structured_loop->indvar()->get_name() == "j") {
            loop_2_ = structured_loop;
        }
    }

    EXPECT_TRUE(loop_2_ != nullptr);
    recorder.apply<transformations::LoopTiling>(*builder_, *analysis_manager_, true, *loop_2_, 16);

    analysis_manager_->invalidate_all();

    auto& loop_analysis_3 = analysis_manager_->get<analysis::LoopAnalysis>();
    structured_control_flow::Map* loop_j_outer = nullptr;
    structured_control_flow::Map* loop_i_tile = nullptr;
    for (auto& loop : loop_analysis_3.loops()) {
        auto* structured_loop = dynamic_cast<structured_control_flow::Map*>(loop);
        if (structured_loop->indvar()->get_name() == "i") {
            loop_i_tile = structured_loop;
        } else if (structured_loop->indvar()->get_name() == "j_tile0") {
            loop_j_outer = structured_loop;
        }
    }
    EXPECT_TRUE(loop_j_outer != nullptr);
    EXPECT_TRUE(loop_i_tile != nullptr);

    auto loop_i_tile_id = loop_i_tile->element_id();
    auto loop_j_outer_id = loop_j_outer->element_id();

    EXPECT_TRUE(loop_i_tile_id != 0);
    EXPECT_TRUE(loop_j_outer_id != 0);

    EXPECT_NO_THROW(recorder.apply<
                    transformations::LoopInterchange>(*builder_, *analysis_manager_, true, *loop_i_tile, *loop_j_outer)
    );

    /**** Save ****/

    std::filesystem::path tmp_file = std::filesystem::temp_directory_path() / "Replay_Transformations.json";
    EXPECT_NO_THROW(recorder.save(tmp_file));
    EXPECT_TRUE(std::filesystem::exists(tmp_file));

    std::ifstream file(tmp_file);
    nlohmann::json j = nlohmann::json::array();
    file >> j;
    file.close();

    EXPECT_TRUE(j.is_array());
    EXPECT_EQ(j.size(), 3);
    EXPECT_EQ(j[0]["transformation_type"], "LoopTiling");
    EXPECT_EQ(j[0]["parameters"]["tile_size"], 32);
    EXPECT_EQ(j[0]["subgraph"]["0"]["element_id"], 1);

    EXPECT_EQ(j[1]["transformation_type"], "LoopTiling");
    EXPECT_EQ(j[1]["parameters"]["tile_size"], 16);
    EXPECT_EQ(j[1]["subgraph"]["0"]["element_id"], 4);

    EXPECT_EQ(j[2]["transformation_type"], "LoopInterchange");
    EXPECT_EQ(j[2]["subgraph"]["0"]["element_id"], 1);
    EXPECT_EQ(j[2]["subgraph"]["1"]["element_id"], 19);
}

TEST_F(RecorderMultiTransformationTest, Replay_Transformations) {
    transformations::Replayer recorder;

    nlohmann::json j_array = nlohmann::json::array();
    nlohmann::json j;
    j["transformation_type"] = "LoopTiling";
    j["subgraph"] = {{"0", {{"element_id", 1}, {"type", "for"}}}};
    j["parameters"] = {{"tile_size", 32}};
    j_array.push_back(j);

    nlohmann::json j1;
    j1["transformation_type"] = "LoopTiling";
    j1["subgraph"] = {{"0", {{"element_id", 4}, {"type", "map"}}}};
    j1["parameters"] = {{"tile_size", 16}};
    j_array.push_back(j1);

    nlohmann::json j2;
    j2["transformation_type"] = "LoopInterchange";
    j2["subgraph"] = {{"0", {{"element_id", 1}, {"type", "for"}}}, {"1", {{"element_id", 19}, {"type", "map"}}}};
    j_array.push_back(j2);

    EXPECT_NO_THROW(recorder.replay(*builder_, *analysis_manager_, j_array));
}

TEST_F(RecorderMultiTransformationTest, Replay_InvalidTransformation) {
    nlohmann::json j_array = nlohmann::json::array();
    nlohmann::json j;
    j["transformation_type"] = "LoopTiling";
    j["subgraph"] = {{"0", {{"element_id", 1}, {"type", "for"}}}};
    j["parameters"] = {{"tile_size", 0}};
    j_array.push_back(j);

    transformations::Replayer replayer;

    EXPECT_THROW(
        replayer.replay(*builder_, *analysis_manager_, j_array, false), transformations::InvalidTransformationException
    );
}

class ReplayerTest : public ::testing::Test {
protected:
    std::unique_ptr<builder::StructuredSDFGBuilder> builder_;
    std::unique_ptr<analysis::AnalysisManager> analysis_manager_;

    void SetUp() override {
        builder_ = std::make_unique<builder::StructuredSDFGBuilder>("sdfg_test", FunctionType_CPU);

        auto& sdfg = builder_->subject();
        auto& root = sdfg.root();

        /**
         * for (i = 0; i < N; i++)
         *   for (j = 0; j < M; j++)
         *     A[i][j] = A[i][j] + 1;
         */

        // Add containers
        types::Scalar base_desc(types::PrimitiveType::Float);
        types::Array base_desc_ptr(base_desc, symbolic::symbol("M"));
        types::Pointer desc(base_desc_ptr);

        builder_->add_container("A", desc, true);

        types::Scalar sym_desc(types::PrimitiveType::UInt64);
        builder_->add_container("N", sym_desc, true);
        builder_->add_container("M", sym_desc, true);
        builder_->add_container("i", sym_desc);
        builder_->add_container("j", sym_desc);

        // Define loop 1
        auto bound1 = symbolic::symbol("N");
        auto indvar1 = symbolic::symbol("i");
        auto& loop_1 = builder_->add_map(
            root,
            indvar1,
            symbolic::Lt(indvar1, bound1),
            symbolic::integer(0),
            symbolic::add(indvar1, symbolic::integer(1)),
            structured_control_flow::ScheduleType_Sequential::create()
        );
        auto& body1 = loop_1.root();

        // Define loop 2
        auto bound2 = symbolic::symbol("M");
        auto indvar2 = symbolic::symbol("j");
        auto& loop_2 = builder_->add_map(
            body1,
            indvar2,
            symbolic::Lt(indvar2, bound2),
            symbolic::integer(0),
            symbolic::add(indvar2, symbolic::integer(1)),
            structured_control_flow::ScheduleType_Sequential::create()
        );
        auto& body2 = loop_2.root();

        // Add computation
        auto& block = builder_->add_block(body2);
        auto& A_in = builder_->add_access(block, "A");
        auto& A_out = builder_->add_access(block, "A");
        auto& one_node = builder_->add_constant(block, "1.0", base_desc);

        auto& tasklet = builder_->add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});

        builder_->add_computational_memlet(block, A_in, tasklet, "_in1", {symbolic::symbol("i"), symbolic::symbol("j")});
        builder_->add_computational_memlet(block, one_node, tasklet, "_in2", {});
        builder_
            ->add_computational_memlet(block, tasklet, "_out", A_out, {symbolic::symbol("i"), symbolic::symbol("j")});

        analysis_manager_ = std::make_unique<analysis::AnalysisManager>(builder_->subject());
    }
    void TearDown() override { analysis_manager_->invalidate<analysis::LoopAnalysis>(); };
};

TEST_F(ReplayerTest, Replay_Transformations) {
    transformations::Replayer recorder;

    nlohmann::json j = nlohmann::json::array();
    j.push_back(
        {{"transformation_type", "LoopTiling"},
         {"subgraph", {{"0", {{"element_id", 1}, {"type", "map"}}}}},
         {"parameters", {{"tile_size", 32}}}}
    );
    j.push_back(
        {{"transformation_type", "LoopTiling"},
         {"subgraph", {{"0", {{"element_id", 4}, {"type", "map"}}}}},
         {"parameters", {{"tile_size", 16}}}}
    );
    j.push_back(
        {{"transformation_type", "LoopInterchange"},
         {"subgraph", {{"0", {{"element_id", 1}, {"type", "map"}}}, {"1", {{"element_id", 19}, {"type", "map"}}}}}}
    );

    EXPECT_NO_THROW(recorder.replay(*builder_, *analysis_manager_, j));
}

TEST_F(ReplayerTest, Replay_InvalidTransformation) {
    nlohmann::json j = nlohmann::json::array();
    j.push_back(
        {{"transformation_type", "LoopTiling"},
         {"subgraph", {{"0", {{"element_id", 1}, {"type", "map"}}}}},
         {"parameters", {{"tile_size", 0}}}}
    );

    transformations::Replayer recorder;

    EXPECT_THROW(recorder.replay(*builder_, *analysis_manager_, j, false), transformations::InvalidTransformationException);
}
