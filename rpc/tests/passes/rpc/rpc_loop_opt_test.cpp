#include <gtest/gtest.h>
#include <memory>
#include <sdfg/passes/rpc/rpc_scheduler.h>


#include "sdfg/analysis/loop_analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/deepcopy/structured_sdfg_deep_copy.h"
#include "sdfg/passes/rpc/rpc_context.h"
#include "sdfg/passes/scheduler/loop_scheduling_pass.h"
#include "sdfg/passes/scheduler/scheduler_registry.h"
#include "sdfg/serializer/json_serializer.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/structured_sdfg.h"
#include "sdfg/transformations/loop_interchange.h"
#include "sdfg/transformations/loop_tiling.h"
#include "sdfg/transformations/recorder.h"
#include "sdfg/types/pointer.h"
#include "sdfg/types/type.h"

using namespace sdfg;

class RPCLoopOptTest : public ::testing::Test {
protected:
    std::unique_ptr<builder::StructuredSDFGBuilder> builder_;
    nlohmann::json desc_;

    void SetUp() override {
        builder_ = std::make_unique<builder::StructuredSDFGBuilder>("sdfg_test", FunctionType_CPU);

        auto& root = builder_->subject().root();
        types::Scalar base_desc(types::PrimitiveType::Float);
        types::Array desc_1(base_desc, symbolic::integer(64));
        types::Pointer desc_2(desc_1);

        builder_->add_container("A", desc_2, true);
        builder_->add_container("B", desc_2, true);
        builder_->add_container("C", desc_2, true);

        types::Scalar sym_desc(types::PrimitiveType::UInt64);
        builder_->add_container("K", sym_desc, true);
        builder_->add_container("N", sym_desc, true);
        builder_->add_container("M", sym_desc, true);
        builder_->add_container("i", sym_desc);
        builder_->add_container("j", sym_desc);
        builder_->add_container("k", sym_desc);

        // Define loop 1
        auto bound = symbolic::integer(64);
        auto indvar = symbolic::symbol("i");

        auto& loop = builder_->add_map(
            root,
            indvar,
            symbolic::Lt(symbolic::symbol("i"), bound),
            symbolic::integer(0),
            symbolic::add(symbolic::symbol("i"), symbolic::integer(1)),
            structured_control_flow::ScheduleType_Sequential::create()
        );
        auto& body = loop.root();

        // Define loop 2
        auto bound_2 = symbolic::integer(64);
        auto indvar_2 = symbolic::symbol("j");

        auto& loop_2 = builder_->add_for(
            body,
            indvar_2,
            symbolic::Lt(symbolic::symbol("j"), bound_2),
            symbolic::integer(0),
            symbolic::add(symbolic::symbol("j"), symbolic::integer(1))
        );

        auto& body_2 = loop_2.root();

        // Define loop 3
        auto bound_3 = symbolic::integer(64);
        auto indvar_3 = symbolic::symbol("k");

        auto& loop_3 = builder_->add_map(
            body_2,
            indvar_3,
            symbolic::Lt(symbolic::symbol("k"), bound_3),
            symbolic::integer(0),
            symbolic::add(symbolic::symbol("k"), symbolic::integer(1)),
            structured_control_flow::ScheduleType_Sequential::create()
        );

        auto& body_3 = loop_3.root();

        // Add computation
        auto& block = builder_->add_block(body_3);
        auto& a_in = builder_->add_access(block, "A");
        auto& b_in = builder_->add_access(block, "B");
        auto& c_in = builder_->add_access(block, "C");
        auto& c_out = builder_->add_access(block, "C");

        {
            auto& tasklet =
                builder_->add_tasklet(block, data_flow::TaskletCode::fp_fma, "_out", {"_in1", "_in2", "_in3"});
            builder_
                ->add_computational_memlet(block, a_in, tasklet, "_in1", {symbolic::symbol("i"), symbolic::symbol("j")});
            builder_
                ->add_computational_memlet(block, b_in, tasklet, "_in2", {symbolic::symbol("j"), symbolic::symbol("k")});
            builder_
                ->add_computational_memlet(block, c_in, tasklet, "_in3", {symbolic::symbol("i"), symbolic::symbol("k")});
            builder_
                ->add_computational_memlet(block, tasklet, "_out", c_out, {symbolic::symbol("i"), symbolic::symbol("k")});
        }
    };

    void TearDown() override {
        // Cleanup if necessary
    };
};

TEST_F(RPCLoopOptTest, Matmul_FMA) {
    auto sdfg_initial = builder_->subject().clone();
    sdfg::builder::StructuredSDFGBuilder builder(sdfg_initial);

    // Transfer tuning replayer

    sdfg::analysis::AnalysisManager analysis_manager(builder_->subject());
    auto& loop_analysis = analysis_manager.get<sdfg::analysis::LoopAnalysis>();
    auto outer_loops = loop_analysis.outermost_loops();

    passes::scheduler::LoopSchedulingPass loop_scheduling_pass({"rpc"}, nullptr);
    loop_scheduling_pass.run(*builder_, analysis_manager);

    sdfg::analysis::AnalysisManager test_analysis_manager(builder_->subject());
    auto& test_loop_analysis = test_analysis_manager.get<sdfg::analysis::LoopAnalysis>();
    auto loop_nest_tree = test_loop_analysis.loop_tree();


    EXPECT_EQ(test_loop_analysis.find_loop_by_indvar("k"), loop_nest_tree[test_loop_analysis.find_loop_by_indvar("j")]);
    EXPECT_EQ(
        test_loop_analysis.find_loop_by_indvar("j_tile0"), loop_nest_tree[test_loop_analysis.find_loop_by_indvar("k")]
    );

    EXPECT_NE(test_loop_analysis.find_loop_by_indvar("k_tile0"), nullptr);
    EXPECT_NE(test_loop_analysis.find_loop_by_indvar("j_tile0"), nullptr);
    EXPECT_NE(test_loop_analysis.find_loop_by_indvar("i_tile0"), nullptr);

    EXPECT_EQ(
        test_loop_analysis.find_loop_by_indvar("k_tile0"),
        loop_nest_tree[test_loop_analysis.find_loop_by_indvar("j_tile0")]
    );
    EXPECT_EQ(
        test_loop_analysis.find_loop_by_indvar("i"), loop_nest_tree[test_loop_analysis.find_loop_by_indvar("k_tile0")]
    );
    EXPECT_EQ(
        test_loop_analysis.find_loop_by_indvar("i_tile0"), loop_nest_tree[test_loop_analysis.find_loop_by_indvar("i")]
    );
};

TEST_F(RPCLoopOptTest, Double_Matmul) {
    {
        analysis::AnalysisManager analysis_manager(builder_->subject());
        auto& loop_analysis = analysis_manager.get<sdfg::analysis::LoopAnalysis>();
        auto outer_loops = loop_analysis.outermost_loops();
        EXPECT_EQ(outer_loops.size(), 1);
        auto loopnest = outer_loops[0];
        sdfg::deepcopy::StructuredSDFGDeepCopy deep_copy(*builder_, builder_->subject().root(), *loopnest);
        deep_copy.copy();
    }

    sdfg::analysis::AnalysisManager analysis_manager(builder_->subject());
    auto& loop_analysis = analysis_manager.get<sdfg::analysis::LoopAnalysis>();
    auto outer_loops = loop_analysis.outermost_loops();

    EXPECT_EQ(outer_loops.size(), 2);

    passes::scheduler::LoopSchedulingPass loop_scheduling_pass({"rpc"}, nullptr);
    loop_scheduling_pass.run(*builder_, analysis_manager);

    sdfg::analysis::AnalysisManager test_analysis_manager(builder_->subject());

    auto& test_loop_analysis = test_analysis_manager.get<sdfg::analysis::LoopAnalysis>();
    auto loop_nest_tree = test_loop_analysis.loop_tree();

    EXPECT_EQ(test_loop_analysis.find_loop_by_indvar("k"), loop_nest_tree[test_loop_analysis.find_loop_by_indvar("j")]);
    EXPECT_EQ(
        test_loop_analysis.find_loop_by_indvar("j_tile0"), loop_nest_tree[test_loop_analysis.find_loop_by_indvar("k")]
    );

    EXPECT_NE(test_loop_analysis.find_loop_by_indvar("k_tile0"), nullptr);
    EXPECT_NE(test_loop_analysis.find_loop_by_indvar("j_tile0"), nullptr);
    EXPECT_NE(test_loop_analysis.find_loop_by_indvar("i_tile0"), nullptr);

    EXPECT_EQ(
        test_loop_analysis.find_loop_by_indvar("k_tile0"),
        loop_nest_tree[test_loop_analysis.find_loop_by_indvar("j_tile0")]
    );
    EXPECT_EQ(
        test_loop_analysis.find_loop_by_indvar("i"), loop_nest_tree[test_loop_analysis.find_loop_by_indvar("k_tile0")]
    );
    EXPECT_EQ(
        test_loop_analysis.find_loop_by_indvar("i_tile0"), loop_nest_tree[test_loop_analysis.find_loop_by_indvar("i")]
    );
}

// Regression test: When the RPC server returns an optimized SDFG with multiple
// children in its root (e.g. H2D + CUDA map + D2H), all children must be moved
// into the parent scope. Previously only child 0 was moved, losing the rest.
//
// This test serializes a multi-child response SDFG to disk and points the
// transfer server at it via SDFG-Result-Path header, then runs
// LoopSchedulingPass and verifies all children ended up in the target SDFG.
class RPCLoopOptMoveChildrenTest : public ::testing::Test {
protected:
    std::unique_ptr<builder::StructuredSDFGBuilder> builder_;
    std::string target_key_ = "rpc_move_children";

    void SetUp() override {
        // Build the input SDFG (same matmul as RPCLoopOptTest)
        builder_ = std::make_unique<builder::StructuredSDFGBuilder>("sdfg_test", FunctionType_CPU);

        auto& root = builder_->subject().root();
        types::Scalar base_desc(types::PrimitiveType::Float);
        types::Array desc_1(base_desc, symbolic::integer(64));
        types::Pointer desc_2(desc_1);

        builder_->add_container("A", desc_2, true);
        builder_->add_container("B", desc_2, true);
        builder_->add_container("C", desc_2, true);

        types::Scalar sym_desc(types::PrimitiveType::UInt64);
        builder_->add_container("K", sym_desc, true);
        builder_->add_container("N", sym_desc, true);
        builder_->add_container("M", sym_desc, true);
        builder_->add_container("i", sym_desc);
        builder_->add_container("j", sym_desc);
        builder_->add_container("k", sym_desc);

        auto& loop = builder_->add_map(
            root,
            symbolic::symbol("i"),
            symbolic::Lt(symbolic::symbol("i"), symbolic::integer(64)),
            symbolic::integer(0),
            symbolic::add(symbolic::symbol("i"), symbolic::integer(1)),
            structured_control_flow::ScheduleType_Sequential::create()
        );
        auto& body = loop.root();

        auto& loop_2 = builder_->add_for(
            body,
            symbolic::symbol("j"),
            symbolic::Lt(symbolic::symbol("j"), symbolic::integer(64)),
            symbolic::integer(0),
            symbolic::add(symbolic::symbol("j"), symbolic::integer(1))
        );
        auto& body_2 = loop_2.root();

        auto& loop_3 = builder_->add_map(
            body_2,
            symbolic::symbol("k"),
            symbolic::Lt(symbolic::symbol("k"), symbolic::integer(64)),
            symbolic::integer(0),
            symbolic::add(symbolic::symbol("k"), symbolic::integer(1)),
            structured_control_flow::ScheduleType_Sequential::create()
        );
        auto& body_3 = loop_3.root();

        auto& block = builder_->add_block(body_3);
        auto& a_in = builder_->add_access(block, "A");
        auto& b_in = builder_->add_access(block, "B");
        auto& c_in = builder_->add_access(block, "C");
        auto& c_out = builder_->add_access(block, "C");

        auto& tasklet = builder_->add_tasklet(block, data_flow::TaskletCode::fp_fma, "_out", {"_in1", "_in2", "_in3"});
        builder_->add_computational_memlet(block, a_in, tasklet, "_in1", {symbolic::symbol("i"), symbolic::symbol("j")});
        builder_->add_computational_memlet(block, b_in, tasklet, "_in2", {symbolic::symbol("j"), symbolic::symbol("k")});
        builder_->add_computational_memlet(block, c_in, tasklet, "_in3", {symbolic::symbol("i"), symbolic::symbol("k")});
        builder_
            ->add_computational_memlet(block, tasklet, "_out", c_out, {symbolic::symbol("i"), symbolic::symbol("k")});

        // Build a response SDFG with 3 children in root (simulating H2D + map + D2H)
        auto response_sdfg = std::make_unique<StructuredSDFG>("response_sdfg", FunctionType_CPU);
        {
            builder::StructuredSDFGBuilder resp_builder(*response_sdfg);

            resp_builder.add_container("A", desc_2, true);
            resp_builder.add_container("B", desc_2, true);
            resp_builder.add_container("C", desc_2, true);
            resp_builder.add_container("K", sym_desc, true);
            resp_builder.add_container("N", sym_desc, true);
            resp_builder.add_container("M", sym_desc, true);
            resp_builder.add_container("i", sym_desc);
            resp_builder.add_container("j", sym_desc);
            resp_builder.add_container("k", sym_desc);

            auto& resp_root = resp_builder.subject().root();

            // Child 1: simulated H2D block
            auto& h2d_block = resp_builder.add_block(resp_root);
            resp_builder.add_access(h2d_block, "A");

            // Child 2: the optimized map (same structure as input)
            auto& opt_loop = resp_builder.add_map(
                resp_root,
                symbolic::symbol("i"),
                symbolic::Lt(symbolic::symbol("i"), symbolic::integer(64)),
                symbolic::integer(0),
                symbolic::add(symbolic::symbol("i"), symbolic::integer(1)),
                structured_control_flow::ScheduleType_Sequential::create()
            );
            {
                auto& b = opt_loop.root();
                auto& l2 = resp_builder.add_for(
                    b,
                    symbolic::symbol("j"),
                    symbolic::Lt(symbolic::symbol("j"), symbolic::integer(64)),
                    symbolic::integer(0),
                    symbolic::add(symbolic::symbol("j"), symbolic::integer(1))
                );
                auto& b2 = l2.root();
                auto& l3 = resp_builder.add_map(
                    b2,
                    symbolic::symbol("k"),
                    symbolic::Lt(symbolic::symbol("k"), symbolic::integer(64)),
                    symbolic::integer(0),
                    symbolic::add(symbolic::symbol("k"), symbolic::integer(1)),
                    structured_control_flow::ScheduleType_Sequential::create()
                );
                auto& b3 = l3.root();
                auto& blk = resp_builder.add_block(b3);
                auto& ai = resp_builder.add_access(blk, "A");
                auto& bi = resp_builder.add_access(blk, "B");
                auto& ci = resp_builder.add_access(blk, "C");
                auto& co = resp_builder.add_access(blk, "C");
                auto& t = resp_builder.add_tasklet(blk, data_flow::TaskletCode::fp_fma, "_out", {"_in1", "_in2", "_in3"});
                resp_builder.add_computational_memlet(blk, ai, t, "_in1", {symbolic::symbol("i"), symbolic::symbol("j")});
                resp_builder.add_computational_memlet(blk, bi, t, "_in2", {symbolic::symbol("j"), symbolic::symbol("k")});
                resp_builder.add_computational_memlet(blk, ci, t, "_in3", {symbolic::symbol("i"), symbolic::symbol("k")});
                resp_builder.add_computational_memlet(blk, t, "_out", co, {symbolic::symbol("i"), symbolic::symbol("k")});
            }

            // Child 3: simulated D2H block
            auto& d2h_block = resp_builder.add_block(resp_root);
            resp_builder.add_access(d2h_block, "C");
        }

        // Serialize the response SDFG and set env var for the local transfer server
        const std::string sdfg_path = "/tmp/rpc_move_children_test.sdfg.json";
        serializer::JSONSerializer::writeToFile(*response_sdfg, sdfg_path);

        const std::string header_value = "SDFG-Result-Path:" + sdfg_path;
        setenv("RPC_HEADER", header_value.c_str(), 1);

        // Build RPC context from env vars (same mechanism as test.cpp main)
        passes::rpc::SimpleRpcContextBuilder ctx_builder;
        auto test_ctx = ctx_builder
                            .initialize_local_default()
                            .from_env()
                            .from_header_env()
                            .build();

        // Register scheduler under a test-specific key
        passes::scheduler::SchedulerRegistry::instance()
            .register_loop_scheduler<passes::rpc::RPCScheduler>(
                target_key_, test_ctx, "sequential", "server", false);
    }

    void TearDown() override {
        unsetenv("RPC_HEADER");
    }
};

TEST_F(RPCLoopOptMoveChildrenTest, MoveAllChildrenFromRPCResult) {
    ASSERT_EQ(builder_->subject().root().size(), 1); // just the one map

    // Run the scheduling pass through the full pipeline
    analysis::AnalysisManager analysis_manager(builder_->subject());
    passes::scheduler::LoopSchedulingPass loop_scheduling_pass({target_key_}, nullptr);
    loop_scheduling_pass.run(*builder_, analysis_manager);

    // After the pass: the single map should be replaced by all 3 children from the response
    EXPECT_EQ(builder_->subject().root().size(), 3)
        << "All children from the RPC response SDFG must be moved into the target. "
           "With the old bug (move_child instead of move_children), only 1 child would appear.";
}