#include <gtest/gtest.h>
#include <memory>
#include <sdfg/passes/rpc/rpc_scheduler.h>


#include "sdfg/analysis/loop_analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/deepcopy/structured_sdfg_deep_copy.h"
#include "sdfg/passes/rpc/rpc_context.h"
#include "sdfg/passes/scheduler/loop_scheduling_pass.h"
#include "sdfg/structured_control_flow/map.h"
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
