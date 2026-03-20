#include <gtest/gtest.h>

#include <memory>

#include "sdfg/analysis/loop_analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/codegen/code_generators/cpp_code_generator.h"
#include "sdfg/codegen/instrumentation/instrumentation_plan.h"
#include "sdfg/cutouts/cutouts.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/structured_sdfg.h"
#include "sdfg/types/pointer.h"
#include "sdfg/types/type.h"

using namespace sdfg;

class CutoutTest : public ::testing::Test {
protected:
    std::unique_ptr<builder::StructuredSDFGBuilder> builder_;

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

TEST_F(CutoutTest, TestCutoutInstrumentation) {
    auto sdfg = builder_->move();

    auto local_builder = sdfg::builder::StructuredSDFGBuilder(sdfg);

    sdfg::analysis::AnalysisManager analysis_manager(local_builder.subject());
    auto& loop_analysis = analysis_manager.get<sdfg::analysis::LoopAnalysis>();
    auto outermost_loops = loop_analysis.outermost_loops();
    EXPECT_EQ(outermost_loops.size(), 1);

    auto region_node = outermost_loops[0];

    EXPECT_TRUE(region_node != nullptr);

    auto cutout_sdfg = sdfg::util::cutout(local_builder, analysis_manager, *region_node);
    EXPECT_TRUE(cutout_sdfg != nullptr);

    EXPECT_GE(cutout_sdfg->root().size(), 1u);
    EXPECT_TRUE(dynamic_cast<const structured_control_flow::Sequence*>(&(cutout_sdfg->root().at(0).first)) == nullptr);

    auto cutout_analysis_manager = sdfg::analysis::AnalysisManager(*cutout_sdfg);
    auto& cutoutloop_analysis = cutout_analysis_manager.get<sdfg::analysis::LoopAnalysis>();
    auto outer_cutout_loops = cutoutloop_analysis.outermost_loops();
    EXPECT_EQ(outer_cutout_loops.size(), 1);

    auto instrumentation_plan_opt = codegen::InstrumentationPlan::outermost_loops_plan(*cutout_sdfg);
    auto arg_capture_plan_opt = sdfg::codegen::ArgCapturePlan::none(*cutout_sdfg);
    analysis::AnalysisManager analysis_manager_opt(*cutout_sdfg);
    codegen::CPPCodeGenerator
        code_generator_opt(*cutout_sdfg, analysis_manager_opt, *instrumentation_plan_opt, *arg_capture_plan_opt);

    EXPECT_TRUE(code_generator_opt.generate());
    EXPECT_TRUE(code_generator_opt.as_source("cutout_sdfg.h", "cutout_sdfg.cpp"));
}
