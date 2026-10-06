#include <gtest/gtest.h>

#include <nlohmann/json.hpp>

#include <memory>
#include <string>

#include "sdfg/analysis/loop_analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/codegen/code_generators/cpp_code_generator.h"
#include "sdfg/codegen/instrumentation/instrumentation_plan.h"
#include "sdfg/cutouts/cutouts.h"
#include "sdfg/metadata/rpc_optimization.h"
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

        types::Scalar sym_desc(types::PrimitiveType::Int64);
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

    auto cutout_sdfg = sdfg::util::cutout(local_builder.subject(), analysis_manager, *region_node);
    EXPECT_TRUE(cutout_sdfg != nullptr);

    EXPECT_GE(cutout_sdfg->root().size(), 1u);
    EXPECT_TRUE(dynamic_cast<const structured_control_flow::Sequence*>(&(cutout_sdfg->root().at(0))) == nullptr);

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

TEST_F(CutoutTest, ProvenanceGroupedInstrumentationSharesLogicalOrigin) {
    auto& root = builder_->subject().root();
    for (const auto* name : {"extra_i", "extra_j"}) {
        builder_->add_container(name, types::Scalar(types::PrimitiveType::Int64));
        auto indvar = symbolic::symbol(name);
        builder_->add_map(
            root,
            indvar,
            symbolic::Lt(indvar, symbolic::integer(4)),
            symbolic::integer(0),
            symbolic::add(indvar, symbolic::integer(1)),
            structured_control_flow::ScheduleType_Sequential::create()
        );
    }

    analysis::AnalysisManager analysis_manager(builder_->subject());
    auto& loop_analysis = analysis_manager.get<analysis::LoopAnalysis>();
    const auto& outermost_loops = loop_analysis.outermost_loops();
    ASSERT_EQ(outermost_loops.size(), 3u);

    for (size_t i = 1; i < outermost_loops.size(); ++i) {
        metadata::set_source_loop_id(*outermost_loops[i], 777);
        metadata::set_rpc_optimization(
            *outermost_loops[i], nlohmann::json{{"speedup", 1.75}, {"custom_metric", "future"}}, 0.125
        );
    }

    auto plan = codegen::InstrumentationPlan::provenance_grouped_outermost_loops_plan(builder_->subject(), true, true);
    for (size_t i = 0; i < outermost_loops.size(); ++i) {
        EXPECT_TRUE(plan->should_instrument(*outermost_loops[i]));
    }
    EXPECT_FALSE(plan->logical_region_id(*outermost_loops[0]).has_value());
    EXPECT_EQ(plan->logical_region_id(*outermost_loops[1]), 777u);
    EXPECT_EQ(plan->logical_region_id(*outermost_loops[2]), 777u);

    auto arg_capture_plan = codegen::ArgCapturePlan::none(builder_->subject());
    codegen::CPPCodeGenerator generator(builder_->subject(), analysis_manager, *plan, *arg_capture_plan);
    ASSERT_TRUE(generator.generate());
    const std::string generated = generator.main().str();
    EXPECT_NE(generated.find("sdfg_test_origin_777"), std::string::npos);
    EXPECT_NE(generated.find("member_loops_json"), std::string::npos);
    EXPECT_NE(generated.find("source_loop_id = 777"), std::string::npos);
    EXPECT_NE(generated.find("expected_performance_json = \"\";"), std::string::npos);
    EXPECT_NE(
        generated.find("expected_performance_json = \"{\\\"custom_metric\\\":\\\"future\\\",\\\"speedup\\\":1.75}\""),
        std::string::npos
    );
    EXPECT_NE(generated.find("vector_distance = 0.125000"), std::string::npos);
    EXPECT_NE(generated.find(std::to_string(outermost_loops[0]->element_id())), std::string::npos);
    EXPECT_NE(generated.find(std::to_string(outermost_loops[1]->element_id())), std::string::npos);

    auto count_occurrences = [&](const std::string& needle) {
        size_t count = 0;
        for (size_t position = 0; (position = generated.find(needle, position)) != std::string::npos;
             position += needle.size()) {
            ++count;
        }
        return count;
    };
    // The two contiguous mapped loops share one span; the unmapped outermost loop has its own OLS region.
    EXPECT_EQ(count_occurrences("__daisy_instrumentation_init("), 2u);
    EXPECT_EQ(count_occurrences("__daisy_instrumentation_enter("), 2u);
    EXPECT_EQ(count_occurrences("__daisy_instrumentation_exit("), 2u);
    EXPECT_EQ(count_occurrences("__daisy_instrumentation_should_continue("), 2u);
    EXPECT_EQ(count_occurrences("__daisy_instrumentation_finalize("), 2u);
}

TEST_F(CutoutTest, ProvenanceGroupedInstrumentationFallsBackForNoncontiguousMembers) {
    auto& root = builder_->subject().root();
    for (const auto* name : {"gap_i", "gap_j"}) {
        builder_->add_container(name, types::Scalar(types::PrimitiveType::Int64));
        auto indvar = symbolic::symbol(name);
        builder_->add_map(
            root,
            indvar,
            symbolic::Lt(indvar, symbolic::integer(4)),
            symbolic::integer(0),
            symbolic::add(indvar, symbolic::integer(1)),
            structured_control_flow::ScheduleType_Sequential::create()
        );
    }

    analysis::AnalysisManager analysis_manager(builder_->subject());
    auto& loop_analysis = analysis_manager.get<analysis::LoopAnalysis>();
    const auto& loops = loop_analysis.outermost_loops();
    ASSERT_EQ(loops.size(), 3u);

    metadata::set_source_loop_id(*loops[0], 900);
    metadata::set_source_loop_id(*loops[2], 900);

    auto plan = codegen::InstrumentationPlan::provenance_grouped_outermost_loops_plan(builder_->subject());
    for (const auto* loop : loops) {
        EXPECT_TRUE(plan->should_instrument(*loop));
        EXPECT_TRUE(plan->group_span_starting_at(*loop) == nullptr);
    }
}

// Regression: an external read inside a cutout used to be emitted both as an
// `external` and as an `argument` of the cutout SDFG, causing the deserializer
// to throw "Container <name> already exists".
TEST(CutoutTest_External, ExternalAndArgumentDoNotCollide) {
    builder::StructuredSDFGBuilder builder("cutout_external_test", FunctionType_CPU);

    types::Scalar float_desc(types::PrimitiveType::Float);

    // External, used inside the cutout region.
    builder.add_external("ext_table", float_desc, LinkageType_External);

    // Regular argument.
    builder.add_container("C", float_desc, true);

    // Loop induction variable + bound.
    types::Scalar sym_desc(types::PrimitiveType::Int64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);

    auto& root = builder.subject().root();
    auto& loop = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1))
    );

    auto& block = builder.add_block(loop.root());
    auto& ext_in = builder.add_access(block, "ext_table");
    auto& c_out = builder.add_access(block, "C");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, ext_in, tasklet, "_in", {});
    builder.add_computational_memlet(block, tasklet, "_out", c_out, {});

    auto sdfg = builder.move();
    auto local_builder = sdfg::builder::StructuredSDFGBuilder(sdfg);
    sdfg::analysis::AnalysisManager analysis_manager(local_builder.subject());

    auto& loop_analysis = analysis_manager.get<sdfg::analysis::LoopAnalysis>();
    auto outermost_loops = loop_analysis.outermost_loops();
    ASSERT_EQ(outermost_loops.size(), 1u);

    std::unique_ptr<StructuredSDFG> cutout_sdfg;
    ASSERT_NO_THROW(cutout_sdfg = sdfg::util::cutout(local_builder.subject(), analysis_manager, *outermost_loops[0]));
    ASSERT_TRUE(cutout_sdfg != nullptr);

    // External must remain external (with its linkage), not promoted to a
    // function argument.
    EXPECT_TRUE(cutout_sdfg->is_external("ext_table"));
    EXPECT_FALSE(cutout_sdfg->is_argument("ext_table"));
}
