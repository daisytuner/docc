#include "sdfg/codegen/instrumentation/arg_capture_plan.h"
#include "sdfg/codegen/code_generators/c_code_generator.h"

#include <gtest/gtest.h>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/pointer.h"
#include "sdfg/types/type.h"

using namespace sdfg;

TEST(ArgCapturePlanTest, CreateCapturePlan_Arguments_Scalar) {
    builder::StructuredSDFGBuilder builder("sdfg_a", FunctionType_CPU);

    types::Scalar int_type(types::PrimitiveType::Int32);
    builder.add_container("arg1", int_type, true);
    builder.add_container("arg2", int_type, true);

    auto& block = builder.add_assignments(
        builder.subject().root(),
        {{symbolic::symbol("arg1"), symbolic::zero()}, {symbolic::symbol("arg2"), symbolic::zero()}}
    );

    analysis::AnalysisManager analysis_manager(builder.subject());

    auto arg_capture_plan =
        codegen::ArgCapturePlan::create_capture_plan(builder.subject(), analysis_manager, builder.subject().root());

    EXPECT_EQ(arg_capture_plan.size(), 2);

    auto& arg1_plan = arg_capture_plan.at("arg1");
    EXPECT_TRUE(arg1_plan.capture_input);
    EXPECT_TRUE(arg1_plan.capture_output);
    EXPECT_EQ(arg1_plan.inner_type, types::PrimitiveType::Int32);

    auto& arg2_plan = arg_capture_plan.at("arg2");
    EXPECT_TRUE(arg2_plan.capture_input);
    EXPECT_TRUE(arg2_plan.capture_output);
    EXPECT_EQ(arg2_plan.inner_type, types::PrimitiveType::Int32);
}

TEST(ArgCapturePlanTest, CreateCapturePlan_Transients_Scalar) {
    builder::StructuredSDFGBuilder builder("sdfg_a", FunctionType_CPU);

    types::Scalar int_type(types::PrimitiveType::Int32);
    builder.add_container("t1", int_type);
    builder.add_container("t2", int_type);

    auto& block1 = builder.add_assignments(
        builder.subject().root(),
        {{symbolic::symbol("t1"), symbolic::zero()}, {symbolic::symbol("t2"), symbolic::zero()}}
    );
    auto& subseq = builder.add_sequence(builder.subject().root());
    auto& block2 = builder.add_assignments(
        subseq,
        {{symbolic::symbol("t1"), symbolic::add(symbolic::symbol("t1"), symbolic::one())},
         {symbolic::symbol("t2"), symbolic::add(symbolic::symbol("t2"), symbolic::one())}}
    );

    analysis::AnalysisManager analysis_manager(builder.subject());

    auto arg_capture_plan = codegen::ArgCapturePlan::create_capture_plan(builder.subject(), analysis_manager, subseq);

    EXPECT_EQ(arg_capture_plan.size(), 2);

    auto& t1_plan = arg_capture_plan.at("t1");
    EXPECT_TRUE(t1_plan.capture_input);
    EXPECT_TRUE(t1_plan.capture_output);
    EXPECT_EQ(t1_plan.inner_type, types::PrimitiveType::Int32);

    auto& t2_plan = arg_capture_plan.at("t2");
    EXPECT_TRUE(t2_plan.capture_input);
    EXPECT_TRUE(t2_plan.capture_output);
    EXPECT_EQ(t2_plan.inner_type, types::PrimitiveType::Int32);
}
