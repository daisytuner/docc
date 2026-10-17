#include "sdfg/fp_math/transformations/unfuse_fma.h"

#include <gtest/gtest.h>

#include <nlohmann/json.hpp>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/fp_math/transformations/fuse_fma.h"
#include "sdfg/transformations/transformation_registry.h"
#include "sdfg/types/scalar.h"

using namespace sdfg;

namespace {

data_flow::Tasklet& build_fma(builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& block) {
    types::Scalar desc(types::PrimitiveType::Float);
    for (auto name : {"a", "b", "c", "d"}) {
        builder.add_container(name, desc, true);
    }
    auto& a = builder.add_access(block, "a");
    auto& b = builder.add_access(block, "b");
    auto& c = builder.add_access(block, "c");
    auto& d = builder.add_access(block, "d");
    auto& fma = builder.add_tasklet(block, data_flow::TaskletCode::fp_fma, "_out", {"_in1", "_in2", "_in3"});
    builder.add_computational_memlet(block, a, fma, "_in1", {});
    builder.add_computational_memlet(block, b, fma, "_in2", {});
    builder.add_computational_memlet(block, c, fma, "_in3", {});
    builder.add_computational_memlet(block, fma, "_out", d, {});
    return fma;
}

std::string input_of(const data_flow::DataFlowGraph& dataflow, const data_flow::Tasklet& tasklet, size_t i) {
    auto inputs = dataflow.in_edges_by_connector(tasklet);
    return static_cast<const data_flow::AccessNode&>(inputs.at(i)->src()).data();
}

} // namespace

TEST(UnfuseFMATest, Simple) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& block = builder.add_block(sdfg.root());
    auto& fma = build_fma(builder, block);

    analysis::AnalysisManager analysis_manager(sdfg);
    fp_math::UnfuseFMA transformation(fma);
    ASSERT_TRUE(transformation.can_be_applied(builder, analysis_manager));
    transformation.apply(builder, analysis_manager);
    EXPECT_NO_THROW(sdfg.validate());

    auto& dataflow = block.dataflow();
    EXPECT_EQ(dataflow.tasklets().size(), 2);
    auto* mul = transformation.multiply();
    auto* add = transformation.add();
    ASSERT_NE(mul, nullptr);
    ASSERT_NE(add, nullptr);
    EXPECT_EQ(mul->code(), data_flow::TaskletCode::fp_mul);
    EXPECT_EQ(add->code(), data_flow::TaskletCode::fp_add);
    EXPECT_EQ(input_of(dataflow, *mul, 0), "a");
    EXPECT_EQ(input_of(dataflow, *mul, 1), "b");
    EXPECT_EQ(input_of(dataflow, *add, 0), transformation.product());
    EXPECT_EQ(input_of(dataflow, *add, 1), "c");
    EXPECT_TRUE(sdfg.is_transient(transformation.product()));
    EXPECT_EQ(sdfg.type(transformation.product()).primitive_type(), types::PrimitiveType::Float);
}

TEST(UnfuseFMATest, InverseOfFuseFMA) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& block = builder.add_block(sdfg.root());
    auto& fma = build_fma(builder, block);

    analysis::AnalysisManager analysis_manager(sdfg);
    fp_math::UnfuseFMA unfuse(fma);
    ASSERT_TRUE(unfuse.can_be_applied(builder, analysis_manager));
    unfuse.apply(builder, analysis_manager);

    fp_math::FuseFMA fuse(*unfuse.add());
    ASSERT_TRUE(fuse.can_be_applied(builder, analysis_manager));
    fuse.apply(builder, analysis_manager);
    EXPECT_NO_THROW(sdfg.validate());

    auto& dataflow = block.dataflow();
    ASSERT_EQ(dataflow.tasklets().size(), 1);
    EXPECT_EQ(fuse.fma()->code(), data_flow::TaskletCode::fp_fma);
    EXPECT_EQ(input_of(dataflow, *fuse.fma(), 0), "a");
    EXPECT_EQ(input_of(dataflow, *fuse.fma(), 1), "b");
    EXPECT_EQ(input_of(dataflow, *fuse.fma(), 2), "c");
    EXPECT_FALSE(sdfg.exists(unfuse.product()));
}

TEST(UnfuseFMATest, NotAnFMA) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();
    types::Scalar desc(types::PrimitiveType::Float);
    builder.add_container("a", desc, true);
    builder.add_container("b", desc, true);
    auto& block = builder.add_block(sdfg.root());
    auto& a = builder.add_access(block, "a");
    auto& b = builder.add_access(block, "b");
    auto& neg = builder.add_tasklet(block, data_flow::TaskletCode::fp_neg, "_out", {"_in"});
    builder.add_computational_memlet(block, a, neg, "_in", {});
    builder.add_computational_memlet(block, neg, "_out", b, {});

    analysis::AnalysisManager analysis_manager(sdfg);
    fp_math::UnfuseFMA transformation(neg);
    EXPECT_FALSE(transformation.can_be_applied(builder, analysis_manager));
}

TEST(UnfuseFMATest, RoundTripJson) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& block = builder.add_block(sdfg.root());
    auto& fma = build_fma(builder, block);

    fp_math::UnfuseFMA transformation(fma);
    nlohmann::json desc;
    transformation.to_json(desc);
    EXPECT_EQ(desc["transformation_type"], "UnfuseFMA");
    std::string error;
    EXPECT_TRUE(transformations::verify_transformation_round_trip(builder, desc, error)) << error;
}
