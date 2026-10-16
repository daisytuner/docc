#include "sdfg/fp_math/transformations/fuse_fma.h"

#include <gtest/gtest.h>

#include <nlohmann/json.hpp>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/transformations/transformation_registry.h"
#include "sdfg/types/pointer.h"
#include "sdfg/types/scalar.h"

using namespace sdfg;

namespace {

/// Builds `t = a * b; d = t + c` (or `d = c + t` if @p product_second) in one block.
data_flow::Tasklet& build_mul_add(
    builder::StructuredSDFGBuilder& builder,
    structured_control_flow::Block& block,
    bool product_second = false,
    const std::string& product = "t"
) {
    auto& a = builder.add_access(block, "a");
    auto& b = builder.add_access(block, "b");
    auto& c = builder.add_access(block, "c");
    auto& t = builder.add_access(block, product);
    auto& d = builder.add_access(block, "d");
    auto& mul = builder.add_tasklet(block, data_flow::TaskletCode::fp_mul, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, a, mul, "_in1", {});
    builder.add_computational_memlet(block, b, mul, "_in2", {});
    builder.add_computational_memlet(block, mul, "_out", t, {});
    auto& add = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, t, add, product_second ? "_in2" : "_in1", {});
    builder.add_computational_memlet(block, c, add, product_second ? "_in1" : "_in2", {});
    builder.add_computational_memlet(block, add, "_out", d, {});
    return add;
}

void add_operands(builder::StructuredSDFGBuilder& builder, types::PrimitiveType precision) {
    types::Scalar desc(precision);
    for (auto name : {"a", "b", "c", "d"}) {
        builder.add_container(name, desc, true);
    }
    builder.add_container("t", desc);
}

std::string input_of(const data_flow::DataFlowGraph& dataflow, const data_flow::Tasklet& tasklet, size_t i) {
    auto inputs = dataflow.in_edges_by_connector(tasklet);
    return static_cast<const data_flow::AccessNode&>(inputs.at(i)->src()).data();
}

} // namespace

TEST(FuseFMATest, ProductFirst) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();
    add_operands(builder, types::PrimitiveType::Float);
    auto& block = builder.add_block(sdfg.root());
    auto& add = build_mul_add(builder, block);

    analysis::AnalysisManager analysis_manager(sdfg);
    fp_math::FuseFMA transformation(add);
    ASSERT_TRUE(transformation.can_be_applied(builder, analysis_manager));
    transformation.apply(builder, analysis_manager);
    EXPECT_NO_THROW(sdfg.validate());

    auto& dataflow = block.dataflow();
    ASSERT_EQ(dataflow.tasklets().size(), 1);
    auto* fma = transformation.fma();
    ASSERT_NE(fma, nullptr);
    EXPECT_EQ(fma->code(), data_flow::TaskletCode::fp_fma);
    EXPECT_EQ(input_of(dataflow, *fma, 0), "a");
    EXPECT_EQ(input_of(dataflow, *fma, 1), "b");
    EXPECT_EQ(input_of(dataflow, *fma, 2), "c");
    EXPECT_EQ(static_cast<const data_flow::AccessNode&>((*dataflow.out_edges(*fma).begin()).dst()).data(), "d");
    EXPECT_FALSE(sdfg.exists("t"));
    EXPECT_EQ(dataflow.data_nodes().size(), 4);
}

TEST(FuseFMATest, ProductSecond) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();
    add_operands(builder, types::PrimitiveType::Double);
    auto& block = builder.add_block(sdfg.root());
    auto& add = build_mul_add(builder, block, true);

    analysis::AnalysisManager analysis_manager(sdfg);
    fp_math::FuseFMA transformation(add);
    ASSERT_TRUE(transformation.can_be_applied(builder, analysis_manager));
    transformation.apply(builder, analysis_manager);
    EXPECT_NO_THROW(sdfg.validate());

    auto& dataflow = block.dataflow();
    auto* fma = transformation.fma();
    ASSERT_NE(fma, nullptr);
    EXPECT_EQ(input_of(dataflow, *fma, 0), "a");
    EXPECT_EQ(input_of(dataflow, *fma, 1), "b");
    EXPECT_EQ(input_of(dataflow, *fma, 2), "c");
}

TEST(FuseFMATest, ArrayOperands) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();
    types::Scalar element(types::PrimitiveType::Float);
    types::Pointer pointer(element);
    for (auto name : {"A", "B", "C", "D"}) {
        builder.add_container(name, pointer, true);
    }
    builder.add_container("t", element);
    auto& block = builder.add_block(sdfg.root());

    auto& a = builder.add_access(block, "A");
    auto& b = builder.add_access(block, "B");
    auto& c = builder.add_access(block, "C");
    auto& t = builder.add_access(block, "t");
    auto& d = builder.add_access(block, "D");
    auto& mul = builder.add_tasklet(block, data_flow::TaskletCode::fp_mul, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, a, mul, "_in1", {symbolic::integer(1)}, pointer);
    builder.add_computational_memlet(block, b, mul, "_in2", {symbolic::integer(2)}, pointer);
    builder.add_computational_memlet(block, mul, "_out", t, {});
    auto& add = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, t, add, "_in1", {});
    builder.add_computational_memlet(block, c, add, "_in2", {symbolic::integer(3)}, pointer);
    builder.add_computational_memlet(block, add, "_out", d, {symbolic::integer(4)}, pointer);

    analysis::AnalysisManager analysis_manager(sdfg);
    fp_math::FuseFMA transformation(add);
    ASSERT_TRUE(transformation.can_be_applied(builder, analysis_manager));
    transformation.apply(builder, analysis_manager);
    EXPECT_NO_THROW(sdfg.validate());

    auto& dataflow = block.dataflow();
    auto* fma = transformation.fma();
    ASSERT_NE(fma, nullptr);
    auto inputs = dataflow.in_edges_by_connector(*fma);
    EXPECT_TRUE(symbolic::eq(inputs.at(0)->subset().at(0), symbolic::integer(1)));
    EXPECT_TRUE(symbolic::eq(inputs.at(1)->subset().at(0), symbolic::integer(2)));
    EXPECT_TRUE(symbolic::eq(inputs.at(2)->subset().at(0), symbolic::integer(3)));
    EXPECT_TRUE(symbolic::eq((*dataflow.out_edges(*fma).begin()).subset().at(0), symbolic::integer(4)));
}

TEST(FuseFMATest, ProductUsedElsewhere) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();
    add_operands(builder, types::PrimitiveType::Float);
    auto& block = builder.add_block(sdfg.root());
    auto& add = build_mul_add(builder, block);

    auto& other = builder.add_block(sdfg.root());
    auto& t = builder.add_access(other, "t");
    auto& a = builder.add_access(other, "a");
    auto& assign = builder.add_tasklet(other, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(other, t, assign, "_in", {});
    builder.add_computational_memlet(other, assign, "_out", a, {});

    analysis::AnalysisManager analysis_manager(sdfg);
    fp_math::FuseFMA transformation(add);
    EXPECT_FALSE(transformation.can_be_applied(builder, analysis_manager));
}

TEST(FuseFMATest, ProductIsArgument) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();
    add_operands(builder, types::PrimitiveType::Float);
    builder.add_container("u", types::Scalar(types::PrimitiveType::Float), true);
    auto& block = builder.add_block(sdfg.root());
    auto& add = build_mul_add(builder, block, false, "u");

    analysis::AnalysisManager analysis_manager(sdfg);
    fp_math::FuseFMA transformation(add);
    EXPECT_FALSE(transformation.can_be_applied(builder, analysis_manager));
}

TEST(FuseFMATest, MixedPrecision) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();
    add_operands(builder, types::PrimitiveType::Float);
    builder.add_container("u", types::Scalar(types::PrimitiveType::Double));
    auto& block = builder.add_block(sdfg.root());
    auto& add = build_mul_add(builder, block, false, "u");

    analysis::AnalysisManager analysis_manager(sdfg);
    fp_math::FuseFMA transformation(add);
    EXPECT_FALSE(transformation.can_be_applied(builder, analysis_manager));
}

TEST(FuseFMATest, NotAnAddition) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();
    add_operands(builder, types::PrimitiveType::Float);
    auto& block = builder.add_block(sdfg.root());
    build_mul_add(builder, block);

    analysis::AnalysisManager analysis_manager(sdfg);
    for (auto* tasklet : block.dataflow().tasklets()) {
        if (tasklet->code() == data_flow::TaskletCode::fp_mul) {
            fp_math::FuseFMA transformation(*tasklet);
            EXPECT_FALSE(transformation.can_be_applied(builder, analysis_manager));
        }
    }
}

TEST(FuseFMATest, RoundTripJson) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();
    add_operands(builder, types::PrimitiveType::Float);
    auto& block = builder.add_block(sdfg.root());
    auto& add = build_mul_add(builder, block);

    fp_math::FuseFMA transformation(add);
    nlohmann::json desc;
    transformation.to_json(desc);
    EXPECT_EQ(desc["transformation_type"], "FuseFMA");
    std::string error;
    EXPECT_TRUE(transformations::verify_transformation_round_trip(builder, desc, error)) << error;
}
