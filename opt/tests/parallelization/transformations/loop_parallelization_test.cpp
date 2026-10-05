#include "sdfg/parallelization/transformations/loop_parallelization.h"

#include <gtest/gtest.h>

#include "sdfg/builder/structured_sdfg_builder.h"

using namespace sdfg;

// for i in [1, N): A[i] = A[i - offset] + 1
static structured_control_flow::For& build_loop(builder::StructuredSDFGBuilder& builder, int64_t offset) {
    types::Scalar idx(types::PrimitiveType::Int64);
    builder.add_container("N", idx, true);
    builder.add_container("i", idx);
    types::Scalar elem(types::PrimitiveType::Double);
    types::Pointer desc(elem);
    builder.add_container("A", desc, true);

    auto i = symbolic::symbol("i");
    auto& loop = builder.add_for(
        builder.subject().root(),
        i,
        symbolic::Lt(i, symbolic::symbol("N")),
        symbolic::one(),
        symbolic::add(i, symbolic::one())
    );
    auto& block = builder.add_block(loop.root());
    auto& a_in = builder.add_access(block, "A");
    auto& a_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, a_in, tasklet, "_in1", {symbolic::sub(i, symbolic::integer(offset))}, desc);
    builder.add_computational_memlet(block, builder.add_constant(block, "1.0", elem), tasklet, "_in2", {});
    builder.add_computational_memlet(block, tasklet, "_out", a_out, {i}, desc);
    return loop;
}

TEST(LoopParallelizationTest, IndependentIterations) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);
    auto& loop = build_loop(builder, 0);
    analysis::AnalysisManager analysis_manager(builder.subject());

    transformations::LoopParallelization transformation(loop);
    ASSERT_TRUE(transformation.can_be_applied(builder, analysis_manager));
    transformation.apply(builder, analysis_manager);

    auto* map = dynamic_cast<structured_control_flow::Map*>(&builder.subject().root().at(0));
    ASSERT_NE(map, nullptr);
    EXPECT_EQ(transformation.map(), map);
    EXPECT_EQ(map->indvar()->get_name(), "i");
}

TEST(LoopParallelizationTest, CarriedDependence) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);
    auto& loop = build_loop(builder, 1);
    analysis::AnalysisManager analysis_manager(builder.subject());

    transformations::LoopParallelization transformation(loop);
    EXPECT_FALSE(transformation.can_be_applied(builder, analysis_manager));
}

TEST(LoopParallelizationTest, Serialization) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);
    auto& loop = build_loop(builder, 0);

    nlohmann::json j;
    transformations::LoopParallelization(loop).to_json(j);
    EXPECT_EQ(j["transformation_type"], "LoopParallelization");

    auto deserialized = transformations::LoopParallelization::from_json(builder, j);
    analysis::AnalysisManager analysis_manager(builder.subject());
    EXPECT_TRUE(deserialized.can_be_applied(builder, analysis_manager));
}
