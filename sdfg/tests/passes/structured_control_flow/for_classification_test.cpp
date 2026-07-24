#include "sdfg/passes/structured_control_flow/for_classification.h"

#include <gtest/gtest.h>

#include <map>
#include <string>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_nodes/math/cmath/cmath_node.h"
#include "sdfg/structured_control_flow/reduce.h"
#include "sdfg/symbolic/symbolic.h"

using namespace sdfg;

TEST(ForClassificationTest, Basic) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);

    types::Pointer opaque_desc;
    builder.add_container("A", opaque_desc, true);
    builder.add_container("B", opaque_desc, true);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);

    // Define loop
    auto bound = symbolic::symbol("N");
    auto indvar = symbolic::symbol("i");
    auto init = symbolic::integer(0);
    auto condition = symbolic::Lt(indvar, bound);
    auto update = symbolic::add(indvar, symbolic::integer(1));

    auto& loop = builder.add_for(root, indvar, condition, init, update);
    auto& body = loop.root();

    // Add computation
    auto& block = builder.add_block(body);
    auto& a = builder.add_access(block, "A");
    auto& b = builder.add_access(block, "B");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, a, tasklet, "_in", {indvar}, desc);
    builder.add_computational_memlet(block, tasklet, "_out", b, {indvar}, desc);

    // Analysis
    auto sdfg_opt = builder.move();
    builder::StructuredSDFGBuilder builder_opt(sdfg_opt);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::ForClassificationPass conversion_pass;
    EXPECT_TRUE(conversion_pass.run(builder_opt, analysis_manager));

    auto& sdfg_map = builder_opt.subject();

    // Check
    auto map = dynamic_cast<const structured_control_flow::Map*>(&sdfg_map.root().at(0));
    EXPECT_TRUE(map != nullptr);
    EXPECT_TRUE(symbolic::eq(map->indvar(), symbolic::symbol("i")));
}

TEST(ForClassificationTest, MultiBound) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);

    types::Pointer opaque_desc;
    builder.add_container("A", opaque_desc, true);
    builder.add_container("B", opaque_desc, true);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("M", sym_desc, true);
    builder.add_container("i", sym_desc);

    // Define loop
    auto bound = symbolic::symbol("N");
    auto indvar = symbolic::symbol("i");
    auto init = symbolic::integer(0);
    auto condition = symbolic::And(symbolic::Lt(indvar, bound), symbolic::Le(indvar, symbolic::symbol("M")));
    auto update = symbolic::add(indvar, symbolic::integer(1));

    auto& loop = builder.add_for(root, indvar, condition, init, update);
    auto& body = loop.root();

    // Add computation
    auto& block = builder.add_block(body);
    auto& a = builder.add_access(block, "A");
    auto& b = builder.add_access(block, "B");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, a, tasklet, "_in", {indvar}, desc);
    builder.add_computational_memlet(block, tasklet, "_out", b, {indvar}, desc);

    // Analysis
    auto sdfg_opt = builder.move();
    builder::StructuredSDFGBuilder builder_opt(sdfg_opt);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::ForClassificationPass conversion_pass;
    EXPECT_TRUE(conversion_pass.run(builder_opt, analysis_manager));

    auto& sdfg_map = builder_opt.subject();

    // Check
    auto map = dynamic_cast<const structured_control_flow::Map*>(&sdfg_map.root().at(0));
    EXPECT_TRUE(map != nullptr);
    EXPECT_TRUE(symbolic::eq(map->indvar(), symbolic::symbol("i")));
}

TEST(ForClassificationTest, NonContiguousDomain) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);

    types::Pointer opaque_desc;
    builder.add_container("A", opaque_desc, true);
    builder.add_container("B", opaque_desc, true);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);

    // Define loop
    auto bound = symbolic::symbol("N");
    auto indvar = symbolic::symbol("i");
    auto init = symbolic::integer(0);
    auto condition = symbolic::Lt(indvar, bound);
    auto update = symbolic::add(indvar, symbolic::integer(2));

    auto& loop = builder.add_for(root, indvar, condition, init, update);
    auto& body = loop.root();

    // Add computation
    auto& block = builder.add_block(body);
    auto& a = builder.add_access(block, "A");
    auto& b = builder.add_access(block, "B");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, a, tasklet, "_in", {indvar}, desc);
    builder.add_computational_memlet(block, tasklet, "_out", b, {indvar}, desc);

    // Analysis
    auto sdfg_opt = builder.move();
    builder::StructuredSDFGBuilder builder_opt(sdfg_opt);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::ForClassificationPass conversion_pass;
    EXPECT_TRUE(conversion_pass.run(builder_opt, analysis_manager));

    auto& sdfg_map = builder_opt.subject();

    // Check
    auto map = dynamic_cast<const structured_control_flow::Map*>(&sdfg_map.root().at(0));
    EXPECT_TRUE(map != nullptr);
    EXPECT_TRUE(symbolic::eq(map->indvar(), symbolic::symbol("i")));
}

TEST(ForClassificationTest, NonCanonicalBound) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);

    types::Pointer opaque_desc;
    builder.add_container("A", opaque_desc, true);
    builder.add_container("B", opaque_desc, true);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);

    // Define loop
    auto bound = symbolic::symbol("N");
    auto indvar = symbolic::symbol("i");
    auto init = symbolic::integer(0);
    auto condition = symbolic::Ge(indvar, bound);
    auto update = symbolic::add(indvar, symbolic::integer(1));

    auto& loop = builder.add_for(root, indvar, condition, init, update);
    auto& body = loop.root();

    // Add computation
    auto& block = builder.add_block(body);
    auto& a = builder.add_access(block, "A");
    auto& b = builder.add_access(block, "B");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, a, tasklet, "_in", {indvar}, desc);
    builder.add_computational_memlet(block, tasklet, "_out", b, {indvar}, desc);

    // Analysis
    auto sdfg_opt = builder.move();
    builder::StructuredSDFGBuilder builder_opt(sdfg_opt);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::ForClassificationPass conversion_pass;
    EXPECT_TRUE(conversion_pass.run(builder_opt, analysis_manager));

    auto& sdfg_map = builder_opt.subject();

    // Check
    auto map = dynamic_cast<const structured_control_flow::Map*>(&sdfg_map.root().at(0));
    EXPECT_TRUE(map != nullptr);
    EXPECT_TRUE(symbolic::eq(map->indvar(), symbolic::symbol("i")));
}

TEST(ForClassificationTest, Shift) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);

    types::Pointer opaque_desc;
    builder.add_container("A", opaque_desc, true);
    builder.add_container("B", opaque_desc, true);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);

    // Define loop
    auto bound = symbolic::symbol("N");
    auto indvar = symbolic::symbol("i");
    auto init = symbolic::one();
    auto condition = symbolic::Lt(indvar, bound);
    auto update = symbolic::add(indvar, symbolic::integer(1));

    auto& loop = builder.add_for(root, indvar, condition, init, update);
    auto& body = loop.root();

    // Add computation
    auto& block = builder.add_block(body);
    auto& a = builder.add_access(block, "A");
    auto& b = builder.add_access(block, "B");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, a, tasklet, "_in", {indvar}, desc);
    builder.add_computational_memlet(block, tasklet, "_out", b, {indvar}, desc);

    // Analysis
    auto sdfg_opt = builder.move();
    builder::StructuredSDFGBuilder builder_opt(sdfg_opt);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::ForClassificationPass conversion_pass;
    EXPECT_TRUE(conversion_pass.run(builder_opt, analysis_manager));

    auto& sdfg_map = builder_opt.subject();

    // Check
    auto map = dynamic_cast<const structured_control_flow::Map*>(&sdfg_map.root().at(0));
    EXPECT_TRUE(map != nullptr);
    EXPECT_TRUE(symbolic::eq(map->indvar(), symbolic::symbol("i")));
}

TEST(ForClassificationTest, LastValue) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);

    types::Pointer opaque_desc;
    builder.add_container("A", opaque_desc, true);
    builder.add_container("B", opaque_desc, true);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);

    // Define loop
    auto bound = symbolic::symbol("N");
    auto indvar = symbolic::symbol("i");
    auto init = symbolic::one();
    auto condition = symbolic::Lt(indvar, bound);
    auto update = symbolic::add(indvar, symbolic::integer(1));

    auto& loop = builder.add_for(root, indvar, condition, init, update);
    auto& body = loop.root();

    // Add computation
    auto& block = builder.add_block(body);
    auto& a = builder.add_access(block, "A");
    auto& b = builder.add_access(block, "B");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, a, tasklet, "_in", {indvar}, desc);
    builder.add_computational_memlet(block, tasklet, "_out", b, {indvar}, desc);

    // Analysis
    auto sdfg_opt = builder.move();
    builder::StructuredSDFGBuilder builder_opt(sdfg_opt);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::ForClassificationPass conversion_pass;
    EXPECT_TRUE(conversion_pass.run(builder_opt, analysis_manager));

    auto& sdfg_map = builder_opt.subject();

    // Check
    auto map = dynamic_cast<const structured_control_flow::Map*>(&sdfg_map.root().at(0));
    EXPECT_TRUE(map != nullptr);
    EXPECT_TRUE(symbolic::eq(map->indvar(), symbolic::symbol("i")));
}

TEST(ForClassificationTest, Tiled) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    types::Pointer opaque_desc;
    builder.add_container("A", opaque_desc, true);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("i_tile", sym_desc);

    // Define loop
    auto bound = symbolic::symbol("N");
    auto indvar = symbolic::symbol("i_tile");
    auto init = symbolic::integer(0);
    auto condition = symbolic::Lt(indvar, bound);
    auto tile_size = symbolic::integer(8);
    auto update = symbolic::add(indvar, tile_size);

    auto& loop_outer = builder.add_for(root, indvar, condition, init, update);
    auto& body = loop_outer.root();

    auto indvar_tile = symbolic::symbol("i");
    auto init_tile = indvar;
    auto condition_tile = symbolic::
        And(symbolic::Lt(indvar_tile, symbolic::symbol("N")),
            symbolic::Lt(indvar_tile, symbolic::add(indvar, tile_size)));
    auto update_tile = symbolic::add(indvar_tile, symbolic::one());

    auto& loop_inner = builder.add_for(body, indvar_tile, condition_tile, init_tile, update_tile);
    auto& body_inner = loop_inner.root();

    // Add computation
    auto& block = builder.add_block(body_inner);
    auto& a_in = builder.add_access(block, "A");
    auto& one_node = builder.add_constant(block, "1.0", base_desc);
    auto& a_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, a_in, tasklet, "_in1", {symbolic::symbol("i")}, desc);
    builder.add_computational_memlet(block, one_node, tasklet, "_in2", {});
    builder.add_computational_memlet(block, tasklet, "_out", a_out, {symbolic::symbol("i")}, desc);

    // Analysis
    auto sdfg_opt = builder.move();
    builder::StructuredSDFGBuilder builder_opt(sdfg_opt);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::ForClassificationPass conversion_pass;
    EXPECT_TRUE(conversion_pass.run(builder_opt, analysis_manager));

    auto& sdfg_map = builder_opt.subject();

    // Check
    auto map_outer = dynamic_cast<const structured_control_flow::Map*>(&sdfg_map.root().at(0));
    EXPECT_TRUE(map_outer != nullptr);
    EXPECT_TRUE(symbolic::eq(map_outer->indvar(), symbolic::symbol("i_tile")));

    auto map_inner = dynamic_cast<const structured_control_flow::Map*>(&map_outer->root().at(0));
    EXPECT_TRUE(map_inner != nullptr);
    EXPECT_TRUE(symbolic::eq(map_inner->indvar(), symbolic::symbol("i")));
}

TEST(ForClassificationTest, NonContiguousMemory) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    types::Pointer desc2(static_cast<const types::IType&>(desc));

    types::Pointer opaque_desc;
    builder.add_container("B_", opaque_desc, true);
    builder.add_container("B", opaque_desc);
    builder.add_container("A", opaque_desc, true);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);

    // Define loop
    auto bound = symbolic::symbol("N");
    auto indvar = symbolic::symbol("i");
    auto init = symbolic::integer(0);
    auto condition = symbolic::Lt(indvar, bound);
    auto update = symbolic::add(indvar, symbolic::integer(2));

    auto& loop = builder.add_for(root, indvar, condition, init, update);
    auto& body = loop.root();

    // Add Dereference
    {
        auto& block = builder.add_block(body);
        auto& b_ = builder.add_access(block, "B_");
        auto& b = builder.add_access(block, "B");
        builder.add_dereference_memlet(block, b_, b, true, desc2);
    }

    // Add computation
    {
        auto& block = builder.add_block(body);
        auto& a = builder.add_access(block, "A");
        auto& b = builder.add_access(block, "B");
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
        builder.add_computational_memlet(block, a, tasklet, "_in", {indvar}, desc);
        builder.add_computational_memlet(block, tasklet, "_out", b, {indvar}, desc);
    }

    // Analysis
    auto sdfg_opt = builder.move();
    builder::StructuredSDFGBuilder builder_opt(sdfg_opt);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::ForClassificationPass conversion_pass;
    EXPECT_TRUE(conversion_pass.run(builder_opt, analysis_manager));
}

TEST(ForClassificationTest, ScalarSumReduction) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);

    types::Pointer opaque_desc;
    builder.add_container("A", opaque_desc, true);
    builder.add_container("sum", base_desc, true);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);

    // Define loop
    auto bound = symbolic::symbol("N");
    auto indvar = symbolic::symbol("i");
    auto init = symbolic::integer(0);
    auto condition = symbolic::Lt(indvar, bound);
    auto update = symbolic::add(indvar, symbolic::integer(1));

    auto& loop = builder.add_for(root, indvar, condition, init, update);
    auto& body = loop.root();

    // sum = sum + A[i]
    auto& block = builder.add_block(body);
    auto& a = builder.add_access(block, "A");
    auto& sum_in = builder.add_access(block, "sum");
    auto& sum_out = builder.add_access(block, "sum");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, a, tasklet, "_in1", {indvar}, desc);
    builder.add_computational_memlet(block, sum_in, tasklet, "_in2", {}, base_desc);
    builder.add_computational_memlet(block, tasklet, "_out", sum_out, {}, base_desc);

    // Analysis
    auto sdfg_opt = builder.move();
    builder::StructuredSDFGBuilder builder_opt(sdfg_opt);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::ForClassificationPass conversion_pass;
    EXPECT_TRUE(conversion_pass.run(builder_opt, analysis_manager));

    auto& sdfg_red = builder_opt.subject();

    // Check
    auto reduce = dynamic_cast<const structured_control_flow::Reduce*>(&sdfg_red.root().at(0));
    ASSERT_TRUE(reduce != nullptr);
    EXPECT_TRUE(symbolic::eq(reduce->indvar(), symbolic::symbol("i")));
    ASSERT_EQ(reduce->reductions().size(), 1);
    EXPECT_EQ(reduce->reductions()[0].operation, structured_control_flow::ReductionOperation::Add);
    EXPECT_EQ(reduce->reductions()[0].container, "sum");
}

TEST(ForClassificationTest, ScalarProductReduction) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);

    types::Pointer opaque_desc;
    builder.add_container("A", opaque_desc, true);
    builder.add_container("prod", base_desc, true);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);

    auto bound = symbolic::symbol("N");
    auto indvar = symbolic::symbol("i");
    auto init = symbolic::integer(0);
    auto condition = symbolic::Lt(indvar, bound);
    auto update = symbolic::add(indvar, symbolic::integer(1));

    auto& loop = builder.add_for(root, indvar, condition, init, update);
    auto& body = loop.root();

    // prod = prod * A[i]
    auto& block = builder.add_block(body);
    auto& a = builder.add_access(block, "A");
    auto& prod_in = builder.add_access(block, "prod");
    auto& prod_out = builder.add_access(block, "prod");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_mul, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, a, tasklet, "_in1", {indvar}, desc);
    builder.add_computational_memlet(block, prod_in, tasklet, "_in2", {}, base_desc);
    builder.add_computational_memlet(block, tasklet, "_out", prod_out, {}, base_desc);

    auto sdfg_opt = builder.move();
    builder::StructuredSDFGBuilder builder_opt(sdfg_opt);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::ForClassificationPass conversion_pass;
    EXPECT_TRUE(conversion_pass.run(builder_opt, analysis_manager));

    auto& sdfg_red = builder_opt.subject();

    auto reduce = dynamic_cast<const structured_control_flow::Reduce*>(&sdfg_red.root().at(0));
    ASSERT_TRUE(reduce != nullptr);
    ASSERT_EQ(reduce->reductions().size(), 1);
    EXPECT_EQ(reduce->reductions()[0].operation, structured_control_flow::ReductionOperation::Mul);
    EXPECT_EQ(reduce->reductions()[0].container, "prod");
}

TEST(ForClassificationTest, FloatMaxReductionCMath) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);

    types::Pointer opaque_desc;
    builder.add_container("A", opaque_desc, true);
    builder.add_container("m", base_desc, true);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);

    auto bound = symbolic::symbol("N");
    auto indvar = symbolic::symbol("i");
    auto init = symbolic::integer(0);
    auto condition = symbolic::Lt(indvar, bound);
    auto update = symbolic::add(indvar, symbolic::integer(1));

    auto& loop = builder.add_for(root, indvar, condition, init, update);
    auto& body = loop.root();

    // m = fmax(m, A[i])
    auto& block = builder.add_block(body);
    auto& a = builder.add_access(block, "A");
    auto& m_in = builder.add_access(block, "m");
    auto& m_out = builder.add_access(block, "m");
    auto& node = static_cast<math::cmath::CMathNode&>(builder.add_library_node<math::cmath::CMathNode>(
        block, DebugInfo(), math::cmath::CMathFunction::fmax, types::PrimitiveType::Float
    ));
    builder.add_computational_memlet(block, m_in, node, "_in1", {}, base_desc);
    builder.add_computational_memlet(block, a, node, "_in2", {indvar}, desc);
    builder.add_computational_memlet(block, node, "_out", m_out, {}, base_desc);

    auto sdfg_opt = builder.move();
    builder::StructuredSDFGBuilder builder_opt(sdfg_opt);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::ForClassificationPass conversion_pass;
    EXPECT_TRUE(conversion_pass.run(builder_opt, analysis_manager));

    auto& sdfg_red = builder_opt.subject();

    auto reduce = dynamic_cast<const structured_control_flow::Reduce*>(&sdfg_red.root().at(0));
    ASSERT_TRUE(reduce != nullptr);
    ASSERT_EQ(reduce->reductions().size(), 1);
    EXPECT_EQ(reduce->reductions()[0].operation, structured_control_flow::ReductionOperation::Max);
    EXPECT_EQ(reduce->reductions()[0].container, "m");
}

TEST(ForClassificationTest, FusedSumAndProductReduction) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);

    types::Pointer opaque_desc;
    builder.add_container("A", opaque_desc, true);
    builder.add_container("sum", base_desc, true);
    builder.add_container("prod", base_desc, true);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);

    auto bound = symbolic::symbol("N");
    auto indvar = symbolic::symbol("i");
    auto init = symbolic::integer(0);
    auto condition = symbolic::Lt(indvar, bound);
    auto update = symbolic::add(indvar, symbolic::integer(1));

    auto& loop = builder.add_for(root, indvar, condition, init, update);
    auto& body = loop.root();

    // sum = sum + A[i]
    {
        auto& block = builder.add_block(body);
        auto& a = builder.add_access(block, "A");
        auto& sum_in = builder.add_access(block, "sum");
        auto& sum_out = builder.add_access(block, "sum");
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
        builder.add_computational_memlet(block, a, tasklet, "_in1", {indvar}, desc);
        builder.add_computational_memlet(block, sum_in, tasklet, "_in2", {}, base_desc);
        builder.add_computational_memlet(block, tasklet, "_out", sum_out, {}, base_desc);
    }
    // prod = prod * A[i]
    {
        auto& block = builder.add_block(body);
        auto& a = builder.add_access(block, "A");
        auto& prod_in = builder.add_access(block, "prod");
        auto& prod_out = builder.add_access(block, "prod");
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_mul, "_out", {"_in1", "_in2"});
        builder.add_computational_memlet(block, a, tasklet, "_in1", {indvar}, desc);
        builder.add_computational_memlet(block, prod_in, tasklet, "_in2", {}, base_desc);
        builder.add_computational_memlet(block, tasklet, "_out", prod_out, {}, base_desc);
    }

    auto sdfg_opt = builder.move();
    builder::StructuredSDFGBuilder builder_opt(sdfg_opt);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::ForClassificationPass conversion_pass;
    EXPECT_TRUE(conversion_pass.run(builder_opt, analysis_manager));

    auto& sdfg_red = builder_opt.subject();

    auto reduce = dynamic_cast<const structured_control_flow::Reduce*>(&sdfg_red.root().at(0));
    ASSERT_TRUE(reduce != nullptr);
    ASSERT_EQ(reduce->reductions().size(), 2);

    std::map<std::string, structured_control_flow::ReductionOperation> ops;
    for (auto& r : reduce->reductions()) {
        ops[r.container] = r.operation;
    }
    ASSERT_EQ(ops.count("sum"), 1);
    ASSERT_EQ(ops.count("prod"), 1);
    EXPECT_EQ(ops["sum"], structured_control_flow::ReductionOperation::Add);
    EXPECT_EQ(ops["prod"], structured_control_flow::ReductionOperation::Mul);
}

TEST(ForClassificationTest, RecurrenceIsNotReduction) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);

    types::Pointer opaque_desc;
    builder.add_container("A", opaque_desc, true);
    builder.add_container("c", base_desc, true);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);

    auto bound = symbolic::symbol("N");
    auto indvar = symbolic::symbol("i");
    auto init = symbolic::one();
    auto condition = symbolic::Lt(indvar, bound);
    auto update = symbolic::add(indvar, symbolic::integer(1));

    auto& loop = builder.add_for(root, indvar, condition, init, update);
    auto& body = loop.root();

    // A[i] = A[i-1] + c  (genuine loop-carried recurrence, not a reduction)
    auto& block = builder.add_block(body);
    auto& a_in = builder.add_access(block, "A");
    auto& c_in = builder.add_access(block, "c");
    auto& a_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, a_in, tasklet, "_in1", {symbolic::sub(indvar, symbolic::integer(1))}, desc);
    builder.add_computational_memlet(block, c_in, tasklet, "_in2", {}, base_desc);
    builder.add_computational_memlet(block, tasklet, "_out", a_out, {indvar}, desc);

    auto sdfg_opt = builder.move();
    builder::StructuredSDFGBuilder builder_opt(sdfg_opt);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::ForClassificationPass conversion_pass;
    EXPECT_FALSE(conversion_pass.run(builder_opt, analysis_manager));

    auto& sdfg_res = builder_opt.subject();

    // Must remain a plain For: neither Map nor Reduce
    EXPECT_TRUE(dynamic_cast<const structured_control_flow::For*>(&sdfg_res.root().at(0)) != nullptr);
    EXPECT_TRUE(dynamic_cast<const structured_control_flow::Map*>(&sdfg_res.root().at(0)) == nullptr);
    EXPECT_TRUE(dynamic_cast<const structured_control_flow::Reduce*>(&sdfg_res.root().at(0)) == nullptr);
}
