#include "sdfg/passes/einsum.h"

#include <gtest/gtest.h>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/function.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/for.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/array.h"
#include "sdfg/types/pointer.h"
#include "sdfg/types/scalar.h"
#include "sdfg/types/type.h"

using namespace sdfg;

TEST(EinsumLiftPassTest, GEMM) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar sym_desc(types::PrimitiveType::Int64);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);
    builder.add_container("k", sym_desc);
    builder.add_container("l", sym_desc, true);
    builder.add_container("m", sym_desc, true);
    builder.add_container("n", sym_desc, true);
    types::Scalar base_desc(types::PrimitiveType::Float);
    builder.add_container("alpha", base_desc, true);
    builder.add_container("tmp", base_desc);
    types::Array array_desc_n(base_desc, symbolic::symbol("n"));
    types::Pointer desc_n(array_desc_n);
    types::Array array_desc_m(base_desc, symbolic::symbol("m"));
    types::Pointer desc_m(array_desc_m);
    builder.add_container("A", desc_n, true);
    builder.add_container("B", desc_m, true);
    builder.add_container("C", desc_m, true);

    // Symbols
    auto zero = symbolic::zero();
    auto one = symbolic::one();
    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");
    auto k = symbolic::symbol("k");
    auto l = symbolic::symbol("l");
    auto m = symbolic::symbol("m");
    auto n = symbolic::symbol("n");

    // Add first map
    auto& map1 =
        builder.add_map(root, i, symbolic::Lt(i, l), zero, symbolic::add(i, one), ScheduleType_Sequential::create());

    // Add second map
    auto& map2 =
        builder
            .add_map(map1.root(), j, symbolic::Lt(j, m), zero, symbolic::add(j, one), ScheduleType_Sequential::create());

    // Add for node
    auto& for_node = builder.add_for(map2.root(), k, symbolic::Lt(k, n), zero, symbolic::add(k, one));

    // Add computation
    {
        auto& block = builder.add_block(for_node.root());
        auto& alpha = builder.add_access(block, "alpha");
        auto& tmp = builder.add_access(block, "tmp");
        auto& A = builder.add_access(block, "A");
        auto& B = builder.add_access(block, "B");
        auto& C1 = builder.add_access(block, "C");
        auto& C2 = builder.add_access(block, "C");
        auto& tasklet1 = builder.add_tasklet(block, data_flow::TaskletCode::fp_mul, "_out", {"_in1", "_in2"});
        builder.add_computational_memlet(block, alpha, tasklet1, "_in1", {});
        builder.add_computational_memlet(block, A, tasklet1, "_in2", {i, k});
        builder.add_computational_memlet(block, tasklet1, "_out", tmp, {});
        auto& tasklet2 = builder.add_tasklet(block, data_flow::TaskletCode::fp_fma, "_out", {"_in1", "_in2", "_in3"});
        builder.add_computational_memlet(block, tmp, tasklet2, "_in1", {});
        builder.add_computational_memlet(block, B, tasklet2, "_in2", {k, j});
        builder.add_computational_memlet(block, C1, tasklet2, "_in3", {i, j});
        builder.add_computational_memlet(block, tasklet2, "_out", C2, {i, j});
    }

    // Run pass
    analysis::AnalysisManager analysis_manager(sdfg);
    passes::EinsumLiftPass einsum_lift_pass;
    EXPECT_TRUE(einsum_lift_pass.run(builder, analysis_manager));

    // Check
    EXPECT_EQ(root.size(), 1);
    ASSERT_GE(root.size(), 1);

    auto* new_map1 = dynamic_cast<structured_control_flow::Map*>(&root.at(0).first);
    ASSERT_TRUE(new_map1);
    ASSERT_EQ(new_map1, &map1);
    EXPECT_EQ(map1.root().size(), 1);
    ASSERT_GE(map1.root().size(), 1);

    auto* new_map2 = dynamic_cast<structured_control_flow::Map*>(&map1.root().at(0).first);
    ASSERT_TRUE(new_map2);
    ASSERT_EQ(new_map2, &map2);
    EXPECT_EQ(map2.root().size(), 1);
    ASSERT_GE(map2.root().size(), 1);

    auto* new_for_node = dynamic_cast<structured_control_flow::For*>(&map2.root().at(0).first);
    ASSERT_TRUE(new_for_node);
    ASSERT_EQ(new_for_node, &for_node);
    EXPECT_EQ(for_node.root().size(), 1);
    ASSERT_GE(for_node.root().size(), 1);

    auto* new_block = dynamic_cast<structured_control_flow::Block*>(&for_node.root().at(0).first);
    ASSERT_TRUE(new_block);
    EXPECT_EQ(new_block->dataflow().data_nodes().size(), 6);
    EXPECT_EQ(new_block->dataflow().tasklets().size(), 1);
    EXPECT_EQ(new_block->dataflow().library_nodes().size(), 1);
}

TEST(EinsumExtendPassTest, GEMM) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar sym_desc(types::PrimitiveType::Int64);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);
    builder.add_container("k", sym_desc);
    builder.add_container("l", sym_desc, true);
    builder.add_container("m", sym_desc, true);
    builder.add_container("n", sym_desc, true);
    types::Scalar base_desc(types::PrimitiveType::Float);
    builder.add_container("alpha", base_desc, true);
    builder.add_container("tmp", base_desc);
    types::Array array_desc_n(base_desc, symbolic::symbol("n"));
    types::Pointer desc_n(array_desc_n);
    types::Array array_desc_m(base_desc, symbolic::symbol("m"));
    types::Pointer desc_m(array_desc_m);
    builder.add_container("A", desc_n, true);
    builder.add_container("B", desc_m, true);
    builder.add_container("C", desc_m, true);

    // Symbols
    auto zero = symbolic::zero();
    auto one = symbolic::one();
    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");
    auto k = symbolic::symbol("k");
    auto l = symbolic::symbol("l");
    auto m = symbolic::symbol("m");
    auto n = symbolic::symbol("n");

    // Add first map
    auto& map1 =
        builder.add_map(root, i, symbolic::Lt(i, l), zero, symbolic::add(i, one), ScheduleType_Sequential::create());

    // Add second map
    auto& map2 =
        builder
            .add_map(map1.root(), j, symbolic::Lt(j, m), zero, symbolic::add(j, one), ScheduleType_Sequential::create());

    // Add for node
    auto& for_node = builder.add_for(map2.root(), k, symbolic::Lt(k, n), zero, symbolic::add(k, one));

    // Add computation
    {
        auto& block = builder.add_block(for_node.root());
        auto& alpha = builder.add_access(block, "alpha");
        auto& tmp = builder.add_access(block, "tmp");
        auto& A = builder.add_access(block, "A");
        auto& B = builder.add_access(block, "B");
        auto& C1 = builder.add_access(block, "C");
        auto& C2 = builder.add_access(block, "C");
        auto& tasklet1 = builder.add_tasklet(block, data_flow::TaskletCode::fp_mul, "_out", {"_in1", "_in2"});
        builder.add_computational_memlet(block, alpha, tasklet1, "_in1", {});
        builder.add_computational_memlet(block, A, tasklet1, "_in2", {i, k});
        builder.add_computational_memlet(block, tasklet1, "_out", tmp, {});
        auto& tasklet2 = builder.add_tasklet(block, data_flow::TaskletCode::fp_fma, "_out", {"_in1", "_in2", "_in3"});
        builder.add_computational_memlet(block, tmp, tasklet2, "_in1", {});
        builder.add_computational_memlet(block, B, tasklet2, "_in2", {k, j});
        builder.add_computational_memlet(block, C1, tasklet2, "_in3", {i, j});
        builder.add_computational_memlet(block, tasklet2, "_out", C2, {i, j});
    }

    // Run pass
    analysis::AnalysisManager analysis_manager(sdfg);
    passes::EinsumLiftPass einsum_lift_pass;
    ASSERT_TRUE(einsum_lift_pass.run(builder, analysis_manager));
    passes::EinsumExtendPass einsum_extend_pass;
    EXPECT_TRUE(einsum_extend_pass.run(builder, analysis_manager));

    // Check
    EXPECT_EQ(root.size(), 1);
    ASSERT_GE(root.size(), 1);

    auto* new_map1 = dynamic_cast<structured_control_flow::Map*>(&root.at(0).first);
    ASSERT_TRUE(new_map1);
    ASSERT_EQ(new_map1, &map1);
    EXPECT_EQ(map1.root().size(), 1);
    ASSERT_GE(map1.root().size(), 1);

    auto* new_map2 = dynamic_cast<structured_control_flow::Map*>(&map1.root().at(0).first);
    ASSERT_TRUE(new_map2);
    ASSERT_EQ(new_map2, &map2);
    EXPECT_EQ(map2.root().size(), 1);
    ASSERT_GE(map2.root().size(), 1);

    auto* new_for_node = dynamic_cast<structured_control_flow::For*>(&map2.root().at(0).first);
    ASSERT_TRUE(new_for_node);
    ASSERT_EQ(new_for_node, &for_node);
    EXPECT_EQ(for_node.root().size(), 1);
    ASSERT_GE(for_node.root().size(), 1);

    auto* new_block = dynamic_cast<structured_control_flow::Block*>(&for_node.root().at(0).first);
    ASSERT_TRUE(new_block);
    EXPECT_EQ(new_block->dataflow().data_nodes().size(), 5);
    EXPECT_EQ(new_block->dataflow().tasklets().size(), 0);
    EXPECT_EQ(new_block->dataflow().library_nodes().size(), 1);
}

TEST(EinsumExpandPassTest, GEMM_1) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar sym_desc(types::PrimitiveType::Int64);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);
    builder.add_container("k", sym_desc);
    builder.add_container("l", sym_desc, true);
    builder.add_container("m", sym_desc, true);
    builder.add_container("n", sym_desc, true);
    types::Scalar base_desc(types::PrimitiveType::Float);
    builder.add_container("alpha", base_desc, true);
    builder.add_container("tmp", base_desc);
    types::Array array_desc_n(base_desc, symbolic::symbol("n"));
    types::Pointer desc_n(array_desc_n);
    types::Array array_desc_m(base_desc, symbolic::symbol("m"));
    types::Pointer desc_m(array_desc_m);
    builder.add_container("A", desc_n, true);
    builder.add_container("B", desc_m, true);
    builder.add_container("C", desc_m, true);

    // Symbols
    auto zero = symbolic::zero();
    auto one = symbolic::one();
    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");
    auto k = symbolic::symbol("k");
    auto l = symbolic::symbol("l");
    auto m = symbolic::symbol("m");
    auto n = symbolic::symbol("n");

    // Add first map
    auto& map1 =
        builder.add_map(root, i, symbolic::Lt(i, l), zero, symbolic::add(i, one), ScheduleType_Sequential::create());

    // Add second map
    auto& map2 =
        builder
            .add_map(map1.root(), j, symbolic::Lt(j, m), zero, symbolic::add(j, one), ScheduleType_Sequential::create());

    // Add for node
    auto& for_node = builder.add_for(map2.root(), k, symbolic::Lt(k, n), zero, symbolic::add(k, one));

    // Add computation
    {
        auto& block = builder.add_block(for_node.root());
        auto& alpha = builder.add_access(block, "alpha");
        auto& tmp = builder.add_access(block, "tmp");
        auto& A = builder.add_access(block, "A");
        auto& B = builder.add_access(block, "B");
        auto& C1 = builder.add_access(block, "C");
        auto& C2 = builder.add_access(block, "C");
        auto& tasklet1 = builder.add_tasklet(block, data_flow::TaskletCode::fp_mul, "_out", {"_in1", "_in2"});
        builder.add_computational_memlet(block, alpha, tasklet1, "_in1", {});
        builder.add_computational_memlet(block, A, tasklet1, "_in2", {i, k});
        builder.add_computational_memlet(block, tasklet1, "_out", tmp, {});
        auto& tasklet2 = builder.add_tasklet(block, data_flow::TaskletCode::fp_fma, "_out", {"_in1", "_in2", "_in3"});
        builder.add_computational_memlet(block, tmp, tasklet2, "_in1", {});
        builder.add_computational_memlet(block, B, tasklet2, "_in2", {k, j});
        builder.add_computational_memlet(block, C1, tasklet2, "_in3", {i, j});
        builder.add_computational_memlet(block, tasklet2, "_out", C2, {i, j});
    }

    // Run pass
    analysis::AnalysisManager analysis_manager(sdfg);
    passes::EinsumLiftPass einsum_lift_pass;
    ASSERT_TRUE(einsum_lift_pass.run(builder, analysis_manager));
    passes::EinsumExtendPass einsum_extend_pass;
    ASSERT_TRUE(einsum_extend_pass.run(builder, analysis_manager));
    passes::EinsumExpandPass einsum_expand_pass_1;
    EXPECT_TRUE(einsum_expand_pass_1.run(builder, analysis_manager));

    // Check
    EXPECT_EQ(root.size(), 1);
    ASSERT_GE(root.size(), 1);

    auto* new_map1 = dynamic_cast<structured_control_flow::Map*>(&root.at(0).first);
    ASSERT_TRUE(new_map1);
    ASSERT_EQ(new_map1, &map1);
    EXPECT_EQ(map1.root().size(), 1);
    ASSERT_GE(map1.root().size(), 1);

    auto* new_map2 = dynamic_cast<structured_control_flow::Map*>(&map1.root().at(0).first);
    ASSERT_TRUE(new_map2);
    ASSERT_EQ(new_map2, &map2);
    EXPECT_EQ(map2.root().size(), 1);
    ASSERT_GE(map2.root().size(), 1);

    auto* new_block = dynamic_cast<structured_control_flow::Block*>(&map2.root().at(0).first);
    ASSERT_TRUE(new_block);
    EXPECT_EQ(new_block->dataflow().data_nodes().size(), 5);
    EXPECT_EQ(new_block->dataflow().tasklets().size(), 0);
    EXPECT_EQ(new_block->dataflow().library_nodes().size(), 1);
}

TEST(EinsumExpandPassTest, GEMM_2) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar sym_desc(types::PrimitiveType::Int64);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);
    builder.add_container("k", sym_desc);
    builder.add_container("l", sym_desc, true);
    builder.add_container("m", sym_desc, true);
    builder.add_container("n", sym_desc, true);
    types::Scalar base_desc(types::PrimitiveType::Float);
    builder.add_container("alpha", base_desc, true);
    builder.add_container("tmp", base_desc);
    types::Array array_desc_n(base_desc, symbolic::symbol("n"));
    types::Pointer desc_n(array_desc_n);
    types::Array array_desc_m(base_desc, symbolic::symbol("m"));
    types::Pointer desc_m(array_desc_m);
    builder.add_container("A", desc_n, true);
    builder.add_container("B", desc_m, true);
    builder.add_container("C", desc_m, true);

    // Symbols
    auto zero = symbolic::zero();
    auto one = symbolic::one();
    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");
    auto k = symbolic::symbol("k");
    auto l = symbolic::symbol("l");
    auto m = symbolic::symbol("m");
    auto n = symbolic::symbol("n");

    // Add first map
    auto& map1 =
        builder.add_map(root, i, symbolic::Lt(i, l), zero, symbolic::add(i, one), ScheduleType_Sequential::create());

    // Add second map
    auto& map2 =
        builder
            .add_map(map1.root(), j, symbolic::Lt(j, m), zero, symbolic::add(j, one), ScheduleType_Sequential::create());

    // Add for node
    auto& for_node = builder.add_for(map2.root(), k, symbolic::Lt(k, n), zero, symbolic::add(k, one));

    // Add computation
    {
        auto& block = builder.add_block(for_node.root());
        auto& alpha = builder.add_access(block, "alpha");
        auto& tmp = builder.add_access(block, "tmp");
        auto& A = builder.add_access(block, "A");
        auto& B = builder.add_access(block, "B");
        auto& C1 = builder.add_access(block, "C");
        auto& C2 = builder.add_access(block, "C");
        auto& tasklet1 = builder.add_tasklet(block, data_flow::TaskletCode::fp_mul, "_out", {"_in1", "_in2"});
        builder.add_computational_memlet(block, alpha, tasklet1, "_in1", {});
        builder.add_computational_memlet(block, A, tasklet1, "_in2", {i, k});
        builder.add_computational_memlet(block, tasklet1, "_out", tmp, {});
        auto& tasklet2 = builder.add_tasklet(block, data_flow::TaskletCode::fp_fma, "_out", {"_in1", "_in2", "_in3"});
        builder.add_computational_memlet(block, tmp, tasklet2, "_in1", {});
        builder.add_computational_memlet(block, B, tasklet2, "_in2", {k, j});
        builder.add_computational_memlet(block, C1, tasklet2, "_in3", {i, j});
        builder.add_computational_memlet(block, tasklet2, "_out", C2, {i, j});
    }

    // Run pass
    analysis::AnalysisManager analysis_manager(sdfg);
    passes::EinsumLiftPass einsum_lift_pass;
    ASSERT_TRUE(einsum_lift_pass.run(builder, analysis_manager));
    passes::EinsumExtendPass einsum_extend_pass;
    ASSERT_TRUE(einsum_extend_pass.run(builder, analysis_manager));
    passes::EinsumExpandPass einsum_expand_pass_1;
    ASSERT_TRUE(einsum_expand_pass_1.run(builder, analysis_manager));
    passes::EinsumExpandPass einsum_expand_pass_2;
    EXPECT_TRUE(einsum_expand_pass_2.run(builder, analysis_manager));

    // Check
    EXPECT_EQ(root.size(), 1);
    ASSERT_GE(root.size(), 1);

    auto* new_map1 = dynamic_cast<structured_control_flow::Map*>(&root.at(0).first);
    ASSERT_TRUE(new_map1);
    ASSERT_EQ(new_map1, &map1);
    EXPECT_EQ(map1.root().size(), 1);
    ASSERT_GE(map1.root().size(), 1);

    auto* new_block = dynamic_cast<structured_control_flow::Block*>(&map1.root().at(0).first);
    ASSERT_TRUE(new_block);
    EXPECT_EQ(new_block->dataflow().data_nodes().size(), 5);
    EXPECT_EQ(new_block->dataflow().tasklets().size(), 0);
    EXPECT_EQ(new_block->dataflow().library_nodes().size(), 1);
}

TEST(EinsumExpandPassTest, GEMM_3) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar sym_desc(types::PrimitiveType::Int64);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);
    builder.add_container("k", sym_desc);
    builder.add_container("l", sym_desc, true);
    builder.add_container("m", sym_desc, true);
    builder.add_container("n", sym_desc, true);
    types::Scalar base_desc(types::PrimitiveType::Float);
    builder.add_container("alpha", base_desc, true);
    builder.add_container("tmp", base_desc);
    types::Array array_desc_n(base_desc, symbolic::symbol("n"));
    types::Pointer desc_n(array_desc_n);
    types::Array array_desc_m(base_desc, symbolic::symbol("m"));
    types::Pointer desc_m(array_desc_m);
    builder.add_container("A", desc_n, true);
    builder.add_container("B", desc_m, true);
    builder.add_container("C", desc_m, true);

    // Symbols
    auto zero = symbolic::zero();
    auto one = symbolic::one();
    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");
    auto k = symbolic::symbol("k");
    auto l = symbolic::symbol("l");
    auto m = symbolic::symbol("m");
    auto n = symbolic::symbol("n");

    // Add first map
    auto& map1 =
        builder.add_map(root, i, symbolic::Lt(i, l), zero, symbolic::add(i, one), ScheduleType_Sequential::create());

    // Add second map
    auto& map2 =
        builder
            .add_map(map1.root(), j, symbolic::Lt(j, m), zero, symbolic::add(j, one), ScheduleType_Sequential::create());

    // Add for node
    auto& for_node = builder.add_for(map2.root(), k, symbolic::Lt(k, n), zero, symbolic::add(k, one));

    // Add computation
    {
        auto& block = builder.add_block(for_node.root());
        auto& alpha = builder.add_access(block, "alpha");
        auto& tmp = builder.add_access(block, "tmp");
        auto& A = builder.add_access(block, "A");
        auto& B = builder.add_access(block, "B");
        auto& C1 = builder.add_access(block, "C");
        auto& C2 = builder.add_access(block, "C");
        auto& tasklet1 = builder.add_tasklet(block, data_flow::TaskletCode::fp_mul, "_out", {"_in1", "_in2"});
        builder.add_computational_memlet(block, alpha, tasklet1, "_in1", {});
        builder.add_computational_memlet(block, A, tasklet1, "_in2", {i, k});
        builder.add_computational_memlet(block, tasklet1, "_out", tmp, {});
        auto& tasklet2 = builder.add_tasklet(block, data_flow::TaskletCode::fp_fma, "_out", {"_in1", "_in2", "_in3"});
        builder.add_computational_memlet(block, tmp, tasklet2, "_in1", {});
        builder.add_computational_memlet(block, B, tasklet2, "_in2", {k, j});
        builder.add_computational_memlet(block, C1, tasklet2, "_in3", {i, j});
        builder.add_computational_memlet(block, tasklet2, "_out", C2, {i, j});
    }

    // Run pass
    analysis::AnalysisManager analysis_manager(sdfg);
    passes::EinsumLiftPass einsum_lift_pass;
    ASSERT_TRUE(einsum_lift_pass.run(builder, analysis_manager));
    passes::EinsumExtendPass einsum_extend_pass;
    ASSERT_TRUE(einsum_extend_pass.run(builder, analysis_manager));
    passes::EinsumExpandPass einsum_expand_pass_1;
    ASSERT_TRUE(einsum_expand_pass_1.run(builder, analysis_manager));
    passes::EinsumExpandPass einsum_expand_pass_2;
    ASSERT_TRUE(einsum_expand_pass_2.run(builder, analysis_manager));
    passes::EinsumExpandPass einsum_expand_pass_3;
    EXPECT_TRUE(einsum_expand_pass_3.run(builder, analysis_manager));

    // Check
    EXPECT_EQ(root.size(), 1);
    ASSERT_GE(root.size(), 1);

    auto* new_block = dynamic_cast<structured_control_flow::Block*>(&root.at(0).first);
    ASSERT_TRUE(new_block);
    EXPECT_EQ(new_block->dataflow().data_nodes().size(), 5);
    EXPECT_EQ(new_block->dataflow().tasklets().size(), 0);
    EXPECT_EQ(new_block->dataflow().library_nodes().size(), 1);
}

TEST(EinsumLowerPassTest, GEMM) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar sym_desc(types::PrimitiveType::Int64);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);
    builder.add_container("k", sym_desc);
    builder.add_container("l", sym_desc, true);
    builder.add_container("m", sym_desc, true);
    builder.add_container("n", sym_desc, true);
    types::Scalar base_desc(types::PrimitiveType::Float);
    builder.add_container("alpha", base_desc, true);
    builder.add_container("tmp", base_desc);
    types::Array array_desc_n(base_desc, symbolic::symbol("n"));
    types::Pointer desc_n(array_desc_n);
    types::Array array_desc_m(base_desc, symbolic::symbol("m"));
    types::Pointer desc_m(array_desc_m);
    builder.add_container("A", desc_n, true);
    builder.add_container("B", desc_m, true);
    builder.add_container("C", desc_m, true);

    // Symbols
    auto zero = symbolic::zero();
    auto one = symbolic::one();
    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");
    auto k = symbolic::symbol("k");
    auto l = symbolic::symbol("l");
    auto m = symbolic::symbol("m");
    auto n = symbolic::symbol("n");

    // Add first map
    auto& map1 =
        builder.add_map(root, i, symbolic::Lt(i, l), zero, symbolic::add(i, one), ScheduleType_Sequential::create());

    // Add second map
    auto& map2 =
        builder
            .add_map(map1.root(), j, symbolic::Lt(j, m), zero, symbolic::add(j, one), ScheduleType_Sequential::create());

    // Add for node
    auto& for_node = builder.add_for(map2.root(), k, symbolic::Lt(k, n), zero, symbolic::add(k, one));

    // Add computation
    {
        auto& block = builder.add_block(for_node.root());
        auto& alpha = builder.add_access(block, "alpha");
        auto& tmp = builder.add_access(block, "tmp");
        auto& A = builder.add_access(block, "A");
        auto& B = builder.add_access(block, "B");
        auto& C1 = builder.add_access(block, "C");
        auto& C2 = builder.add_access(block, "C");
        auto& tasklet1 = builder.add_tasklet(block, data_flow::TaskletCode::fp_mul, "_out", {"_in1", "_in2"});
        builder.add_computational_memlet(block, alpha, tasklet1, "_in1", {});
        builder.add_computational_memlet(block, A, tasklet1, "_in2", {i, k});
        builder.add_computational_memlet(block, tasklet1, "_out", tmp, {});
        auto& tasklet2 = builder.add_tasklet(block, data_flow::TaskletCode::fp_fma, "_out", {"_in1", "_in2", "_in3"});
        builder.add_computational_memlet(block, tmp, tasklet2, "_in1", {});
        builder.add_computational_memlet(block, B, tasklet2, "_in2", {k, j});
        builder.add_computational_memlet(block, C1, tasklet2, "_in3", {i, j});
        builder.add_computational_memlet(block, tasklet2, "_out", C2, {i, j});
    }

    // Run pass
    analysis::AnalysisManager analysis_manager(sdfg);
    passes::EinsumLiftPass einsum_lift_pass;
    ASSERT_TRUE(einsum_lift_pass.run(builder, analysis_manager));
    passes::EinsumExtendPass einsum_extend_pass;
    ASSERT_TRUE(einsum_extend_pass.run(builder, analysis_manager));
    passes::EinsumExpandPass einsum_expand_pass_1;
    ASSERT_TRUE(einsum_expand_pass_1.run(builder, analysis_manager));
    passes::EinsumExpandPass einsum_expand_pass_2;
    ASSERT_TRUE(einsum_expand_pass_2.run(builder, analysis_manager));
    passes::EinsumExpandPass einsum_expand_pass_3;
    ASSERT_TRUE(einsum_expand_pass_3.run(builder, analysis_manager));
    passes::EinsumLowerPass einsum_lower_pass;
    EXPECT_TRUE(einsum_lower_pass.run(builder, analysis_manager));

    // Check
    EXPECT_EQ(root.size(), 1);
    ASSERT_GE(root.size(), 1);

    auto* new_einsum_map1 = dynamic_cast<structured_control_flow::Map*>(&root.at(0).first);
    ASSERT_TRUE(new_einsum_map1);
    EXPECT_EQ(new_einsum_map1->root().size(), 1);
    ASSERT_GE(new_einsum_map1->root().size(), 1);

    auto* new_einsum_map2 = dynamic_cast<structured_control_flow::Map*>(&new_einsum_map1->root().at(0).first);
    ASSERT_TRUE(new_einsum_map2);
    EXPECT_EQ(new_einsum_map2->root().size(), 1);
    ASSERT_GE(new_einsum_map2->root().size(), 1);

    auto* new_einsum_for_node = dynamic_cast<structured_control_flow::For*>(&new_einsum_map2->root().at(0).first);
    ASSERT_TRUE(new_einsum_for_node);
    EXPECT_EQ(new_einsum_for_node->root().size(), 1);
    ASSERT_GE(new_einsum_for_node->root().size(), 1);

    auto* new_block = dynamic_cast<structured_control_flow::Block*>(&new_einsum_for_node->root().at(0).first);
    ASSERT_TRUE(new_block);
    EXPECT_EQ(new_block->dataflow().data_nodes().size(), 6);
    EXPECT_EQ(new_block->dataflow().tasklets().size(), 2);
    EXPECT_EQ(new_block->dataflow().library_nodes().size(), 0);
}

TEST(EinsumLiftPassTest, Means) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar sym_desc(types::PrimitiveType::Int64);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);
    builder.add_container("m", sym_desc, true);
    builder.add_container("n", sym_desc, true);
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Array array_desc_m(base_desc, symbolic::symbol("m"));
    types::Pointer desc_m(array_desc_m);
    types::Pointer desc(base_desc);
    builder.add_container("A", desc_m, true);
    builder.add_container("y", desc, true);
    builder.add_container("m_tmp", base_desc);

    // Symbols
    auto zero = symbolic::zero();
    auto one = symbolic::one();
    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");
    auto m = symbolic::symbol("m");
    auto n = symbolic::symbol("n");

    // Add first for loop
    auto& for_node1 = builder.add_for(root, i, symbolic::Lt(i, m), zero, symbolic::add(i, one));

    // Add initialization
    auto& block_init = builder.add_block(for_node1.root());
    auto& zero_init = builder.add_constant(block_init, "0.0", base_desc);
    auto& y_init = builder.add_access(block_init, "y");
    auto& tasklet_init = builder.add_tasklet(block_init, data_flow::TaskletCode::assign, {"_out"}, {"_in"});
    builder.add_computational_memlet(block_init, zero_init, tasklet_init, "_in", {});
    builder.add_computational_memlet(block_init, tasklet_init, "_out", y_init, {i});

    // Add second for loop
    auto& for_node2 = builder.add_for(for_node1.root(), j, symbolic::Lt(j, n), zero, symbolic::add(j, one));

    // Add computation
    auto& block = builder.add_block(for_node2.root());
    auto& A = builder.add_access(block, "A");
    auto& y1 = builder.add_access(block, "y");
    auto& y2 = builder.add_access(block, "y");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, A, tasklet, "_in1", {i, j});
    builder.add_computational_memlet(block, y1, tasklet, "_in2", {i});
    builder.add_computational_memlet(block, tasklet, "_out", y2, {i});

    // Add division
    auto& block_div = builder.add_block(for_node1.root());
    auto& m_div = builder.add_access(block_div, "m");
    auto& m_tmp = builder.add_access(block_div, "m_tmp");
    auto& y_div1 = builder.add_access(block_div, "y");
    auto& y_div2 = builder.add_access(block_div, "y");
    auto& tasklet_div1 = builder.add_tasklet(block_div, data_flow::TaskletCode::assign, {"_out"}, {"_in"});
    builder.add_computational_memlet(block_div, m_div, tasklet_div1, "_in", {});
    builder.add_computational_memlet(block_div, tasklet_div1, "_out", m_tmp, {});
    auto& tasklet_div2 = builder.add_tasklet(block_div, data_flow::TaskletCode::fp_div, {"_out"}, {"_in1", "_in2"});
    builder.add_computational_memlet(block_div, y_div1, tasklet_div2, "_in1", {i});
    builder.add_computational_memlet(block_div, m_tmp, tasklet_div2, "_in2", {});
    builder.add_computational_memlet(block_div, tasklet_div2, "_out", y_div2, {i});

    // Run pass
    analysis::AnalysisManager analysis_manager(sdfg);
    passes::EinsumLiftPass einsum_lift_pass;
    EXPECT_TRUE(einsum_lift_pass.run(builder, analysis_manager));

    // Check
    EXPECT_EQ(root.size(), 1);
    ASSERT_GE(root.size(), 1);

    auto* new_for_node1 = dynamic_cast<structured_control_flow::For*>(&root.at(0).first);
    ASSERT_TRUE(new_for_node1);
    ASSERT_EQ(new_for_node1, &for_node1);
    EXPECT_EQ(for_node1.root().size(), 3);
    ASSERT_GE(for_node1.root().size(), 3);

    auto* new_block_init = dynamic_cast<structured_control_flow::Block*>(&for_node1.root().at(0).first);
    ASSERT_TRUE(new_block_init);
    ASSERT_EQ(new_block_init, &block_init);
    EXPECT_EQ(block_init.dataflow().data_nodes().size(), 2);
    EXPECT_EQ(block_init.dataflow().tasklets().size(), 1);
    EXPECT_EQ(block_init.dataflow().library_nodes().size(), 0);

    auto* new_for_node2 = dynamic_cast<structured_control_flow::For*>(&for_node1.root().at(1).first);
    ASSERT_TRUE(new_for_node2);
    ASSERT_EQ(new_for_node2, &for_node2);
    EXPECT_EQ(for_node2.root().size(), 1);
    ASSERT_GE(for_node2.root().size(), 1);

    auto* new_block = dynamic_cast<structured_control_flow::Block*>(&for_node2.root().at(0).first);
    ASSERT_TRUE(new_block);
    ASSERT_EQ(new_block, &block);
    EXPECT_EQ(block.dataflow().data_nodes().size(), 3);
    EXPECT_EQ(block.dataflow().tasklets().size(), 0);
    EXPECT_EQ(block.dataflow().library_nodes().size(), 1);

    auto* new_block_div = dynamic_cast<structured_control_flow::Block*>(&for_node1.root().at(2).first);
    ASSERT_TRUE(new_block_div);
    ASSERT_EQ(new_block_div, &block_div);
    EXPECT_EQ(block_div.dataflow().data_nodes().size(), 4);
    EXPECT_EQ(block_div.dataflow().tasklets().size(), 2);
    EXPECT_EQ(block_div.dataflow().library_nodes().size(), 0);
}

TEST(EinsumExtendPassTest, Means) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar sym_desc(types::PrimitiveType::Int64);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);
    builder.add_container("m", sym_desc, true);
    builder.add_container("n", sym_desc, true);
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Array array_desc_m(base_desc, symbolic::symbol("m"));
    types::Pointer desc_m(array_desc_m);
    types::Pointer desc(base_desc);
    builder.add_container("A", desc_m, true);
    builder.add_container("y", desc, true);
    builder.add_container("m_tmp", base_desc);

    // Symbols
    auto zero = symbolic::zero();
    auto one = symbolic::one();
    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");
    auto m = symbolic::symbol("m");
    auto n = symbolic::symbol("n");

    // Add first for loop
    auto& for_node1 = builder.add_for(root, i, symbolic::Lt(i, m), zero, symbolic::add(i, one));

    // Add initialization
    auto& block_init = builder.add_block(for_node1.root());
    auto& zero_init = builder.add_constant(block_init, "0.0", base_desc);
    auto& y_init = builder.add_access(block_init, "y");
    auto& tasklet_init = builder.add_tasklet(block_init, data_flow::TaskletCode::assign, {"_out"}, {"_in"});
    builder.add_computational_memlet(block_init, zero_init, tasklet_init, "_in", {});
    builder.add_computational_memlet(block_init, tasklet_init, "_out", y_init, {i});

    // Add second for loop
    auto& for_node2 = builder.add_for(for_node1.root(), j, symbolic::Lt(j, n), zero, symbolic::add(j, one));

    // Add computation
    auto& block = builder.add_block(for_node2.root());
    auto& A = builder.add_access(block, "A");
    auto& y1 = builder.add_access(block, "y");
    auto& y2 = builder.add_access(block, "y");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, A, tasklet, "_in1", {i, j});
    builder.add_computational_memlet(block, y1, tasklet, "_in2", {i});
    builder.add_computational_memlet(block, tasklet, "_out", y2, {i});

    // Add division
    auto& block_div = builder.add_block(for_node1.root());
    auto& m_div = builder.add_access(block_div, "m");
    auto& m_tmp = builder.add_access(block_div, "m_tmp");
    auto& y_div1 = builder.add_access(block_div, "y");
    auto& y_div2 = builder.add_access(block_div, "y");
    auto& tasklet_div1 = builder.add_tasklet(block_div, data_flow::TaskletCode::assign, {"_out"}, {"_in"});
    builder.add_computational_memlet(block_div, m_div, tasklet_div1, "_in", {});
    builder.add_computational_memlet(block_div, tasklet_div1, "_out", m_tmp, {});
    auto& tasklet_div2 = builder.add_tasklet(block_div, data_flow::TaskletCode::fp_div, {"_out"}, {"_in1", "_in2"});
    builder.add_computational_memlet(block_div, y_div1, tasklet_div2, "_in1", {i});
    builder.add_computational_memlet(block_div, m_tmp, tasklet_div2, "_in2", {});
    builder.add_computational_memlet(block_div, tasklet_div2, "_out", y_div2, {i});

    // Run pass
    analysis::AnalysisManager analysis_manager(sdfg);
    passes::EinsumLiftPass einsum_lift_pass;
    ASSERT_TRUE(einsum_lift_pass.run(builder, analysis_manager));
    passes::EinsumExtendPass einsum_extend_pass;
    EXPECT_FALSE(einsum_extend_pass.run(builder, analysis_manager));

    // Check
    EXPECT_EQ(root.size(), 1);
    ASSERT_GE(root.size(), 1);

    auto* new_for_node1 = dynamic_cast<structured_control_flow::For*>(&root.at(0).first);
    ASSERT_TRUE(new_for_node1);
    ASSERT_EQ(new_for_node1, &for_node1);
    EXPECT_EQ(for_node1.root().size(), 3);
    ASSERT_GE(for_node1.root().size(), 3);

    auto* new_block_init = dynamic_cast<structured_control_flow::Block*>(&for_node1.root().at(0).first);
    ASSERT_TRUE(new_block_init);
    ASSERT_EQ(new_block_init, &block_init);
    EXPECT_EQ(block_init.dataflow().data_nodes().size(), 2);
    EXPECT_EQ(block_init.dataflow().tasklets().size(), 1);
    EXPECT_EQ(block_init.dataflow().library_nodes().size(), 0);

    auto* new_for_node2 = dynamic_cast<structured_control_flow::For*>(&for_node1.root().at(1).first);
    ASSERT_TRUE(new_for_node2);
    ASSERT_EQ(new_for_node2, &for_node2);
    EXPECT_EQ(for_node2.root().size(), 1);
    ASSERT_GE(for_node2.root().size(), 1);

    auto* new_block = dynamic_cast<structured_control_flow::Block*>(&for_node2.root().at(0).first);
    ASSERT_TRUE(new_block);
    ASSERT_EQ(new_block, &block);
    EXPECT_EQ(block.dataflow().data_nodes().size(), 3);
    EXPECT_EQ(block.dataflow().tasklets().size(), 0);
    EXPECT_EQ(block.dataflow().library_nodes().size(), 1);

    auto* new_block_div = dynamic_cast<structured_control_flow::Block*>(&for_node1.root().at(2).first);
    ASSERT_TRUE(new_block_div);
    ASSERT_EQ(new_block_div, &block_div);
    EXPECT_EQ(block_div.dataflow().data_nodes().size(), 4);
    EXPECT_EQ(block_div.dataflow().tasklets().size(), 2);
    EXPECT_EQ(block_div.dataflow().library_nodes().size(), 0);
}

TEST(EinsumExpandPassTest, Means) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar sym_desc(types::PrimitiveType::Int64);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);
    builder.add_container("m", sym_desc, true);
    builder.add_container("n", sym_desc, true);
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Array array_desc_m(base_desc, symbolic::symbol("m"));
    types::Pointer desc_m(array_desc_m);
    types::Pointer desc(base_desc);
    builder.add_container("A", desc_m, true);
    builder.add_container("y", desc, true);
    builder.add_container("m_tmp", base_desc);

    // Symbols
    auto zero = symbolic::zero();
    auto one = symbolic::one();
    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");
    auto m = symbolic::symbol("m");
    auto n = symbolic::symbol("n");

    // Add first for loop
    auto& for_node1 = builder.add_for(root, i, symbolic::Lt(i, m), zero, symbolic::add(i, one));

    // Add initialization
    auto& block_init = builder.add_block(for_node1.root());
    auto& zero_init = builder.add_constant(block_init, "0.0", base_desc);
    auto& y_init = builder.add_access(block_init, "y");
    auto& tasklet_init = builder.add_tasklet(block_init, data_flow::TaskletCode::assign, {"_out"}, {"_in"});
    builder.add_computational_memlet(block_init, zero_init, tasklet_init, "_in", {});
    builder.add_computational_memlet(block_init, tasklet_init, "_out", y_init, {i});

    // Add second for loop
    auto& for_node2 = builder.add_for(for_node1.root(), j, symbolic::Lt(j, n), zero, symbolic::add(j, one));

    // Add computation
    auto& block = builder.add_block(for_node2.root());
    auto& A = builder.add_access(block, "A");
    auto& y1 = builder.add_access(block, "y");
    auto& y2 = builder.add_access(block, "y");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, A, tasklet, "_in1", {i, j});
    builder.add_computational_memlet(block, y1, tasklet, "_in2", {i});
    builder.add_computational_memlet(block, tasklet, "_out", y2, {i});

    // Add division
    auto& block_div = builder.add_block(for_node1.root());
    auto& m_div = builder.add_access(block_div, "m");
    auto& m_tmp = builder.add_access(block_div, "m_tmp");
    auto& y_div1 = builder.add_access(block_div, "y");
    auto& y_div2 = builder.add_access(block_div, "y");
    auto& tasklet_div1 = builder.add_tasklet(block_div, data_flow::TaskletCode::assign, {"_out"}, {"_in"});
    builder.add_computational_memlet(block_div, m_div, tasklet_div1, "_in", {});
    builder.add_computational_memlet(block_div, tasklet_div1, "_out", m_tmp, {});
    auto& tasklet_div2 = builder.add_tasklet(block_div, data_flow::TaskletCode::fp_div, {"_out"}, {"_in1", "_in2"});
    builder.add_computational_memlet(block_div, y_div1, tasklet_div2, "_in1", {i});
    builder.add_computational_memlet(block_div, m_tmp, tasklet_div2, "_in2", {});
    builder.add_computational_memlet(block_div, tasklet_div2, "_out", y_div2, {i});

    // Run pass
    analysis::AnalysisManager analysis_manager(sdfg);
    passes::EinsumLiftPass einsum_lift_pass;
    ASSERT_TRUE(einsum_lift_pass.run(builder, analysis_manager));
    passes::EinsumExtendPass einsum_extend_pass;
    ASSERT_FALSE(einsum_extend_pass.run(builder, analysis_manager));
    passes::EinsumExpandPass einsum_expand_pass_1;
    EXPECT_TRUE(einsum_expand_pass_1.run(builder, analysis_manager));

    // Check
    EXPECT_EQ(root.size(), 1);
    ASSERT_GE(root.size(), 1);

    auto* new_for_node1 = dynamic_cast<structured_control_flow::For*>(&root.at(0).first);
    ASSERT_TRUE(new_for_node1);
    ASSERT_EQ(new_for_node1, &for_node1);
    EXPECT_EQ(for_node1.root().size(), 3);
    ASSERT_GE(for_node1.root().size(), 3);

    auto* new_block_init = dynamic_cast<structured_control_flow::Block*>(&for_node1.root().at(0).first);
    ASSERT_TRUE(new_block_init);
    ASSERT_EQ(new_block_init, &block_init);
    EXPECT_EQ(block_init.dataflow().data_nodes().size(), 2);
    EXPECT_EQ(block_init.dataflow().tasklets().size(), 1);
    EXPECT_EQ(block_init.dataflow().library_nodes().size(), 0);

    auto* new_block = dynamic_cast<structured_control_flow::Block*>(&for_node1.root().at(1).first);
    ASSERT_TRUE(new_block);
    EXPECT_EQ(new_block->dataflow().data_nodes().size(), 3);
    EXPECT_EQ(new_block->dataflow().tasklets().size(), 0);
    EXPECT_EQ(new_block->dataflow().library_nodes().size(), 1);

    auto* new_block_div = dynamic_cast<structured_control_flow::Block*>(&for_node1.root().at(2).first);
    ASSERT_TRUE(new_block_div);
    ASSERT_EQ(new_block_div, &block_div);
    EXPECT_EQ(block_div.dataflow().data_nodes().size(), 4);
    EXPECT_EQ(block_div.dataflow().tasklets().size(), 2);
    EXPECT_EQ(block_div.dataflow().library_nodes().size(), 0);
}

TEST(EinsumLowerPassTest, Means) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar sym_desc(types::PrimitiveType::Int64);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);
    builder.add_container("m", sym_desc, true);
    builder.add_container("n", sym_desc, true);
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Array array_desc_m(base_desc, symbolic::symbol("m"));
    types::Pointer desc_m(array_desc_m);
    types::Pointer desc(base_desc);
    builder.add_container("A", desc_m, true);
    builder.add_container("y", desc, true);
    builder.add_container("m_tmp", base_desc);

    // Symbols
    auto zero = symbolic::zero();
    auto one = symbolic::one();
    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");
    auto m = symbolic::symbol("m");
    auto n = symbolic::symbol("n");

    // Add first for loop
    auto& for_node1 = builder.add_for(root, i, symbolic::Lt(i, m), zero, symbolic::add(i, one));

    // Add initialization
    auto& block_init = builder.add_block(for_node1.root());
    auto& zero_init = builder.add_constant(block_init, "0.0", base_desc);
    auto& y_init = builder.add_access(block_init, "y");
    auto& tasklet_init = builder.add_tasklet(block_init, data_flow::TaskletCode::assign, {"_out"}, {"_in"});
    builder.add_computational_memlet(block_init, zero_init, tasklet_init, "_in", {});
    builder.add_computational_memlet(block_init, tasklet_init, "_out", y_init, {i});

    // Add second for loop
    auto& for_node2 = builder.add_for(for_node1.root(), j, symbolic::Lt(j, n), zero, symbolic::add(j, one));

    // Add computation
    auto& block = builder.add_block(for_node2.root());
    auto& A = builder.add_access(block, "A");
    auto& y1 = builder.add_access(block, "y");
    auto& y2 = builder.add_access(block, "y");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, A, tasklet, "_in1", {i, j});
    builder.add_computational_memlet(block, y1, tasklet, "_in2", {i});
    builder.add_computational_memlet(block, tasklet, "_out", y2, {i});

    // Add division
    auto& block_div = builder.add_block(for_node1.root());
    auto& m_div = builder.add_access(block_div, "m");
    auto& m_tmp = builder.add_access(block_div, "m_tmp");
    auto& y_div1 = builder.add_access(block_div, "y");
    auto& y_div2 = builder.add_access(block_div, "y");
    auto& tasklet_div1 = builder.add_tasklet(block_div, data_flow::TaskletCode::assign, {"_out"}, {"_in"});
    builder.add_computational_memlet(block_div, m_div, tasklet_div1, "_in", {});
    builder.add_computational_memlet(block_div, tasklet_div1, "_out", m_tmp, {});
    auto& tasklet_div2 = builder.add_tasklet(block_div, data_flow::TaskletCode::fp_div, {"_out"}, {"_in1", "_in2"});
    builder.add_computational_memlet(block_div, y_div1, tasklet_div2, "_in1", {i});
    builder.add_computational_memlet(block_div, m_tmp, tasklet_div2, "_in2", {});
    builder.add_computational_memlet(block_div, tasklet_div2, "_out", y_div2, {i});

    // Run pass
    analysis::AnalysisManager analysis_manager(sdfg);
    passes::EinsumLiftPass einsum_lift_pass;
    ASSERT_TRUE(einsum_lift_pass.run(builder, analysis_manager));
    passes::EinsumExtendPass einsum_extend_pass;
    ASSERT_FALSE(einsum_extend_pass.run(builder, analysis_manager));
    passes::EinsumExpandPass einsum_expand_pass_1;
    ASSERT_TRUE(einsum_expand_pass_1.run(builder, analysis_manager));
    passes::EinsumLowerPass einsum_lower_pass;
    EXPECT_TRUE(einsum_lower_pass.run(builder, analysis_manager));

    // Check

    // Check
    EXPECT_EQ(root.size(), 1);
    ASSERT_GE(root.size(), 1);

    auto* new_for_node1 = dynamic_cast<structured_control_flow::For*>(&root.at(0).first);
    ASSERT_TRUE(new_for_node1);
    ASSERT_EQ(new_for_node1, &for_node1);
    EXPECT_EQ(for_node1.root().size(), 3);
    ASSERT_GE(for_node1.root().size(), 3);

    auto* new_block_init = dynamic_cast<structured_control_flow::Block*>(&for_node1.root().at(0).first);
    ASSERT_TRUE(new_block_init);
    ASSERT_EQ(new_block_init, &block_init);
    EXPECT_EQ(block_init.dataflow().data_nodes().size(), 2);
    EXPECT_EQ(block_init.dataflow().tasklets().size(), 1);
    EXPECT_EQ(block_init.dataflow().library_nodes().size(), 0);

    auto* new_einsum_for_node = dynamic_cast<structured_control_flow::For*>(&for_node1.root().at(1).first);
    ASSERT_TRUE(new_einsum_for_node);
    EXPECT_EQ(new_einsum_for_node->root().size(), 1);
    ASSERT_GE(new_einsum_for_node->root().size(), 1);

    auto* new_block = dynamic_cast<structured_control_flow::Block*>(&new_einsum_for_node->root().at(0).first);
    ASSERT_TRUE(new_block);
    EXPECT_EQ(new_block->dataflow().data_nodes().size(), 3);
    EXPECT_EQ(new_block->dataflow().tasklets().size(), 1);
    EXPECT_EQ(new_block->dataflow().library_nodes().size(), 0);

    auto* new_block_div = dynamic_cast<structured_control_flow::Block*>(&for_node1.root().at(2).first);
    ASSERT_TRUE(new_block_div);
    ASSERT_EQ(new_block_div, &block_div);
    EXPECT_EQ(block_div.dataflow().data_nodes().size(), 4);
    EXPECT_EQ(block_div.dataflow().tasklets().size(), 2);
    EXPECT_EQ(block_div.dataflow().library_nodes().size(), 0);
}

TEST(EinsumLiftPassTest, Mean) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar sym_desc(types::PrimitiveType::Int64);
    builder.add_container("i", sym_desc);
    builder.add_container("m", sym_desc, true);
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("a", desc, true);
    builder.add_container("y", base_desc, true);
    builder.add_container("m_tmp", base_desc);

    // Symbols
    auto zero = symbolic::zero();
    auto i = symbolic::symbol("i");
    auto m = symbolic::symbol("m");

    // Add initialization
    auto& block_init = builder.add_block(root);
    auto& zero_init = builder.add_constant(block_init, "0.0", base_desc);
    auto& y_init = builder.add_access(block_init, "y");
    auto& tasklet_init = builder.add_tasklet(block_init, data_flow::TaskletCode::assign, {"_out"}, {"_in"});
    builder.add_computational_memlet(block_init, zero_init, tasklet_init, "_in", {});
    builder.add_computational_memlet(block_init, tasklet_init, "_out", y_init, {});

    // Add for loop
    auto& for_node = builder.add_for(root, i, symbolic::Lt(i, m), zero, symbolic::add(i, symbolic::one()));

    // Add computation
    auto& block = builder.add_block(for_node.root());
    auto& a = builder.add_access(block, "a");
    auto& y1 = builder.add_access(block, "y");
    auto& y2 = builder.add_access(block, "y");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, a, tasklet, "_in1", {i});
    builder.add_computational_memlet(block, y1, tasklet, "_in2", {});
    builder.add_computational_memlet(block, tasklet, "_out", y2, {});

    // Add division
    auto& block_div = builder.add_block(root);
    auto& m_div = builder.add_access(block_div, "m");
    auto& m_tmp = builder.add_access(block_div, "m_tmp");
    auto& y_div1 = builder.add_access(block_div, "y");
    auto& y_div2 = builder.add_access(block_div, "y");
    auto& tasklet_div1 = builder.add_tasklet(block_div, data_flow::TaskletCode::assign, {"_out"}, {"_in"});
    builder.add_computational_memlet(block_div, m_div, tasklet_div1, "_in", {});
    builder.add_computational_memlet(block_div, tasklet_div1, "_out", m_tmp, {});
    auto& tasklet_div2 = builder.add_tasklet(block_div, data_flow::TaskletCode::fp_div, {"_out"}, {"_in1", "_in2"});
    builder.add_computational_memlet(block_div, y_div1, tasklet_div2, "_in1", {});
    builder.add_computational_memlet(block_div, m_tmp, tasklet_div2, "_in2", {});
    builder.add_computational_memlet(block_div, tasklet_div2, "_out", y_div2, {});

    // Run pass
    analysis::AnalysisManager analysis_manager(sdfg);
    passes::EinsumLiftPass einsum_lift_pass;
    EXPECT_TRUE(einsum_lift_pass.run(builder, analysis_manager));

    // Check
    EXPECT_EQ(root.size(), 3);
    ASSERT_GE(root.size(), 3);

    auto* new_block_init = dynamic_cast<structured_control_flow::Block*>(&root.at(0).first);
    ASSERT_TRUE(new_block_init);
    ASSERT_EQ(new_block_init, &block_init);
    EXPECT_EQ(block_init.dataflow().data_nodes().size(), 2);
    EXPECT_EQ(block_init.dataflow().tasklets().size(), 1);
    EXPECT_EQ(block_init.dataflow().library_nodes().size(), 0);

    auto* new_for_node = dynamic_cast<structured_control_flow::For*>(&root.at(1).first);
    ASSERT_TRUE(new_for_node);
    ASSERT_EQ(new_for_node, &for_node);
    EXPECT_EQ(for_node.root().size(), 1);
    ASSERT_GE(for_node.root().size(), 1);

    auto* new_block = dynamic_cast<structured_control_flow::Block*>(&for_node.root().at(0).first);
    ASSERT_TRUE(new_block);
    ASSERT_EQ(new_block, &block);
    EXPECT_EQ(block.dataflow().data_nodes().size(), 3);
    EXPECT_EQ(block.dataflow().tasklets().size(), 0);
    EXPECT_EQ(block.dataflow().library_nodes().size(), 1);

    auto* new_block_div = dynamic_cast<structured_control_flow::Block*>(&root.at(2).first);
    ASSERT_TRUE(new_block_div);
    ASSERT_EQ(new_block_div, &block_div);
    EXPECT_EQ(block_div.dataflow().data_nodes().size(), 4);
    EXPECT_EQ(block_div.dataflow().tasklets().size(), 2);
    EXPECT_EQ(block_div.dataflow().library_nodes().size(), 0);
}

TEST(EinsumExtendPassTest, Mean) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar sym_desc(types::PrimitiveType::Int64);
    builder.add_container("i", sym_desc);
    builder.add_container("m", sym_desc, true);
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("a", desc, true);
    builder.add_container("y", base_desc, true);
    builder.add_container("m_tmp", base_desc);

    // Symbols
    auto zero = symbolic::zero();
    auto i = symbolic::symbol("i");
    auto m = symbolic::symbol("m");

    // Add initialization
    auto& block_init = builder.add_block(root);
    auto& zero_init = builder.add_constant(block_init, "0.0", base_desc);
    auto& y_init = builder.add_access(block_init, "y");
    auto& tasklet_init = builder.add_tasklet(block_init, data_flow::TaskletCode::assign, {"_out"}, {"_in"});
    builder.add_computational_memlet(block_init, zero_init, tasklet_init, "_in", {});
    builder.add_computational_memlet(block_init, tasklet_init, "_out", y_init, {});

    // Add for loop
    auto& for_node = builder.add_for(root, i, symbolic::Lt(i, m), zero, symbolic::add(i, symbolic::one()));

    // Add computation
    auto& block = builder.add_block(for_node.root());
    auto& a = builder.add_access(block, "a");
    auto& y1 = builder.add_access(block, "y");
    auto& y2 = builder.add_access(block, "y");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, a, tasklet, "_in1", {i});
    builder.add_computational_memlet(block, y1, tasklet, "_in2", {});
    builder.add_computational_memlet(block, tasklet, "_out", y2, {});

    // Add division
    auto& block_div = builder.add_block(root);
    auto& m_div = builder.add_access(block_div, "m");
    auto& m_tmp = builder.add_access(block_div, "m_tmp");
    auto& y_div1 = builder.add_access(block_div, "y");
    auto& y_div2 = builder.add_access(block_div, "y");
    auto& tasklet_div1 = builder.add_tasklet(block_div, data_flow::TaskletCode::assign, {"_out"}, {"_in"});
    builder.add_computational_memlet(block_div, m_div, tasklet_div1, "_in", {});
    builder.add_computational_memlet(block_div, tasklet_div1, "_out", m_tmp, {});
    auto& tasklet_div2 = builder.add_tasklet(block_div, data_flow::TaskletCode::fp_div, {"_out"}, {"_in1", "_in2"});
    builder.add_computational_memlet(block_div, y_div1, tasklet_div2, "_in1", {});
    builder.add_computational_memlet(block_div, m_tmp, tasklet_div2, "_in2", {});
    builder.add_computational_memlet(block_div, tasklet_div2, "_out", y_div2, {});

    // Run pass
    analysis::AnalysisManager analysis_manager(sdfg);
    passes::EinsumLiftPass einsum_lift_pass;
    ASSERT_TRUE(einsum_lift_pass.run(builder, analysis_manager));
    passes::EinsumExtendPass einsum_extend_pass;
    EXPECT_FALSE(einsum_extend_pass.run(builder, analysis_manager));

    // Check
    EXPECT_EQ(root.size(), 3);
    ASSERT_GE(root.size(), 3);

    auto* new_block_init = dynamic_cast<structured_control_flow::Block*>(&root.at(0).first);
    ASSERT_TRUE(new_block_init);
    ASSERT_EQ(new_block_init, &block_init);
    EXPECT_EQ(block_init.dataflow().data_nodes().size(), 2);
    EXPECT_EQ(block_init.dataflow().tasklets().size(), 1);
    EXPECT_EQ(block_init.dataflow().library_nodes().size(), 0);

    auto* new_for_node = dynamic_cast<structured_control_flow::For*>(&root.at(1).first);
    ASSERT_TRUE(new_for_node);
    ASSERT_EQ(new_for_node, &for_node);
    EXPECT_EQ(for_node.root().size(), 1);
    ASSERT_GE(for_node.root().size(), 1);

    auto* new_block = dynamic_cast<structured_control_flow::Block*>(&for_node.root().at(0).first);
    ASSERT_TRUE(new_block);
    ASSERT_EQ(new_block, &block);
    EXPECT_EQ(block.dataflow().data_nodes().size(), 3);
    EXPECT_EQ(block.dataflow().tasklets().size(), 0);
    EXPECT_EQ(block.dataflow().library_nodes().size(), 1);

    auto* new_block_div = dynamic_cast<structured_control_flow::Block*>(&root.at(2).first);
    ASSERT_TRUE(new_block_div);
    ASSERT_EQ(new_block_div, &block_div);
    EXPECT_EQ(block_div.dataflow().data_nodes().size(), 4);
    EXPECT_EQ(block_div.dataflow().tasklets().size(), 2);
    EXPECT_EQ(block_div.dataflow().library_nodes().size(), 0);
}

TEST(EinsumExpandPassTest, Mean) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar sym_desc(types::PrimitiveType::Int64);
    builder.add_container("i", sym_desc);
    builder.add_container("m", sym_desc, true);
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("a", desc, true);
    builder.add_container("y", base_desc, true);
    builder.add_container("m_tmp", base_desc);

    // Symbols
    auto zero = symbolic::zero();
    auto i = symbolic::symbol("i");
    auto m = symbolic::symbol("m");

    // Add initialization
    auto& block_init = builder.add_block(root);
    auto& zero_init = builder.add_constant(block_init, "0.0", base_desc);
    auto& y_init = builder.add_access(block_init, "y");
    auto& tasklet_init = builder.add_tasklet(block_init, data_flow::TaskletCode::assign, {"_out"}, {"_in"});
    builder.add_computational_memlet(block_init, zero_init, tasklet_init, "_in", {});
    builder.add_computational_memlet(block_init, tasklet_init, "_out", y_init, {});

    // Add for loop
    auto& for_node = builder.add_for(root, i, symbolic::Lt(i, m), zero, symbolic::add(i, symbolic::one()));

    // Add computation
    auto& block = builder.add_block(for_node.root());
    auto& a = builder.add_access(block, "a");
    auto& y1 = builder.add_access(block, "y");
    auto& y2 = builder.add_access(block, "y");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, a, tasklet, "_in1", {i});
    builder.add_computational_memlet(block, y1, tasklet, "_in2", {});
    builder.add_computational_memlet(block, tasklet, "_out", y2, {});

    // Add division
    auto& block_div = builder.add_block(root);
    auto& m_div = builder.add_access(block_div, "m");
    auto& m_tmp = builder.add_access(block_div, "m_tmp");
    auto& y_div1 = builder.add_access(block_div, "y");
    auto& y_div2 = builder.add_access(block_div, "y");
    auto& tasklet_div1 = builder.add_tasklet(block_div, data_flow::TaskletCode::assign, {"_out"}, {"_in"});
    builder.add_computational_memlet(block_div, m_div, tasklet_div1, "_in", {});
    builder.add_computational_memlet(block_div, tasklet_div1, "_out", m_tmp, {});
    auto& tasklet_div2 = builder.add_tasklet(block_div, data_flow::TaskletCode::fp_div, {"_out"}, {"_in1", "_in2"});
    builder.add_computational_memlet(block_div, y_div1, tasklet_div2, "_in1", {});
    builder.add_computational_memlet(block_div, m_tmp, tasklet_div2, "_in2", {});
    builder.add_computational_memlet(block_div, tasklet_div2, "_out", y_div2, {});

    // Run pass
    analysis::AnalysisManager analysis_manager(sdfg);
    passes::EinsumLiftPass einsum_lift_pass;
    ASSERT_TRUE(einsum_lift_pass.run(builder, analysis_manager));
    passes::EinsumExtendPass einsum_extend_pass;
    ASSERT_FALSE(einsum_extend_pass.run(builder, analysis_manager));
    passes::EinsumExpandPass einsum_expand_pass;
    EXPECT_TRUE(einsum_expand_pass.run(builder, analysis_manager));

    // Check
    EXPECT_EQ(root.size(), 3);
    ASSERT_GE(root.size(), 3);

    auto* new_block_init = dynamic_cast<structured_control_flow::Block*>(&root.at(0).first);
    ASSERT_TRUE(new_block_init);
    ASSERT_EQ(new_block_init, &block_init);
    EXPECT_EQ(block_init.dataflow().data_nodes().size(), 2);
    EXPECT_EQ(block_init.dataflow().tasklets().size(), 1);
    EXPECT_EQ(block_init.dataflow().library_nodes().size(), 0);

    auto* new_block = dynamic_cast<structured_control_flow::Block*>(&root.at(1).first);
    ASSERT_TRUE(new_block);
    EXPECT_EQ(new_block->dataflow().data_nodes().size(), 3);
    EXPECT_EQ(new_block->dataflow().tasklets().size(), 0);
    EXPECT_EQ(new_block->dataflow().library_nodes().size(), 1);

    auto* new_block_div = dynamic_cast<structured_control_flow::Block*>(&root.at(2).first);
    ASSERT_TRUE(new_block_div);
    ASSERT_EQ(new_block_div, &block_div);
    EXPECT_EQ(block_div.dataflow().data_nodes().size(), 4);
    EXPECT_EQ(block_div.dataflow().tasklets().size(), 2);
    EXPECT_EQ(block_div.dataflow().library_nodes().size(), 0);
}

TEST(EinsumLowerPassTest, Mean) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar sym_desc(types::PrimitiveType::Int64);
    builder.add_container("i", sym_desc);
    builder.add_container("m", sym_desc, true);
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("a", desc, true);
    builder.add_container("y", base_desc, true);
    builder.add_container("m_tmp", base_desc);

    // Symbols
    auto zero = symbolic::zero();
    auto i = symbolic::symbol("i");
    auto m = symbolic::symbol("m");

    // Add initialization
    auto& block_init = builder.add_block(root);
    auto& zero_init = builder.add_constant(block_init, "0.0", base_desc);
    auto& y_init = builder.add_access(block_init, "y");
    auto& tasklet_init = builder.add_tasklet(block_init, data_flow::TaskletCode::assign, {"_out"}, {"_in"});
    builder.add_computational_memlet(block_init, zero_init, tasklet_init, "_in", {});
    builder.add_computational_memlet(block_init, tasklet_init, "_out", y_init, {});

    // Add for loop
    auto& for_node = builder.add_for(root, i, symbolic::Lt(i, m), zero, symbolic::add(i, symbolic::one()));

    // Add computation
    auto& block = builder.add_block(for_node.root());
    auto& a = builder.add_access(block, "a");
    auto& y1 = builder.add_access(block, "y");
    auto& y2 = builder.add_access(block, "y");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, a, tasklet, "_in1", {i});
    builder.add_computational_memlet(block, y1, tasklet, "_in2", {});
    builder.add_computational_memlet(block, tasklet, "_out", y2, {});

    // Add division
    auto& block_div = builder.add_block(root);
    auto& m_div = builder.add_access(block_div, "m");
    auto& m_tmp = builder.add_access(block_div, "m_tmp");
    auto& y_div1 = builder.add_access(block_div, "y");
    auto& y_div2 = builder.add_access(block_div, "y");
    auto& tasklet_div1 = builder.add_tasklet(block_div, data_flow::TaskletCode::assign, {"_out"}, {"_in"});
    builder.add_computational_memlet(block_div, m_div, tasklet_div1, "_in", {});
    builder.add_computational_memlet(block_div, tasklet_div1, "_out", m_tmp, {});
    auto& tasklet_div2 = builder.add_tasklet(block_div, data_flow::TaskletCode::fp_div, {"_out"}, {"_in1", "_in2"});
    builder.add_computational_memlet(block_div, y_div1, tasklet_div2, "_in1", {});
    builder.add_computational_memlet(block_div, m_tmp, tasklet_div2, "_in2", {});
    builder.add_computational_memlet(block_div, tasklet_div2, "_out", y_div2, {});

    // Run pass
    analysis::AnalysisManager analysis_manager(sdfg);
    passes::EinsumLiftPass einsum_lift_pass;
    ASSERT_TRUE(einsum_lift_pass.run(builder, analysis_manager));
    passes::EinsumExtendPass einsum_extend_pass;
    ASSERT_FALSE(einsum_extend_pass.run(builder, analysis_manager));
    passes::EinsumExpandPass einsum_expand_pass;
    ASSERT_TRUE(einsum_expand_pass.run(builder, analysis_manager));
    passes::EinsumLowerPass einsum_lower_pass;
    EXPECT_TRUE(einsum_lower_pass.run(builder, analysis_manager));

    // Check
    EXPECT_EQ(root.size(), 3);
    ASSERT_GE(root.size(), 3);

    auto* new_block_init = dynamic_cast<structured_control_flow::Block*>(&root.at(0).first);
    ASSERT_TRUE(new_block_init);
    ASSERT_EQ(new_block_init, &block_init);
    EXPECT_EQ(block_init.dataflow().data_nodes().size(), 2);
    EXPECT_EQ(block_init.dataflow().tasklets().size(), 1);
    EXPECT_EQ(block_init.dataflow().library_nodes().size(), 0);

    auto* new_einsum_for_node = dynamic_cast<structured_control_flow::For*>(&root.at(1).first);
    ASSERT_TRUE(new_einsum_for_node);
    EXPECT_EQ(new_einsum_for_node->root().size(), 1);
    ASSERT_GE(new_einsum_for_node->root().size(), 1);

    auto* new_block = dynamic_cast<structured_control_flow::Block*>(&new_einsum_for_node->root().at(0).first);
    ASSERT_TRUE(new_block);
    EXPECT_EQ(new_block->dataflow().data_nodes().size(), 3);
    EXPECT_EQ(new_block->dataflow().tasklets().size(), 1);
    EXPECT_EQ(new_block->dataflow().library_nodes().size(), 0);

    auto* new_block_div = dynamic_cast<structured_control_flow::Block*>(&root.at(2).first);
    ASSERT_TRUE(new_block_div);
    ASSERT_EQ(new_block_div, &block_div);
    EXPECT_EQ(block_div.dataflow().data_nodes().size(), 4);
    EXPECT_EQ(block_div.dataflow().tasklets().size(), 2);
    EXPECT_EQ(block_div.dataflow().library_nodes().size(), 0);
}
