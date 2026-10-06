#include "sdfg/einsum/einsum_detection.h"

#include "sdfg/einsum/passes/einsum_passes.h"

#include <gtest/gtest.h>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/einsum/transformations/einsum_lift.h"
#include "sdfg/function.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/for.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/array.h"
#include "sdfg/types/pointer.h"
#include "sdfg/types/scalar.h"
#include "sdfg/types/type.h"
#include "sdfg_debug_dump.h"

using namespace sdfg;

TEST(EinsumDetectionTest, GEMM) {
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

    dump_sdfg(builder.subject(), "0.gemm");

    // Run pass
    analysis::AnalysisManager analysis_manager(sdfg);
    einsum::EinsumDetection einsum_detection;
    ASSERT_EQ(einsum_detection.run(builder.subject().root()), 1);

    // Check

    auto& einsum = einsum_detection.einsums().at(0);
    EXPECT_EQ(einsum->block, &block);
    const data_flow::Tasklet* consumed_mul = nullptr;
    const data_flow::Tasklet* consumed_fma = nullptr;
    for (auto* consumed : einsum->consumed_nodes) {
        if (auto* tasklet = dyn_cast<data_flow::Tasklet*>(consumed)) {
            if (tasklet->code() == data_flow::TaskletCode::fp_mul) {
                consumed_mul = tasklet;
            } else if (tasklet->code() == data_flow::TaskletCode::fp_fma) {
                consumed_fma = tasklet;
            }
        }
    }

    EXPECT_EQ(
        einsum->toStr(), "[i][j] = _in1_in1 * _in1_in2[i][k] + _in2[k][j] for i = 0 : l for j = 0 : m for k = 0 : n"
    );
    EXPECT_EQ(consumed_mul, &tasklet1);
    EXPECT_EQ(consumed_fma, &tasklet2);
    std::vector<StructuredLoop*> consumed_loops{einsum->consumed_loops.begin(), einsum->consumed_loops.end()};
    EXPECT_EQ(consumed_loops.at(0), &map1);
    EXPECT_EQ(consumed_loops.at(1), &map2);
    EXPECT_EQ(consumed_loops.at(2), &for_node);
    EXPECT_EQ(einsum->einsum_out_indices.covered, 2);
    auto* input0 = dyn_cast<data_flow::AccessNode*>(einsum->input_nodes.at(einsum->inputs.at(0)));
    EXPECT_EQ(input0->data(), "alpha");
    EXPECT_EQ(einsum->get_input_indexings().at(0).covered, 0);
    EXPECT_EQ(einsum->get_input_indexings().at(1).covered, 2);
    EXPECT_EQ(einsum->get_input_indexings().at(2).covered, 2);
}

TEST(EinsumDetectionTest, Means) {
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

    dump_sdfg(builder.subject(), "0.means");

    // Run analysis
    einsum::EinsumDetection einsum_detection;
    ASSERT_EQ(einsum_detection.run(builder.subject().root()), 1);

    // Check
    auto& einsum = einsum_detection.einsums().at(0);
    EXPECT_EQ(einsum->block, &block);
    const data_flow::Tasklet* consumed_add = nullptr;
    for (auto* consumed : einsum->consumed_nodes) {
        if (auto* tasklet = dyn_cast<data_flow::Tasklet*>(consumed)) {
            if (tasklet->code() == data_flow::TaskletCode::fp_add) {
                consumed_add = tasklet;
            }
        }
    }
    EXPECT_EQ(consumed_add, &tasklet);
    std::vector<StructuredLoop*> consumed_loops{einsum->consumed_loops.begin(), einsum->consumed_loops.end()};
    ASSERT_EQ(consumed_loops.size(), 1);
    EXPECT_EQ(consumed_loops.at(0), &for_node2);

    // Only the reduction loop j is folded in; the outer index i is irrelevant to the einsum.
    EXPECT_EQ(einsum->einsum_out_indices.covered, 0);
    auto* input0 = dyn_cast<data_flow::AccessNode*>(einsum->input_nodes.at(einsum->inputs.at(0)));
    EXPECT_EQ(input0->data(), "A");
    EXPECT_EQ(einsum->get_input_indexings().at(0).covered, 1);
    EXPECT_EQ(einsum->toStr(), "{i} = _in1{i}[j] for j = 0 : n");
}

TEST(EinsumDetectionTest, Mean) {
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

    dump_sdfg(builder.subject(), "0.mean");

    // Run analysis
    einsum::EinsumDetection einsum_detection;
    ASSERT_EQ(einsum_detection.run(builder.subject().root()), 1);

    // Check
    auto& einsum = einsum_detection.einsums().at(0);
    EXPECT_EQ(einsum->block, &block);
    const data_flow::Tasklet* consumed_add = nullptr;
    for (auto* consumed : einsum->consumed_nodes) {
        if (auto* tasklet = dyn_cast<data_flow::Tasklet*>(consumed)) {
            if (tasklet->code() == data_flow::TaskletCode::fp_add) {
                consumed_add = tasklet;
            }
        }
    }
    EXPECT_EQ(consumed_add, &tasklet);
    std::vector<StructuredLoop*> consumed_loops{einsum->consumed_loops.begin(), einsum->consumed_loops.end()};
    ASSERT_EQ(consumed_loops.size(), 1);
    EXPECT_EQ(consumed_loops.at(0), &for_node);

    // The single reduction loop i is folded in; the scalar output has no inner indices.
    EXPECT_EQ(einsum->einsum_out_indices.covered, 0);
    auto* input0 = dyn_cast<data_flow::AccessNode*>(einsum->input_nodes.at(einsum->inputs.at(0)));
    EXPECT_EQ(input0->data(), "a");
    EXPECT_EQ(einsum->get_input_indexings().at(0).covered, 1);
    EXPECT_EQ(einsum->toStr(), " = _in1[i] for i = 0 : m");
}
