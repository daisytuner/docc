#include "sdfg/analysis/loop_carried_dependency_analysis.h"

#include <gtest/gtest.h>

#include <map>

#include "sdfg/analysis/analysis.h"
#include "sdfg/analysis/data_dependency_analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_nodes/math/cmath/cmath_node.h"
#include "sdfg/symbolic/symbolic.h"

using namespace sdfg;

namespace {

// Build a simple loop: for (i = 0; i < N; ++i) { B = A[i]; }
// Expect WAW on scalar B (no in-loop subset to differentiate).
TEST(LoopCarriedDependencyAnalysisTest, Reduce_WriteScalar) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer edge_desc(base_desc);
    types::Pointer ptr_desc;

    builder.add_container("A", ptr_desc, true);
    builder.add_container("B", base_desc, true);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);

    auto bound = symbolic::symbol("N");
    auto indvar = symbolic::symbol("i");
    auto init = symbolic::integer(0);
    auto condition = symbolic::Lt(indvar, bound);
    auto update = symbolic::add(indvar, symbolic::integer(1));

    auto& loop = builder.add_for(root, indvar, condition, init, update);
    auto& body = loop.root();

    auto& block = builder.add_block(body);
    auto& a1 = builder.add_access(block, "A");
    auto& b_out = builder.add_access(block, "B");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, a1, tasklet, "_in", {indvar}, edge_desc);
    builder.add_computational_memlet(block, tasklet, "_out", b_out, {});

    analysis::AnalysisManager analysis_manager(sdfg);
    auto& lcd = analysis_manager.get<analysis::LoopCarriedDependencyAnalysis>();

    EXPECT_TRUE(lcd.available(loop));
    auto& deps = lcd.dependencies(loop);
    ASSERT_EQ(deps.count("B"), 1u);
    EXPECT_EQ(deps.at("B").type, analysis::LOOP_CARRIED_DEPENDENCY_WRITE_WRITE);
    EXPECT_TRUE(lcd.has_loop_carried(loop));
    EXPECT_FALSE(lcd.has_loop_carried_raw(loop));

    // Last-value write (B = A[i]) is a WAW hazard, NOT a reduction.
    EXPECT_FALSE(lcd.has_reductions(loop));
    EXPECT_TRUE(lcd.reductions(loop).empty());
    EXPECT_FALSE(lcd.is_reduction_only(loop));
}

// for (i = 0; i < N; ++i) { B = B + A[i]; }
// Expect RAW on scalar B (read+write same scalar in body).
TEST(LoopCarriedDependencyAnalysisTest, Sum_RawOnScalar) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer edge_desc(base_desc);
    types::Pointer ptr_desc;

    builder.add_container("A", ptr_desc, true);
    builder.add_container("B", base_desc, true);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);

    auto bound = symbolic::symbol("N");
    auto indvar = symbolic::symbol("i");
    auto init = symbolic::integer(0);
    auto condition = symbolic::Lt(indvar, bound);
    auto update = symbolic::add(indvar, symbolic::integer(1));

    auto& loop = builder.add_for(root, indvar, condition, init, update);
    auto& body = loop.root();

    auto& block = builder.add_block(body);
    auto& a1 = builder.add_access(block, "A");
    auto& b_in = builder.add_access(block, "B");
    auto& b_out = builder.add_access(block, "B");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, a1, tasklet, "_in1", {indvar}, edge_desc);
    builder.add_computational_memlet(block, b_in, tasklet, "_in2", {});
    builder.add_computational_memlet(block, tasklet, "_out", b_out, {});

    analysis::AnalysisManager analysis_manager(sdfg);
    auto& lcd = analysis_manager.get<analysis::LoopCarriedDependencyAnalysis>();

    auto& deps = lcd.dependencies(loop);
    ASSERT_EQ(deps.count("B"), 1u);
    EXPECT_EQ(deps.at("B").type, analysis::LOOP_CARRIED_DEPENDENCY_READ_WRITE);
    EXPECT_TRUE(lcd.has_loop_carried_raw(loop));

    // B = B + A[i] is an additive reduction on the scalar accumulator B.
    ASSERT_TRUE(lcd.has_reductions(loop));
    auto& reductions = lcd.reductions(loop);
    ASSERT_EQ(reductions.size(), 1u);
    EXPECT_EQ(reductions[0].operation, structured_control_flow::ReductionOperation::Add);
    EXPECT_EQ(reductions[0].container, "B");
    EXPECT_TRUE(lcd.is_reduction_only(loop));
}

// for (i = 0; i < N; ++i) { B = A[i] * C[i] + B; }
// Fused multiply-add (_out = _in1 * _in2 + _in3) with the accumulator B as the
// addend operand is an additive reduction.
TEST(LoopCarriedDependencyAnalysisTest, Fma_AddendAccumulator_IsReduction) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer edge_desc(base_desc);
    types::Pointer ptr_desc;

    builder.add_container("A", ptr_desc, true);
    builder.add_container("C", ptr_desc, true);
    builder.add_container("B", base_desc, true);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);

    auto bound = symbolic::symbol("N");
    auto indvar = symbolic::symbol("i");
    auto init = symbolic::integer(0);
    auto condition = symbolic::Lt(indvar, bound);
    auto update = symbolic::add(indvar, symbolic::integer(1));

    auto& loop = builder.add_for(root, indvar, condition, init, update);
    auto& body = loop.root();

    auto& block = builder.add_block(body);
    auto& a1 = builder.add_access(block, "A");
    auto& c1 = builder.add_access(block, "C");
    auto& b_in = builder.add_access(block, "B");
    auto& b_out = builder.add_access(block, "B");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_fma, "_out", {"_in1", "_in2", "_in3"});
    builder.add_computational_memlet(block, a1, tasklet, "_in1", {indvar}, edge_desc);
    builder.add_computational_memlet(block, c1, tasklet, "_in2", {indvar}, edge_desc);
    builder.add_computational_memlet(block, b_in, tasklet, "_in3", {});
    builder.add_computational_memlet(block, tasklet, "_out", b_out, {});

    analysis::AnalysisManager analysis_manager(sdfg);
    auto& lcd = analysis_manager.get<analysis::LoopCarriedDependencyAnalysis>();

    auto& deps = lcd.dependencies(loop);
    ASSERT_EQ(deps.count("B"), 1u);
    EXPECT_EQ(deps.at("B").type, analysis::LOOP_CARRIED_DEPENDENCY_READ_WRITE);

    ASSERT_TRUE(lcd.has_reductions(loop));
    auto& reductions = lcd.reductions(loop);
    ASSERT_EQ(reductions.size(), 1u);
    EXPECT_EQ(reductions[0].operation, structured_control_flow::ReductionOperation::Add);
    EXPECT_EQ(reductions[0].container, "B");
    EXPECT_TRUE(lcd.is_reduction_only(loop));
}

// for (i = 0; i < N; ++i) { B = B * A[i] + C[i]; }
// Fused multiply-add with the accumulator B as a *multiplicand* is not a
// reorderable reduction (the update is acc = acc * b + c).
TEST(LoopCarriedDependencyAnalysisTest, Fma_MultiplicandAccumulator_NotReduction) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer edge_desc(base_desc);
    types::Pointer ptr_desc;

    builder.add_container("A", ptr_desc, true);
    builder.add_container("C", ptr_desc, true);
    builder.add_container("B", base_desc, true);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);

    auto bound = symbolic::symbol("N");
    auto indvar = symbolic::symbol("i");
    auto init = symbolic::integer(0);
    auto condition = symbolic::Lt(indvar, bound);
    auto update = symbolic::add(indvar, symbolic::integer(1));

    auto& loop = builder.add_for(root, indvar, condition, init, update);
    auto& body = loop.root();

    auto& block = builder.add_block(body);
    auto& b_in = builder.add_access(block, "B");
    auto& a1 = builder.add_access(block, "A");
    auto& c1 = builder.add_access(block, "C");
    auto& b_out = builder.add_access(block, "B");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_fma, "_out", {"_in1", "_in2", "_in3"});
    builder.add_computational_memlet(block, b_in, tasklet, "_in1", {});
    builder.add_computational_memlet(block, a1, tasklet, "_in2", {indvar}, edge_desc);
    builder.add_computational_memlet(block, c1, tasklet, "_in3", {indvar}, edge_desc);
    builder.add_computational_memlet(block, tasklet, "_out", b_out, {});

    analysis::AnalysisManager analysis_manager(sdfg);
    auto& lcd = analysis_manager.get<analysis::LoopCarriedDependencyAnalysis>();

    auto& deps = lcd.dependencies(loop);
    ASSERT_EQ(deps.count("B"), 1u);
    EXPECT_EQ(deps.at("B").type, analysis::LOOP_CARRIED_DEPENDENCY_READ_WRITE);

    // Accumulator on a multiplicand port => not a reduction.
    EXPECT_FALSE(lcd.has_reductions(loop));
    EXPECT_TRUE(lcd.reductions(loop).empty());
    EXPECT_FALSE(lcd.is_reduction_only(loop));
}

// for (i = 0; i < N; ++i) { A[i] = ...; }
// No loop-carried dependency: each iter writes a distinct element.
TEST(LoopCarriedDependencyAnalysisTest, IndependentArrayWrite_NoCarry) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer edge_desc(base_desc);
    types::Pointer ptr_desc;

    builder.add_container("A", ptr_desc, true);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);

    auto bound = symbolic::symbol("N");
    auto indvar = symbolic::symbol("i");
    auto init = symbolic::integer(0);
    auto condition = symbolic::Lt(indvar, bound);
    auto update = symbolic::add(indvar, symbolic::integer(1));

    auto& loop = builder.add_for(root, indvar, condition, init, update);
    auto& body = loop.root();

    auto& block = builder.add_block(body);
    auto& a_out = builder.add_access(block, "A");
    auto& zero = builder.add_constant(block, "0", base_desc);
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, zero, tasklet, "_in", {});
    builder.add_computational_memlet(block, tasklet, "_out", a_out, {indvar}, edge_desc);

    analysis::AnalysisManager analysis_manager(sdfg);
    auto& lcd = analysis_manager.get<analysis::LoopCarriedDependencyAnalysis>();

    EXPECT_TRUE(lcd.available(loop));
    EXPECT_FALSE(lcd.has_loop_carried(loop));
    EXPECT_TRUE(lcd.dependencies(loop).empty());
    EXPECT_TRUE(lcd.pairs(loop).empty());

    // No carried dependency at all => no reductions.
    EXPECT_FALSE(lcd.has_reductions(loop));
    EXPECT_TRUE(lcd.reductions(loop).empty());
    EXPECT_FALSE(lcd.is_reduction_only(loop));
}

// Shift dependency: for (i = 1; i < N; ++i) { A[i] = A[i-1]; }
// Expect RAW on A.
TEST(LoopCarriedDependencyAnalysisTest, Shift_Raw) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer edge_desc(base_desc);
    types::Pointer ptr_desc;

    builder.add_container("A", ptr_desc, true);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);

    auto bound = symbolic::symbol("N");
    auto indvar = symbolic::symbol("i");
    auto init = symbolic::integer(1);
    auto condition = symbolic::Lt(indvar, bound);
    auto update = symbolic::add(indvar, symbolic::integer(1));

    auto& loop = builder.add_for(root, indvar, condition, init, update);
    auto& body = loop.root();

    auto& block = builder.add_block(body);
    auto& a_in = builder.add_access(block, "A");
    auto& a_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder
        .add_computational_memlet(block, a_in, tasklet, "_in", {symbolic::sub(indvar, symbolic::integer(1))}, edge_desc);
    builder.add_computational_memlet(block, tasklet, "_out", a_out, {indvar}, edge_desc);

    analysis::AnalysisManager analysis_manager(sdfg);
    auto& lcd = analysis_manager.get<analysis::LoopCarriedDependencyAnalysis>();

    auto& deps = lcd.dependencies(loop);
    ASSERT_EQ(deps.count("A"), 1u);
    EXPECT_EQ(deps.at("A").type, analysis::LOOP_CARRIED_DEPENDENCY_READ_WRITE);
    EXPECT_TRUE(lcd.has_loop_carried_raw(loop));

    // A[i] = A[i-1] is a genuine recurrence (shift), not a reorderable reduction.
    EXPECT_FALSE(lcd.has_reductions(loop));
    EXPECT_TRUE(lcd.reductions(loop).empty());
    EXPECT_FALSE(lcd.is_reduction_only(loop));
}

TEST(LoopCarriedDependencyAnalysisTest, Last_1D) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer edge_desc(base_desc);
    types::Pointer desc;

    builder.add_container("A", desc, true);
    builder.add_container("B", base_desc, true);

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
    auto& a1 = builder.add_access(block, "A");
    auto& b_out = builder.add_access(block, "B");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, a1, tasklet, "_in", {indvar}, edge_desc);
    builder.add_computational_memlet(block, tasklet, "_out", b_out, {});

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::LoopCarriedDependencyAnalysis>();
    auto& dependencies = analysis.dependencies(loop);

    // Check
    EXPECT_EQ(dependencies.size(), 1);
    EXPECT_EQ(dependencies.at("B").type, analysis::LoopCarriedDependency::LOOP_CARRIED_DEPENDENCY_WRITE_WRITE);
}

TEST(LoopCarriedDependencyAnalysisTest, Sum_1D) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer edge_desc(base_desc);
    types::Pointer desc;

    builder.add_container("A", desc, true);
    builder.add_container("B", base_desc, true);

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
    auto& a1 = builder.add_access(block, "A");
    auto& b_in = builder.add_access(block, "B");
    auto& b_out = builder.add_access(block, "B");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, a1, tasklet, "_in1", {indvar}, edge_desc);
    builder.add_computational_memlet(block, b_in, tasklet, "_in2", {});
    builder.add_computational_memlet(block, tasklet, "_out", b_out, {});

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::LoopCarriedDependencyAnalysis>();
    auto& dependencies = analysis.dependencies(loop);

    // Check
    EXPECT_EQ(dependencies.size(), 1);
    EXPECT_EQ(dependencies.at("B").type, analysis::LoopCarriedDependency::LOOP_CARRIED_DEPENDENCY_READ_WRITE);

    // B = B + A[i] is an additive reduction.
    ASSERT_TRUE(analysis.has_reductions(loop));
    ASSERT_EQ(analysis.reductions(loop).size(), 1u);
    EXPECT_EQ(analysis.reductions(loop)[0].operation, structured_control_flow::ReductionOperation::Add);
    EXPECT_EQ(analysis.reductions(loop)[0].container, "B");
    EXPECT_TRUE(analysis.is_reduction_only(loop));
}

TEST(LoopCarriedDependencyAnalysisTest, Shift_1D) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer edge_desc(base_desc);
    types::Pointer desc;

    builder.add_container("A", desc, true);

    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);

    // Define loop
    auto bound = symbolic::symbol("N");
    auto indvar = symbolic::symbol("i");
    auto init = symbolic::integer(1);
    auto condition = symbolic::Lt(indvar, bound);
    auto update = symbolic::add(indvar, symbolic::integer(1));

    auto& loop = builder.add_for(root, indvar, condition, init, update);
    auto& body = loop.root();

    // Add computation
    auto& block = builder.add_block(body);
    auto& a1 = builder.add_access(block, "A");
    auto& a2 = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder
        .add_computational_memlet(block, a1, tasklet, "_in", {symbolic::sub(indvar, symbolic::integer(1))}, edge_desc);
    builder.add_computational_memlet(block, tasklet, "_out", a2, {indvar}, edge_desc);

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::LoopCarriedDependencyAnalysis>();
    auto& dependencies = analysis.dependencies(loop);

    // Check
    EXPECT_EQ(dependencies.size(), 1);
    EXPECT_EQ(dependencies.at("A").type, analysis::LoopCarriedDependency::LOOP_CARRIED_DEPENDENCY_READ_WRITE);
}

TEST(LoopCarriedDependencyAnalysisTest, PartialSum_1D) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer edge_desc(base_desc);
    types::Pointer desc;

    builder.add_container("A", desc, true);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);

    // Define loop
    auto bound = symbolic::symbol("N");
    auto indvar = symbolic::symbol("i");
    auto init = symbolic::integer(1);
    auto condition = symbolic::Lt(indvar, bound);
    auto update = symbolic::add(indvar, symbolic::integer(1));

    auto& loop = builder.add_for(root, indvar, condition, init, update);
    auto& body = loop.root();

    // Add computation
    auto& block = builder.add_block(body);
    auto& A1 = builder.add_access(block, "A");
    auto& A2 = builder.add_access(block, "A");
    auto& A3 = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder
        .add_computational_memlet(block, A1, tasklet, "_in1", {symbolic::sub(indvar, symbolic::integer(1))}, edge_desc);
    builder.add_computational_memlet(block, A2, tasklet, "_in2", {indvar}, edge_desc);
    builder.add_computational_memlet(block, tasklet, "_out", A3, {indvar}, edge_desc);

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::LoopCarriedDependencyAnalysis>();
    auto& dependencies = analysis.dependencies(loop);

    // Check
    EXPECT_EQ(dependencies.size(), 1);
    EXPECT_EQ(dependencies.at("A").type, analysis::LoopCarriedDependency::LOOP_CARRIED_DEPENDENCY_READ_WRITE);
}

TEST(LoopCarriedDependencyAnalysisTest, LoopLocal_1D) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer edge_desc(base_desc);
    types::Pointer desc;

    builder.add_container("A", desc, true);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("tmp", sym_desc);

    // Define loop
    auto bound = symbolic::symbol("N");
    auto indvar = symbolic::symbol("i");
    auto init = symbolic::integer(0);
    auto condition = symbolic::Lt(indvar, bound);
    auto update = symbolic::add(indvar, symbolic::integer(1));

    auto& loop = builder.add_for(root, indvar, condition, init, update);
    auto& body = loop.root();

    // Add computation
    auto& block_1 = builder.add_block(body);
    auto& i_in = builder.add_access(block_1, "i");
    auto& tmp_out = builder.add_access(block_1, "tmp");
    auto& tasklet_1 = builder.add_tasklet(block_1, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block_1, i_in, tasklet_1, "_in", {});
    builder.add_computational_memlet(block_1, tasklet_1, "_out", tmp_out, {});

    auto& block_2 = builder.add_block(body);
    auto& tmp_in = builder.add_access(block_2, "tmp");
    auto& a_out = builder.add_access(block_2, "A");
    auto& tasklet = builder.add_tasklet(block_2, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block_2, tmp_in, tasklet, "_in", {});
    builder.add_computational_memlet(block_2, tasklet, "_out", a_out, {indvar}, edge_desc);

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::LoopCarriedDependencyAnalysis>();
    auto& dependencies = analysis.dependencies(loop);

    // Check
    EXPECT_EQ(dependencies.size(), 1);
    EXPECT_EQ(dependencies.at("tmp").type, analysis::LoopCarriedDependency::LOOP_CARRIED_DEPENDENCY_WRITE_WRITE);
}

TEST(LoopCarriedDependencyAnalysisTest, LoopLocal_Conditional) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer edge_desc(base_desc);
    types::Pointer desc;

    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("tmp", sym_desc);

    builder.add_container("A", desc, true);

    // Define loop
    auto bound1 = symbolic::symbol("N");
    auto indvar1 = symbolic::symbol("i");
    auto init1 = symbolic::integer(0);
    auto condition1 = symbolic::Lt(indvar1, bound1);
    auto update1 = symbolic::add(indvar1, symbolic::integer(1));

    auto& loop1 = builder.add_for(root, indvar1, condition1, init1, update1);
    auto& body1 = loop1.root();

    auto& ifelse = builder.add_if_else(body1);
    auto& branch1 = builder.add_case(ifelse, symbolic::Eq(indvar1, symbolic::integer(0)));
    auto& block1 = builder.add_assignments(branch1, {{symbolic::symbol("tmp"), symbolic::zero()}});
    auto& branch2 = builder.add_case(ifelse, symbolic::Ne(indvar1, symbolic::integer(0)));
    auto& block2 = builder.add_assignments(branch2, {{symbolic::symbol("tmp"), symbolic::one()}});

    // Add computation
    auto& block = builder.add_block(body1);
    auto& A_in = builder.add_access(block, "tmp");
    auto& A_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, A_in, tasklet, "_in", {});
    builder.add_computational_memlet(block, tasklet, "_out", A_out, {indvar1}, edge_desc);

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::LoopCarriedDependencyAnalysis>();
    auto& dependencies = analysis.dependencies(loop1);

    // Check
    EXPECT_EQ(dependencies.size(), 1);
    EXPECT_EQ(dependencies.at("tmp").type, analysis::LoopCarriedDependency::LOOP_CARRIED_DEPENDENCY_WRITE_WRITE);
}

TEST(LoopCarriedDependencyAnalysisTest, LoopLocal_Conditional_Incomplete) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer edge_desc(base_desc);
    types::Pointer desc;

    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("tmp", sym_desc);
    builder.add_container("A", desc, true);

    // Define loop
    auto bound1 = symbolic::symbol("N");
    auto indvar1 = symbolic::symbol("i");
    auto init1 = symbolic::integer(0);
    auto condition1 = symbolic::Lt(indvar1, bound1);
    auto update1 = symbolic::add(indvar1, symbolic::integer(1));

    auto& loop1 = builder.add_for(root, indvar1, condition1, init1, update1);
    auto& body1 = loop1.root();

    auto& ifelse = builder.add_if_else(body1);
    auto& branch1 = builder.add_case(ifelse, symbolic::Eq(indvar1, symbolic::integer(0)));
    auto& block1 = builder.add_assignments(branch1, {{symbolic::symbol("tmp"), symbolic::zero()}});

    // Add computation
    auto& block = builder.add_block(body1);
    auto& A_in = builder.add_access(block, "tmp");
    auto& A_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, A_in, tasklet, "_in", {});
    builder.add_computational_memlet(block, tasklet, "_out", A_out, {indvar1}, edge_desc);

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::LoopCarriedDependencyAnalysis>();
    auto& dependencies = analysis.dependencies(loop1);

    // Check
    EXPECT_EQ(dependencies.size(), 1);
    EXPECT_EQ(dependencies.at("tmp").type, analysis::LoopCarriedDependency::LOOP_CARRIED_DEPENDENCY_READ_WRITE);
}

TEST(LoopCarriedDependencyAnalysisTest, Store_1D) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer edge_desc(base_desc);
    types::Pointer desc;

    builder.add_container("A", desc, true);
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
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"0"});
    builder.add_computational_memlet(block, tasklet, "_out", a, {indvar}, edge_desc);

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::LoopCarriedDependencyAnalysis>();
    auto& dependencies = analysis.dependencies(loop);

    // Check
    EXPECT_EQ(dependencies.size(), 0);
}

TEST(LoopCarriedDependencyAnalysisTest, Copy_1D) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer edge_desc(base_desc);
    types::Pointer desc;

    builder.add_container("A", desc, true);
    builder.add_container("B", desc, true);
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
    builder.add_computational_memlet(block, a, tasklet, "_in", {symbolic::symbol("i")}, edge_desc);
    builder.add_computational_memlet(block, tasklet, "_out", b, {symbolic::symbol("i")}, edge_desc);

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::LoopCarriedDependencyAnalysis>();
    auto& dependencies = analysis.dependencies(loop);

    // Check
    EXPECT_EQ(dependencies.size(), 0);
}

TEST(LoopCarriedDependencyAnalysisTest, Map_1D) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer edge_desc(base_desc);
    types::Pointer desc;

    builder.add_container("A", desc, true);
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
    auto& a_in = builder.add_access(block, "A");
    auto& one_node = builder.add_constant(block, "1.0", base_desc);
    auto& a_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, a_in, tasklet, "_in1", {symbolic::symbol("i")}, edge_desc);
    builder.add_computational_memlet(block, one_node, tasklet, "_in2", {});
    builder.add_computational_memlet(block, tasklet, "_out", a_out, {symbolic::symbol("i")}, edge_desc);

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::LoopCarriedDependencyAnalysis>();
    auto& dependencies = analysis.dependencies(loop);

    // Check
    EXPECT_EQ(dependencies.size(), 0);
}

TEST(LoopCarriedDependencyAnalysisTest, Map_1D_Disjoint) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer edge_desc(base_desc);
    types::Pointer desc;

    builder.add_container("A", desc, true);
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
    auto& a_in = builder.add_access(block, "A");
    auto& one_node = builder.add_constant(block, "1.0", base_desc);
    auto& a_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, a_in, tasklet, "_in1", {symbolic::zero()}, edge_desc);
    builder.add_computational_memlet(block, one_node, tasklet, "_in2", {});
    builder.add_computational_memlet(
        block, tasklet, "_out", a_out, {symbolic::add(symbolic::symbol("i"), symbolic::one())}, edge_desc
    );

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::LoopCarriedDependencyAnalysis>();
    auto& dependencies = analysis.dependencies(loop);

    // Check
    EXPECT_EQ(dependencies.size(), 0);
}

TEST(LoopCarriedDependencyAnalysisTest, Map_1D_Strided) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer edge_desc(base_desc);
    types::Pointer desc;

    builder.add_container("A", desc, true);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);

    // Define loop
    auto bound = symbolic::sub(symbolic::symbol("N"), symbolic::integer(1));
    auto indvar = symbolic::symbol("i");
    auto init = symbolic::integer(0);
    auto condition = symbolic::Lt(indvar, bound);
    auto update = symbolic::add(indvar, symbolic::integer(2));

    auto& loop = builder.add_for(root, indvar, condition, init, update);
    auto& body = loop.root();

    // Add computation
    auto& block = builder.add_block(body);
    auto& a_in = builder.add_access(block, "A");
    auto& a_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, a_in, tasklet, "_in", {symbolic::symbol("i")}, edge_desc);
    builder.add_computational_memlet(
        block, tasklet, "_out", a_out, {symbolic::add(symbolic::symbol("i"), symbolic::one())}, edge_desc
    );

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::LoopCarriedDependencyAnalysis>();
    auto& dependencies = analysis.dependencies(loop);

    // Check
    EXPECT_EQ(dependencies.size(), 0);
}

TEST(LoopCarriedDependencyAnalysisTest, Map_1D_Strided2) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer edge_desc(base_desc);
    types::Pointer desc;

    builder.add_container("A", desc, true);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);

    // Define loop
    auto bound = symbolic::symbol("N");
    auto indvar = symbolic::symbol("i");
    auto init = symbolic::integer(1);
    auto condition = symbolic::Lt(indvar, bound);
    auto update = symbolic::add(indvar, symbolic::integer(2));

    auto& loop = builder.add_for(root, indvar, condition, init, update);
    auto& body = loop.root();

    // Add computation
    auto& block = builder.add_block(body);
    auto& a_in = builder.add_access(block, "A");
    auto& a_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, a_in, tasklet, "_in", {symbolic::symbol("i")}, edge_desc);
    builder.add_computational_memlet(
        block, tasklet, "_out", a_out, {symbolic::sub(symbolic::symbol("i"), symbolic::one())}, edge_desc
    );

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::LoopCarriedDependencyAnalysis>();
    auto& dependencies = analysis.dependencies(loop);

    // Check
    EXPECT_EQ(dependencies.size(), 0);
}

TEST(LoopCarriedDependencyAnalysisTest, Map_1D_Tiled) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer edge_desc(base_desc);
    types::Pointer desc;

    builder.add_container("A", desc, true);
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
    builder.add_computational_memlet(block, a_in, tasklet, "_in1", {symbolic::symbol("i")}, edge_desc);
    builder.add_computational_memlet(block, one_node, tasklet, "_in2", {});
    builder.add_computational_memlet(block, tasklet, "_out", a_out, {symbolic::symbol("i")}, edge_desc);

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::LoopCarriedDependencyAnalysis>();
    auto& dependencies1 = analysis.dependencies(loop_outer);
    auto& dependencies2 = analysis.dependencies(loop_inner);

    // Check
    EXPECT_EQ(dependencies1.size(), 1);
    EXPECT_EQ(dependencies1.at("i").type, analysis::LoopCarriedDependency::LOOP_CARRIED_DEPENDENCY_WRITE_WRITE);
    EXPECT_EQ(dependencies2.size(), 0);
}

TEST(LoopCarriedDependencyAnalysisTest, Map_1D_Incomplete) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar bool_desc(types::PrimitiveType::Bool);
    types::Scalar sym_desc(types::PrimitiveType::Int32);
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer edge_desc(base_desc);
    types::Pointer desc;

    builder.add_container("A", desc, true);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("tmp", base_desc);
    builder.add_container("k", bool_desc);

    // Define loop
    auto bound = symbolic::symbol("N");
    auto indvar = symbolic::symbol("i");
    auto init = symbolic::integer(0);
    auto condition = symbolic::Lt(indvar, bound);
    auto update = symbolic::add(indvar, symbolic::integer(1));

    auto& loop = builder.add_for(root, indvar, condition, init, update);
    auto& body = loop.root();

    // tmp = A[i]
    auto& block = builder.add_block(body);
    auto& a_in = builder.add_access(block, "A");
    auto& tmp_out = builder.add_access(block, "tmp");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, a_in, tasklet, "_in", {indvar}, edge_desc);
    builder.add_computational_memlet(block, tasklet, "_out", tmp_out, {});

    // switch = tmp > 0
    auto& block_switch = builder.add_block(body);
    auto& tmp_in = builder.add_access(block_switch, "tmp");
    auto& zero_node = builder.add_constant(block_switch, "0.0", base_desc);
    auto& switch_out = builder.add_access(block_switch, "k");
    auto& tasklet_switch = builder.add_tasklet(block_switch, data_flow::TaskletCode::fp_oge, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block_switch, tmp_in, tasklet_switch, "_in1", {});
    builder.add_computational_memlet(block_switch, zero_node, tasklet_switch, "_in2", {});
    builder.add_computational_memlet(block_switch, tasklet_switch, "_out", switch_out, {});

    auto switch_condition = symbolic::Eq(symbolic::symbol("k"), symbolic::__true__());
    auto& ifelse = builder.add_if_else(body);

    // if (switch) A[i] = tmp
    auto& branch1 = builder.add_case(ifelse, switch_condition);
    auto& block1 = builder.add_block(branch1);
    auto& tmp_in1 = builder.add_access(block1, "tmp");
    auto& a_out1 = builder.add_access(block1, "A");
    auto& tasklet1 = builder.add_tasklet(block1, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block1, tmp_in1, tasklet1, "_in", {});
    builder.add_computational_memlet(block1, tasklet1, "_out", a_out1, {indvar}, edge_desc);

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::LoopCarriedDependencyAnalysis>();
    auto& dependencies = analysis.dependencies(loop);

    // Check
    EXPECT_EQ(dependencies.size(), 2);
    EXPECT_EQ(dependencies.at("tmp").type, analysis::LoopCarriedDependency::LOOP_CARRIED_DEPENDENCY_WRITE_WRITE);
    EXPECT_EQ(dependencies.at("k").type, analysis::LoopCarriedDependency::LOOP_CARRIED_DEPENDENCY_WRITE_WRITE);
}

TEST(LoopCarriedDependencyAnalysisTest, MapParameterized_1D) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer edge_desc(base_desc);
    types::Pointer desc;

    builder.add_container("A", desc, true);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("m", sym_desc, true);
    builder.add_container("b", sym_desc, true);

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
    auto& A_in = builder.add_access(block, "A");
    auto& A_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in1"});
    builder.add_computational_memlet(
        block,
        A_in,
        tasklet,
        "_in1",
        {symbolic::add(symbolic::mul(symbolic::symbol("m"), symbolic::symbol("i")), symbolic::symbol("b"))},
        edge_desc
    );
    builder.add_computational_memlet(
        block,
        tasklet,
        "_out",
        A_out,
        {symbolic::add(symbolic::mul(symbolic::symbol("m"), symbolic::symbol("i")), symbolic::symbol("b"))},
        edge_desc
    );

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::LoopCarriedDependencyAnalysis>();
    auto& dependencies = analysis.dependencies(loop);

    // Check
    // m == 0 -> all iterations access the same location
    EXPECT_EQ(dependencies.size(), 1);
    EXPECT_EQ(dependencies.at("A").type, analysis::LoopCarriedDependency::LOOP_CARRIED_DEPENDENCY_READ_WRITE);
}

TEST(LoopCarriedDependencyAnalysisTest, Stencil_1D) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer edge_desc(base_desc);
    types::Pointer desc;

    builder.add_container("A", desc, true);
    builder.add_container("B", desc, true);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);

    // Define loop
    auto bound = symbolic::symbol("N");
    auto indvar = symbolic::symbol("i");
    auto init = symbolic::integer(1);
    auto condition = symbolic::Lt(indvar, bound);
    auto update = symbolic::add(indvar, symbolic::integer(1));

    auto& loop = builder.add_for(root, indvar, condition, init, update);
    auto& body = loop.root();

    // Add computation
    auto& block = builder.add_block(body);
    auto& A1 = builder.add_access(block, "A");
    auto& A2 = builder.add_access(block, "A");
    auto& A3 = builder.add_access(block, "A");
    auto& B = builder.add_access(block, "B");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_fma, "_out", {"_in1", "_in2", "_in3"});
    builder.add_computational_memlet(
        block, A1, tasklet, "_in1", {symbolic::sub(symbolic::symbol("i"), symbolic::integer(1))}, edge_desc
    );
    builder.add_computational_memlet(block, A2, tasklet, "_in2", {symbolic::symbol("i")}, edge_desc);
    builder.add_computational_memlet(
        block, A3, tasklet, "_in3", {symbolic::add(symbolic::symbol("i"), symbolic::integer(1))}, edge_desc
    );
    builder.add_computational_memlet(block, tasklet, "_out", B, {symbolic::symbol("i")}, edge_desc);

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::LoopCarriedDependencyAnalysis>();
    auto& dependencies = analysis.dependencies(loop);

    // Check
    EXPECT_EQ(dependencies.size(), 0);
}

TEST(LoopCarriedDependencyAnalysisTest, Gather_1D) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    types::Scalar base_desc(types::PrimitiveType::Int64);
    types::Pointer edge_desc(base_desc);
    types::Pointer desc;

    builder.add_container("A", desc, true);
    builder.add_container("B", desc, true);
    builder.add_container("C", desc, true);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("b", sym_desc);

    // Define loop
    auto bound = symbolic::symbol("N");
    auto indvar = symbolic::symbol("i");
    auto init = symbolic::integer(0);
    auto condition = symbolic::Lt(indvar, bound);
    auto update = symbolic::add(indvar, symbolic::integer(1));

    auto& loop = builder.add_for(root, indvar, condition, init, update);
    auto& body = loop.root();

    // Add computation
    auto& block_1 = builder.add_block(body);

    // Define indirection
    auto& A = builder.add_access(block_1, "A");
    auto& b = builder.add_access(block_1, "b");
    auto& tasklet1 = builder.add_tasklet(block_1, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block_1, A, tasklet1, "_in", {indvar}, edge_desc);
    builder.add_computational_memlet(block_1, tasklet1, "_out", b, {});

    auto& block_2 = builder.add_block(body);
    auto& B = builder.add_access(block_2, "B");
    auto& C = builder.add_access(block_2, "C");
    auto& tasklet = builder.add_tasklet(block_2, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block_2, B, tasklet, "_in", {symbolic::symbol("b")}, edge_desc);
    builder.add_computational_memlet(block_2, tasklet, "_out", C, {symbolic::symbol("i")}, edge_desc);

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::LoopCarriedDependencyAnalysis>();
    auto& dependencies = analysis.dependencies(loop);

    // Check
    EXPECT_EQ(dependencies.size(), 1);
    EXPECT_EQ(dependencies.at("b").type, analysis::LoopCarriedDependency::LOOP_CARRIED_DEPENDENCY_WRITE_WRITE);
}

TEST(LoopCarriedDependencyAnalysisTest, Scatter_1D) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    types::Scalar base_desc(types::PrimitiveType::Int64);
    types::Pointer edge_desc(base_desc);
    types::Pointer desc;

    builder.add_container("A", desc, true);
    builder.add_container("B", desc, true);
    builder.add_container("C", desc, true);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("b", sym_desc);

    // Define loop
    auto bound = symbolic::symbol("N");
    auto indvar = symbolic::symbol("i");
    auto init = symbolic::integer(0);
    auto condition = symbolic::Lt(indvar, bound);
    auto update = symbolic::add(indvar, symbolic::integer(1));

    auto& loop = builder.add_for(root, indvar, condition, init, update);
    auto& body = loop.root();

    // Define indirection
    auto& block_1 = builder.add_block(body);
    auto& A = builder.add_access(block_1, "A");
    auto& b = builder.add_access(block_1, "b");
    auto& tasklet1 = builder.add_tasklet(block_1, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block_1, A, tasklet1, "_in", {symbolic::symbol("i")}, edge_desc);
    builder.add_computational_memlet(block_1, tasklet1, "_out", b, {});

    auto& block_2 = builder.add_block(body);
    auto& B = builder.add_access(block_2, "B");
    auto& C = builder.add_access(block_2, "C");
    auto& tasklet = builder.add_tasklet(block_2, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block_2, B, tasklet, "_in", {symbolic::symbol("i")}, edge_desc);
    builder.add_computational_memlet(block_2, tasklet, "_out", C, {symbolic::symbol("b")}, edge_desc);

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::LoopCarriedDependencyAnalysis>();
    auto& dependencies = analysis.dependencies(loop);

    // Check
    EXPECT_EQ(dependencies.size(), 2);
    EXPECT_EQ(dependencies.at("b").type, analysis::LoopCarriedDependency::LOOP_CARRIED_DEPENDENCY_WRITE_WRITE);
    EXPECT_EQ(dependencies.at("C").type, analysis::LoopCarriedDependency::LOOP_CARRIED_DEPENDENCY_WRITE_WRITE);
}

TEST(LoopCarriedDependencyAnalysisTest, MapDeg2_1D) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer edge_desc(base_desc);
    types::Pointer desc;

    builder.add_container("A", desc, true);
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
    auto& A = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"0"});
    builder.add_computational_memlet(
        block, tasklet, "_out", A, {symbolic::mul(symbolic::symbol("i"), symbolic::symbol("i"))}, edge_desc
    );

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::LoopCarriedDependencyAnalysis>();
    auto& dependencies = analysis.dependencies(loop);

    // Check
    EXPECT_EQ(dependencies.size(), 0);
}

TEST(LoopCarriedDependencyAnalysisTest, Map_2D) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Array array_desc(base_desc, symbolic::symbol("M"));
    types::Pointer edge_desc(array_desc);
    types::Pointer desc;

    builder.add_container("A", desc, true);
    builder.add_container("N", sym_desc, true);
    builder.add_container("M", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);

    // Define loop 1
    auto bound = symbolic::symbol("N");
    auto indvar = symbolic::symbol("i");
    auto init = symbolic::integer(0);
    auto condition = symbolic::Lt(indvar, bound);
    auto update = symbolic::add(indvar, symbolic::integer(1));

    auto& loop = builder.add_for(root, indvar, condition, init, update);
    auto& body = loop.root();

    // Define loop 2
    auto bound_2 = symbolic::symbol("M");
    auto indvar_2 = symbolic::symbol("j");
    auto init_2 = symbolic::integer(0);
    auto condition_2 = symbolic::Lt(indvar_2, bound_2);
    auto update_2 = symbolic::add(indvar_2, symbolic::integer(1));

    auto& loop_2 = builder.add_for(body, indvar_2, condition_2, init_2, update_2);
    auto& body_2 = loop_2.root();

    // Add computation
    auto& block = builder.add_block(body_2);
    auto& a_in = builder.add_access(block, "A");
    auto& one_node = builder.add_constant(block, "1.0", base_desc);
    auto& a_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder
        .add_computational_memlet(block, a_in, tasklet, "_in1", {symbolic::symbol("i"), symbolic::symbol("j")}, edge_desc);
    builder.add_computational_memlet(block, one_node, tasklet, "_in2", {});
    builder
        .add_computational_memlet(block, tasklet, "_out", a_out, {symbolic::symbol("i"), symbolic::symbol("j")}, edge_desc);

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::LoopCarriedDependencyAnalysis>();

    // Check
    auto& dependencies = analysis.dependencies(loop);
    EXPECT_EQ(dependencies.size(), 1);
    EXPECT_EQ(dependencies.at("j").type, analysis::LoopCarriedDependency::LOOP_CARRIED_DEPENDENCY_WRITE_WRITE);

    // Check loop 2
    auto& dependencies_2 = analysis.dependencies(loop_2);
    EXPECT_EQ(dependencies_2.size(), 0);
}

TEST(LoopCarriedDependencyAnalysisTest, PartialSumInner_2D) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Array array_desc(base_desc, symbolic::symbol("N"));
    types::Pointer edge_desc(array_desc);
    types::Pointer desc;

    builder.add_container("A", desc, true);
    builder.add_container("B", array_desc, true);

    // Define loop
    auto bound1 = symbolic::symbol("N");
    auto indvar1 = symbolic::symbol("i");
    auto init1 = symbolic::integer(0);
    auto condition1 = symbolic::Lt(indvar1, bound1);
    auto update1 = symbolic::add(indvar1, symbolic::integer(1));

    auto& loop1 = builder.add_for(root, indvar1, condition1, init1, update1);
    auto& body1 = loop1.root();

    auto bound2 = symbolic::symbol("N");
    auto indvar2 = symbolic::symbol("j");
    auto init2 = symbolic::integer(0);
    auto condition2 = symbolic::Lt(indvar2, bound2);
    auto update3 = symbolic::add(indvar2, symbolic::integer(1));

    auto& loop2 = builder.add_for(body1, indvar2, condition2, init2, update3);
    auto& body2 = loop2.root();

    // Add computation
    auto& block = builder.add_block(body2);
    auto& A = builder.add_access(block, "A");
    auto& B1 = builder.add_access(block, "B");
    auto& B2 = builder.add_access(block, "B");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder
        .add_computational_memlet(block, A, tasklet, "_in1", {symbolic::symbol("i"), symbolic::symbol("j")}, edge_desc);
    builder.add_computational_memlet(block, B1, tasklet, "_in2", {symbolic::symbol("i")}, array_desc);
    builder.add_computational_memlet(block, tasklet, "_out", B2, {symbolic::symbol("i")}, array_desc);

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::LoopCarriedDependencyAnalysis>();
    auto& dependencies1 = analysis.dependencies(loop1);
    auto& dependencies2 = analysis.dependencies(loop2);

    // Check
    EXPECT_EQ(dependencies1.size(), 1);
    EXPECT_EQ(dependencies1.at("j").type, analysis::LoopCarriedDependency::LOOP_CARRIED_DEPENDENCY_WRITE_WRITE);

    EXPECT_EQ(dependencies2.size(), 1);
    EXPECT_EQ(dependencies2.at("B").type, analysis::LoopCarriedDependency::LOOP_CARRIED_DEPENDENCY_READ_WRITE);

    // Inner j-loop: B[i] = A[i,j] + B[i] is an additive reduction (accumulator
    // B[i] is invariant w.r.t. the inner induction variable j).
    ASSERT_TRUE(analysis.has_reductions(loop2));
    ASSERT_EQ(analysis.reductions(loop2).size(), 1u);
    EXPECT_EQ(analysis.reductions(loop2)[0].operation, structured_control_flow::ReductionOperation::Add);
    EXPECT_EQ(analysis.reductions(loop2)[0].container, "B");
    EXPECT_TRUE(analysis.is_reduction_only(loop2));

    // Outer i-loop: the combine lives in the nested j-loop body, so the outer
    // loop sees no in-body reduction combine (only a WAW on j).
    EXPECT_FALSE(analysis.has_reductions(loop1));
    EXPECT_TRUE(analysis.reductions(loop1).empty());
}

TEST(LoopCarriedDependencyAnalysisTest, PartialSumOuter_2D) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Array array_desc(base_desc, symbolic::symbol("N"));
    types::Pointer edge_desc(array_desc);
    types::Pointer desc;

    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);
    builder.add_container("A", desc, true);
    builder.add_container("B", array_desc, true);

    // Define loop
    auto bound1 = symbolic::symbol("N");
    auto indvar1 = symbolic::symbol("i");
    auto init1 = symbolic::integer(0);
    auto condition1 = symbolic::Lt(indvar1, bound1);
    auto update1 = symbolic::add(indvar1, symbolic::integer(1));

    auto& loop1 = builder.add_for(root, indvar1, condition1, init1, update1);
    auto& body1 = loop1.root();

    auto bound2 = symbolic::symbol("N");
    auto indvar2 = symbolic::symbol("j");
    auto init2 = symbolic::integer(0);
    auto condition2 = symbolic::Lt(indvar2, bound2);
    auto update3 = symbolic::add(indvar2, symbolic::integer(1));

    auto& loop2 = builder.add_for(body1, indvar2, condition2, init2, update3);
    auto& body2 = loop2.root();

    // Add computation
    auto& block = builder.add_block(body2);
    auto& A = builder.add_access(block, "A");
    auto& B1 = builder.add_access(block, "B");
    auto& B2 = builder.add_access(block, "B");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, A, tasklet, "_in1", {indvar1, indvar2}, edge_desc);
    builder.add_computational_memlet(block, B1, tasklet, "_in2", {indvar2}, array_desc);
    builder.add_computational_memlet(block, tasklet, "_out", B2, {indvar2}, array_desc);

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::LoopCarriedDependencyAnalysis>();
    auto& dependencies1 = analysis.dependencies(loop1);
    auto& dependencies2 = analysis.dependencies(loop2);

    // Check
    EXPECT_EQ(dependencies1.size(), 2);
    EXPECT_EQ(dependencies1.at("j").type, analysis::LoopCarriedDependency::LOOP_CARRIED_DEPENDENCY_WRITE_WRITE);
    EXPECT_EQ(dependencies1.at("B").type, analysis::LoopCarriedDependency::LOOP_CARRIED_DEPENDENCY_READ_WRITE);

    EXPECT_EQ(dependencies2.size(), 0);
}

TEST(LoopCarriedDependencyAnalysisTest, PartialSum_1D_Triangle) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer ptr_desc(base_desc);

    types::Pointer desc;
    builder.add_container("A", desc, true);

    // Define loop
    auto indvar1 = symbolic::symbol("i");
    auto init1 = symbolic::integer(0);
    auto condition1 = symbolic::Lt(indvar1, symbolic::symbol("N"));
    auto update1 = symbolic::add(indvar1, symbolic::integer(1));

    auto& loop1 = builder.add_for(root, indvar1, condition1, init1, update1);
    auto& body1 = loop1.root();

    // init block
    auto& init_block = builder.add_block(body1);
    auto& A_init = builder.add_access(init_block, "A");
    auto& zero_node = builder.add_constant(init_block, "0.0", base_desc);
    auto& tasklet_init = builder.add_tasklet(init_block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(init_block, zero_node, tasklet_init, "_in", {});
    builder.add_computational_memlet(init_block, tasklet_init, "_out", A_init, {indvar1}, ptr_desc);

    auto indvar2 = symbolic::symbol("j");
    auto init2 = symbolic::zero();
    auto condition2 = symbolic::Lt(indvar2, indvar1);
    auto update2 = symbolic::add(indvar2, symbolic::integer(1));

    auto& loop2 = builder.add_for(body1, indvar2, condition2, init2, update2);
    auto& body2 = loop2.root();

    // Reduction block
    auto& block = builder.add_block(body2);
    auto& A_in = builder.add_access(block, "A");
    auto& A_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, A_in, tasklet, "_in1", {indvar2}, ptr_desc);
    builder.add_computational_memlet(block, A_in, tasklet, "_in2", {indvar1}, ptr_desc);
    builder.add_computational_memlet(block, tasklet, "_out", A_out, {indvar1}, ptr_desc);

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::LoopCarriedDependencyAnalysis>();
    auto& dependencies1 = analysis.dependencies(loop1);
    auto& dependencies2 = analysis.dependencies(loop2);

    // Check
    EXPECT_EQ(dependencies1.size(), 2);
    EXPECT_EQ(dependencies1.at("A").type, analysis::LoopCarriedDependency::LOOP_CARRIED_DEPENDENCY_READ_WRITE);
    EXPECT_EQ(dependencies1.at("j").type, analysis::LoopCarriedDependency::LOOP_CARRIED_DEPENDENCY_WRITE_WRITE);
    EXPECT_EQ(dependencies2.size(), 1);
    EXPECT_EQ(dependencies2.at("A").type, analysis::LoopCarriedDependency::LOOP_CARRIED_DEPENDENCY_READ_WRITE);
}

TEST(LoopCarriedDependencyAnalysisTest, Transpose_2D) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("M", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Array array_desc(base_desc, symbolic::symbol("M"));
    types::Pointer edge_desc(array_desc);
    types::Pointer desc;

    builder.add_container("A", desc, true);
    builder.add_container("B", desc, true);

    // Define loop
    auto bound1 = symbolic::symbol("N");
    auto indvar1 = symbolic::symbol("i");
    auto init1 = symbolic::integer(0);
    auto condition1 = symbolic::Lt(indvar1, bound1);
    auto update1 = symbolic::add(indvar1, symbolic::integer(1));

    auto& loop1 = builder.add_for(root, indvar1, condition1, init1, update1);
    auto& body1 = loop1.root();

    auto bound2 = symbolic::symbol("M");
    auto indvar2 = symbolic::symbol("j");
    auto init2 = symbolic::integer(0);
    auto condition2 = symbolic::Lt(indvar2, bound2);
    auto update3 = symbolic::add(indvar2, symbolic::integer(1));

    auto& loop2 = builder.add_for(body1, indvar2, condition2, init2, update3);
    auto& body2 = loop2.root();

    // Add computation
    auto& block = builder.add_block(body2);
    auto& A = builder.add_access(block, "A");
    auto& B = builder.add_access(block, "B");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder
        .add_computational_memlet(block, A, tasklet, "_in", {symbolic::symbol("i"), symbolic::symbol("j")}, edge_desc);
    builder
        .add_computational_memlet(block, tasklet, "_out", B, {symbolic::symbol("j"), symbolic::symbol("i")}, edge_desc);

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::LoopCarriedDependencyAnalysis>();
    auto& dependencies1 = analysis.dependencies(loop1);
    auto& dependencies2 = analysis.dependencies(loop2);

    // Check
    EXPECT_EQ(dependencies2.size(), 0);
    EXPECT_EQ(dependencies1.size(), 1);
    EXPECT_EQ(dependencies1.at("j").type, analysis::LoopCarriedDependency::LOOP_CARRIED_DEPENDENCY_WRITE_WRITE);
}

TEST(LoopCarriedDependencyAnalysisTest, TransposeTriangle_2D) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Array array_desc(base_desc, symbolic::symbol("N"));
    types::Pointer edge_desc(array_desc);
    types::Pointer desc;

    builder.add_container("A", desc, true);

    // Define loop
    auto indvar1 = symbolic::symbol("i");
    auto init1 = symbolic::integer(0);
    auto condition1 = symbolic::Lt(indvar1, symbolic::symbol("N"));
    auto update1 = symbolic::add(indvar1, symbolic::integer(1));

    auto& loop1 = builder.add_for(root, indvar1, condition1, init1, update1);
    auto& body1 = loop1.root();

    auto indvar2 = symbolic::symbol("j");
    auto init2 = symbolic::add(indvar1, symbolic::integer(1));
    auto condition2 = symbolic::Lt(indvar2, symbolic::symbol("N"));
    auto update2 = symbolic::add(indvar2, symbolic::integer(1));

    auto& loop2 = builder.add_for(body1, indvar2, condition2, init2, update2);
    auto& body2 = loop2.root();

    // Add computation
    auto& block = builder.add_block(body2);
    auto& A_in = builder.add_access(block, "A");
    auto& A_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder
        .add_computational_memlet(block, A_in, tasklet, "_in", {symbolic::symbol("i"), symbolic::symbol("j")}, edge_desc);
    builder
        .add_computational_memlet(block, tasklet, "_out", A_out, {symbolic::symbol("j"), symbolic::symbol("i")}, edge_desc);

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::LoopCarriedDependencyAnalysis>();
    auto& dependencies1 = analysis.dependencies(loop1);
    auto& dependencies2 = analysis.dependencies(loop2);

    // Check
    EXPECT_EQ(dependencies1.size(), 1);
    EXPECT_EQ(dependencies1.at("j").type, analysis::LoopCarriedDependency::LOOP_CARRIED_DEPENDENCY_WRITE_WRITE);
    EXPECT_EQ(dependencies2.size(), 0);
}

TEST(LoopCarriedDependencyAnalysisTest, TransposeTriangleWithDiagonal_2D) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Array array_desc(base_desc, symbolic::symbol("N"));
    types::Pointer edge_desc(array_desc);
    types::Pointer desc;

    builder.add_container("A", desc, true);

    // Define loop
    auto indvar1 = symbolic::symbol("i");
    auto init1 = symbolic::integer(0);
    auto condition1 = symbolic::Lt(indvar1, symbolic::symbol("N"));
    auto update1 = symbolic::add(indvar1, symbolic::integer(1));

    auto& loop1 = builder.add_for(root, indvar1, condition1, init1, update1);
    auto& body1 = loop1.root();

    auto indvar2 = symbolic::symbol("j");
    auto init2 = indvar1;
    auto condition2 = symbolic::Lt(indvar2, symbolic::symbol("N"));
    auto update2 = symbolic::add(indvar2, symbolic::integer(1));

    auto& loop2 = builder.add_for(body1, indvar2, condition2, init2, update2);
    auto& body2 = loop2.root();

    // Add computation
    auto& block = builder.add_block(body2);
    auto& A_in = builder.add_access(block, "A");
    auto& A_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder
        .add_computational_memlet(block, A_in, tasklet, "_in", {symbolic::symbol("i"), symbolic::symbol("j")}, edge_desc);
    builder
        .add_computational_memlet(block, tasklet, "_out", A_out, {symbolic::symbol("j"), symbolic::symbol("i")}, edge_desc);

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::LoopCarriedDependencyAnalysis>();
    auto& dependencies1 = analysis.dependencies(loop1);
    auto& dependencies2 = analysis.dependencies(loop2);

    // Check
    EXPECT_EQ(dependencies1.size(), 1);
    EXPECT_EQ(dependencies1.at("j").type, analysis::LoopCarriedDependency::LOOP_CARRIED_DEPENDENCY_WRITE_WRITE);
    EXPECT_EQ(dependencies2.size(), 0);
}

TEST(LoopCarriedDependencyAnalysisTest, TransposeSquare_2D) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Array array_desc(base_desc, symbolic::symbol("N"));
    types::Pointer edge_desc(array_desc);
    types::Pointer desc;

    builder.add_container("A", desc, true);

    // Define loop
    auto bound1 = symbolic::symbol("N");
    auto indvar1 = symbolic::symbol("i");
    auto init1 = symbolic::integer(0);
    auto condition1 = symbolic::Lt(indvar1, bound1);
    auto update1 = symbolic::add(indvar1, symbolic::integer(1));

    auto& loop1 = builder.add_for(root, indvar1, condition1, init1, update1);
    auto& body1 = loop1.root();

    auto bound2 = symbolic::symbol("N");
    auto indvar2 = symbolic::symbol("j");
    auto init2 = symbolic::integer(0);
    auto condition2 = symbolic::Lt(indvar2, bound2);
    auto update3 = symbolic::add(indvar2, symbolic::integer(1));

    auto& loop2 = builder.add_for(body1, indvar2, condition2, init2, update3);
    auto& body2 = loop2.root();

    // Add computation
    auto& block = builder.add_block(body2);
    auto& A_in = builder.add_access(block, "A");
    auto& A_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, A_in, tasklet, "_in", {indvar1, indvar2}, edge_desc);
    builder.add_computational_memlet(block, tasklet, "_out", A_out, {indvar2, indvar1}, edge_desc);

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::LoopCarriedDependencyAnalysis>();
    auto& dependencies1 = analysis.dependencies(loop1);
    auto& dependencies2 = analysis.dependencies(loop2);

    // Check
    EXPECT_EQ(dependencies2.size(), 0);
    EXPECT_EQ(dependencies1.size(), 2);
    EXPECT_EQ(dependencies1.at("A").type, analysis::LoopCarriedDependency::LOOP_CARRIED_DEPENDENCY_READ_WRITE);
    EXPECT_EQ(dependencies1.at("j").type, analysis::LoopCarriedDependency::LOOP_CARRIED_DEPENDENCY_WRITE_WRITE);
}

TEST(LoopCarriedDependencyAnalysisTest, ReductionWithLocalStorage) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer ptr_desc(base_desc);
    types::Array array_desc(base_desc, symbolic::integer(2));

    types::Pointer opaque_desc;
    builder.add_container("A", opaque_desc, true);
    builder.add_container("B", opaque_desc, true);
    builder.add_container("C", opaque_desc, true);
    builder.add_container("local", array_desc);

    // Define loop
    auto indvar1 = symbolic::symbol("i");
    auto init1 = symbolic::integer(0);
    auto condition1 = symbolic::Lt(indvar1, symbolic::symbol("N"));
    auto update1 = symbolic::add(indvar1, symbolic::integer(1));

    auto& loop1 = builder.add_for(root, indvar1, condition1, init1, update1);
    auto& body1 = loop1.root();

    // local[0] = 0.0 block
    {
        auto& init_block = builder.add_block(body1);
        auto& local_init_0 = builder.add_access(init_block, "local");
        auto& zero_node = builder.add_constant(init_block, "0.0", base_desc);
        auto& tasklet_init = builder.add_tasklet(init_block, data_flow::TaskletCode::assign, "_out", {"_in"});
        builder.add_computational_memlet(init_block, zero_node, tasklet_init, "_in", {});
        builder.add_computational_memlet(init_block, tasklet_init, "_out", local_init_0, {symbolic::zero()}, array_desc);
    }

    // local[1] = 0.0 block
    {
        auto& init_block = builder.add_block(body1);
        auto& local_init_0 = builder.add_access(init_block, "local");
        auto& zero_node = builder.add_constant(init_block, "0.0", base_desc);
        auto& tasklet_init = builder.add_tasklet(init_block, data_flow::TaskletCode::assign, "_out", {"_in"});
        builder.add_computational_memlet(init_block, zero_node, tasklet_init, "_in", {});
        builder.add_computational_memlet(init_block, tasklet_init, "_out", local_init_0, {symbolic::one()}, array_desc);
    }

    auto indvar2 = symbolic::symbol("j");
    auto init2 = symbolic::zero();
    auto condition2 = symbolic::Lt(indvar2, indvar1);
    auto update2 = symbolic::add(indvar2, symbolic::integer(1));

    auto& loop2 = builder.add_for(body1, indvar2, condition2, init2, update2);
    auto& body2 = loop2.root();

    // Reduction: local[0] += A[j] block
    {
        auto& block = builder.add_block(body2);
        auto& A_in = builder.add_access(block, "A");
        auto& local_in = builder.add_access(block, "local");
        auto& local_out = builder.add_access(block, "local");
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
        builder.add_computational_memlet(block, A_in, tasklet, "_in1", {indvar2}, ptr_desc);
        builder.add_computational_memlet(block, local_in, tasklet, "_in2", {symbolic::zero()}, array_desc);
        builder.add_computational_memlet(block, tasklet, "_out", local_out, {symbolic::zero()}, array_desc);
    }

    // Reduction: local[1] *= A[j] block
    {
        auto& block = builder.add_block(body2);
        auto& A_in = builder.add_access(block, "A");
        auto& local_in = builder.add_access(block, "local");
        auto& local_out = builder.add_access(block, "local");
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_mul, "_out", {"_in1", "_in2"});
        builder.add_computational_memlet(block, A_in, tasklet, "_in1", {indvar2}, ptr_desc);
        builder.add_computational_memlet(block, local_in, tasklet, "_in2", {symbolic::one()}, array_desc);
        builder.add_computational_memlet(block, tasklet, "_out", local_out, {symbolic::one()}, array_desc);
    }

    // Writeback block: B[i] = local[0]; C[i] = local[1]
    {
        auto& block = builder.add_block(body1);
        auto& local_in_0 = builder.add_access(block, "local");
        auto& local_in_1 = builder.add_access(block, "local");
        auto& B_out = builder.add_access(block, "B");
        auto& C_out = builder.add_access(block, "C");
        auto& tasklet_0 = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
        builder.add_computational_memlet(block, local_in_0, tasklet_0, "_in", {symbolic::zero()}, array_desc);
        builder.add_computational_memlet(block, tasklet_0, "_out", B_out, {indvar1}, ptr_desc);

        auto& tasklet_1 = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
        builder.add_computational_memlet(block, local_in_1, tasklet_1, "_in", {symbolic::one()}, array_desc);
        builder.add_computational_memlet(block, tasklet_1, "_out", C_out, {indvar1}, ptr_desc);
    }

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::LoopCarriedDependencyAnalysis>();
    auto& dependencies1 = analysis.dependencies(loop1);
    auto& dependencies2 = analysis.dependencies(loop2);

    // Check
    EXPECT_EQ(dependencies1.size(), 2);
    EXPECT_EQ(dependencies1.at("local").type, analysis::LoopCarriedDependency::LOOP_CARRIED_DEPENDENCY_WRITE_WRITE);
    EXPECT_EQ(dependencies1.at("j").type, analysis::LoopCarriedDependency::LOOP_CARRIED_DEPENDENCY_WRITE_WRITE);
    EXPECT_EQ(dependencies2.size(), 1);
    EXPECT_EQ(dependencies2.at("local").type, analysis::LoopCarriedDependency::LOOP_CARRIED_DEPENDENCY_READ_WRITE);

    // The two inner combines (local[0] += ... and local[1] *= ...) write the
    // SAME container `local` with DIFFERENT operators. Reduction detection keys
    // by container name, so it conservatively rejects this as a reduction.
    EXPECT_FALSE(analysis.has_reductions(loop2));
    EXPECT_TRUE(analysis.reductions(loop2).empty());
    EXPECT_FALSE(analysis.is_reduction_only(loop2));
}

TEST(LoopCarriedDependencyAnalysisTest, Cholesky_Full) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::Int64);
    types::Scalar usym_desc(types::PrimitiveType::UInt64); // Unsigned type for inner loop indices
    types::Scalar base_desc(types::PrimitiveType::Double);
    types::Pointer ptr_desc(base_desc);
    types::Pointer opaque_desc;

    builder.add_container("A", opaque_desc, true);
    builder.add_container("_s0", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);
    builder.add_container("_i0", usym_desc); // UInt64 - causes signed/unsigned mismatch
    builder.add_container("_i1", usym_desc); // UInt64 - causes signed/unsigned mismatch
    builder.add_container("_dot_res_3", base_desc);
    builder.add_container("_dot_res_5", base_desc);
    builder.add_container("tmp_0", base_desc);
    builder.add_container("_tmp_1", base_desc);
    builder.add_container("tmp_5", base_desc);
    builder.add_container("_tmp_6", base_desc);

    auto sym_i = symbolic::symbol("i");
    auto sym_s0 = symbolic::symbol("_s0");

    // === Initial diagonal element (A[0]) computation ===

    // Block: tmp_0 = A[0]
    {
        auto& block = builder.add_block(root);
        auto& A_in = builder.add_access(block, "A");
        auto& tmp_out = builder.add_access(block, "tmp_0");
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
        builder.add_computational_memlet(block, A_in, tasklet, "_in", {symbolic::integer(0)}, ptr_desc);
        builder.add_computational_memlet(block, tasklet, "_out", tmp_out, {});
    }

    // Block: _tmp_1 = sqrt(tmp_0)
    {
        auto& block = builder.add_block(root);
        auto& tmp_in = builder.add_access(block, "tmp_0");
        auto& tmp_out = builder.add_access(block, "_tmp_1");
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_neg, "_out", {"_in1"});
        builder.add_computational_memlet(block, tmp_in, tasklet, "_in1", {});
        builder.add_computational_memlet(block, tasklet, "_out", tmp_out, {});
    }

    // Block: A[0] = _tmp_1
    {
        auto& block = builder.add_block(root);
        auto& tmp_in = builder.add_access(block, "_tmp_1");
        auto& A_out = builder.add_access(block, "A");
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
        builder.add_computational_memlet(block, tmp_in, tasklet, "_in", {});
        builder.add_computational_memlet(block, tasklet, "_out", A_out, {symbolic::integer(0)}, ptr_desc);
    }

    // Outer loop: for (i = 1; i < _s0; i++)
    auto indvar_i = symbolic::symbol("i");
    auto init_i = symbolic::integer(1);
    auto condition_i = symbolic::Lt(indvar_i, symbolic::symbol("_s0"));
    auto update_i = symbolic::add(indvar_i, symbolic::integer(1));

    auto& loop_i = builder.add_for(root, indvar_i, condition_i, init_i, update_i);
    auto& body_i = loop_i.root();

    // Middle loop: for (j = 0; j < i; j++)
    auto indvar_j = symbolic::symbol("j");
    auto init_j = symbolic::integer(0);
    auto condition_j = symbolic::Lt(indvar_j, indvar_i);
    auto update_j = symbolic::add(indvar_j, symbolic::integer(1));

    auto& loop_j = builder.add_for(body_i, indvar_j, condition_j, init_j, update_j);
    auto& body_j = loop_j.root();

    // Block: _dot_res_3 = 0.0
    {
        auto& block = builder.add_block(body_j);
        auto& zero = builder.add_constant(block, "0.0", base_desc);
        auto& out = builder.add_access(block, "_dot_res_3");
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
        builder.add_computational_memlet(block, zero, tasklet, "_in", {});
        builder.add_computational_memlet(block, tasklet, "_out", out, {});
    }

    // Inner loop: for (_i0 = 0; _i0 < j; _i0++)
    auto indvar_i0 = symbolic::symbol("_i0");
    auto init_i0 = symbolic::integer(0);
    auto condition_i0 = symbolic::Lt(indvar_i0, indvar_j);
    auto update_i0 = symbolic::add(indvar_i0, symbolic::integer(1));

    auto& loop_i0 = builder.add_for(body_j, indvar_i0, condition_i0, init_i0, update_i0);
    auto& body_i0 = loop_i0.root();

    // Inner body: _dot_res_3 = A[_i0 + i*_s0] * A[_i0 + j*_s0] + _dot_res_3 (FMA)
    {
        auto& block = builder.add_block(body_i0);
        auto& A_in1 = builder.add_access(block, "A");
        auto& A_in2 = builder.add_access(block, "A");
        auto& dot_in = builder.add_access(block, "_dot_res_3");
        auto& dot_out = builder.add_access(block, "_dot_res_3");

        auto& fma_tasklet =
            builder.add_tasklet(block, data_flow::TaskletCode::fp_fma, "__out", {"_in1", "_in2", "_in3"});
        auto read_idx1 = symbolic::add(indvar_i0, symbolic::mul(sym_i, sym_s0));
        auto read_idx2 = symbolic::add(indvar_i0, symbolic::mul(indvar_j, sym_s0));
        builder.add_computational_memlet(block, A_in1, fma_tasklet, "_in1", {read_idx1}, ptr_desc);
        builder.add_computational_memlet(block, A_in2, fma_tasklet, "_in2", {read_idx2}, ptr_desc);
        builder.add_computational_memlet(block, dot_in, fma_tasklet, "_in3", {});
        builder.add_computational_memlet(block, fma_tasklet, "__out", dot_out, {});
    }

    // Block: A[j + i*_s0] -= _dot_res_3
    {
        auto& block = builder.add_block(body_j);
        auto& A_in = builder.add_access(block, "A");
        auto& dot_in = builder.add_access(block, "_dot_res_3");
        auto& A_out = builder.add_access(block, "A");
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_sub, "_out", {"_in1", "_in2"});
        auto idx = symbolic::add(indvar_j, symbolic::mul(sym_i, sym_s0));
        builder.add_computational_memlet(block, A_in, tasklet, "_in1", {idx}, ptr_desc);
        builder.add_computational_memlet(block, dot_in, tasklet, "_in2", {});
        builder.add_computational_memlet(block, tasklet, "_out", A_out, {idx}, ptr_desc);
    }

    // Block: A[j + i*_s0] /= A[j + j*_s0]
    {
        auto& block = builder.add_block(body_j);
        auto& A_in1 = builder.add_access(block, "A");
        auto& A_in2 = builder.add_access(block, "A");
        auto& A_out = builder.add_access(block, "A");
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_div, "_out", {"_in1", "_in2"});
        auto idx1 = symbolic::add(indvar_j, symbolic::mul(sym_i, sym_s0));
        auto idx2 = symbolic::add(indvar_j, symbolic::mul(indvar_j, sym_s0));
        builder.add_computational_memlet(block, A_in1, tasklet, "_in1", {idx1}, ptr_desc);
        builder.add_computational_memlet(block, A_in2, tasklet, "_in2", {idx2}, ptr_desc);
        builder.add_computational_memlet(block, tasklet, "_out", A_out, {idx1}, ptr_desc);
    }

    // === Diagonal element computation (after j-loop, still in body_i) ===

    // Block: _dot_res_5 = 0.0
    {
        auto& block = builder.add_block(body_i);
        auto& zero = builder.add_constant(block, "0.0", base_desc);
        auto& out = builder.add_access(block, "_dot_res_5");
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
        builder.add_computational_memlet(block, zero, tasklet, "_in", {});
        builder.add_computational_memlet(block, tasklet, "_out", out, {});
    }

    // Diagonal dot product loop: for (_i1 = 0; _i1 < i; _i1++)
    auto indvar_i1 = symbolic::symbol("_i1");
    auto init_i1 = symbolic::integer(0);
    auto condition_i1 = symbolic::Lt(indvar_i1, indvar_i);
    auto update_i1 = symbolic::add(indvar_i1, symbolic::integer(1));

    auto& loop_i1 = builder.add_for(body_i, indvar_i1, condition_i1, init_i1, update_i1);
    auto& body_i1 = loop_i1.root();

    // Loop body: _dot_res_5 = A[_i1 + i*_s0] * A[_i1 + i*_s0] + _dot_res_5 (FMA)
    {
        auto& block = builder.add_block(body_i1);
        auto& A_in1 = builder.add_access(block, "A");
        auto& A_in2 = builder.add_access(block, "A");
        auto& dot_in = builder.add_access(block, "_dot_res_5");
        auto& dot_out = builder.add_access(block, "_dot_res_5");

        auto& fma_tasklet =
            builder.add_tasklet(block, data_flow::TaskletCode::fp_fma, "__out", {"_in1", "_in2", "_in3"});
        auto read_idx = symbolic::add(indvar_i1, symbolic::mul(sym_i, sym_s0));
        builder.add_computational_memlet(block, A_in1, fma_tasklet, "_in1", {read_idx}, ptr_desc);
        builder.add_computational_memlet(block, A_in2, fma_tasklet, "_in2", {read_idx}, ptr_desc);
        builder.add_computational_memlet(block, dot_in, fma_tasklet, "_in3", {});
        builder.add_computational_memlet(block, fma_tasklet, "__out", dot_out, {});
    }

    // Block: A[i + i*_s0] -= _dot_res_5
    {
        auto& block = builder.add_block(body_i);
        auto& A_in = builder.add_access(block, "A");
        auto& dot_in = builder.add_access(block, "_dot_res_5");
        auto& A_out = builder.add_access(block, "A");
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_sub, "_out", {"_in1", "_in2"});
        auto diag_idx = symbolic::add(sym_i, symbolic::mul(sym_i, sym_s0));
        builder.add_computational_memlet(block, A_in, tasklet, "_in1", {diag_idx}, ptr_desc);
        builder.add_computational_memlet(block, dot_in, tasklet, "_in2", {});
        builder.add_computational_memlet(block, tasklet, "_out", A_out, {diag_idx}, ptr_desc);
    }

    // Block: tmp_5 = A[i + i*_s0]
    {
        auto& block = builder.add_block(body_i);
        auto& A_in = builder.add_access(block, "A");
        auto& tmp_out = builder.add_access(block, "tmp_5");
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
        auto diag_idx = symbolic::add(sym_i, symbolic::mul(sym_i, sym_s0));
        builder.add_computational_memlet(block, A_in, tasklet, "_in", {diag_idx}, ptr_desc);
        builder.add_computational_memlet(block, tasklet, "_out", tmp_out, {});
    }

    // Block: _tmp_6 = sqrt(tmp_5) (using fp_neg as placeholder for sqrt)
    {
        auto& block = builder.add_block(body_i);
        auto& tmp_in = builder.add_access(block, "tmp_5");
        auto& tmp_out = builder.add_access(block, "_tmp_6");
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_neg, "_out", {"_in1"});
        builder.add_computational_memlet(block, tmp_in, tasklet, "_in1", {});
        builder.add_computational_memlet(block, tasklet, "_out", tmp_out, {});
    }

    // Block: A[i + i*_s0] = _tmp_6
    {
        auto& block = builder.add_block(body_i);
        auto& tmp_in = builder.add_access(block, "_tmp_6");
        auto& A_out = builder.add_access(block, "A");
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
        auto diag_idx = symbolic::add(sym_i, symbolic::mul(sym_i, sym_s0));
        builder.add_computational_memlet(block, tmp_in, tasklet, "_in", {});
        builder.add_computational_memlet(block, tasklet, "_out", A_out, {diag_idx}, ptr_desc);
    }

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::LoopCarriedDependencyAnalysis>();
    auto& dependencies_j = analysis.dependencies(loop_j);

    // j-loop: A has RAW dependency (inner loop reads A[_i0+i*_s0], body writes A[j+i*_s0])
    EXPECT_NE(dependencies_j.find("A"), dependencies_j.end());
    EXPECT_EQ(dependencies_j.at("A").type, analysis::LoopCarriedDependency::LOOP_CARRIED_DEPENDENCY_READ_WRITE);
}

// ============================================================================
// LU factorization (un-tiled). Mirrors LUFixture in
// docc/opt/tests/transformations/optimizations/blocking_test.cpp.
//
// Goal: with MLA delinearization in place, surface what LCDA sees for each
// loop level of LU (i, j, k, j2, k2). Each assertion captures the current
// behavior so that future LCDA / MLA refinements have a regression baseline.
// ============================================================================
TEST(LoopCarriedDependencyAnalysisTest, LU_Factorization_Diagnostic) {
    builder::StructuredSDFGBuilder builder("lu_lcd", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);
    builder.add_container("k", sym_desc);
    builder.add_container("j2", sym_desc);
    builder.add_container("k2", sym_desc);

    types::Scalar elem_desc(types::PrimitiveType::Double);
    builder.add_container("tmp_2", elem_desc);
    builder.add_container("tmp_8", elem_desc);

    types::Pointer ptr_desc(elem_desc);
    builder.add_container("A", ptr_desc, true);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");
    auto k = symbolic::symbol("k");
    auto j2 = symbolic::symbol("j2");
    auto k2 = symbolic::symbol("k2");
    auto N = symbolic::symbol("N");
    auto one = symbolic::integer(1);
    auto zero = symbolic::integer(0);

    auto& i_loop = builder.add_for(root, i, symbolic::Lt(i, N), zero, symbolic::add(i, one));
    auto& j_loop = builder.add_for(i_loop.root(), j, symbolic::Lt(j, i), zero, symbolic::add(j, one));
    auto& k_loop = builder.add_for(j_loop.root(), k, symbolic::Lt(k, j), zero, symbolic::add(k, one));

    // mul block
    {
        auto& block = builder.add_block(k_loop.root());
        auto& a1 = builder.add_access(block, "A");
        auto& a2 = builder.add_access(block, "A");
        auto& mul_t = builder.add_tasklet(block, data_flow::TaskletCode::fp_mul, "_out", {"_in1", "_in2"});
        auto& tmp_out = builder.add_access(block, "tmp_2");
        builder.add_computational_memlet(block, a1, mul_t, "_in1", {symbolic::add(symbolic::mul(i, N), k)}, ptr_desc);
        builder.add_computational_memlet(block, a2, mul_t, "_in2", {symbolic::add(symbolic::mul(k, N), j)}, ptr_desc);
        builder.add_computational_memlet(block, mul_t, "_out", tmp_out, {});
    }
    // sub block (in k loop)
    {
        auto& block = builder.add_block(k_loop.root());
        auto& a_in = builder.add_access(block, "A");
        auto& tmp_in = builder.add_access(block, "tmp_2");
        auto& sub_t = builder.add_tasklet(block, data_flow::TaskletCode::fp_sub, "_out", {"_in1", "_in2"});
        auto& a_out = builder.add_access(block, "A");
        builder.add_computational_memlet(block, a_in, sub_t, "_in1", {symbolic::add(symbolic::mul(i, N), j)}, ptr_desc);
        builder.add_computational_memlet(block, tmp_in, sub_t, "_in2", {});
        builder.add_computational_memlet(block, sub_t, "_out", a_out, {symbolic::add(symbolic::mul(i, N), j)}, ptr_desc);
    }
    // div block (in j loop)
    {
        auto& block = builder.add_block(j_loop.root());
        auto& a_in = builder.add_access(block, "A");
        auto& a_div = builder.add_access(block, "A");
        auto& div_t = builder.add_tasklet(block, data_flow::TaskletCode::fp_div, "_out", {"_in1", "_in2"});
        auto& a_out = builder.add_access(block, "A");
        builder.add_computational_memlet(block, a_in, div_t, "_in1", {symbolic::add(symbolic::mul(i, N), j)}, ptr_desc);
        builder.add_computational_memlet(block, a_div, div_t, "_in2", {symbolic::add(symbolic::mul(j, N), j)}, ptr_desc);
        builder.add_computational_memlet(block, div_t, "_out", a_out, {symbolic::add(symbolic::mul(i, N), j)}, ptr_desc);
    }

    // j2 / k2 (trailing update)
    auto& j2_loop =
        builder.add_for(i_loop.root(), j2, symbolic::Lt(j2, symbolic::sub(N, i)), zero, symbolic::add(j2, one));
    auto& k2_loop = builder.add_for(j2_loop.root(), k2, symbolic::Lt(k2, i), zero, symbolic::add(k2, one));
    {
        auto& block = builder.add_block(k2_loop.root());
        auto& a1 = builder.add_access(block, "A");
        auto& a2 = builder.add_access(block, "A");
        auto& mul_t = builder.add_tasklet(block, data_flow::TaskletCode::fp_mul, "_out", {"_in1", "_in2"});
        auto& tmp_out = builder.add_access(block, "tmp_8");
        builder.add_computational_memlet(block, a1, mul_t, "_in1", {symbolic::add(symbolic::mul(i, N), k2)}, ptr_desc);
        builder.add_computational_memlet(
            block, a2, mul_t, "_in2", {symbolic::add(symbolic::mul(k2, N), symbolic::add(i, j2))}, ptr_desc
        );
        builder.add_computational_memlet(block, mul_t, "_out", tmp_out, {});
    }
    {
        auto& block = builder.add_block(k2_loop.root());
        auto& a_in = builder.add_access(block, "A");
        auto& tmp_in = builder.add_access(block, "tmp_8");
        auto& sub_t = builder.add_tasklet(block, data_flow::TaskletCode::fp_sub, "_out", {"_in1", "_in2"});
        auto& a_out = builder.add_access(block, "A");
        builder.add_computational_memlet(
            block, a_in, sub_t, "_in1", {symbolic::add(symbolic::mul(i, N), symbolic::add(i, j2))}, ptr_desc
        );
        builder.add_computational_memlet(block, tmp_in, sub_t, "_in2", {});
        builder.add_computational_memlet(
            block, sub_t, "_out", a_out, {symbolic::add(symbolic::mul(i, N), symbolic::add(i, j2))}, ptr_desc
        );
    }

    analysis::AnalysisManager am(sdfg);
    auto& lcd = am.get<analysis::LoopCarriedDependencyAnalysis>();

    auto kind_str = [](analysis::LoopCarriedDependency t) {
        return t == analysis::LOOP_CARRIED_DEPENDENCY_READ_WRITE ? "RAW" : "WAW";
    };

    auto dump_loop = [&](const char* name, structured_control_flow::StructuredLoop& loop) {
        std::cerr << "[LCD-LU] loop=" << name << " (" << loop.indvar()->get_name() << ")\n";
        if (!lcd.available(loop)) {
            std::cerr << "  (not analyzed)\n";
            return;
        }
        for (const auto& p : lcd.pairs(loop)) {
            std::cerr << "  " << kind_str(p.type) << " on '" << p.writer->container() << "': deltas='"
                      << p.deltas.deltas_str << "' empty=" << p.deltas.empty << "\n";
        }
        std::cerr << "  containers with carry:";
        for (const auto& [c, info] : lcd.dependencies(loop)) {
            std::cerr << " " << c << "(" << kind_str(info.type) << ")";
        }
        std::cerr << "\n";
    };

    dump_loop("i", i_loop);
    dump_loop("j", j_loop);
    dump_loop("k", k_loop);
    dump_loop("j2", j2_loop);
    dump_loop("k2", k2_loop);

    // ------------------------------------------------------------------------
    // Expected behavior under MLA-delinearized subsets:
    //
    // i-loop (outermost): A is updated in row i in iteration i (panel column &
    // diagonal divide), and rows j > i are updated by the trailing update at
    // iteration j (k2 loop). Earlier iterations i' < i write row i' which is
    // later read by j_loop's k iteration when k = i' (read A[k*N + j], k < j
    // so j' = k matches a previous panel column). So i has RAW carry on A.
    //
    // j-loop: for fixed i, j varies. The j-loop reads A[k*N + j] with k < j
    // and writes A[i*N + j]. The A[i*N + j] write of iteration j' is read in
    // a later iteration j (j > j') only if some k = j' and the read column
    // matches the written column j. There's also the diagonal A[j*N + j]
    // read which depends on writes from earlier j' = k and a potentially
    // earlier panel iteration. j has RAW carry on A.
    //
    // k-loop: for fixed (i, j), k varies; the body reads A[i*N+k] and A[k*N+j]
    // and reads/writes A[i*N+j]. The A[i*N+j] write/read pair within k is
    // loop-carried (RAW + WAW).
    //
    // j2-loop: for fixed i, the trailing update writes A[i*N + (i+j2)] for
    // j2 in [0, N-i). Different j2 -> different column, so no carry on A
    // across j2 iterations.
    //
    // k2-loop: reads A[i*N+k2], A[k2*N+(i+j2)] and reads/writes A[i*N+(i+j2)].
    // A[i*N+(i+j2)] write/read pair is the carry: RAW + WAW on k2.
    // ------------------------------------------------------------------------

    ASSERT_TRUE(lcd.available(i_loop));
    ASSERT_TRUE(lcd.available(j_loop));
    ASSERT_TRUE(lcd.available(k_loop));
    ASSERT_TRUE(lcd.available(j2_loop));
    ASSERT_TRUE(lcd.available(k2_loop));

    // k2-loop: A has RAW carry from the trailing update accumulating into A[i*N + i+j2].
    EXPECT_NE(lcd.dependencies(k2_loop).find("A"), lcd.dependencies(k2_loop).end());

    // j2-loop: with MLA, no carry on A expected (different j2 -> different column).
    EXPECT_EQ(lcd.dependencies(j2_loop).find("A"), lcd.dependencies(j2_loop).end())
        << "j2 should have no A carry under MLA-delinearized subsets";

    // k-loop: A[i*N+j] read/write within k -> RAW carry on A.
    EXPECT_NE(lcd.dependencies(k_loop).find("A"), lcd.dependencies(k_loop).end());

    // j-loop and i-loop: under MLA-precise subsets, the writes within row i
    // are read in later iterations of i (cross-row diagonal reads via j*N + j
    // etc.). Document the observed carry rather than predicting it.
    SUCCEED() << "j/i carry shape printed in the trace above; refine when LCDA is tightened.";
}

// ---------------------------------------------------------------------------
// Reduction-detection gap-closing tests
// ---------------------------------------------------------------------------

// for (i) { s = s + A[i]; }  integer additive reduction (int_add) -> Add.
TEST(LoopCarriedDependencyAnalysisTest, IntSumReduction) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    types::Scalar base_desc(types::PrimitiveType::Int64);
    types::Pointer edge_desc(base_desc);
    types::Pointer desc;

    builder.add_container("A", desc, true);
    builder.add_container("s", base_desc, true);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);

    auto indvar = symbolic::symbol("i");
    auto& loop = builder.add_for(
        root,
        indvar,
        symbolic::Lt(indvar, symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(indvar, symbolic::integer(1))
    );
    auto& block = builder.add_block(loop.root());
    auto& a = builder.add_access(block, "A");
    auto& s_in = builder.add_access(block, "s");
    auto& s_out = builder.add_access(block, "s");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::int_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, a, tasklet, "_in1", {indvar}, edge_desc);
    builder.add_computational_memlet(block, s_in, tasklet, "_in2", {});
    builder.add_computational_memlet(block, tasklet, "_out", s_out, {});

    analysis::AnalysisManager analysis_manager(sdfg);
    auto& lcd = analysis_manager.get<analysis::LoopCarriedDependencyAnalysis>();

    ASSERT_TRUE(lcd.has_reductions(loop));
    ASSERT_EQ(lcd.reductions(loop).size(), 1u);
    EXPECT_EQ(lcd.reductions(loop)[0].operation, structured_control_flow::ReductionOperation::Add);
    EXPECT_EQ(lcd.reductions(loop)[0].container, "s");
    EXPECT_TRUE(lcd.is_reduction_only(loop));
}

// for (i) { m = smax(m, A[i]); }  signed-integer max reduction -> Max.
TEST(LoopCarriedDependencyAnalysisTest, IntMaxReduction) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    types::Scalar base_desc(types::PrimitiveType::Int64);
    types::Pointer edge_desc(base_desc);
    types::Pointer desc;

    builder.add_container("A", desc, true);
    builder.add_container("m", base_desc, true);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);

    auto indvar = symbolic::symbol("i");
    auto& loop = builder.add_for(
        root,
        indvar,
        symbolic::Lt(indvar, symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(indvar, symbolic::integer(1))
    );
    auto& block = builder.add_block(loop.root());
    auto& a = builder.add_access(block, "A");
    auto& m_in = builder.add_access(block, "m");
    auto& m_out = builder.add_access(block, "m");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::int_smax, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, m_in, tasklet, "_in1", {});
    builder.add_computational_memlet(block, a, tasklet, "_in2", {indvar}, edge_desc);
    builder.add_computational_memlet(block, tasklet, "_out", m_out, {});

    analysis::AnalysisManager analysis_manager(sdfg);
    auto& lcd = analysis_manager.get<analysis::LoopCarriedDependencyAnalysis>();

    ASSERT_TRUE(lcd.has_reductions(loop));
    ASSERT_EQ(lcd.reductions(loop).size(), 1u);
    EXPECT_EQ(lcd.reductions(loop)[0].operation, structured_control_flow::ReductionOperation::Max);
    EXPECT_EQ(lcd.reductions(loop)[0].container, "m");
    EXPECT_TRUE(lcd.is_reduction_only(loop));
}

// for (i) { m = smin(m, A[i]); }  signed-integer min reduction -> Min.
TEST(LoopCarriedDependencyAnalysisTest, IntMinReduction) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    types::Scalar base_desc(types::PrimitiveType::Int64);
    types::Pointer edge_desc(base_desc);
    types::Pointer desc;

    builder.add_container("A", desc, true);
    builder.add_container("m", base_desc, true);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);

    auto indvar = symbolic::symbol("i");
    auto& loop = builder.add_for(
        root,
        indvar,
        symbolic::Lt(indvar, symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(indvar, symbolic::integer(1))
    );
    auto& block = builder.add_block(loop.root());
    auto& a = builder.add_access(block, "A");
    auto& m_in = builder.add_access(block, "m");
    auto& m_out = builder.add_access(block, "m");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::int_smin, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, m_in, tasklet, "_in1", {});
    builder.add_computational_memlet(block, a, tasklet, "_in2", {indvar}, edge_desc);
    builder.add_computational_memlet(block, tasklet, "_out", m_out, {});

    analysis::AnalysisManager analysis_manager(sdfg);
    auto& lcd = analysis_manager.get<analysis::LoopCarriedDependencyAnalysis>();

    ASSERT_TRUE(lcd.has_reductions(loop));
    ASSERT_EQ(lcd.reductions(loop).size(), 1u);
    EXPECT_EQ(lcd.reductions(loop)[0].operation, structured_control_flow::ReductionOperation::Min);
    EXPECT_EQ(lcd.reductions(loop)[0].container, "m");
    EXPECT_TRUE(lcd.is_reduction_only(loop));
}

// for (i) { m = fmax(m, A[i]); }  floating-point max via CMath library node -> Max.
TEST(LoopCarriedDependencyAnalysisTest, CMathFloatMaxReduction) {
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

    auto indvar = symbolic::symbol("i");
    auto& loop = builder.add_for(
        root,
        indvar,
        symbolic::Lt(indvar, symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(indvar, symbolic::integer(1))
    );
    auto& block = builder.add_block(loop.root());
    auto& a = builder.add_access(block, "A");
    auto& m_in = builder.add_access(block, "m");
    auto& m_out = builder.add_access(block, "m");
    auto& node = static_cast<math::cmath::CMathNode&>(builder.add_library_node<math::cmath::CMathNode>(
        block, DebugInfo(), math::cmath::CMathFunction::fmax, types::PrimitiveType::Float
    ));
    builder.add_computational_memlet(block, m_in, node, "_in1", {}, base_desc);
    builder.add_computational_memlet(block, a, node, "_in2", {indvar}, desc);
    builder.add_computational_memlet(block, node, "_out", m_out, {}, base_desc);

    analysis::AnalysisManager analysis_manager(sdfg);
    auto& lcd = analysis_manager.get<analysis::LoopCarriedDependencyAnalysis>();

    ASSERT_TRUE(lcd.has_reductions(loop));
    ASSERT_EQ(lcd.reductions(loop).size(), 1u);
    EXPECT_EQ(lcd.reductions(loop)[0].operation, structured_control_flow::ReductionOperation::Max);
    EXPECT_EQ(lcd.reductions(loop)[0].container, "m");
    EXPECT_TRUE(lcd.is_reduction_only(loop));
}

// for (i) { s = fma(A[i], C[i], s); }  CMath fma (a*b + c) with the accumulator
// as the addend operand -> additive reduction.
TEST(LoopCarriedDependencyAnalysisTest, CMathFmaAddendReduction) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    types::Pointer opaque_desc;

    builder.add_container("A", opaque_desc, true);
    builder.add_container("C", opaque_desc, true);
    builder.add_container("s", base_desc, true);
    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);

    auto indvar = symbolic::symbol("i");
    auto& loop = builder.add_for(
        root,
        indvar,
        symbolic::Lt(indvar, symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(indvar, symbolic::integer(1))
    );
    auto& block = builder.add_block(loop.root());
    auto& a = builder.add_access(block, "A");
    auto& c = builder.add_access(block, "C");
    auto& s_in = builder.add_access(block, "s");
    auto& s_out = builder.add_access(block, "s");
    auto& node = static_cast<math::cmath::CMathNode&>(builder.add_library_node<math::cmath::CMathNode>(
        block, DebugInfo(), math::cmath::CMathFunction::fma, types::PrimitiveType::Float
    ));
    builder.add_computational_memlet(block, a, node, "_in1", {indvar}, desc);
    builder.add_computational_memlet(block, c, node, "_in2", {indvar}, desc);
    builder.add_computational_memlet(block, s_in, node, "_in3", {}, base_desc);
    builder.add_computational_memlet(block, node, "_out", s_out, {}, base_desc);

    analysis::AnalysisManager analysis_manager(sdfg);
    auto& lcd = analysis_manager.get<analysis::LoopCarriedDependencyAnalysis>();

    ASSERT_TRUE(lcd.has_reductions(loop));
    ASSERT_EQ(lcd.reductions(loop).size(), 1u);
    EXPECT_EQ(lcd.reductions(loop)[0].operation, structured_control_flow::ReductionOperation::Add);
    EXPECT_EQ(lcd.reductions(loop)[0].container, "s");
    EXPECT_TRUE(lcd.is_reduction_only(loop));
}

// for (i) { B = B - A[i]; }  fp_sub is neither associative nor commutative, so
// despite the RAW carry it is NOT recognised as a reduction.
TEST(LoopCarriedDependencyAnalysisTest, NonAssociativeSub_NotReduction) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer edge_desc(base_desc);
    types::Pointer desc;

    builder.add_container("A", desc, true);
    builder.add_container("B", base_desc, true);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);

    auto indvar = symbolic::symbol("i");
    auto& loop = builder.add_for(
        root,
        indvar,
        symbolic::Lt(indvar, symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(indvar, symbolic::integer(1))
    );
    auto& block = builder.add_block(loop.root());
    auto& a = builder.add_access(block, "A");
    auto& b_in = builder.add_access(block, "B");
    auto& b_out = builder.add_access(block, "B");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_sub, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, b_in, tasklet, "_in1", {});
    builder.add_computational_memlet(block, a, tasklet, "_in2", {indvar}, edge_desc);
    builder.add_computational_memlet(block, tasklet, "_out", b_out, {});

    analysis::AnalysisManager analysis_manager(sdfg);
    auto& lcd = analysis_manager.get<analysis::LoopCarriedDependencyAnalysis>();

    // RAW carry exists ...
    EXPECT_TRUE(lcd.has_loop_carried_raw(loop));
    // ... but it is not a reduction.
    EXPECT_FALSE(lcd.has_reductions(loop));
    EXPECT_TRUE(lcd.reductions(loop).empty());
    EXPECT_FALSE(lcd.is_reduction_only(loop));
}

// for (i) { s = s + A[i]; p = p * A[i]; }  two reductions on DISTINCT scalar
// containers are both detected and the loop is reduction-only.
TEST(LoopCarriedDependencyAnalysisTest, FusedDistinctScalarReductions) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer edge_desc(base_desc);
    types::Pointer desc;

    builder.add_container("A", desc, true);
    builder.add_container("s", base_desc, true);
    builder.add_container("p", base_desc, true);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);

    auto indvar = symbolic::symbol("i");
    auto& loop = builder.add_for(
        root,
        indvar,
        symbolic::Lt(indvar, symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(indvar, symbolic::integer(1))
    );
    auto& body = loop.root();

    // s = s + A[i]
    auto& block1 = builder.add_block(body);
    auto& a1 = builder.add_access(block1, "A");
    auto& s_in = builder.add_access(block1, "s");
    auto& s_out = builder.add_access(block1, "s");
    auto& tasklet1 = builder.add_tasklet(block1, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block1, a1, tasklet1, "_in1", {indvar}, edge_desc);
    builder.add_computational_memlet(block1, s_in, tasklet1, "_in2", {});
    builder.add_computational_memlet(block1, tasklet1, "_out", s_out, {});

    // p = p * A[i]
    auto& block2 = builder.add_block(body);
    auto& a2 = builder.add_access(block2, "A");
    auto& p_in = builder.add_access(block2, "p");
    auto& p_out = builder.add_access(block2, "p");
    auto& tasklet2 = builder.add_tasklet(block2, data_flow::TaskletCode::fp_mul, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block2, a2, tasklet2, "_in1", {indvar}, edge_desc);
    builder.add_computational_memlet(block2, p_in, tasklet2, "_in2", {});
    builder.add_computational_memlet(block2, tasklet2, "_out", p_out, {});

    analysis::AnalysisManager analysis_manager(sdfg);
    auto& lcd = analysis_manager.get<analysis::LoopCarriedDependencyAnalysis>();

    ASSERT_TRUE(lcd.has_reductions(loop));
    ASSERT_EQ(lcd.reductions(loop).size(), 2u);
    std::map<std::string, structured_control_flow::ReductionOperation> by_container;
    for (auto& r : lcd.reductions(loop)) {
        by_container[r.container] = r.operation;
    }
    ASSERT_EQ(by_container.count("s"), 1u);
    ASSERT_EQ(by_container.count("p"), 1u);
    EXPECT_EQ(by_container.at("s"), structured_control_flow::ReductionOperation::Add);
    EXPECT_EQ(by_container.at("p"), structured_control_flow::ReductionOperation::Mul);
    EXPECT_TRUE(lcd.is_reduction_only(loop));
}

// for (i=1) { B = B + A[i]; C[i] = C[i-1]; }  a reduction on B combined with a
// non-reduction recurrence on C: has_reductions is true but is_reduction_only
// is false (the C carry is not a reduction).
TEST(LoopCarriedDependencyAnalysisTest, MixedReductionAndRecurrence_NotReductionOnly) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer edge_desc(base_desc);
    types::Pointer desc;

    builder.add_container("A", desc, true);
    builder.add_container("B", base_desc, true);
    builder.add_container("C", desc, true);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);

    auto indvar = symbolic::symbol("i");
    auto& loop = builder.add_for(
        root,
        indvar,
        symbolic::Lt(indvar, symbolic::symbol("N")),
        symbolic::integer(1),
        symbolic::add(indvar, symbolic::integer(1))
    );
    auto& body = loop.root();

    // B = B + A[i]
    auto& block1 = builder.add_block(body);
    auto& a1 = builder.add_access(block1, "A");
    auto& b_in = builder.add_access(block1, "B");
    auto& b_out = builder.add_access(block1, "B");
    auto& tasklet1 = builder.add_tasklet(block1, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block1, a1, tasklet1, "_in1", {indvar}, edge_desc);
    builder.add_computational_memlet(block1, b_in, tasklet1, "_in2", {});
    builder.add_computational_memlet(block1, tasklet1, "_out", b_out, {});

    // C[i] = C[i-1]
    auto& block2 = builder.add_block(body);
    auto& c_in = builder.add_access(block2, "C");
    auto& c_out = builder.add_access(block2, "C");
    auto& tasklet2 = builder.add_tasklet(block2, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder
        .add_computational_memlet(block2, c_in, tasklet2, "_in", {symbolic::sub(indvar, symbolic::integer(1))}, edge_desc);
    builder.add_computational_memlet(block2, tasklet2, "_out", c_out, {indvar}, edge_desc);

    analysis::AnalysisManager analysis_manager(sdfg);
    auto& lcd = analysis_manager.get<analysis::LoopCarriedDependencyAnalysis>();

    ASSERT_TRUE(lcd.has_reductions(loop));
    ASSERT_EQ(lcd.reductions(loop).size(), 1u);
    EXPECT_EQ(lcd.reductions(loop)[0].operation, structured_control_flow::ReductionOperation::Add);
    EXPECT_EQ(lcd.reductions(loop)[0].container, "B");
    // The C recurrence is a non-reduction RAW carry => not reduction-only.
    EXPECT_FALSE(lcd.is_reduction_only(loop));
}

// for (i=1) { A[i] = A[i-1] + B[i]; }  fp_add, but the accumulator location
// A[i] is loop-variant (different element each iteration), so it is a prefix
// sum / recurrence, NOT a reduction.
TEST(LoopCarriedDependencyAnalysisTest, PrefixSum_NotReduction) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer edge_desc(base_desc);
    types::Pointer desc;

    builder.add_container("A", desc, true);
    builder.add_container("B", desc, true);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);

    auto indvar = symbolic::symbol("i");
    auto& loop = builder.add_for(
        root,
        indvar,
        symbolic::Lt(indvar, symbolic::symbol("N")),
        symbolic::integer(1),
        symbolic::add(indvar, symbolic::integer(1))
    );
    auto& block = builder.add_block(loop.root());
    auto& a_in = builder.add_access(block, "A");
    auto& b = builder.add_access(block, "B");
    auto& a_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder
        .add_computational_memlet(block, a_in, tasklet, "_in1", {symbolic::sub(indvar, symbolic::integer(1))}, edge_desc);
    builder.add_computational_memlet(block, b, tasklet, "_in2", {indvar}, edge_desc);
    builder.add_computational_memlet(block, tasklet, "_out", a_out, {indvar}, edge_desc);

    analysis::AnalysisManager analysis_manager(sdfg);
    auto& lcd = analysis_manager.get<analysis::LoopCarriedDependencyAnalysis>();

    // RAW carry on A exists ...
    EXPECT_TRUE(lcd.has_loop_carried_raw(loop));
    // ... but the loop-variant accumulator means it is not a reduction.
    EXPECT_FALSE(lcd.has_reductions(loop));
    EXPECT_TRUE(lcd.reductions(loop).empty());
    EXPECT_FALSE(lcd.is_reduction_only(loop));
}

// A linearized accumulator address written and read back in different (but
// domain-equal) parametric forms: C[i*N + k] = C[k + N*i] + A[...]. The reduction
// runs over the inner induction variable j, w.r.t. which the accumulator cell is
// invariant. The structural form of the written vs. read-back index differs by
// reassociation, so detection relies on the semantic `equal_on_domain` check
// (with delinearization) rather than a syntactic comparison.
TEST(LoopCarriedDependencyAnalysisTest, LinearizedParametricReduction) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer edge_desc(base_desc);
    types::Pointer desc;

    builder.add_container("A", desc, true);
    builder.add_container("C", desc, true);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("k", sym_desc);
    builder.add_container("j", sym_desc);

    auto N = symbolic::symbol("N");
    auto i = symbolic::symbol("i");
    auto k = symbolic::symbol("k");
    auto j = symbolic::symbol("j");

    auto& loop_i =
        builder.add_for(root, i, symbolic::Lt(i, N), symbolic::integer(0), symbolic::add(i, symbolic::integer(1)));
    auto& loop_k =
        builder
            .add_for(loop_i.root(), k, symbolic::Lt(k, N), symbolic::integer(0), symbolic::add(k, symbolic::integer(1)));
    auto& loop_j =
        builder
            .add_for(loop_k.root(), j, symbolic::Lt(j, N), symbolic::integer(0), symbolic::add(j, symbolic::integer(1)));

    auto& block = builder.add_block(loop_j.root());
    auto& a_in = builder.add_access(block, "A");
    auto& c_in = builder.add_access(block, "C");
    auto& c_out = builder.add_access(block, "C");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});

    auto acc_addr_write = symbolic::add(symbolic::mul(i, N), k); // i*N + k
    auto acc_addr_read = symbolic::add(k, symbolic::mul(N, i)); // k + N*i (reassociated)
    auto a_addr = symbolic::add(symbolic::mul(acc_addr_write, N), j);

    builder.add_computational_memlet(block, a_in, tasklet, "_in1", {a_addr}, edge_desc);
    builder.add_computational_memlet(block, c_in, tasklet, "_in2", {acc_addr_read}, edge_desc);
    builder.add_computational_memlet(block, tasklet, "_out", c_out, {acc_addr_write}, edge_desc);

    analysis::AnalysisManager analysis_manager(sdfg);
    auto& lcd = analysis_manager.get<analysis::LoopCarriedDependencyAnalysis>();

    ASSERT_TRUE(lcd.has_reductions(loop_j));
    ASSERT_EQ(lcd.reductions(loop_j).size(), 1u);
    EXPECT_EQ(lcd.reductions(loop_j)[0].operation, structured_control_flow::ReductionOperation::Add);
    EXPECT_EQ(lcd.reductions(loop_j)[0].container, "C");
    EXPECT_TRUE(lcd.is_reduction_only(loop_j));
}

// A 2-D linearized array `A[i*N + j]` is summed over both nested loops into a
// single scalar accumulator `sum`. The accumulator address is the empty
// (scalar) subset, hence invariant across every reduction loop, while the
// source operand carries a multi-dimensional linearized index. The innermost
// loop (which directly holds the combine block) is the reduction loop.
TEST(LoopCarriedDependencyAnalysisTest, LinearizedMultiDimReductionToScalar) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer edge_desc(base_desc);
    types::Pointer desc;

    builder.add_container("A", desc, true);
    builder.add_container("sum", base_desc, true);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);

    auto N = symbolic::symbol("N");
    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");

    auto& loop_i =
        builder.add_for(root, i, symbolic::Lt(i, N), symbolic::integer(0), symbolic::add(i, symbolic::integer(1)));
    auto& loop_j =
        builder
            .add_for(loop_i.root(), j, symbolic::Lt(j, N), symbolic::integer(0), symbolic::add(j, symbolic::integer(1)));

    auto& block = builder.add_block(loop_j.root());
    auto& a_in = builder.add_access(block, "A");
    auto& sum_in = builder.add_access(block, "sum");
    auto& sum_out = builder.add_access(block, "sum");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});

    auto a_addr = symbolic::add(symbolic::mul(i, N), j); // i*N + j

    builder.add_computational_memlet(block, a_in, tasklet, "_in1", {a_addr}, edge_desc);
    builder.add_computational_memlet(block, sum_in, tasklet, "_in2", {});
    builder.add_computational_memlet(block, tasklet, "_out", sum_out, {});

    analysis::AnalysisManager analysis_manager(sdfg);
    auto& lcd = analysis_manager.get<analysis::LoopCarriedDependencyAnalysis>();

    ASSERT_TRUE(lcd.has_reductions(loop_j));
    ASSERT_EQ(lcd.reductions(loop_j).size(), 1u);
    EXPECT_EQ(lcd.reductions(loop_j)[0].operation, structured_control_flow::ReductionOperation::Add);
    EXPECT_EQ(lcd.reductions(loop_j)[0].container, "sum");
    EXPECT_TRUE(lcd.is_reduction_only(loop_j));
}
} // namespace
