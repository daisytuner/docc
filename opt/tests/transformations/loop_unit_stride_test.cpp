#include <gtest/gtest.h>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/structured_control_flow/for.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/transformations/loop_unit_stride.h"
#include "sdfg/types/array.h"
#include "sdfg/types/scalar.h"

using namespace sdfg;

// Test 1: Basic positive stride (stride = 2)
TEST(LoopUnitStrideTest, BasicPositiveStride) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);

    types::Scalar elem_desc(types::PrimitiveType::Double);
    types::Array arr_desc(elem_desc, symbolic::integer(100));
    builder.add_container("A", arr_desc, true);

    auto& root = builder.subject().root();

    // for i = 0; i < 10; i += 2 (iterations: 0, 2, 4, 6, 8)
    auto& for_loop = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::integer(10)),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(2))
    );

    auto& block = builder.add_block(for_loop.root());
    auto& a_node = builder.add_access(block, "A");
    auto& const_node = builder.add_constant(block, "1.0", elem_desc);
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in1"});
    builder.add_computational_memlet(block, const_node, tasklet, "_in1", {});
    builder.add_computational_memlet(block, tasklet, "_out", a_node, {symbolic::symbol("i")});

    auto sdfg = builder.move();
    builder::StructuredSDFGBuilder builder2(sdfg);
    analysis::AnalysisManager am(builder2.subject());

    auto* loop = dynamic_cast<structured_control_flow::For*>(&builder2.subject().root().at(0).first);
    ASSERT_NE(loop, nullptr);

    // Apply LoopUnitStride
    transformations::LoopUnitStride unit_stride(*loop);
    ASSERT_TRUE(unit_stride.can_be_applied(builder2, am));
    unit_stride.apply(builder2, am);

    // Verify unit stride
    auto new_stride = loop->stride();
    ASSERT_FALSE(new_stride.is_null());
    EXPECT_EQ(new_stride->as_int(), 1);

    // New init = 0
    EXPECT_TRUE(symbolic::eq(loop->init(), symbolic::integer(0)));

    // New condition: substituted condition 2*i < 10
    auto strided_expr = symbolic::mul(symbolic::integer(2), symbolic::symbol("i"));
    auto expected_cond = symbolic::Lt(strided_expr, symbolic::integer(10));
    EXPECT_TRUE(symbolic::eq(loop->condition(), expected_cond));

    // Verify container was created
    EXPECT_EQ(unit_stride.strided_container_name(), "__i_orig__");

    // Verify assignment was added
    ASSERT_EQ(loop->root().size(), 2); // new block + original block
    auto first_child = loop->root().at(0);
    auto& transition = first_child.second;
    ASSERT_EQ(transition.assignments().size(), 1);

    // Assignment: __i_orig__ = 0 + 2 * i = 2 * i
    auto strided_var = symbolic::symbol(unit_stride.strided_container_name());
    ASSERT_TRUE(transition.assignments().count(strided_var) > 0);
    auto assigned_value = transition.assignments().at(strided_var);
    auto expected = symbolic::mul(symbolic::integer(2), symbolic::symbol("i"));
    EXPECT_TRUE(symbolic::eq(assigned_value, expected));
}

// Test 2: Positive stride with symbolic bound
TEST(LoopUnitStrideTest, SymbolicBoundPositiveStride) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);

    types::Scalar elem_desc(types::PrimitiveType::Double);
    types::Array arr_desc(elem_desc, symbolic::symbol("N"));
    builder.add_container("A", arr_desc, true);

    auto& root = builder.subject().root();

    // for i = 0; i < N; i += 4
    auto& for_loop = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(4))
    );

    auto& block = builder.add_block(for_loop.root());
    auto& a_node = builder.add_access(block, "A");
    auto& const_node = builder.add_constant(block, "1.0", elem_desc);
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in1"});
    builder.add_computational_memlet(block, const_node, tasklet, "_in1", {});
    builder.add_computational_memlet(block, tasklet, "_out", a_node, {symbolic::symbol("i")});

    auto sdfg = builder.move();
    builder::StructuredSDFGBuilder builder2(sdfg);
    analysis::AnalysisManager am(builder2.subject());

    auto* loop = dynamic_cast<structured_control_flow::For*>(&builder2.subject().root().at(0).first);
    ASSERT_NE(loop, nullptr);

    // Apply LoopUnitStride
    transformations::LoopUnitStride unit_stride(*loop);
    ASSERT_TRUE(unit_stride.can_be_applied(builder2, am));
    unit_stride.apply(builder2, am);

    // Verify unit stride
    auto new_stride = loop->stride();
    ASSERT_FALSE(new_stride.is_null());
    EXPECT_EQ(new_stride->as_int(), 1);

    // New init = 0
    EXPECT_TRUE(symbolic::eq(loop->init(), symbolic::integer(0)));

    // New condition: substituted condition 4*i < N
    auto strided_expr = symbolic::mul(symbolic::integer(4), symbolic::symbol("i"));
    auto expected_cond = symbolic::Lt(strided_expr, symbolic::symbol("N"));
    EXPECT_TRUE(symbolic::eq(loop->condition(), expected_cond));

    // Assignment: __i_orig__ = 0 + 4 * i = 4 * i
    auto first_child = loop->root().at(0);
    auto& transition = first_child.second;
    auto strided_var = symbolic::symbol(unit_stride.strided_container_name());
    auto assigned_value = transition.assignments().at(strided_var);
    auto expected = symbolic::mul(symbolic::integer(4), symbolic::symbol("i"));
    EXPECT_TRUE(symbolic::eq(assigned_value, expected));
}

// Test 3: Cannot apply to non-zero init (requires LoopShift first)
TEST(LoopUnitStrideTest, CannotApplyNonZeroInit) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);

    types::Scalar elem_desc(types::PrimitiveType::Double);
    types::Array arr_desc(elem_desc, symbolic::integer(100));
    builder.add_container("A", arr_desc, true);

    auto& root = builder.subject().root();

    // for i = 5; i < 20; i += 3 (init != 0, requires LoopShift first)
    auto& for_loop = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::integer(20)),
        symbolic::integer(5),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(3))
    );

    auto& block = builder.add_block(for_loop.root());
    auto& a_node = builder.add_access(block, "A");
    auto& const_node = builder.add_constant(block, "1.0", elem_desc);
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in1"});
    builder.add_computational_memlet(block, const_node, tasklet, "_in1", {});
    builder.add_computational_memlet(block, tasklet, "_out", a_node, {symbolic::symbol("i")});

    auto sdfg = builder.move();
    builder::StructuredSDFGBuilder builder2(sdfg);
    analysis::AnalysisManager am(builder2.subject());

    auto* loop = dynamic_cast<structured_control_flow::For*>(&builder2.subject().root().at(0).first);
    ASSERT_NE(loop, nullptr);

    // Cannot apply to non-zero init - need LoopShift first
    transformations::LoopUnitStride unit_stride(*loop);
    EXPECT_FALSE(unit_stride.can_be_applied(builder2, am));
}

// Test 4: Negative stride with init=0 (stride = -2) - direction is preserved
TEST(LoopUnitStrideTest, NegativeStridePreservesDirection) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::Int64); // Signed for negative values
    builder.add_container("i", sym_desc);

    types::Scalar elem_desc(types::PrimitiveType::Double);
    types::Array arr_desc(elem_desc, symbolic::integer(100));
    builder.add_container("A", arr_desc, true);

    auto& root = builder.subject().root();

    // for i = 0; -10 < i; i -= 2 (iterations: 0, -2, -4, -6, -8)
    auto& for_loop = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::integer(-10), symbolic::symbol("i")), // -10 < i
        symbolic::integer(0),
        symbolic::sub(symbolic::symbol("i"), symbolic::integer(2)) // i -= 2
    );

    auto& block = builder.add_block(for_loop.root());
    auto& a_node = builder.add_access(block, "A");
    auto& const_node = builder.add_constant(block, "1.0", elem_desc);
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in1"});
    builder.add_computational_memlet(block, const_node, tasklet, "_in1", {});
    // Use absolute value of i for indexing: 50 + i maps -8..0 to 42..50
    builder.add_computational_memlet(
        block, tasklet, "_out", a_node, {symbolic::add(symbolic::integer(50), symbolic::symbol("i"))}
    );

    auto sdfg = builder.move();
    builder::StructuredSDFGBuilder builder2(sdfg);
    analysis::AnalysisManager am(builder2.subject());

    auto* loop = dynamic_cast<structured_control_flow::For*>(&builder2.subject().root().at(0).first);
    ASSERT_NE(loop, nullptr);

    // Apply LoopUnitStride
    transformations::LoopUnitStride unit_stride(*loop);
    ASSERT_TRUE(unit_stride.can_be_applied(builder2, am));
    unit_stride.apply(builder2, am);

    // Verify NEGATIVE unit stride - direction is preserved!
    // Original: i -= 2, After: i -= 1
    auto new_stride = loop->stride();
    ASSERT_FALSE(new_stride.is_null());
    EXPECT_EQ(new_stride->as_int(), -1) << "Direction must be preserved: negative stride -> negative unit stride";

    // New init = 0
    EXPECT_TRUE(symbolic::eq(loop->init(), symbolic::integer(0)));

    // New condition: substituted with |stride| * i = 2*i
    // -10 < i becomes -10 < 2*i
    auto strided_expr = symbolic::mul(symbolic::integer(2), symbolic::symbol("i"));
    auto expected_cond = symbolic::Lt(symbolic::integer(-10), strided_expr);
    EXPECT_TRUE(symbolic::eq(loop->condition(), expected_cond));

    // Assignment: __i_orig__ = 2 * i (using absolute value)
    // When i' = 0, -1, -2, -3, -4, we get __i_orig__ = 0, -2, -4, -6, -8
    auto first_child = loop->root().at(0);
    auto& transition = first_child.second;
    auto strided_var = symbolic::symbol(unit_stride.strided_container_name());
    auto assigned_value = transition.assignments().at(strided_var);
    EXPECT_TRUE(symbolic::eq(assigned_value, strided_expr));
}

// Test 5: Positive stride direction is preserved
TEST(LoopUnitStrideTest, PositiveStridePreservesDirection) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);

    types::Scalar elem_desc(types::PrimitiveType::Double);
    types::Array arr_desc(elem_desc, symbolic::integer(100));
    builder.add_container("A", arr_desc, true);

    auto& root = builder.subject().root();

    // for i = 0; i < 12; i += 3 (iterations: 0, 3, 6, 9)
    auto& for_loop = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::integer(12)),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(3))
    );

    auto& block = builder.add_block(for_loop.root());
    auto& a_node = builder.add_access(block, "A");
    auto& const_node = builder.add_constant(block, "1.0", elem_desc);
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in1"});
    builder.add_computational_memlet(block, const_node, tasklet, "_in1", {});
    builder.add_computational_memlet(block, tasklet, "_out", a_node, {symbolic::symbol("i")});

    auto sdfg = builder.move();
    builder::StructuredSDFGBuilder builder2(sdfg);
    analysis::AnalysisManager am(builder2.subject());

    auto* loop = dynamic_cast<structured_control_flow::For*>(&builder2.subject().root().at(0).first);
    ASSERT_NE(loop, nullptr);

    // Apply LoopUnitStride
    transformations::LoopUnitStride unit_stride(*loop);
    ASSERT_TRUE(unit_stride.can_be_applied(builder2, am));
    unit_stride.apply(builder2, am);

    // Verify POSITIVE unit stride - direction is preserved!
    // Original: i += 3, After: i += 1
    auto new_stride = loop->stride();
    ASSERT_FALSE(new_stride.is_null());
    EXPECT_EQ(new_stride->as_int(), 1) << "Direction must be preserved: positive stride -> positive unit stride";

    // When i' = 0, 1, 2, 3, we get __i_orig__ = 0, 3, 6, 9
    auto strided_expr = symbolic::mul(symbolic::integer(3), symbolic::symbol("i"));
    auto first_child = loop->root().at(0);
    auto& transition = first_child.second;
    auto strided_var = symbolic::symbol(unit_stride.strided_container_name());
    auto assigned_value = transition.assignments().at(strided_var);
    EXPECT_TRUE(symbolic::eq(assigned_value, strided_expr));
}

// Test 6: Cannot apply to unit stride loop
TEST(LoopUnitStrideTest, CannotApplyUnitStride) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);

    types::Scalar elem_desc(types::PrimitiveType::Double);
    types::Array arr_desc(elem_desc, symbolic::integer(100));
    builder.add_container("A", arr_desc, true);

    auto& root = builder.subject().root();

    // for i = 0; i < 10; i++ (stride = 1)
    auto& for_loop = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::integer(10)),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1)) // stride = 1
    );

    auto& block = builder.add_block(for_loop.root());
    auto& a_node = builder.add_access(block, "A");
    auto& const_node = builder.add_constant(block, "1.0", elem_desc);
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in1"});
    builder.add_computational_memlet(block, const_node, tasklet, "_in1", {});
    builder.add_computational_memlet(block, tasklet, "_out", a_node, {symbolic::symbol("i")});

    auto sdfg = builder.move();
    builder::StructuredSDFGBuilder builder2(sdfg);
    analysis::AnalysisManager am(builder2.subject());

    auto* loop = dynamic_cast<structured_control_flow::For*>(&builder2.subject().root().at(0).first);
    ASSERT_NE(loop, nullptr);

    // Cannot apply to unit stride
    transformations::LoopUnitStride unit_stride(*loop);
    EXPECT_FALSE(unit_stride.can_be_applied(builder2, am));
}

// Test 7: Cannot apply to negative unit stride
TEST(LoopUnitStrideTest, CannotApplyNegativeUnitStride) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);

    types::Scalar elem_desc(types::PrimitiveType::Double);
    types::Array arr_desc(elem_desc, symbolic::integer(100));
    builder.add_container("A", arr_desc, true);

    auto& root = builder.subject().root();

    // for i = 10; 0 < i; i-- (stride = -1)
    auto& for_loop = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::integer(0), symbolic::symbol("i")),
        symbolic::integer(10),
        symbolic::sub(symbolic::symbol("i"), symbolic::integer(1)) // stride = -1
    );

    auto& block = builder.add_block(for_loop.root());
    auto& a_node = builder.add_access(block, "A");
    auto& const_node = builder.add_constant(block, "1.0", elem_desc);
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in1"});
    builder.add_computational_memlet(block, const_node, tasklet, "_in1", {});
    builder.add_computational_memlet(block, tasklet, "_out", a_node, {symbolic::symbol("i")});

    auto sdfg = builder.move();
    builder::StructuredSDFGBuilder builder2(sdfg);
    analysis::AnalysisManager am(builder2.subject());

    auto* loop = dynamic_cast<structured_control_flow::For*>(&builder2.subject().root().at(0).first);
    ASSERT_NE(loop, nullptr);

    // Cannot apply to stride = -1 (use LoopRotate instead)
    transformations::LoopUnitStride unit_stride(*loop);
    EXPECT_FALSE(unit_stride.can_be_applied(builder2, am));
}

// Test 8: Map loop with non-unit stride
TEST(LoopUnitStrideTest, MapLoop) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);

    types::Scalar elem_desc(types::PrimitiveType::Double);
    types::Array arr_desc(elem_desc, symbolic::integer(100));
    builder.add_container("A", arr_desc, true);

    auto& root = builder.subject().root();

    // map i = 0; i < 20; i += 5
    auto& map_loop = builder.add_map(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::integer(20)),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(5)),
        structured_control_flow::ScheduleType_Sequential::create()
    );

    auto& block = builder.add_block(map_loop.root());
    auto& a_node = builder.add_access(block, "A");
    auto& const_node = builder.add_constant(block, "1.0", elem_desc);
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in1"});
    builder.add_computational_memlet(block, const_node, tasklet, "_in1", {});
    builder.add_computational_memlet(block, tasklet, "_out", a_node, {symbolic::symbol("i")});

    auto sdfg = builder.move();
    builder::StructuredSDFGBuilder builder2(sdfg);
    analysis::AnalysisManager am(builder2.subject());

    auto* loop = dynamic_cast<structured_control_flow::Map*>(&builder2.subject().root().at(0).first);
    ASSERT_NE(loop, nullptr);

    // Apply LoopUnitStride
    transformations::LoopUnitStride unit_stride(*loop);
    ASSERT_TRUE(unit_stride.can_be_applied(builder2, am));
    unit_stride.apply(builder2, am);

    // Verify unit stride
    auto new_stride = loop->stride();
    ASSERT_FALSE(new_stride.is_null());
    EXPECT_EQ(new_stride->as_int(), 1);

    // New condition: substituted condition 5*i < 20
    auto strided_expr = symbolic::mul(symbolic::integer(5), symbolic::symbol("i"));
    auto expected_cond = symbolic::Lt(strided_expr, symbolic::integer(20));
    EXPECT_TRUE(symbolic::eq(loop->condition(), expected_cond));
}

// Test 9: JSON serialization round-trip
TEST(LoopUnitStrideTest, JsonRoundTrip) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);

    types::Scalar elem_desc(types::PrimitiveType::Double);
    types::Array arr_desc(elem_desc, symbolic::integer(100));
    builder.add_container("A", arr_desc, true);

    auto& root = builder.subject().root();

    auto& for_loop = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::integer(10)),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(2))
    );

    auto& block = builder.add_block(for_loop.root());
    auto& a_node = builder.add_access(block, "A");
    auto& const_node = builder.add_constant(block, "1.0", elem_desc);
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in1"});
    builder.add_computational_memlet(block, const_node, tasklet, "_in1", {});
    builder.add_computational_memlet(block, tasklet, "_out", a_node, {symbolic::symbol("i")});

    auto sdfg = builder.move();
    builder::StructuredSDFGBuilder builder2(sdfg);
    analysis::AnalysisManager am(builder2.subject());

    auto* loop = dynamic_cast<structured_control_flow::For*>(&builder2.subject().root().at(0).first);
    ASSERT_NE(loop, nullptr);

    // Create transformation and serialize
    transformations::LoopUnitStride unit_stride(*loop);
    nlohmann::json j;
    unit_stride.to_json(j);

    // Deserialize
    auto restored = transformations::LoopUnitStride::from_json(builder2, j);
    EXPECT_EQ(restored.name(), "LoopUnitStride");

    // Verify the restored transformation can be applied
    EXPECT_TRUE(restored.can_be_applied(builder2, am));
}

// Test 10: Large stride value
TEST(LoopUnitStrideTest, LargeStride) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);

    types::Scalar elem_desc(types::PrimitiveType::Double);
    types::Array arr_desc(elem_desc, symbolic::integer(1024));
    builder.add_container("A", arr_desc, true);

    auto& root = builder.subject().root();

    // for i = 0; i < 1024; i += 64 (iterations: 0, 64, 128, ..., 960)
    auto& for_loop = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::integer(1024)),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(64))
    );

    auto& block = builder.add_block(for_loop.root());
    auto& a_node = builder.add_access(block, "A");
    auto& const_node = builder.add_constant(block, "1.0", elem_desc);
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in1"});
    builder.add_computational_memlet(block, const_node, tasklet, "_in1", {});
    builder.add_computational_memlet(block, tasklet, "_out", a_node, {symbolic::symbol("i")});

    auto sdfg = builder.move();
    builder::StructuredSDFGBuilder builder2(sdfg);
    analysis::AnalysisManager am(builder2.subject());

    auto* loop = dynamic_cast<structured_control_flow::For*>(&builder2.subject().root().at(0).first);
    ASSERT_NE(loop, nullptr);

    // Apply LoopUnitStride
    transformations::LoopUnitStride unit_stride(*loop);
    ASSERT_TRUE(unit_stride.can_be_applied(builder2, am));
    unit_stride.apply(builder2, am);

    // New condition: substituted condition 64*i < 1024
    auto strided_expr = symbolic::mul(symbolic::integer(64), symbolic::symbol("i"));
    auto expected_cond = symbolic::Lt(strided_expr, symbolic::integer(1024));
    EXPECT_TRUE(symbolic::eq(loop->condition(), expected_cond));

    // Assignment: __i_orig__ = 64 * i
    auto first_child = loop->root().at(0);
    auto& transition = first_child.second;
    auto strided_var = symbolic::symbol(unit_stride.strided_container_name());
    auto assigned_value = transition.assignments().at(strided_var);
    auto expected = symbolic::mul(symbolic::integer(64), symbolic::symbol("i"));
    EXPECT_TRUE(symbolic::eq(assigned_value, expected));
}

// Test: IndvarFinalValueAfterLoop - verify reconstruction after loop
// Original: for(i=0; i<10; i+=2) body(); // iterations 0,2,4,6,8, exits with i=10
// After unit stride: for(i=0; 2*i<10; i++) body(); // iterations 0,1,2,3,4, exits with i=5
// Reconstruction: i = 2*i  // restores i=10 after loop
TEST(LoopUnitStrideTest, IndvarFinalValueAfterLoop) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);

    types::Scalar elem_desc(types::PrimitiveType::Double);
    types::Array arr_desc(elem_desc, symbolic::integer(100));
    builder.add_container("A", arr_desc, true);

    auto& root = builder.subject().root();

    // for i = 0; i < 10; i += 2 (iterations: 0, 2, 4, 6, 8; exits with i=10)
    auto& for_loop = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::integer(10)),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(2))
    );

    auto& block = builder.add_block(for_loop.root());
    auto& a_node = builder.add_access(block, "A");
    auto& const_node = builder.add_constant(block, "1.0", elem_desc);
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in1"});
    builder.add_computational_memlet(block, const_node, tasklet, "_in1", {});
    builder.add_computational_memlet(block, tasklet, "_out", a_node, {symbolic::symbol("i")});

    auto sdfg = builder.move();
    builder::StructuredSDFGBuilder builder2(sdfg);
    analysis::AnalysisManager am(builder2.subject());

    auto& sdfg_root = builder2.subject().root();
    ASSERT_EQ(sdfg_root.size(), 1); // Just the loop

    auto* loop = dynamic_cast<structured_control_flow::For*>(&sdfg_root.at(0).first);
    ASSERT_NE(loop, nullptr);

    // Apply LoopUnitStride
    transformations::LoopUnitStride unit_stride(*loop);
    ASSERT_TRUE(unit_stride.can_be_applied(builder2, am));
    unit_stride.apply(builder2, am);

    // After transformation, root should have: loop + reconstruction block
    ASSERT_EQ(sdfg_root.size(), 2);

    // Verify the second element is a block (reconstruction)
    auto* post_loop_block = dynamic_cast<structured_control_flow::Block*>(&sdfg_root.at(1).first);
    ASSERT_NE(post_loop_block, nullptr);

    // Check the reconstruction block's transition contains the reconstruction assignment
    // (add_block_after puts assignments in the new block's transition)
    auto& post_loop_transition = sdfg_root.at(1).second;
    ASSERT_EQ(post_loop_transition.assignments().size(), 1);

    // Assignment should be: i = 2 * i
    auto indvar = symbolic::symbol("i");
    ASSERT_TRUE(post_loop_transition.assignments().count(indvar) > 0);
    auto reconstruction = post_loop_transition.assignments().at(indvar);
    auto expected = symbolic::mul(symbolic::integer(2), indvar);
    EXPECT_TRUE(symbolic::eq(reconstruction, expected));
}
