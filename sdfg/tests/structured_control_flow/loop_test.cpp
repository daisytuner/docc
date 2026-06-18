#include <gtest/gtest.h>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/structured_control_flow/for.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/structured_control_flow/while.h"
#include "sdfg/symbolic/symbolic.h"

using namespace sdfg;

namespace sdfg::structured_control_flow {

// Test For loop structure and pointers
TEST(ForLoopTest, BasicStructure) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);

    types::Scalar int_type(types::PrimitiveType::Int32);
    builder.add_container("i", int_type);

    auto& root = builder.subject().root();

    auto& for_loop = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::integer(10)),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1))
    );

    // Verify for_loop is a StructuredLoop
    EXPECT_TRUE(dynamic_cast<const StructuredLoop*>(&for_loop) != nullptr);

    // Verify for_loop is a For
    EXPECT_TRUE(dynamic_cast<const For*>(&for_loop) != nullptr);
}

// Test For loop parameters
TEST(ForLoopTest, LoopParameters) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);

    types::Scalar int_type(types::PrimitiveType::Int32);
    builder.add_container("i", int_type);

    auto& root = builder.subject().root();

    auto indvar = symbolic::symbol("i");
    auto init = symbolic::integer(0);
    auto update = symbolic::add(symbolic::symbol("i"), symbolic::integer(1));
    auto condition = symbolic::Lt(symbolic::symbol("i"), symbolic::integer(10));

    auto& for_loop = builder.add_for(root, indvar, condition, init, update);

    // Verify parameters
    EXPECT_TRUE(symbolic::eq(for_loop.indvar(), indvar));
    EXPECT_TRUE(symbolic::eq(for_loop.init(), init));
    EXPECT_TRUE(symbolic::eq(for_loop.update(), update));
    EXPECT_TRUE(symbolic::eq(for_loop.condition(), condition));
}

// Test For loop root sequence
TEST(ForLoopTest, RootSequence) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);

    types::Scalar int_type(types::PrimitiveType::Int32);
    builder.add_container("i", int_type);

    auto& root = builder.subject().root();

    auto& for_loop = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::integer(10)),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1))
    );

    // Verify loop has a root sequence
    auto& loop_root = for_loop.root();
    EXPECT_TRUE(dynamic_cast<const Sequence*>(&loop_root) != nullptr);

    // Add block to loop body
    builder.add_block(loop_root, control_flow::Assignments{});
    EXPECT_EQ(loop_root.size(), 1);
}

// Test While loop structure and pointers
TEST(WhileLoopTest, BasicStructure) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);

    types::Scalar int_type(types::PrimitiveType::Int32);
    builder.add_container("x", int_type);

    auto& root = builder.subject().root();

    auto& while_loop = builder.add_while(root);

    // Verify while_loop is a ControlFlowNode
    EXPECT_TRUE(dynamic_cast<const ControlFlowNode*>(&while_loop) != nullptr);

    // Verify while_loop is a While
    EXPECT_TRUE(dynamic_cast<const While*>(&while_loop) != nullptr);
}

// Test While loop root sequence
TEST(WhileLoopTest, RootSequence) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);

    auto& root = builder.subject().root();
    auto& while_loop = builder.add_while(root);

    // Verify loop has a root sequence
    auto& loop_root = while_loop.root();
    EXPECT_TRUE(dynamic_cast<const Sequence*>(&loop_root) != nullptr);

    // Add block to loop body
    builder.add_block(loop_root, control_flow::Assignments{});
    EXPECT_EQ(loop_root.size(), 1);

    // Const access
    const auto& const_while = while_loop;
    const auto& const_root = const_while.root();
    EXPECT_TRUE(dynamic_cast<const Sequence*>(&const_root) != nullptr);
}

// Test Map structure and pointers
TEST(MapTest, BasicStructure) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);

    types::Scalar int_type(types::PrimitiveType::Int32);
    builder.add_container("i", int_type);

    auto& root = builder.subject().root();

    auto& map = builder.add_map(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::integer(10)),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1)),
        ScheduleType_Sequential::create()
    );

    // Verify map is a StructuredLoop
    EXPECT_TRUE(dynamic_cast<const StructuredLoop*>(&map) != nullptr);

    // Verify map is a Map
    EXPECT_TRUE(dynamic_cast<const Map*>(&map) != nullptr);
}

// Test Map parameters
TEST(MapTest, LoopParameters) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);

    types::Scalar int_type(types::PrimitiveType::Int32);
    builder.add_container("i", int_type);

    auto& root = builder.subject().root();

    auto indvar = symbolic::symbol("i");
    auto init = symbolic::integer(0);
    auto update = symbolic::add(symbolic::symbol("i"), symbolic::integer(1));
    auto condition = symbolic::Lt(symbolic::symbol("i"), symbolic::integer(100));

    auto& map = builder.add_map(root, indvar, condition, init, update, ScheduleType_Sequential::create());

    // Verify parameters
    EXPECT_TRUE(symbolic::eq(map.indvar(), indvar));
    EXPECT_TRUE(symbolic::eq(map.init(), init));
    EXPECT_TRUE(symbolic::eq(map.update(), update));
    EXPECT_TRUE(symbolic::eq(map.condition(), condition));
}

// Test Map schedule type
TEST(MapTest, ScheduleType) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);

    types::Scalar int_type(types::PrimitiveType::Int32);
    builder.add_container("i", int_type);

    auto& root = builder.subject().root();

    // Test Sequential schedule
    auto schedule_seq = ScheduleType_Sequential::create();
    auto& map_seq = builder.add_map(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::integer(10)),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1)),
        schedule_seq
    );

    EXPECT_EQ(map_seq.schedule_type().value(), ScheduleType_Sequential::value());

    // Test CPU Parallel schedule
    auto schedule_par = ScheduleType_Sequential::create();
    auto& map_par = builder.add_map(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::integer(10)),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1)),
        schedule_par
    );

    EXPECT_EQ(map_par.schedule_type().value(), ScheduleType_Sequential::value());
}

// Test Break and Continue structures
TEST(BreakContinueTest, BasicStructure) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);

    types::Scalar int_type(types::PrimitiveType::Int32);
    builder.add_container("i", int_type);

    auto& root = builder.subject().root();

    auto& while_loop = builder.add_while(root);
    auto& loop_root = while_loop.root();

    // Add break
    auto& break_node = builder.add_break(loop_root);
    EXPECT_TRUE(dynamic_cast<const Break*>(&break_node) != nullptr);
    EXPECT_TRUE(dynamic_cast<const ControlFlowNode*>(&break_node) != nullptr);

    // Add continue
    auto& continue_node = builder.add_continue(loop_root);
    EXPECT_TRUE(dynamic_cast<const Continue*>(&continue_node) != nullptr);
    EXPECT_TRUE(dynamic_cast<const ControlFlowNode*>(&continue_node) != nullptr);
}

// Test nested loops
TEST(LoopTest, NestedLoops) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);

    types::Scalar int_type(types::PrimitiveType::Int32);
    builder.add_container("i", int_type);
    builder.add_container("j", int_type);

    auto& root = builder.subject().root();

    // Outer loop
    auto& outer_loop = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::integer(10)),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1))
    );

    auto& outer_root = outer_loop.root();

    // Inner loop
    auto& inner_loop = builder.add_for(
        outer_root,
        symbolic::symbol("j"),
        symbolic::Lt(symbolic::symbol("j"), symbolic::integer(10)),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("j"), symbolic::integer(1))
    );

    // Verify nesting
    EXPECT_EQ(root.size(), 1);
    EXPECT_EQ(outer_root.size(), 1);

    auto& inner_root = inner_loop.root();
    builder.add_block(inner_root, control_flow::Assignments{});
    EXPECT_EQ(inner_root.size(), 1);
}

// ============================================================================
// Tests for stride() and canonical_bound() methods
// ============================================================================

// Test stride extraction for positive unit stride
TEST(StructuredLoopTest, StridePositiveUnit) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);

    types::Scalar int_type(types::PrimitiveType::Int32);
    builder.add_container("i", int_type);

    auto& root = builder.subject().root();
    auto& loop = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::integer(10)),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1)) // i = i + 1
    );

    auto stride = loop.stride();
    ASSERT_FALSE(stride.is_null());
    EXPECT_EQ(stride->as_int(), 1);
}

// Test stride extraction for negative unit stride
TEST(StructuredLoopTest, StrideNegativeUnit) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);

    types::Scalar int_type(types::PrimitiveType::Int32);
    builder.add_container("i", int_type);

    auto& root = builder.subject().root();
    auto& loop = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::integer(0), symbolic::symbol("i")),
        symbolic::integer(10),
        symbolic::sub(symbolic::symbol("i"), symbolic::integer(1)) // i = i - 1
    );

    auto stride = loop.stride();
    ASSERT_FALSE(stride.is_null());
    EXPECT_EQ(stride->as_int(), -1);
}

// Test stride extraction for positive non-unit stride
TEST(StructuredLoopTest, StridePositiveNonUnit) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);

    types::Scalar int_type(types::PrimitiveType::Int32);
    builder.add_container("i", int_type);

    auto& root = builder.subject().root();
    auto& loop = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::integer(100)),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(32)) // i = i + 32
    );

    auto stride = loop.stride();
    ASSERT_FALSE(stride.is_null());
    EXPECT_EQ(stride->as_int(), 32);
}

// Test canonical bound for simple i < N
TEST(StructuredLoopTest, CanonicalBoundSimple) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);

    types::Scalar int_type(types::PrimitiveType::Int32);
    builder.add_container("i", int_type);
    builder.add_container("N", int_type, true);

    auto& root = builder.subject().root();
    auto& loop = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1))
    );

    auto bound = loop.canonical_bound();
    ASSERT_FALSE(bound.is_null());
    EXPECT_TRUE(symbolic::eq(bound, symbolic::symbol("N")));
}

// Test canonical bound for i < N - 1
TEST(StructuredLoopTest, CanonicalBoundWithOffset) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);

    types::Scalar int_type(types::PrimitiveType::Int32);
    builder.add_container("i", int_type);
    builder.add_container("N", int_type, true);

    auto& root = builder.subject().root();
    auto& loop = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::sub(symbolic::symbol("N"), symbolic::integer(1))),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1))
    );

    auto bound = loop.canonical_bound();
    ASSERT_FALSE(bound.is_null());
    // N - 1
    EXPECT_TRUE(symbolic::eq(bound, symbolic::sub(symbolic::symbol("N"), symbolic::integer(1))));
}

// Test canonical bound for i + offset < N (should isolate to i < N - offset)
TEST(StructuredLoopTest, CanonicalBoundIndvarWithOffset) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);

    types::Scalar int_type(types::PrimitiveType::Int32);
    builder.add_container("i", int_type);
    builder.add_container("N", int_type, true);

    auto& root = builder.subject().root();
    // Condition: i + 1 < N  =>  bound = N - 1
    auto& loop = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::add(symbolic::symbol("i"), symbolic::integer(1)), symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1))
    );

    auto bound = loop.canonical_bound();
    ASSERT_FALSE(bound.is_null());
    // N - 1
    EXPECT_TRUE(symbolic::eq(bound, symbolic::sub(symbolic::symbol("N"), symbolic::integer(1))));
}

// Test canonical bound for skewed loop: i - 32*t < N (should give N + 32*t)
TEST(StructuredLoopTest, CanonicalBoundSkewed) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);

    types::Scalar int_type(types::PrimitiveType::Int32);
    builder.add_container("i", int_type);
    builder.add_container("t", int_type);
    builder.add_container("N", int_type, true);

    auto& root = builder.subject().root();
    // Condition: i - 32*t < N  =>  bound = N + 32*t
    auto& loop = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::
            Lt(symbolic::sub(symbolic::symbol("i"), symbolic::mul(symbolic::integer(32), symbolic::symbol("t"))),
               symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1))
    );

    auto bound = loop.canonical_bound();
    ASSERT_FALSE(bound.is_null());
    // N + 32*t
    auto expected = symbolic::add(symbolic::symbol("N"), symbolic::mul(symbolic::integer(32), symbolic::symbol("t")));
    EXPECT_TRUE(symbolic::eq(bound, expected));
}

// Test canonical bound for i <= N (should give N + 1)
TEST(StructuredLoopTest, CanonicalBoundLessEqual) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);

    types::Scalar int_type(types::PrimitiveType::Int32);
    builder.add_container("i", int_type);
    builder.add_container("N", int_type, true);

    auto& root = builder.subject().root();
    auto& loop = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Le(symbolic::symbol("i"), symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1))
    );

    auto bound = loop.canonical_bound();
    ASSERT_FALSE(bound.is_null());
    // N + 1
    EXPECT_TRUE(symbolic::eq(bound, symbolic::add(symbolic::symbol("N"), symbolic::integer(1))));
}

// Test canonical bound for negative stride: bound < i (lower bound)
TEST(StructuredLoopTest, CanonicalBoundNegativeStride) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);

    types::Scalar int_type(types::PrimitiveType::Int32);
    builder.add_container("i", int_type);

    auto& root = builder.subject().root();
    // Loop from 10 down to > 0: i > 0  =>  bound = 0
    auto& loop = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::integer(0), symbolic::symbol("i")), // 0 < i
        symbolic::integer(10),
        symbolic::sub(symbolic::symbol("i"), symbolic::integer(1)) // i = i - 1
    );

    auto bound = loop.canonical_bound();
    ASSERT_FALSE(bound.is_null());
    // For negative stride, this should extract lower bound = 0
    EXPECT_TRUE(symbolic::eq(bound, symbolic::integer(0)));
}

// Test canonical bound for integer constant
TEST(StructuredLoopTest, CanonicalBoundIntegerConstant) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);

    types::Scalar int_type(types::PrimitiveType::Int32);
    builder.add_container("i", int_type);

    auto& root = builder.subject().root();
    auto& loop = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::integer(100)),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1))
    );

    auto bound = loop.canonical_bound();
    ASSERT_FALSE(bound.is_null());
    EXPECT_TRUE(symbolic::eq(bound, symbolic::integer(100)));
}

// Test canonical bound for Map (should work the same as For)
TEST(StructuredLoopTest, CanonicalBoundMap) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);

    types::Scalar int_type(types::PrimitiveType::Int32);
    builder.add_container("i", int_type);
    builder.add_container("N", int_type, true);

    auto& root = builder.subject().root();
    auto& map = builder.add_map(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1)),
        ScheduleType_Sequential::create()
    );

    auto bound = map.canonical_bound();
    ASSERT_FALSE(bound.is_null());
    EXPECT_TRUE(symbolic::eq(bound, symbolic::symbol("N")));
}

// Test canonical bound with complex offset: 1 + i - 32*t < NX (the FDTD2D case)
TEST(StructuredLoopTest, CanonicalBoundComplexOffset) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);

    types::Scalar int_type(types::PrimitiveType::Int32);
    builder.add_container("i", int_type);
    builder.add_container("t", int_type);
    builder.add_container("NX", int_type, true);

    auto& root = builder.subject().root();
    // Condition: 1 + i - 32*t < NX  =>  bound = NX - 1 + 32*t
    auto lhs = symbolic::
        add(symbolic::integer(1),
            symbolic::sub(symbolic::symbol("i"), symbolic::mul(symbolic::integer(32), symbolic::symbol("t"))));
    auto& loop = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(lhs, symbolic::symbol("NX")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1))
    );

    auto bound = loop.canonical_bound();
    ASSERT_FALSE(bound.is_null());
    // NX - 1 + 32*t
    auto expected = symbolic::
        add(symbolic::sub(symbolic::symbol("NX"), symbolic::integer(1)),
            symbolic::mul(symbolic::integer(32), symbolic::symbol("t")));
    EXPECT_TRUE(symbolic::eq(bound, expected));
}

// ============================================================================
// Additional rejection / negative tests for stride()
// ============================================================================

// Stride with negative non-unit step (i = i - 4)
TEST(StructuredLoopTest, StrideNegativeNonUnit) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    types::Scalar int_type(types::PrimitiveType::Int32);
    builder.add_container("i", int_type);

    auto& root = builder.subject().root();
    auto& loop = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::integer(0), symbolic::symbol("i")),
        symbolic::integer(100),
        symbolic::sub(symbolic::symbol("i"), symbolic::integer(4))
    );

    auto stride = loop.stride();
    ASSERT_FALSE(stride.is_null());
    EXPECT_EQ(stride->as_int(), -4);
}

// Stride rejected for symbolic update (i = i + N)
TEST(StructuredLoopTest, StrideSymbolicUpdateRejected) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    types::Scalar int_type(types::PrimitiveType::Int32);
    builder.add_container("i", int_type);
    builder.add_container("N", int_type, true);

    auto& root = builder.subject().root();
    auto& loop = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::integer(100)),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::symbol("N"))
    );

    EXPECT_TRUE(loop.stride().is_null());
}

// Stride rejected for non-affine update (i = i * 2)
TEST(StructuredLoopTest, StrideNonAffineUpdateRejected) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    types::Scalar int_type(types::PrimitiveType::Int32);
    builder.add_container("i", int_type);

    auto& root = builder.subject().root();
    auto& loop = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::integer(100)),
        symbolic::integer(1),
        symbolic::mul(symbolic::symbol("i"), symbolic::integer(2))
    );

    EXPECT_TRUE(loop.stride().is_null());
}

// Stride rejected when indvar coefficient is not 1 (i = 2*i + 1)
TEST(StructuredLoopTest, StrideCoeffNotOneRejected) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    types::Scalar int_type(types::PrimitiveType::Int32);
    builder.add_container("i", int_type);

    auto& root = builder.subject().root();
    auto& loop = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::integer(100)),
        symbolic::integer(1),
        symbolic::add(symbolic::mul(symbolic::integer(2), symbolic::symbol("i")), symbolic::integer(1))
    );

    EXPECT_TRUE(loop.stride().is_null());
}

// ============================================================================
// Tests for is_contiguous() and is_monotonic()
// ============================================================================

TEST(StructuredLoopTest, IsContiguousTrueUnitStride) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    types::Scalar int_type(types::PrimitiveType::Int32);
    builder.add_container("i", int_type);

    auto& root = builder.subject().root();
    auto& loop = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::integer(10)),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1))
    );

    EXPECT_TRUE(loop.is_contiguous());
}

TEST(StructuredLoopTest, IsContiguousFalseNonUnitStride) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    types::Scalar int_type(types::PrimitiveType::Int32);
    builder.add_container("i", int_type);

    auto& root = builder.subject().root();
    auto& loop = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::integer(10)),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(2))
    );

    EXPECT_FALSE(loop.is_contiguous());
}

TEST(StructuredLoopTest, IsContiguousFalseNegativeStride) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    types::Scalar int_type(types::PrimitiveType::Int32);
    builder.add_container("i", int_type);

    auto& root = builder.subject().root();
    auto& loop = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::integer(0), symbolic::symbol("i")),
        symbolic::integer(10),
        symbolic::sub(symbolic::symbol("i"), symbolic::integer(1))
    );

    EXPECT_FALSE(loop.is_contiguous());
}

TEST(StructuredLoopTest, IsContiguousFalseNullStride) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    types::Scalar int_type(types::PrimitiveType::Int32);
    builder.add_container("i", int_type);

    auto& root = builder.subject().root();
    auto& loop = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::integer(10)),
        symbolic::integer(1),
        symbolic::mul(symbolic::symbol("i"), symbolic::integer(2))
    );

    EXPECT_FALSE(loop.is_contiguous());
}

TEST(StructuredLoopTest, IsMonotonicTrueUnitStride) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    types::Scalar int_type(types::PrimitiveType::Int32);
    builder.add_container("i", int_type);

    auto& root = builder.subject().root();
    auto& loop = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::integer(10)),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1))
    );

    EXPECT_TRUE(loop.is_monotonic());
}

TEST(StructuredLoopTest, IsMonotonicTrueNonUnitPositive) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    types::Scalar int_type(types::PrimitiveType::Int32);
    builder.add_container("i", int_type);

    auto& root = builder.subject().root();
    auto& loop = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::integer(100)),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(32))
    );

    EXPECT_TRUE(loop.is_monotonic());
}

TEST(StructuredLoopTest, IsMonotonicFalseNegativeStride) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    types::Scalar int_type(types::PrimitiveType::Int32);
    builder.add_container("i", int_type);

    auto& root = builder.subject().root();
    auto& loop = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::integer(0), symbolic::symbol("i")),
        symbolic::integer(10),
        symbolic::sub(symbolic::symbol("i"), symbolic::integer(1))
    );

    EXPECT_FALSE(loop.is_monotonic());
}

TEST(StructuredLoopTest, IsMonotonicFalseNullStride) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    types::Scalar int_type(types::PrimitiveType::Int32);
    builder.add_container("i", int_type);

    auto& root = builder.subject().root();
    auto& loop = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::integer(10)),
        symbolic::integer(1),
        symbolic::mul(symbolic::symbol("i"), symbolic::integer(2))
    );

    EXPECT_FALSE(loop.is_monotonic());
}

// ============================================================================
// Tests for canonical_bound_upper() / canonical_bound_lower() directly
// ============================================================================

// Upper bound for i <= N gives N + 1
TEST(StructuredLoopTest, CanonicalBoundUpperLessEqualDirect) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    types::Scalar int_type(types::PrimitiveType::Int32);
    builder.add_container("i", int_type);
    builder.add_container("N", int_type, true);

    auto& root = builder.subject().root();
    auto& loop = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Le(symbolic::symbol("i"), symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1))
    );

    auto bound = loop.canonical_bound_upper();
    ASSERT_FALSE(bound.is_null());
    EXPECT_TRUE(symbolic::eq(bound, symbolic::add(symbolic::symbol("N"), symbolic::integer(1))));
}

// Upper bound for conjunction i < N && i < M gives min(N, M)
TEST(StructuredLoopTest, CanonicalBoundUpperConjunction) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    types::Scalar int_type(types::PrimitiveType::Int32);
    builder.add_container("i", int_type);
    builder.add_container("N", int_type, true);
    builder.add_container("M", int_type, true);

    auto& root = builder.subject().root();
    auto& loop = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::
            And(symbolic::Lt(symbolic::symbol("i"), symbolic::symbol("N")),
                symbolic::Lt(symbolic::symbol("i"), symbolic::symbol("M"))),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1))
    );

    auto bound = loop.canonical_bound_upper();
    ASSERT_FALSE(bound.is_null());
    auto expected = symbolic::min(symbolic::symbol("N"), symbolic::symbol("M"));
    EXPECT_TRUE(symbolic::eq(bound, expected));
}

// Upper bound rejected when only a lower-bound clause is present (0 < i)
TEST(StructuredLoopTest, CanonicalBoundUpperRejectsLowerBoundOnly) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    types::Scalar int_type(types::PrimitiveType::Int32);
    builder.add_container("i", int_type);

    auto& root = builder.subject().root();
    auto& loop = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::integer(0), symbolic::symbol("i")),
        symbolic::integer(10),
        symbolic::sub(symbolic::symbol("i"), symbolic::integer(1))
    );

    EXPECT_TRUE(loop.canonical_bound_upper().is_null());
}

// Lower bound for 1 <= i gives 0
TEST(StructuredLoopTest, CanonicalBoundLowerLessEqualDirect) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    types::Scalar int_type(types::PrimitiveType::Int32);
    builder.add_container("i", int_type);

    auto& root = builder.subject().root();
    auto& loop = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Le(symbolic::integer(1), symbolic::symbol("i")),
        symbolic::integer(10),
        symbolic::sub(symbolic::symbol("i"), symbolic::integer(1))
    );

    auto bound = loop.canonical_bound_lower();
    ASSERT_FALSE(bound.is_null());
    // 1 <= i  =>  i > 0  =>  bound = 0
    EXPECT_TRUE(symbolic::eq(bound, symbolic::integer(0)));
}

// Lower bound rejected when only an upper-bound clause is present (i < N)
TEST(StructuredLoopTest, CanonicalBoundLowerRejectsUpperBoundOnly) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    types::Scalar int_type(types::PrimitiveType::Int32);
    builder.add_container("i", int_type);
    builder.add_container("N", int_type, true);

    auto& root = builder.subject().root();
    auto& loop = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1))
    );

    EXPECT_TRUE(loop.canonical_bound_lower().is_null());
}

// canonical_bound() now allows non-unit positive stride and dispatches to upper
TEST(StructuredLoopTest, CanonicalBoundDispatchPositiveNonUnit) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    types::Scalar int_type(types::PrimitiveType::Int32);
    builder.add_container("i", int_type);

    auto& root = builder.subject().root();
    auto& loop = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::integer(100)),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(4))
    );

    auto bound = loop.canonical_bound();
    ASSERT_FALSE(bound.is_null());
    EXPECT_TRUE(symbolic::eq(bound, symbolic::integer(100)));
}

// canonical_bound() with non-unit negative stride dispatches to lower
TEST(StructuredLoopTest, CanonicalBoundDispatchNegativeNonUnit) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    types::Scalar int_type(types::PrimitiveType::Int32);
    builder.add_container("i", int_type);

    auto& root = builder.subject().root();
    auto& loop = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::integer(0), symbolic::symbol("i")),
        symbolic::integer(100),
        symbolic::sub(symbolic::symbol("i"), symbolic::integer(4))
    );

    auto bound = loop.canonical_bound();
    ASSERT_FALSE(bound.is_null());
    EXPECT_TRUE(symbolic::eq(bound, symbolic::integer(0)));
}

// canonical_bound() returns null when stride is undefined
TEST(StructuredLoopTest, CanonicalBoundNullStride) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    types::Scalar int_type(types::PrimitiveType::Int32);
    builder.add_container("i", int_type);

    auto& root = builder.subject().root();
    auto& loop = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::integer(10)),
        symbolic::integer(1),
        symbolic::mul(symbolic::symbol("i"), symbolic::integer(2))
    );

    EXPECT_TRUE(loop.canonical_bound().is_null());
}

// ============================================================================
// Tests for num_iterations()
// ============================================================================

// Simple positive unit stride: for(i=0; i<10; ++i) -> 10
TEST(StructuredLoopTest, NumIterationsSimplePositive) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    types::Scalar int_type(types::PrimitiveType::Int32);
    builder.add_container("i", int_type);

    auto& root = builder.subject().root();
    auto& loop = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::integer(10)),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1))
    );

    auto n = loop.num_iterations();
    ASSERT_FALSE(n.is_null());
    EXPECT_TRUE(symbolic::eq(n, symbolic::integer(10)));
}

// Symbolic bound returns symbolic count: for(i=0; i<N; ++i) -> N (clamped non-negative)
TEST(StructuredLoopTest, NumIterationsSymbolicBound) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    types::Scalar int_type(types::PrimitiveType::Int32);
    builder.add_container("i", int_type);
    builder.add_container("N", int_type, true);

    auto& root = builder.subject().root();
    auto& loop = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1))
    );

    auto n = loop.num_iterations();
    ASSERT_FALSE(n.is_null());
    // max(0, N)
    auto expected = symbolic::simplify(symbolic::max(symbolic::zero(), symbolic::symbol("N")));
    EXPECT_TRUE(symbolic::eq(n, expected));
}

// Negative unit stride: for(i=10; 0<i; --i) -> 10 (was a latent bug pre-fix)
TEST(StructuredLoopTest, NumIterationsNegativeUnitStride) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    types::Scalar int_type(types::PrimitiveType::Int32);
    builder.add_container("i", int_type);

    auto& root = builder.subject().root();
    auto& loop = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::integer(0), symbolic::symbol("i")),
        symbolic::integer(10),
        symbolic::sub(symbolic::symbol("i"), symbolic::integer(1))
    );

    auto n = loop.num_iterations();
    ASSERT_FALSE(n.is_null());
    EXPECT_TRUE(symbolic::eq(n, symbolic::integer(10)));
}

// Stride 2, divisible: for(i=0; i<10; i+=2) -> 5
TEST(StructuredLoopTest, NumIterationsStrideTwoExact) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    types::Scalar int_type(types::PrimitiveType::Int32);
    builder.add_container("i", int_type);

    auto& root = builder.subject().root();
    auto& loop = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::integer(10)),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(2))
    );

    auto n = loop.num_iterations();
    ASSERT_FALSE(n.is_null());
    EXPECT_TRUE(symbolic::eq(n, symbolic::integer(5)));
}

// Stride 2, non-divisible: for(i=0; i<9; i+=2) -> ceil(9/2) = 5 (iters: 0,2,4,6,8)
TEST(StructuredLoopTest, NumIterationsStrideTwoCeiling) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    types::Scalar int_type(types::PrimitiveType::Int32);
    builder.add_container("i", int_type);

    auto& root = builder.subject().root();
    auto& loop = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::integer(9)),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(2))
    );

    auto n = loop.num_iterations();
    ASSERT_FALSE(n.is_null());
    EXPECT_TRUE(symbolic::eq(n, symbolic::integer(5)));
}

// Stride 3, non-divisible: for(i=0; i<10; i+=3) -> ceil(10/3) = 4 (iters: 0,3,6,9)
TEST(StructuredLoopTest, NumIterationsStrideThreeCeiling) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    types::Scalar int_type(types::PrimitiveType::Int32);
    builder.add_container("i", int_type);

    auto& root = builder.subject().root();
    auto& loop = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::integer(10)),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(3))
    );

    auto n = loop.num_iterations();
    ASSERT_FALSE(n.is_null());
    EXPECT_TRUE(symbolic::eq(n, symbolic::integer(4)));
}

// Negative stride -2: for(i=10; 0<i; i-=2) -> iters: 10,8,6,4,2 -> 5
TEST(StructuredLoopTest, NumIterationsStrideMinusTwo) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    types::Scalar int_type(types::PrimitiveType::Int32);
    builder.add_container("i", int_type);

    auto& root = builder.subject().root();
    auto& loop = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::integer(0), symbolic::symbol("i")),
        symbolic::integer(10),
        symbolic::sub(symbolic::symbol("i"), symbolic::integer(2))
    );

    auto n = loop.num_iterations();
    ASSERT_FALSE(n.is_null());
    EXPECT_TRUE(symbolic::eq(n, symbolic::integer(5)));
}

// Empty loop: for(i=10; i<5; ++i) -> 0 (clamped by max(0, ...))
TEST(StructuredLoopTest, NumIterationsNegativeDeltaClampsToZero) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    types::Scalar int_type(types::PrimitiveType::Int32);
    builder.add_container("i", int_type);

    auto& root = builder.subject().root();
    auto& loop = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::integer(5)),
        symbolic::integer(10),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1))
    );

    auto n = loop.num_iterations();
    ASSERT_FALSE(n.is_null());
    EXPECT_TRUE(symbolic::eq(n, symbolic::integer(0)));
}

// Rejected: stride cannot be determined
TEST(StructuredLoopTest, NumIterationsNullStrideRejected) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    types::Scalar int_type(types::PrimitiveType::Int32);
    builder.add_container("i", int_type);

    auto& root = builder.subject().root();
    auto& loop = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::integer(10)),
        symbolic::integer(1),
        symbolic::mul(symbolic::symbol("i"), symbolic::integer(2))
    );

    EXPECT_TRUE(loop.num_iterations().is_null());
}

// Rejected: condition has no extractable bound for the indvar
TEST(StructuredLoopTest, NumIterationsNoBoundRejected) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    types::Scalar int_type(types::PrimitiveType::Int32);
    builder.add_container("i", int_type);
    builder.add_container("N", int_type, true);

    auto& root = builder.subject().root();
    // Condition does not mention indvar at all -> no clauses extract a bound.
    auto& loop = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::integer(0), symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1))
    );

    EXPECT_TRUE(loop.num_iterations().is_null());
}

// ============================================================================
// Tests for num_iterations_approx()
// ============================================================================

// Same as exact when no min/max is present
TEST(StructuredLoopTest, NumIterationsApproxMatchesExactSimple) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    types::Scalar int_type(types::PrimitiveType::Int32);
    builder.add_container("i", int_type);

    auto& root = builder.subject().root();
    auto& loop = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::integer(10)),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1))
    );

    auto n = loop.num_iterations_approx();
    ASSERT_FALSE(n.is_null());
    EXPECT_TRUE(symbolic::eq(n, symbolic::integer(10)));
}

// Tile-style pattern: for(k = k_tile0; k < N && k < k_tile0 + 8; ++k)
// canonical_bound() = min(N, k_tile0 + 8); approx collapses to constant 8.
TEST(StructuredLoopTest, NumIterationsApproxTilePattern) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    types::Scalar int_type(types::PrimitiveType::Int32);
    builder.add_container("k", int_type);
    builder.add_container("k_tile0", int_type, true);
    builder.add_container("N", int_type, true);

    auto k = symbolic::symbol("k");
    auto k_tile0 = symbolic::symbol("k_tile0");
    auto N = symbolic::symbol("N");

    auto& root = builder.subject().root();
    auto& loop = builder.add_for(
        root,
        k,
        symbolic::And(symbolic::Lt(k, N), symbolic::Lt(k, symbolic::add(k_tile0, symbolic::integer(8)))),
        k_tile0,
        symbolic::add(k, symbolic::integer(1))
    );

    auto approx = loop.num_iterations_approx();
    ASSERT_FALSE(approx.is_null());
    EXPECT_TRUE(symbolic::eq(approx, symbolic::integer(8)));
}

// Tile pattern with non-unit stride: stride 2 should give ceil(8/2) = 4.
TEST(StructuredLoopTest, NumIterationsApproxTilePatternNonUnitStride) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    types::Scalar int_type(types::PrimitiveType::Int32);
    builder.add_container("k", int_type);
    builder.add_container("k_tile0", int_type, true);
    builder.add_container("N", int_type, true);

    auto k = symbolic::symbol("k");
    auto k_tile0 = symbolic::symbol("k_tile0");
    auto N = symbolic::symbol("N");

    auto& root = builder.subject().root();
    auto& loop = builder.add_for(
        root,
        k,
        symbolic::And(symbolic::Lt(k, N), symbolic::Lt(k, symbolic::add(k_tile0, symbolic::integer(8)))),
        k_tile0,
        symbolic::add(k, symbolic::integer(2))
    );

    auto approx = loop.num_iterations_approx();
    ASSERT_FALSE(approx.is_null());
    EXPECT_TRUE(symbolic::eq(approx, symbolic::integer(4)));
}

// num_iterations_approx is always >= num_iterations (sanity for trivial case)
TEST(StructuredLoopTest, NumIterationsApproxNullStrideRejected) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    types::Scalar int_type(types::PrimitiveType::Int32);
    builder.add_container("i", int_type);

    auto& root = builder.subject().root();
    auto& loop = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::integer(10)),
        symbolic::integer(1),
        symbolic::mul(symbolic::symbol("i"), symbolic::integer(2))
    );

    EXPECT_TRUE(loop.num_iterations_approx().is_null());
}

// ============================================================================
// Tests for is_loop_normal_form()
// ============================================================================

TEST(StructuredLoopTest, IsLoopNormalFormTrue) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    types::Scalar int_type(types::PrimitiveType::Int32);
    builder.add_container("i", int_type);
    builder.add_container("N", int_type, true);

    auto& root = builder.subject().root();
    auto& loop = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1))
    );

    EXPECT_TRUE(loop.is_loop_normal_form());
}

TEST(StructuredLoopTest, IsLoopNormalFormFalseNonZeroInit) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    types::Scalar int_type(types::PrimitiveType::Int32);
    builder.add_container("i", int_type);

    auto& root = builder.subject().root();
    auto& loop = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::integer(10)),
        symbolic::integer(5),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1))
    );

    EXPECT_FALSE(loop.is_loop_normal_form());
}

TEST(StructuredLoopTest, IsLoopNormalFormFalseNonUnitStride) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    types::Scalar int_type(types::PrimitiveType::Int32);
    builder.add_container("i", int_type);

    auto& root = builder.subject().root();
    auto& loop = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::integer(10)),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(2))
    );

    EXPECT_FALSE(loop.is_loop_normal_form());
}

TEST(StructuredLoopTest, IsLoopNormalFormFalseNegativeStride) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    types::Scalar int_type(types::PrimitiveType::Int32);
    builder.add_container("i", int_type);

    auto& root = builder.subject().root();
    auto& loop = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::integer(0), symbolic::symbol("i")),
        symbolic::integer(10),
        symbolic::sub(symbolic::symbol("i"), symbolic::integer(1))
    );

    EXPECT_FALSE(loop.is_loop_normal_form());
}

TEST(StructuredLoopTest, IsLoopNormalFormFalseNoCanonicalBound) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    types::Scalar int_type(types::PrimitiveType::Int32);
    builder.add_container("i", int_type);
    builder.add_container("N", int_type, true);

    auto& root = builder.subject().root();
    // Condition does not constrain the indvar -> no canonical bound.
    auto& loop = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::integer(0), symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1))
    );

    EXPECT_FALSE(loop.is_loop_normal_form());
}

} // namespace sdfg::structured_control_flow
