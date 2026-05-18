#include <gtest/gtest.h>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/if_else.h"
#include "sdfg/symbolic/symbolic.h"

using namespace sdfg;

namespace sdfg::structured_control_flow {

// Test basic IfElse structure and pointers
TEST(IfElseTest, BasicStructure) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);

    types::Scalar int_type(types::PrimitiveType::Int32);
    builder.add_container("x", int_type);

    auto& root = builder.subject().root();

    // Create an if-else with two branches
    auto& if_else = builder.add_if_else(root);
    auto& if_branch = builder.add_case(if_else, symbolic::Gt(symbolic::symbol("x"), symbolic::integer(0)));
    auto& else_branch = builder.add_case(if_else, symbolic::Le(symbolic::symbol("x"), symbolic::integer(0)));

    // Verify structure
    EXPECT_EQ(if_else.size(), 2);

    // Verify pointers - cases should be sequences
    auto [seq1, cond1] = if_else.at(0);
    auto [seq2, cond2] = if_else.at(1);

    EXPECT_TRUE(dynamic_cast<const Sequence*>(&seq1) != nullptr);
    EXPECT_TRUE(dynamic_cast<const Sequence*>(&seq2) != nullptr);

    // Verify conditions
    EXPECT_TRUE(SymEngine::is_a_Boolean(*cond1));
    EXPECT_TRUE(SymEngine::is_a_Boolean(*cond2));
}

// Test is_complete() with positive cases - complete if-else
TEST(IfElseTest, IsCompleteTrue_SimpleIfElse) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);

    types::Scalar int_type(types::PrimitiveType::Int32);
    builder.add_container("x", int_type);

    auto& root = builder.subject().root();
    auto& if_else = builder.add_if_else(root);

    // x > 0
    builder.add_case(if_else, symbolic::Gt(symbolic::symbol("x"), symbolic::integer(0)));
    // x <= 0 (covers all other cases)
    builder.add_case(if_else, symbolic::Le(symbolic::symbol("x"), symbolic::integer(0)));

    EXPECT_TRUE(if_else.is_complete());
}

// Test is_complete() with positive cases - complete with three branches
TEST(IfElseTest, IsCompleteTrue_ThreeBranches) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);

    types::Scalar int_type(types::PrimitiveType::Int32);
    builder.add_container("x", int_type);

    auto& root = builder.subject().root();
    auto& if_else = builder.add_if_else(root);

    // x > 0
    builder.add_case(if_else, symbolic::Gt(symbolic::symbol("x"), symbolic::integer(0)));
    // x == 0
    builder.add_case(if_else, symbolic::Eq(symbolic::symbol("x"), symbolic::integer(0)));
    // x < 0
    builder.add_case(if_else, symbolic::Lt(symbolic::symbol("x"), symbolic::integer(0)));

    EXPECT_TRUE(if_else.is_complete());
}

// Test is_complete() with positive cases - complete with OR condition
TEST(IfElseTest, IsCompleteTrue_OrCondition) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);

    types::Scalar int_type(types::PrimitiveType::Int32);
    builder.add_container("x", int_type);
    builder.add_container("y", int_type);

    auto& root = builder.subject().root();
    auto& if_else = builder.add_if_else(root);

    // x > 0 OR y > 0
    builder.add_case(
        if_else,
        symbolic::
            Or(symbolic::Gt(symbolic::symbol("x"), symbolic::integer(0)),
               symbolic::Gt(symbolic::symbol("y"), symbolic::integer(0)))
    );
    // x <= 0 AND y <= 0
    builder.add_case(
        if_else,
        symbolic::
            And(symbolic::Le(symbolic::symbol("x"), symbolic::integer(0)),
                symbolic::Le(symbolic::symbol("y"), symbolic::integer(0)))
    );

    EXPECT_TRUE(if_else.is_complete());
}

// Test is_complete() with positive cases - always true condition
TEST(IfElseTest, IsCompleteTrue_AlwaysTrue) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);

    auto& root = builder.subject().root();
    auto& if_else = builder.add_if_else(root);

    // Always true condition
    builder.add_case(if_else, symbolic::__true__());

    EXPECT_TRUE(if_else.is_complete());
}

// Test is_complete() with negative cases - incomplete if without else
TEST(IfElseTest, IsCompleteFalse_OnlyIfBranch) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);

    types::Scalar int_type(types::PrimitiveType::Int32);
    builder.add_container("x", int_type);

    auto& root = builder.subject().root();
    auto& if_else = builder.add_if_else(root);

    // Only x > 0, doesn't cover x <= 0
    builder.add_case(if_else, symbolic::Gt(symbolic::symbol("x"), symbolic::integer(0)));

    EXPECT_FALSE(if_else.is_complete());
}

// Test is_complete() with negative cases - non-exhaustive conditions
TEST(IfElseTest, IsCompleteFalse_NonExhaustive) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);

    types::Scalar int_type(types::PrimitiveType::Int32);
    builder.add_container("x", int_type);

    auto& root = builder.subject().root();
    auto& if_else = builder.add_if_else(root);

    // x > 10
    builder.add_case(if_else, symbolic::Gt(symbolic::symbol("x"), symbolic::integer(10)));
    // x < 5 (doesn't cover 5 <= x <= 10)
    builder.add_case(if_else, symbolic::Lt(symbolic::symbol("x"), symbolic::integer(5)));

    EXPECT_FALSE(if_else.is_complete());
}

// Test is_complete() with negative cases - incomplete with gaps
TEST(IfElseTest, IsCompleteFalse_WithGaps) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);

    types::Scalar int_type(types::PrimitiveType::Int32);
    builder.add_container("x", int_type);
    builder.add_container("y", int_type);

    auto& root = builder.subject().root();
    auto& if_else = builder.add_if_else(root);

    // x > 0 AND y > 0
    builder.add_case(
        if_else,
        symbolic::
            And(symbolic::Gt(symbolic::symbol("x"), symbolic::integer(0)),
                symbolic::Gt(symbolic::symbol("y"), symbolic::integer(0)))
    );
    // x < 0 AND y < 0 (doesn't cover cases where x and y have different signs or are zero)
    builder.add_case(
        if_else,
        symbolic::
            And(symbolic::Lt(symbolic::symbol("x"), symbolic::integer(0)),
                symbolic::Lt(symbolic::symbol("y"), symbolic::integer(0)))
    );

    EXPECT_FALSE(if_else.is_complete());
}

// Test is_complete() with negative cases - always false condition
TEST(IfElseTest, IsCompleteFalse_AlwaysFalse) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);

    auto& root = builder.subject().root();
    auto& if_else = builder.add_if_else(root);

    // Always false condition
    builder.add_case(if_else, symbolic::__false__());

    EXPECT_FALSE(if_else.is_complete());
}

// Test size() method
TEST(IfElseTest, SizeMethod) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);

    types::Scalar int_type(types::PrimitiveType::Int32);
    builder.add_container("x", int_type);

    auto& root = builder.subject().root();
    auto& if_else = builder.add_if_else(root);

    EXPECT_EQ(if_else.size(), 0);

    builder.add_case(if_else, symbolic::Gt(symbolic::symbol("x"), symbolic::integer(0)));
    EXPECT_EQ(if_else.size(), 1);

    builder.add_case(if_else, symbolic::Le(symbolic::symbol("x"), symbolic::integer(0)));
    EXPECT_EQ(if_else.size(), 2);
}

// Test at() method for accessing cases
TEST(IfElseTest, AtMethod) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);

    types::Scalar int_type(types::PrimitiveType::Int32);
    builder.add_container("x", int_type);

    auto& root = builder.subject().root();
    auto& if_else = builder.add_if_else(root);

    auto cond1 = symbolic::Gt(symbolic::symbol("x"), symbolic::integer(0));
    auto cond2 = symbolic::Le(symbolic::symbol("x"), symbolic::integer(0));

    builder.add_case(if_else, cond1);
    builder.add_case(if_else, cond2);

    // Test const version
    const auto& const_if_else = if_else;
    auto [seq1, ret_cond1] = const_if_else.at(0);
    auto [seq2, ret_cond2] = const_if_else.at(1);

    EXPECT_TRUE(symbolic::eq(ret_cond1, cond1));
    EXPECT_TRUE(symbolic::eq(ret_cond2, cond2));

    // Test non-const version
    auto [mut_seq1, mut_cond1] = if_else.at(0);
    auto [mut_seq2, mut_cond2] = if_else.at(1);

    EXPECT_TRUE(symbolic::eq(mut_cond1, cond1));
    EXPECT_TRUE(symbolic::eq(mut_cond2, cond2));
}

// Test conditions and sequences together
TEST(IfElseTest, ConditionsAndSequences) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);

    types::Scalar int_type(types::PrimitiveType::Int32);
    builder.add_container("x", int_type);

    auto& root = builder.subject().root();
    auto& if_else = builder.add_if_else(root);

    auto& if_seq = builder.add_case(if_else, symbolic::Gt(symbolic::symbol("x"), symbolic::integer(0)));
    auto& else_seq = builder.add_case(if_else, symbolic::Le(symbolic::symbol("x"), symbolic::integer(0)));

    // Add blocks to sequences to verify they're properly connected
    builder.add_block(if_seq, control_flow::Assignments{});
    builder.add_block(else_seq, control_flow::Assignments{});

    // Verify sequences have content
    auto [seq1, cond1] = if_else.at(0);
    auto [seq2, cond2] = if_else.at(1);

    EXPECT_EQ(seq1.size(), 1);
    EXPECT_EQ(seq2.size(), 1);
}

} // namespace sdfg::structured_control_flow
