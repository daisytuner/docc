#include <gtest/gtest.h>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/structured_control_flow/return.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/scalar.h"

using namespace sdfg;

namespace sdfg::structured_control_flow {

// Test basic Return structure and pointers
TEST(ReturnTest, BasicStructure) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU, types::Scalar(types::PrimitiveType::Int32));

    types::Scalar int_type(types::PrimitiveType::Int32);
    builder.add_container("x", int_type);

    auto& root = builder.subject().root();

    auto& return_node = builder.add_return(root, "x");

    // Verify return_node is a ControlFlowNode
    EXPECT_TRUE(dynamic_cast<const ControlFlowNode*>(&return_node) != nullptr);

    // Verify return_node is a Return
    EXPECT_TRUE(dynamic_cast<const Return*>(&return_node) != nullptr);
}

// Test data return
TEST(ReturnTest, DataReturn) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU, types::Scalar(types::PrimitiveType::Int32));

    types::Scalar int_type(types::PrimitiveType::Int32);
    builder.add_container("result", int_type);

    auto& root = builder.subject().root();

    auto& return_node = builder.add_return(root, "result");

    // Verify it's a data return
    EXPECT_TRUE(return_node.is_data());
    EXPECT_FALSE(return_node.is_constant());
    EXPECT_EQ(return_node.data(), "result");
}

// Test constant return
TEST(ReturnTest, ConstantReturn) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU, types::Scalar(types::PrimitiveType::Int32));

    types::Scalar int_type(types::PrimitiveType::Int32);
    builder.add_container("constant_val", int_type);

    auto& root = builder.subject().root();

    // Add return for a constant container
    auto& return_node = builder.add_return(root, "constant_val");

    // Verify it's returning data (container name)
    EXPECT_TRUE(return_node.is_data());
    EXPECT_EQ(return_node.data(), "constant_val");
}

// Test return type
TEST(ReturnTest, ReturnType) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU, types::Scalar(types::PrimitiveType::Float));

    types::Scalar float_type(types::PrimitiveType::Float);
    builder.add_container("value", float_type);

    auto& root = builder.subject().root();

    auto& return_node = builder.add_return(root, "value");

    EXPECT_TRUE(return_node.is_data());
    EXPECT_FALSE(return_node.is_constant());
    EXPECT_EQ(return_node.data(), "value");
}

// Test return with different types
TEST(ReturnTest, DifferentTypes) {
    // Int32
    {
        builder::StructuredSDFGBuilder
            builder("test_sdfg", FunctionType_CPU, types::Scalar(types::PrimitiveType::Int32));
        types::Scalar int_type(types::PrimitiveType::Int32);
        builder.add_container("int_val", int_type);
        auto& root = builder.subject().root();
        auto& return_node = builder.add_return(root, "int_val");
        EXPECT_TRUE(return_node.is_data());
    }

    // Float
    {
        builder::StructuredSDFGBuilder
            builder("test_sdfg", FunctionType_CPU, types::Scalar(types::PrimitiveType::Float));
        types::Scalar float_type(types::PrimitiveType::Float);
        builder.add_container("float_val", float_type);
        auto& root = builder.subject().root();
        auto& return_node = builder.add_return(root, "float_val");
        EXPECT_TRUE(return_node.is_data());
    }

    // Bool
    {
        builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU, types::Scalar(types::PrimitiveType::Bool));
        types::Scalar bool_type(types::PrimitiveType::Bool);
        builder.add_container("bool_val", bool_type);
        auto& root = builder.subject().root();
        auto& return_node = builder.add_return(root, "bool_val");
        EXPECT_TRUE(return_node.is_data());
    }
}

// Test return in conditional
TEST(ReturnTest, ReturnInConditional) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU, types::Scalar(types::PrimitiveType::Int32));

    types::Scalar int_type(types::PrimitiveType::Int32);
    builder.add_container("x", int_type);
    builder.add_container("y", int_type);

    auto& root = builder.subject().root();

    auto& if_else = builder.add_if_else(root);

    // Return x if condition is true
    auto& if_seq = builder.add_case(if_else, symbolic::Gt(symbolic::symbol("x"), symbolic::integer(0)));
    auto& return_x = builder.add_return(if_seq, "x");
    EXPECT_TRUE(return_x.is_data());
    EXPECT_EQ(return_x.data(), "x");

    // Return y otherwise
    auto& else_seq = builder.add_case(if_else, symbolic::Le(symbolic::symbol("x"), symbolic::integer(0)));
    auto& return_y = builder.add_return(else_seq, "y");
    EXPECT_TRUE(return_y.is_data());
    EXPECT_EQ(return_y.data(), "y");
}

// Test return at end of sequence
TEST(ReturnTest, ReturnAtEndOfSequence) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU, types::Scalar(types::PrimitiveType::Int32));

    types::Scalar int_type(types::PrimitiveType::Int32);
    builder.add_container("result", int_type);

    auto& root = builder.subject().root();

    // Add some blocks
    builder.add_block(root);
    builder.add_block(root);

    // Add return at the end
    auto& return_node = builder.add_return(root, "result");

    EXPECT_EQ(root.size(), 3);

    // Verify last element is the return
    auto& node = root.at(2);
    EXPECT_TRUE(dynamic_cast<const Return*>(&node) != nullptr);
}

// Test multiple return paths
TEST(ReturnTest, MultipleReturnPaths) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU, types::Scalar(types::PrimitiveType::Int32));

    types::Scalar int_type(types::PrimitiveType::Int32);
    builder.add_container("x", int_type);
    builder.add_container("val1", int_type);
    builder.add_container("val2", int_type);
    builder.add_container("val3", int_type);

    auto& root = builder.subject().root();
    auto& if_else = builder.add_if_else(root);

    // First case
    auto& seq1 = builder.add_case(if_else, symbolic::Gt(symbolic::symbol("x"), symbolic::integer(10)));
    builder.add_return(seq1, "val1");

    // Second case
    auto& seq2 = builder.add_case(
        if_else,
        symbolic::
            And(symbolic::Gt(symbolic::symbol("x"), symbolic::integer(0)),
                symbolic::Le(symbolic::symbol("x"), symbolic::integer(10)))
    );
    builder.add_return(seq2, "val2");

    // Third case (else)
    auto& seq3 = builder.add_case(if_else, symbolic::Le(symbolic::symbol("x"), symbolic::integer(0)));
    builder.add_return(seq3, "val3");

    // All three branches have returns
    EXPECT_EQ(seq1.size(), 1);
    EXPECT_EQ(seq2.size(), 1);
    EXPECT_EQ(seq3.size(), 1);
}

} // namespace sdfg::structured_control_flow
