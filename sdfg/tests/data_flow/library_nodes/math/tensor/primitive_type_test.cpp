#include "gtest/gtest.h"
#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"

#include "sdfg/data_flow/library_nodes/math/cmath/cmath_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/elementwise_ops/abs_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/elementwise_ops/add_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/elementwise_ops/cast_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/elementwise_ops/div_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/elementwise_ops/exp_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/elementwise_ops/maximum_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/elementwise_ops/minimum_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/reduce_ops/max_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/reduce_ops/sum_node.h"

using namespace sdfg;

// Test integer support for operations that should support integers
TEST(TensorPrimitiveTypeTest, MaximumNodeInt32) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();

    types::Scalar desc(types::PrimitiveType::Int32);
    types::Pointer desc_ptr(desc);

    builder.add_container("a", desc_ptr);
    builder.add_container("b", desc_ptr);
    builder.add_container("c", desc_ptr);

    auto& block = builder.add_block(sdfg.root());
    auto& a_node = builder.add_access(block, "a");
    auto& b_node = builder.add_access(block, "b");
    auto& c_node = builder.add_access(block, "c");

    std::vector<symbolic::Expression> shape = {symbolic::integer(10)};
    types::Tensor tensor_type(types::PrimitiveType::Int32, shape);
    auto& node = static_cast<
        math::tensor::TensorNode&>(builder.add_library_node<math::tensor::MaximumNode>(block, DebugInfo(), shape));

    builder.add_computational_memlet(block, a_node, node, "A", {}, tensor_type, block.debug_info());
    builder.add_computational_memlet(block, b_node, node, "B", {}, tensor_type, block.debug_info());
    builder.add_computational_memlet(block, node, "C", c_node, {}, tensor_type, block.debug_info());

    // Should not throw and should validate successfully
    EXPECT_NO_THROW(node.validate(sdfg));

    analysis::AnalysisManager analysis_manager(sdfg);
    EXPECT_TRUE(node.expand(builder, analysis_manager));
}

// Test float support for operations
TEST(TensorPrimitiveTypeTest, MaximumNodeFloat) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();

    types::Scalar desc(types::PrimitiveType::Float);
    types::Pointer desc_ptr(desc);

    builder.add_container("a", desc_ptr);
    builder.add_container("b", desc_ptr);
    builder.add_container("c", desc_ptr);

    auto& block = builder.add_block(sdfg.root());
    auto& a_node = builder.add_access(block, "a");
    auto& b_node = builder.add_access(block, "b");
    auto& c_node = builder.add_access(block, "c");

    std::vector<symbolic::Expression> shape = {symbolic::integer(10)};
    types::Tensor tensor_type(types::PrimitiveType::Float, shape);
    auto& node = static_cast<
        math::tensor::TensorNode&>(builder.add_library_node<math::tensor::MaximumNode>(block, DebugInfo(), shape));

    builder.add_computational_memlet(block, a_node, node, "A", {}, tensor_type, block.debug_info());
    builder.add_computational_memlet(block, b_node, node, "B", {}, tensor_type, block.debug_info());
    builder.add_computational_memlet(block, node, "C", c_node, {}, tensor_type, block.debug_info());

    EXPECT_NO_THROW(node.validate(sdfg));

    analysis::AnalysisManager analysis_manager(sdfg);
    EXPECT_TRUE(node.expand(builder, analysis_manager));
}

// Test double support for operations
TEST(TensorPrimitiveTypeTest, MaximumNodeDouble) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();

    types::Scalar desc(types::PrimitiveType::Double);
    types::Pointer desc_ptr(desc);

    builder.add_container("a", desc_ptr);
    builder.add_container("b", desc_ptr);
    builder.add_container("c", desc_ptr);

    auto& block = builder.add_block(sdfg.root());
    auto& a_node = builder.add_access(block, "a");
    auto& b_node = builder.add_access(block, "b");
    auto& c_node = builder.add_access(block, "c");

    std::vector<symbolic::Expression> shape = {symbolic::integer(10)};
    types::Tensor tensor_type(types::PrimitiveType::Double, shape);
    auto& node = static_cast<
        math::tensor::TensorNode&>(builder.add_library_node<math::tensor::MaximumNode>(block, DebugInfo(), shape));

    builder.add_computational_memlet(block, a_node, node, "A", {}, tensor_type, block.debug_info());
    builder.add_computational_memlet(block, b_node, node, "B", {}, tensor_type, block.debug_info());
    builder.add_computational_memlet(block, node, "C", c_node, {}, tensor_type, block.debug_info());

    EXPECT_NO_THROW(node.validate(sdfg));

    analysis::AnalysisManager analysis_manager(sdfg);
    EXPECT_TRUE(node.expand(builder, analysis_manager));
}

// Test unsigned integer support
TEST(TensorPrimitiveTypeTest, MaximumNodeUInt32) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();

    types::Scalar desc(types::PrimitiveType::UInt32);
    types::Pointer desc_ptr(desc);

    builder.add_container("a", desc_ptr);
    builder.add_container("b", desc_ptr);
    builder.add_container("c", desc_ptr);

    auto& block = builder.add_block(sdfg.root());
    auto& a_node = builder.add_access(block, "a");
    auto& b_node = builder.add_access(block, "b");
    auto& c_node = builder.add_access(block, "c");

    std::vector<symbolic::Expression> shape = {symbolic::integer(10)};
    types::Tensor tensor_type(types::PrimitiveType::UInt32, shape);
    auto& node = static_cast<
        math::tensor::TensorNode&>(builder.add_library_node<math::tensor::MaximumNode>(block, DebugInfo(), shape));

    builder.add_computational_memlet(block, a_node, node, "A", {}, tensor_type, block.debug_info());
    builder.add_computational_memlet(block, b_node, node, "B", {}, tensor_type, block.debug_info());
    builder.add_computational_memlet(block, node, "C", c_node, {}, tensor_type, block.debug_info());

    EXPECT_NO_THROW(node.validate(sdfg));

    analysis::AnalysisManager analysis_manager(sdfg);
    EXPECT_TRUE(node.expand(builder, analysis_manager));
}

// Test that float-only operations reject integer types
TEST(TensorPrimitiveTypeTest, ExpNodeRejectsInt32) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();

    types::Scalar desc(types::PrimitiveType::Int32);
    types::Pointer desc_ptr(desc);

    builder.add_container("a", desc_ptr);
    builder.add_container("b", desc_ptr);

    auto& block = builder.add_block(sdfg.root());
    auto& a_node = builder.add_access(block, "a");
    auto& b_node = builder.add_access(block, "b");

    std::vector<symbolic::Expression> shape = {symbolic::integer(10)};
    types::Tensor tensor_type(types::PrimitiveType::Int32, shape);
    auto& node =
        static_cast<math::tensor::TensorNode&>(builder.add_library_node<math::tensor::ExpNode>(block, DebugInfo(), shape)
        );

    builder.add_computational_memlet(block, a_node, node, "X", {}, tensor_type, block.debug_info());
    builder.add_computational_memlet(block, node, "Y", b_node, {}, tensor_type, block.debug_info());

    // Should throw because ExpNode doesn't support integer types
    EXPECT_THROW(node.validate(sdfg), InvalidSDFGException);
}

// Test that mixed types are rejected
TEST(TensorPrimitiveTypeTest, MixedTypesRejected) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();

    types::Scalar desc_float(types::PrimitiveType::Float);
    types::Scalar desc_double(types::PrimitiveType::Double);
    types::Pointer desc_float_ptr(desc_float);
    types::Pointer desc_double_ptr(desc_double);

    builder.add_container("a", desc_float_ptr);
    builder.add_container("b", desc_double_ptr);
    builder.add_container("c", desc_float_ptr);

    auto& block = builder.add_block(sdfg.root());
    auto& a_node = builder.add_access(block, "a");
    auto& b_node = builder.add_access(block, "b");
    auto& c_node = builder.add_access(block, "c");

    std::vector<symbolic::Expression> shape = {symbolic::integer(10)};
    types::Tensor tensor_float(types::PrimitiveType::Float, shape);
    types::Tensor tensor_double(types::PrimitiveType::Double, shape);
    auto& node =
        static_cast<math::tensor::TensorNode&>(builder.add_library_node<math::tensor::AddNode>(block, DebugInfo(), shape)
        );

    builder.add_computational_memlet(block, a_node, node, "A", {}, tensor_float, block.debug_info());
    builder.add_computational_memlet(block, b_node, node, "B", {}, tensor_double, block.debug_info());
    builder.add_computational_memlet(block, node, "C", c_node, {}, tensor_float, block.debug_info());

    // Should throw because types are mixed (Float and Double)
    EXPECT_THROW(node.validate(sdfg), InvalidSDFGException);
}

// Test AbsNode with integers
TEST(TensorPrimitiveTypeTest, AbsNodeInt32) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();

    types::Scalar desc(types::PrimitiveType::Int32);
    types::Pointer desc_ptr(desc);

    builder.add_container("a", desc_ptr);
    builder.add_container("b", desc_ptr);

    auto& block = builder.add_block(sdfg.root());
    auto& a_node = builder.add_access(block, "a");
    auto& b_node = builder.add_access(block, "b");

    std::vector<symbolic::Expression> shape = {symbolic::integer(10)};
    types::Tensor tensor_type(types::PrimitiveType::Int32, shape);
    auto& node =
        static_cast<math::tensor::TensorNode&>(builder.add_library_node<math::tensor::AbsNode>(block, DebugInfo(), shape)
        );

    builder.add_computational_memlet(block, a_node, node, "X", {}, tensor_type, block.debug_info());
    builder.add_computational_memlet(block, node, "Y", b_node, {}, tensor_type, block.debug_info());

    EXPECT_NO_THROW(node.validate(sdfg));

    analysis::AnalysisManager analysis_manager(sdfg);
    EXPECT_TRUE(node.expand(builder, analysis_manager));
}

// Test reduce max with integers
TEST(TensorPrimitiveTypeTest, ReduceMaxInt64) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();

    types::Scalar desc(types::PrimitiveType::Int64);
    types::Pointer desc_ptr(desc);

    builder.add_container("a", desc_ptr);
    builder.add_container("b", desc_ptr);

    auto& block = builder.add_block(sdfg.root());
    auto& a_node = builder.add_access(block, "a");
    auto& b_node = builder.add_access(block, "b");

    std::vector<symbolic::Expression> shape = {symbolic::integer(10), symbolic::integer(20)};
    types::Tensor tensor_type_input(types::PrimitiveType::Int64, shape);
    types::Tensor
        tensor_type_output(types::PrimitiveType::Int64, std::vector<symbolic::Expression>{symbolic::integer(10)});
    std::vector<int64_t> axes = {1};
    auto& node = static_cast<math::tensor::TensorNode&>(builder.add_library_node<
                                                        math::tensor::MaxNode>(block, DebugInfo(), shape, axes, false));

    builder.add_computational_memlet(block, a_node, node, "X", {}, tensor_type_input, block.debug_info());
    builder.add_computational_memlet(block, node, "Y", b_node, {}, tensor_type_output, block.debug_info());

    EXPECT_NO_THROW(node.validate(sdfg));

    analysis::AnalysisManager analysis_manager(sdfg);
    EXPECT_TRUE(node.expand(builder, analysis_manager));
}

// Test reduce sum with integers
TEST(TensorPrimitiveTypeTest, ReduceSumInt32) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();

    types::Scalar desc(types::PrimitiveType::Int32);
    types::Pointer desc_ptr(desc);

    builder.add_container("a", desc_ptr);
    builder.add_container("b", desc_ptr);

    auto& block = builder.add_block(sdfg.root());
    auto& a_node = builder.add_access(block, "a");
    auto& b_node = builder.add_access(block, "b");

    std::vector<symbolic::Expression> shape = {symbolic::integer(10), symbolic::integer(20)};
    std::vector<int64_t> axes = {0};
    types::Tensor tensor_type_input(types::PrimitiveType::Int32, shape);
    types::Tensor
        tensor_type_output(types::PrimitiveType::Int32, std::vector<symbolic::Expression>{symbolic::integer(20)});
    auto& node = static_cast<math::tensor::TensorNode&>(builder.add_library_node<
                                                        math::tensor::SumNode>(block, DebugInfo(), shape, axes, false));

    builder.add_computational_memlet(block, a_node, node, "X", {}, tensor_type_input, block.debug_info());
    builder.add_computational_memlet(block, node, "Y", b_node, {}, tensor_type_output, block.debug_info());

    EXPECT_NO_THROW(node.validate(sdfg));

    analysis::AnalysisManager analysis_manager(sdfg);
    EXPECT_TRUE(node.expand(builder, analysis_manager));
}

// Test with scalar inputs (not pointers)
TEST(TensorPrimitiveTypeTest, ScalarInputFloat) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();

    types::Scalar desc_scalar(types::PrimitiveType::Float);
    types::Pointer desc_ptr(desc_scalar);

    builder.add_container("a", desc_scalar);
    builder.add_container("b", desc_ptr);
    builder.add_container("c", desc_ptr);

    auto& block = builder.add_block(sdfg.root());
    auto& a_node = builder.add_access(block, "a");
    auto& b_node = builder.add_access(block, "b");
    auto& c_node = builder.add_access(block, "c");

    std::vector<symbolic::Expression> shape = {symbolic::integer(10)};
    types::Tensor tensor_type_float(types::PrimitiveType::Float, shape);
    types::Tensor tensor_type_A(types::PrimitiveType::Float, {});
    auto& node =
        static_cast<math::tensor::TensorNode&>(builder.add_library_node<math::tensor::AddNode>(block, DebugInfo(), shape)
        );

    // Connect scalar and pointer - should still validate
    builder.add_computational_memlet(block, a_node, node, "A", {}, tensor_type_A, block.debug_info());
    builder.add_computational_memlet(block, b_node, node, "B", {}, tensor_type_float, block.debug_info());
    builder.add_computational_memlet(block, node, "C", c_node, {}, tensor_type_float, block.debug_info());

    EXPECT_NO_THROW(node.validate(sdfg));
}

// Test that the correct tasklets are generated for integer operations
TEST(TensorPrimitiveTypeTest, AddNodeInt32GeneratesIntAddTasklet) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();

    types::Scalar desc(types::PrimitiveType::Int32);
    types::Pointer desc_ptr(desc);

    builder.add_container("a", desc_ptr);
    builder.add_container("b", desc_ptr);
    builder.add_container("c", desc_ptr);

    auto& block = builder.add_block(sdfg.root());
    auto& a_node = builder.add_access(block, "a");
    auto& b_node = builder.add_access(block, "b");
    auto& c_node = builder.add_access(block, "c");

    std::vector<symbolic::Expression> shape = {symbolic::integer(10)};
    types::Tensor tensor_type(types::PrimitiveType::Int32, shape);
    auto& node =
        static_cast<math::tensor::TensorNode&>(builder.add_library_node<math::tensor::AddNode>(block, DebugInfo(), shape)
        );

    builder.add_computational_memlet(block, a_node, node, "A", {}, tensor_type, block.debug_info());
    builder.add_computational_memlet(block, b_node, node, "B", {}, tensor_type, block.debug_info());
    builder.add_computational_memlet(block, node, "C", c_node, {}, tensor_type, block.debug_info());

    analysis::AnalysisManager analysis_manager(sdfg);
    EXPECT_TRUE(node.expand(builder, analysis_manager));

    // Navigate to innermost block
    auto& new_sequence = dynamic_cast<structured_control_flow::Sequence&>(sdfg.root().at(0).first);
    structured_control_flow::Sequence* current_scope = &new_sequence;
    auto map_loop = dynamic_cast<structured_control_flow::Map*>(&current_scope->at(0).first);
    ASSERT_NE(map_loop, nullptr);
    current_scope = &map_loop->root();
    auto code_block = dynamic_cast<structured_control_flow::Block*>(&current_scope->at(0).first);
    ASSERT_NE(code_block, nullptr);

    // Check that there's an int_add tasklet
    EXPECT_FALSE(code_block->dataflow().tasklets().empty());
    auto* tasklet = *code_block->dataflow().tasklets().begin();
    EXPECT_EQ(tasklet->code(), data_flow::TaskletCode::int_add);
}

// Test that the correct tasklets are generated for float operations
TEST(TensorPrimitiveTypeTest, AddNodeFloatGeneratesFpAddTasklet) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();

    types::Scalar desc(types::PrimitiveType::Float);
    types::Pointer desc_ptr(desc);

    builder.add_container("a", desc_ptr);
    builder.add_container("b", desc_ptr);
    builder.add_container("c", desc_ptr);

    auto& block = builder.add_block(sdfg.root());
    auto& a_node = builder.add_access(block, "a");
    auto& b_node = builder.add_access(block, "b");
    auto& c_node = builder.add_access(block, "c");

    std::vector<symbolic::Expression> shape = {symbolic::integer(10)};
    types::Tensor tensor_type(types::PrimitiveType::Float, shape);
    auto& node =
        static_cast<math::tensor::TensorNode&>(builder.add_library_node<math::tensor::AddNode>(block, DebugInfo(), shape)
        );

    builder.add_computational_memlet(block, a_node, node, "A", {}, tensor_type, block.debug_info());
    builder.add_computational_memlet(block, b_node, node, "B", {}, tensor_type, block.debug_info());
    builder.add_computational_memlet(block, node, "C", c_node, {}, tensor_type, block.debug_info());

    analysis::AnalysisManager analysis_manager(sdfg);
    EXPECT_TRUE(node.expand(builder, analysis_manager));

    // Navigate to innermost block
    auto& new_sequence = dynamic_cast<structured_control_flow::Sequence&>(sdfg.root().at(0).first);
    structured_control_flow::Sequence* current_scope = &new_sequence;
    auto map_loop = dynamic_cast<structured_control_flow::Map*>(&current_scope->at(0).first);
    ASSERT_NE(map_loop, nullptr);
    current_scope = &map_loop->root();
    auto code_block = dynamic_cast<structured_control_flow::Block*>(&current_scope->at(0).first);
    ASSERT_NE(code_block, nullptr);

    // Check that there's an fp_add tasklet
    EXPECT_FALSE(code_block->dataflow().tasklets().empty());
    auto* tasklet = *code_block->dataflow().tasklets().begin();
    EXPECT_EQ(tasklet->code(), data_flow::TaskletCode::fp_add);
}

// Test signed division generates int_sdiv
TEST(TensorPrimitiveTypeTest, DivNodeInt32GeneratesIntSdivTasklet) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();

    types::Scalar desc(types::PrimitiveType::Int32);
    types::Pointer desc_ptr(desc);

    builder.add_container("a", desc_ptr);
    builder.add_container("b", desc_ptr);
    builder.add_container("c", desc_ptr);

    auto& block = builder.add_block(sdfg.root());
    auto& a_node = builder.add_access(block, "a");
    auto& b_node = builder.add_access(block, "b");
    auto& c_node = builder.add_access(block, "c");

    std::vector<symbolic::Expression> shape = {symbolic::integer(10)};
    types::Tensor tensor_type(types::PrimitiveType::Int32, shape);
    auto& node =
        static_cast<math::tensor::TensorNode&>(builder.add_library_node<math::tensor::DivNode>(block, DebugInfo(), shape)
        );

    builder.add_computational_memlet(block, a_node, node, "A", {}, tensor_type, block.debug_info());
    builder.add_computational_memlet(block, b_node, node, "B", {}, tensor_type, block.debug_info());
    builder.add_computational_memlet(block, node, "C", c_node, {}, tensor_type, block.debug_info());

    analysis::AnalysisManager analysis_manager(sdfg);
    EXPECT_TRUE(node.expand(builder, analysis_manager));

    // Navigate to innermost block
    auto& new_sequence = dynamic_cast<structured_control_flow::Sequence&>(sdfg.root().at(0).first);
    structured_control_flow::Sequence* current_scope = &new_sequence;
    auto map_loop = dynamic_cast<structured_control_flow::Map*>(&current_scope->at(0).first);
    ASSERT_NE(map_loop, nullptr);
    current_scope = &map_loop->root();
    auto code_block = dynamic_cast<structured_control_flow::Block*>(&current_scope->at(0).first);
    ASSERT_NE(code_block, nullptr);

    // Check that there's an int_sdiv tasklet
    EXPECT_FALSE(code_block->dataflow().tasklets().empty());
    auto* tasklet = *code_block->dataflow().tasklets().begin();
    EXPECT_EQ(tasklet->code(), data_flow::TaskletCode::int_sdiv);
}

// Test unsigned division generates int_udiv
TEST(TensorPrimitiveTypeTest, DivNodeUInt32GeneratesIntUdivTasklet) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();

    types::Scalar desc(types::PrimitiveType::UInt32);
    types::Pointer desc_ptr(desc);

    builder.add_container("a", desc_ptr);
    builder.add_container("b", desc_ptr);
    builder.add_container("c", desc_ptr);

    auto& block = builder.add_block(sdfg.root());
    auto& a_node = builder.add_access(block, "a");
    auto& b_node = builder.add_access(block, "b");
    auto& c_node = builder.add_access(block, "c");

    std::vector<symbolic::Expression> shape = {symbolic::integer(10)};
    types::Tensor tensor_type(types::PrimitiveType::UInt32, shape);
    auto& node =
        static_cast<math::tensor::TensorNode&>(builder.add_library_node<math::tensor::DivNode>(block, DebugInfo(), shape)
        );

    builder.add_computational_memlet(block, a_node, node, "A", {}, tensor_type, block.debug_info());
    builder.add_computational_memlet(block, b_node, node, "B", {}, tensor_type, block.debug_info());
    builder.add_computational_memlet(block, node, "C", c_node, {}, tensor_type, block.debug_info());

    analysis::AnalysisManager analysis_manager(sdfg);
    EXPECT_TRUE(node.expand(builder, analysis_manager));

    // Navigate to innermost block
    auto& new_sequence = dynamic_cast<structured_control_flow::Sequence&>(sdfg.root().at(0).first);
    structured_control_flow::Sequence* current_scope = &new_sequence;
    auto map_loop = dynamic_cast<structured_control_flow::Map*>(&current_scope->at(0).first);
    ASSERT_NE(map_loop, nullptr);
    current_scope = &map_loop->root();
    auto code_block = dynamic_cast<structured_control_flow::Block*>(&current_scope->at(0).first);
    ASSERT_NE(code_block, nullptr);

    // Check that there's an int_udiv tasklet
    EXPECT_FALSE(code_block->dataflow().tasklets().empty());
    auto* tasklet = *code_block->dataflow().tasklets().begin();
    EXPECT_EQ(tasklet->code(), data_flow::TaskletCode::int_udiv);
}

// Test MaximumNode generates int_smax tasklet for signed integers
TEST(TensorPrimitiveTypeTest, MaximumNodeInt32GeneratesIntSmaxTasklet) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();

    types::Scalar desc(types::PrimitiveType::Int32);
    types::Pointer desc_ptr(desc);

    builder.add_container("a", desc_ptr);
    builder.add_container("b", desc_ptr);
    builder.add_container("c", desc_ptr);

    auto& block = builder.add_block(sdfg.root());
    auto& a_node = builder.add_access(block, "a");
    auto& b_node = builder.add_access(block, "b");
    auto& c_node = builder.add_access(block, "c");

    std::vector<symbolic::Expression> shape = {symbolic::integer(10)};
    types::Tensor tensor_type(types::PrimitiveType::Int32, shape);
    auto& node = static_cast<
        math::tensor::TensorNode&>(builder.add_library_node<math::tensor::MaximumNode>(block, DebugInfo(), shape));

    builder.add_computational_memlet(block, a_node, node, "A", {}, tensor_type, block.debug_info());
    builder.add_computational_memlet(block, b_node, node, "B", {}, tensor_type, block.debug_info());
    builder.add_computational_memlet(block, node, "C", c_node, {}, tensor_type, block.debug_info());

    analysis::AnalysisManager analysis_manager(sdfg);
    EXPECT_TRUE(node.expand(builder, analysis_manager));

    // Navigate to innermost block
    auto& new_sequence = dynamic_cast<structured_control_flow::Sequence&>(sdfg.root().at(0).first);
    structured_control_flow::Sequence* current_scope = &new_sequence;
    auto map_loop = dynamic_cast<structured_control_flow::Map*>(&current_scope->at(0).first);
    ASSERT_NE(map_loop, nullptr);
    current_scope = &map_loop->root();
    auto code_block = dynamic_cast<structured_control_flow::Block*>(&current_scope->at(0).first);
    ASSERT_NE(code_block, nullptr);

    // Check that there's an int_smax tasklet
    EXPECT_FALSE(code_block->dataflow().tasklets().empty());
    auto* tasklet = *code_block->dataflow().tasklets().begin();
    EXPECT_EQ(tasklet->code(), data_flow::TaskletCode::int_smax);
}

// Test MaximumNode generates fmaxf intrinsic for float
TEST(TensorPrimitiveTypeTest, MaximumNodeFloatGeneratesFmaxfIntrinsic) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();

    types::Scalar desc(types::PrimitiveType::Float);
    types::Pointer desc_ptr(desc);

    builder.add_container("a", desc_ptr);
    builder.add_container("b", desc_ptr);
    builder.add_container("c", desc_ptr);

    auto& block = builder.add_block(sdfg.root());
    auto& a_node = builder.add_access(block, "a");
    auto& b_node = builder.add_access(block, "b");
    auto& c_node = builder.add_access(block, "c");

    std::vector<symbolic::Expression> shape = {symbolic::integer(10)};
    types::Tensor tensor_type(types::PrimitiveType::Float, shape);
    auto& node = static_cast<
        math::tensor::TensorNode&>(builder.add_library_node<math::tensor::MaximumNode>(block, DebugInfo(), shape));

    builder.add_computational_memlet(block, a_node, node, "A", {}, tensor_type, block.debug_info());
    builder.add_computational_memlet(block, b_node, node, "B", {}, tensor_type, block.debug_info());
    builder.add_computational_memlet(block, node, "C", c_node, {}, tensor_type, block.debug_info());

    analysis::AnalysisManager analysis_manager(sdfg);
    EXPECT_TRUE(node.expand(builder, analysis_manager));

    // Navigate to innermost block
    auto& new_sequence = dynamic_cast<structured_control_flow::Sequence&>(sdfg.root().at(0).first);
    structured_control_flow::Sequence* current_scope = &new_sequence;
    auto map_loop = dynamic_cast<structured_control_flow::Map*>(&current_scope->at(0).first);
    ASSERT_NE(map_loop, nullptr);
    current_scope = &map_loop->root();
    auto code_block = dynamic_cast<structured_control_flow::Block*>(&current_scope->at(0).first);
    ASSERT_NE(code_block, nullptr);

    // Check that there's a fmaxf CMathNode
    EXPECT_FALSE(code_block->dataflow().library_nodes().empty());
    auto* libnode = *code_block->dataflow().library_nodes().begin();
    auto* cmath_node = dynamic_cast<math::cmath::CMathNode*>(libnode);
    ASSERT_NE(cmath_node, nullptr);
    EXPECT_EQ(cmath_node->name(), "fmaxf");
}

// Test MaximumNode generates fmax intrinsic for double
TEST(TensorPrimitiveTypeTest, MaximumNodeDoubleGeneratesFmaxIntrinsic) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();

    types::Scalar desc(types::PrimitiveType::Double);
    types::Pointer desc_ptr(desc);

    builder.add_container("a", desc_ptr);
    builder.add_container("b", desc_ptr);
    builder.add_container("c", desc_ptr);

    auto& block = builder.add_block(sdfg.root());
    auto& a_node = builder.add_access(block, "a");
    auto& b_node = builder.add_access(block, "b");
    auto& c_node = builder.add_access(block, "c");

    std::vector<symbolic::Expression> shape = {symbolic::integer(10)};
    types::Tensor tensor_type(types::PrimitiveType::Double, shape);
    auto& node = static_cast<
        math::tensor::TensorNode&>(builder.add_library_node<math::tensor::MaximumNode>(block, DebugInfo(), shape));

    builder.add_computational_memlet(block, a_node, node, "A", {}, tensor_type, block.debug_info());
    builder.add_computational_memlet(block, b_node, node, "B", {}, tensor_type, block.debug_info());
    builder.add_computational_memlet(block, node, "C", c_node, {}, tensor_type, block.debug_info());

    analysis::AnalysisManager analysis_manager(sdfg);
    EXPECT_TRUE(node.expand(builder, analysis_manager));

    // Navigate to innermost block
    auto& new_sequence = dynamic_cast<structured_control_flow::Sequence&>(sdfg.root().at(0).first);
    structured_control_flow::Sequence* current_scope = &new_sequence;
    auto map_loop = dynamic_cast<structured_control_flow::Map*>(&current_scope->at(0).first);
    ASSERT_NE(map_loop, nullptr);
    current_scope = &map_loop->root();
    auto code_block = dynamic_cast<structured_control_flow::Block*>(&current_scope->at(0).first);
    ASSERT_NE(code_block, nullptr);

    // Check that there's a fmax CMathNode
    EXPECT_FALSE(code_block->dataflow().library_nodes().empty());
    auto* libnode = *code_block->dataflow().library_nodes().begin();
    auto* cmath_node = dynamic_cast<math::cmath::CMathNode*>(libnode);
    ASSERT_NE(cmath_node, nullptr);
    EXPECT_EQ(cmath_node->name(), "fmax");
}

// Test ExpNode generates expf intrinsic for float
TEST(TensorPrimitiveTypeTest, ExpNodeFloatGeneratesExpfIntrinsic) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();

    types::Scalar desc(types::PrimitiveType::Float);
    types::Pointer desc_ptr(desc);

    builder.add_container("a", desc_ptr);
    builder.add_container("b", desc_ptr);

    auto& block = builder.add_block(sdfg.root());
    auto& a_node = builder.add_access(block, "a");
    auto& b_node = builder.add_access(block, "b");

    std::vector<symbolic::Expression> shape = {symbolic::integer(10)};
    types::Tensor tensor_type(types::PrimitiveType::Float, shape);
    auto& node =
        static_cast<math::tensor::TensorNode&>(builder.add_library_node<math::tensor::ExpNode>(block, DebugInfo(), shape)
        );

    builder.add_computational_memlet(block, a_node, node, "X", {}, tensor_type, block.debug_info());
    builder.add_computational_memlet(block, node, "Y", b_node, {}, tensor_type, block.debug_info());

    analysis::AnalysisManager analysis_manager(sdfg);
    EXPECT_TRUE(node.expand(builder, analysis_manager));

    // Navigate to innermost block
    auto& new_sequence = dynamic_cast<structured_control_flow::Sequence&>(sdfg.root().at(0).first);
    structured_control_flow::Sequence* current_scope = &new_sequence;
    auto map_loop = dynamic_cast<structured_control_flow::Map*>(&current_scope->at(0).first);
    ASSERT_NE(map_loop, nullptr);
    current_scope = &map_loop->root();
    auto code_block = dynamic_cast<structured_control_flow::Block*>(&current_scope->at(0).first);
    ASSERT_NE(code_block, nullptr);

    // Check that there's an expf CMathNode
    EXPECT_FALSE(code_block->dataflow().library_nodes().empty());
    auto* libnode = *code_block->dataflow().library_nodes().begin();
    auto* cmath_node = dynamic_cast<math::cmath::CMathNode*>(libnode);
    ASSERT_NE(cmath_node, nullptr);
    EXPECT_EQ(cmath_node->name(), "expf");
}

// Test CastNode with Int32 to Float conversion
TEST(TensorPrimitiveTypeTest, CastNodeInt32ToFloat) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();

    types::Scalar input_desc(types::PrimitiveType::Int32);
    types::Scalar output_desc(types::PrimitiveType::Float);
    types::Pointer input_ptr(input_desc);
    types::Pointer output_ptr(output_desc);

    builder.add_container("a", input_ptr);
    builder.add_container("b", output_ptr);

    auto& block = builder.add_block(sdfg.root());
    auto& a_node = builder.add_access(block, "a");
    auto& b_node = builder.add_access(block, "b");

    std::vector<symbolic::Expression> shape = {symbolic::integer(10)};
    types::Tensor tensor_type(types::PrimitiveType::Int32, shape);
    types::Tensor tensor_type_output(types::PrimitiveType::Float, shape);
    auto& node = static_cast<
        math::tensor::TensorNode&>(builder.add_library_node<
                                   math::tensor::CastNode>(block, DebugInfo(), shape, types::PrimitiveType::Float));

    builder.add_computational_memlet(block, a_node, node, "X", {}, tensor_type, block.debug_info());
    builder.add_computational_memlet(block, node, "Y", b_node, {}, tensor_type_output, block.debug_info());

    EXPECT_NO_THROW(node.validate(sdfg));

    analysis::AnalysisManager analysis_manager(sdfg);
    EXPECT_TRUE(node.expand(builder, analysis_manager));

    // Navigate to innermost block
    auto& new_sequence = dynamic_cast<structured_control_flow::Sequence&>(sdfg.root().at(0).first);
    structured_control_flow::Sequence* current_scope = &new_sequence;
    auto map_loop = dynamic_cast<structured_control_flow::Map*>(&current_scope->at(0).first);
    ASSERT_NE(map_loop, nullptr);
    current_scope = &map_loop->root();
    auto code_block = dynamic_cast<structured_control_flow::Block*>(&current_scope->at(0).first);
    ASSERT_NE(code_block, nullptr);

    // Check that there's an assign tasklet for casting
    EXPECT_FALSE(code_block->dataflow().tasklets().empty());
    auto* tasklet = *code_block->dataflow().tasklets().begin();
    EXPECT_EQ(tasklet->code(), data_flow::TaskletCode::assign);

    // Verify input and output types
    auto& dataflow = tasklet->get_parent();
    for (auto& edge : dataflow.in_edges(*tasklet)) {
        EXPECT_EQ(edge.result_type(sdfg)->primitive_type(), types::PrimitiveType::Int32);
    }
    for (auto& edge : dataflow.out_edges(*tasklet)) {
        EXPECT_EQ(edge.result_type(sdfg)->primitive_type(), types::PrimitiveType::Float);
    }
}

// Test CastNode with Float to Int32 conversion
TEST(TensorPrimitiveTypeTest, CastNodeFloatToInt32) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();

    types::Scalar input_desc(types::PrimitiveType::Float);
    types::Scalar output_desc(types::PrimitiveType::Int32);
    types::Pointer input_ptr(input_desc);
    types::Pointer output_ptr(output_desc);

    builder.add_container("a", input_ptr);
    builder.add_container("b", output_ptr);

    auto& block = builder.add_block(sdfg.root());
    auto& a_node = builder.add_access(block, "a");
    auto& b_node = builder.add_access(block, "b");

    std::vector<symbolic::Expression> shape = {symbolic::integer(10)};
    types::Tensor tensor_type(types::PrimitiveType::Float, shape);
    types::Tensor tensor_type_output(types::PrimitiveType::Int32, shape);
    auto& node = static_cast<
        math::tensor::TensorNode&>(builder.add_library_node<
                                   math::tensor::CastNode>(block, DebugInfo(), shape, types::PrimitiveType::Int32));

    builder.add_computational_memlet(block, a_node, node, "X", {}, tensor_type, block.debug_info());
    builder.add_computational_memlet(block, node, "Y", b_node, {}, tensor_type_output, block.debug_info());

    EXPECT_NO_THROW(node.validate(sdfg));

    analysis::AnalysisManager analysis_manager(sdfg);
    EXPECT_TRUE(node.expand(builder, analysis_manager));

    // Navigate to innermost block
    auto& new_sequence = dynamic_cast<structured_control_flow::Sequence&>(sdfg.root().at(0).first);
    structured_control_flow::Sequence* current_scope = &new_sequence;
    auto map_loop = dynamic_cast<structured_control_flow::Map*>(&current_scope->at(0).first);
    ASSERT_NE(map_loop, nullptr);
    current_scope = &map_loop->root();
    auto code_block = dynamic_cast<structured_control_flow::Block*>(&current_scope->at(0).first);
    ASSERT_NE(code_block, nullptr);

    // Check that there's an assign tasklet for casting
    EXPECT_FALSE(code_block->dataflow().tasklets().empty());
    auto* tasklet = *code_block->dataflow().tasklets().begin();
    EXPECT_EQ(tasklet->code(), data_flow::TaskletCode::assign);

    // Verify input and output types
    auto& dataflow = tasklet->get_parent();
    for (auto& edge : dataflow.in_edges(*tasklet)) {
        EXPECT_EQ(edge.result_type(sdfg)->primitive_type(), types::PrimitiveType::Float);
    }
    for (auto& edge : dataflow.out_edges(*tasklet)) {
        EXPECT_EQ(edge.result_type(sdfg)->primitive_type(), types::PrimitiveType::Int32);
    }
}

// Test CastNode with Float to Double conversion
TEST(TensorPrimitiveTypeTest, CastNodeFloatToDouble) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();

    types::Scalar input_desc(types::PrimitiveType::Float);
    types::Scalar output_desc(types::PrimitiveType::Double);
    types::Pointer input_ptr(input_desc);
    types::Pointer output_ptr(output_desc);

    builder.add_container("a", input_ptr);
    builder.add_container("b", output_ptr);

    auto& block = builder.add_block(sdfg.root());
    auto& a_node = builder.add_access(block, "a");
    auto& b_node = builder.add_access(block, "b");

    std::vector<symbolic::Expression> shape = {symbolic::integer(10)};
    types::Tensor tensor_type(types::PrimitiveType::Float, shape);
    types::Tensor tensor_type_output(types::PrimitiveType::Double, shape);
    auto& node = static_cast<
        math::tensor::TensorNode&>(builder.add_library_node<
                                   math::tensor::CastNode>(block, DebugInfo(), shape, types::PrimitiveType::Double));

    builder.add_computational_memlet(block, a_node, node, "X", {}, tensor_type, block.debug_info());
    builder.add_computational_memlet(block, node, "Y", b_node, {}, tensor_type_output, block.debug_info());

    EXPECT_NO_THROW(node.validate(sdfg));

    analysis::AnalysisManager analysis_manager(sdfg);
    EXPECT_TRUE(node.expand(builder, analysis_manager));

    // Navigate to innermost block and verify types
    auto& new_sequence = dynamic_cast<structured_control_flow::Sequence&>(sdfg.root().at(0).first);
    structured_control_flow::Sequence* current_scope = &new_sequence;
    auto map_loop = dynamic_cast<structured_control_flow::Map*>(&current_scope->at(0).first);
    ASSERT_NE(map_loop, nullptr);
    current_scope = &map_loop->root();
    auto code_block = dynamic_cast<structured_control_flow::Block*>(&current_scope->at(0).first);
    ASSERT_NE(code_block, nullptr);

    // Check that there's an assign tasklet for casting
    EXPECT_FALSE(code_block->dataflow().tasklets().empty());
    auto* tasklet = *code_block->dataflow().tasklets().begin();
    EXPECT_EQ(tasklet->code(), data_flow::TaskletCode::assign);

    // Verify input and output types
    auto& dataflow = tasklet->get_parent();
    for (auto& edge : dataflow.in_edges(*tasklet)) {
        EXPECT_EQ(edge.result_type(sdfg)->primitive_type(), types::PrimitiveType::Float);
    }
    for (auto& edge : dataflow.out_edges(*tasklet)) {
        EXPECT_EQ(edge.result_type(sdfg)->primitive_type(), types::PrimitiveType::Double);
    }
}

// Test CastNode with UInt32 to Int64 conversion
TEST(TensorPrimitiveTypeTest, CastNodeUInt32ToInt64) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();

    types::Scalar input_desc(types::PrimitiveType::UInt32);
    types::Scalar output_desc(types::PrimitiveType::Int64);
    types::Pointer input_ptr(input_desc);
    types::Pointer output_ptr(output_desc);

    builder.add_container("a", input_ptr);
    builder.add_container("b", output_ptr);

    auto& block = builder.add_block(sdfg.root());
    auto& a_node = builder.add_access(block, "a");
    auto& b_node = builder.add_access(block, "b");

    std::vector<symbolic::Expression> shape = {symbolic::integer(10)};
    types::Tensor tensor_type(types::PrimitiveType::UInt32, shape);
    types::Tensor tensor_type_output(types::PrimitiveType::Int64, shape);
    auto& node = static_cast<
        math::tensor::TensorNode&>(builder.add_library_node<
                                   math::tensor::CastNode>(block, DebugInfo(), shape, types::PrimitiveType::Int64));

    builder.add_computational_memlet(block, a_node, node, "X", {}, tensor_type, block.debug_info());
    builder.add_computational_memlet(block, node, "Y", b_node, {}, tensor_type_output, block.debug_info());

    EXPECT_NO_THROW(node.validate(sdfg));

    analysis::AnalysisManager analysis_manager(sdfg);
    EXPECT_TRUE(node.expand(builder, analysis_manager));

    // Verify the expanded structure
    auto& new_sequence = dynamic_cast<structured_control_flow::Sequence&>(sdfg.root().at(0).first);
    structured_control_flow::Sequence* current_scope = &new_sequence;
    auto map_loop = dynamic_cast<structured_control_flow::Map*>(&current_scope->at(0).first);
    ASSERT_NE(map_loop, nullptr);
    current_scope = &map_loop->root();
    auto code_block = dynamic_cast<structured_control_flow::Block*>(&current_scope->at(0).first);
    ASSERT_NE(code_block, nullptr);

    // Check that there's an assign tasklet for casting
    EXPECT_FALSE(code_block->dataflow().tasklets().empty());
    auto* tasklet = *code_block->dataflow().tasklets().begin();
    EXPECT_EQ(tasklet->code(), data_flow::TaskletCode::assign);

    // Verify input and output types
    auto& dataflow = tasklet->get_parent();
    for (auto& edge : dataflow.in_edges(*tasklet)) {
        EXPECT_EQ(edge.result_type(sdfg)->primitive_type(), types::PrimitiveType::UInt32);
    }
    for (auto& edge : dataflow.out_edges(*tasklet)) {
        EXPECT_EQ(edge.result_type(sdfg)->primitive_type(), types::PrimitiveType::Int64);
    }
}
