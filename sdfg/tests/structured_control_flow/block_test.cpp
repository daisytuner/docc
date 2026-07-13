#include <gtest/gtest.h>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/data_flow_graph.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/symbolic/symbolic.h"

using namespace sdfg;

namespace sdfg::structured_control_flow {

// Test basic Block structure and pointers
TEST(BlockTest, BasicStructure) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);

    // Verify block is a ControlFlowNode
    EXPECT_TRUE(dynamic_cast<const ControlFlowNode*>(&block) != nullptr);

    // Verify block has a dataflow graph
    const auto& dataflow = block.dataflow();
    EXPECT_TRUE(dynamic_cast<const data_flow::DataFlowGraph*>(&dataflow) != nullptr);
}

// Test dataflow access
TEST(BlockTest, DataflowAccess) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);

    types::Scalar int_type(types::PrimitiveType::Int32);
    builder.add_container("x", int_type);
    builder.add_container("y", int_type);

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);

    // Access the dataflow graph
    auto& dataflow = block.dataflow();

    // Add a tasklet to the block
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "y", {"x"});

    // Verify tasklet was added
    EXPECT_TRUE(dynamic_cast<const data_flow::Tasklet*>(&tasklet) != nullptr);
}

// Test block with assignments
TEST(BlockTest, AssignmentBlock) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);

    types::Scalar int_type(types::PrimitiveType::Int32);
    builder.add_container("N", int_type);

    auto& root = builder.subject().root();

    control_flow::Assignments assignments{{symbolic::symbol("N"), symbolic::integer(10)}};

    auto& block = builder.add_assignments(root, assignments);

    // Verify block exists
    EXPECT_EQ(root.size(), 1);

    auto& node = root.at(0);
    auto* assignment_block = dynamic_cast<const AssignmentBlock*>(&node);
    EXPECT_TRUE(assignment_block != nullptr);
    EXPECT_EQ(assignment_block->assignments().size(), 1);
}

// Test multiple blocks in sequence
TEST(BlockTest, MultipleBlocks) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);

    auto& root = builder.subject().root();

    auto& block1 = builder.add_block(root);
    auto& block2 = builder.add_block(root);
    auto& block3 = builder.add_block(root);

    EXPECT_EQ(root.size(), 3);

    // Verify all are blocks
    auto& node1 = root.at(0);
    auto& node2 = root.at(1);
    auto& node3 = root.at(2);

    EXPECT_TRUE(dynamic_cast<const Block*>(&node1) != nullptr);
    EXPECT_TRUE(dynamic_cast<const Block*>(&node2) != nullptr);
    EXPECT_TRUE(dynamic_cast<const Block*>(&node3) != nullptr);
}

// Test const and non-const dataflow access
TEST(BlockTest, ConstAndNonConstAccess) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);

    // Non-const access
    auto& mut_dataflow = block.dataflow();
    EXPECT_TRUE(dynamic_cast<data_flow::DataFlowGraph*>(&mut_dataflow) != nullptr);

    // Const access
    const auto& const_block = block;
    const auto& const_dataflow = const_block.dataflow();
    EXPECT_TRUE(dynamic_cast<const data_flow::DataFlowGraph*>(&const_dataflow) != nullptr);
}

} // namespace sdfg::structured_control_flow
