#include "gtest/gtest.h"
#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"

#include "sdfg/data_flow/library_nodes/math/tensor/elementwise_ops/abs_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/elementwise_ops/add_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/elementwise_ops/cast_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/elementwise_ops/div_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/elementwise_ops/elu_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/elementwise_ops/erf_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/elementwise_ops/exp_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/elementwise_ops/hard_sigmoid_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/elementwise_ops/leaky_relu_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/elementwise_ops/mul_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/elementwise_ops/pow_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/elementwise_ops/relu_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/elementwise_ops/sigmoid_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/elementwise_ops/sqrt_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/elementwise_ops/sub_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/elementwise_ops/tanh_node.h"

using namespace sdfg;

template<typename NodeType>
void TestUnary(std::vector<size_t> shape_dims) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();

    types::Scalar desc(types::PrimitiveType::Double);
    types::Pointer desc_ptr(desc);

    builder.add_container("a", desc_ptr);
    builder.add_container("b", desc_ptr);

    auto& block = builder.add_block(sdfg.root());

    auto& a_node = builder.add_access(block, "a");
    auto& b_node = builder.add_access(block, "b");

    std::vector<symbolic::Expression> shape;
    for (auto d : shape_dims) {
        shape.push_back(symbolic::integer(d));
    }
    types::Tensor tensor_type(types::PrimitiveType::Double, shape);

    auto& node = static_cast<NodeType&>(builder.add_library_node<NodeType>(block, DebugInfo(), shape));

    builder.add_computational_memlet(block, a_node, node, "X", {}, tensor_type, block.debug_info());
    builder.add_computational_memlet(block, node, "Y", b_node, {}, tensor_type, block.debug_info());

    sdfg.validate();
    analysis::AnalysisManager analysis_manager(sdfg);
    EXPECT_TRUE(node.expand(builder, analysis_manager));

    auto& new_sequence = dynamic_cast<structured_control_flow::Sequence&>(sdfg.root().at(0).first);

    // Navigate to the innermost map
    structured_control_flow::Sequence* current_scope = &new_sequence;
    for (size_t i = 0; i < shape_dims.size(); ++i) {
        auto map_loop = dynamic_cast<structured_control_flow::Map*>(&current_scope->at(0).first);
        ASSERT_NE(map_loop, nullptr);
        current_scope = &map_loop->root();
    }

    auto code_block = dynamic_cast<structured_control_flow::Block*>(&current_scope->at(0).first);
    ASSERT_NE(code_block, nullptr);

    // Check that the block is not empty (contains either tasklets or library nodes)
    bool has_content = !code_block->dataflow().tasklets().empty() || !code_block->dataflow().library_nodes().empty();
    EXPECT_TRUE(has_content) << "Inner block is empty for " << typeid(NodeType).name();

    // Check subsets of the first node's edges
    data_flow::DataFlowNode* inner_node = nullptr;
    if (!code_block->dataflow().library_nodes().empty()) {
        inner_node = *code_block->dataflow().library_nodes().begin();
    } else if (!code_block->dataflow().tasklets().empty()) {
        inner_node = *code_block->dataflow().tasklets().begin();
    }
    ASSERT_NE(inner_node, nullptr);

    auto& dataflow = inner_node->get_parent();

    // Check input edges
    for (auto& edge : dataflow.in_edges(*inner_node)) {
        if (dynamic_cast<data_flow::ConstantNode*>(&edge.src()) != nullptr) {
            continue; // Skip constant nodes
        }
        if (edge.subset().size() != shape_dims.size()) {
            EXPECT_EQ(edge.subset().size(), shape_dims.size())
                << "Input subset size is not " << shape_dims.size() << " for " << typeid(NodeType).name();
        }
    }

    // Check output edges
    for (auto& edge : dataflow.out_edges(*inner_node)) {
        if (edge.subset().size() != shape_dims.size()) {
            EXPECT_EQ(edge.subset().size(), shape_dims.size())
                << "Output subset size is not " << shape_dims.size() << " for " << typeid(NodeType).name();
        }
    }
}

template<typename NodeType>
void TestBinary(std::vector<size_t> shape_dims) {
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

    std::vector<symbolic::Expression> shape;
    for (auto d : shape_dims) {
        shape.push_back(symbolic::integer(d));
    }
    types::Tensor tensor_type(types::PrimitiveType::Double, shape);

    auto& node = static_cast<NodeType&>(builder.add_library_node<NodeType>(block, DebugInfo(), shape));

    builder.add_computational_memlet(block, a_node, node, "A", {}, tensor_type, block.debug_info());
    builder.add_computational_memlet(block, b_node, node, "B", {}, tensor_type, block.debug_info());
    builder.add_computational_memlet(block, node, "C", c_node, {}, tensor_type, block.debug_info());

    sdfg.validate();
    analysis::AnalysisManager analysis_manager(sdfg);
    EXPECT_TRUE(node.expand(builder, analysis_manager));

    auto& new_sequence = dynamic_cast<structured_control_flow::Sequence&>(sdfg.root().at(0).first);

    // Navigate to the innermost map
    structured_control_flow::Sequence* current_scope = &new_sequence;
    for (size_t i = 0; i < shape_dims.size(); ++i) {
        auto map_loop = dynamic_cast<structured_control_flow::Map*>(&current_scope->at(0).first);
        ASSERT_NE(map_loop, nullptr);
        current_scope = &map_loop->root();
    }

    auto code_block = dynamic_cast<structured_control_flow::Block*>(&current_scope->at(0).first);
    ASSERT_NE(code_block, nullptr);

    bool has_content = !code_block->dataflow().tasklets().empty() || !code_block->dataflow().library_nodes().empty();
    EXPECT_TRUE(has_content) << "Inner block is empty for " << typeid(NodeType).name();

    data_flow::DataFlowNode* inner_node = nullptr;
    if (!code_block->dataflow().library_nodes().empty()) {
        inner_node = *code_block->dataflow().library_nodes().begin();
    } else if (!code_block->dataflow().tasklets().empty()) {
        inner_node = *code_block->dataflow().tasklets().begin();
    }
    ASSERT_NE(inner_node, nullptr);

    auto& dataflow = inner_node->get_parent();

    for (auto& edge : dataflow.in_edges(*inner_node)) {
        if (dynamic_cast<data_flow::ConstantNode*>(&edge.src()) != nullptr) {
            continue; // Skip constant nodes
        }
        if (edge.subset().size() != shape_dims.size()) {
            EXPECT_EQ(edge.subset().size(), shape_dims.size())
                << "Input subset size is not " << shape_dims.size() << " for " << typeid(NodeType).name();
        }
    }

    for (auto& edge : dataflow.out_edges(*inner_node)) {
        if (edge.subset().size() != shape_dims.size()) {
            EXPECT_EQ(edge.subset().size(), shape_dims.size())
                << "Output subset size is not " << shape_dims.size() << " for " << typeid(NodeType).name();
        }
    }
}

#define REGISTER_UNARY_TEST(NodeType, Dim)                \
    TEST(ElementWiseTest, NodeType##_##Dim##D) {          \
        std::vector<size_t> dims;                         \
        for (int i = 0; i < Dim; ++i) dims.push_back(32); \
        TestUnary<math::tensor::NodeType>(dims);          \
    }

#define REGISTER_BINARY_TEST(NodeType, Dim)               \
    TEST(ElementWiseTest, NodeType##_##Dim##D) {          \
        std::vector<size_t> dims;                         \
        for (int i = 0; i < Dim; ++i) dims.push_back(32); \
        TestBinary<math::tensor::NodeType>(dims);         \
    }

// Unary Tests
REGISTER_UNARY_TEST(AbsNode, 1)
REGISTER_UNARY_TEST(AbsNode, 2)
REGISTER_UNARY_TEST(AbsNode, 3)
REGISTER_UNARY_TEST(AbsNode, 4)

REGISTER_UNARY_TEST(SqrtNode, 1)
REGISTER_UNARY_TEST(SqrtNode, 2)
REGISTER_UNARY_TEST(SqrtNode, 3)
REGISTER_UNARY_TEST(SqrtNode, 4)

REGISTER_UNARY_TEST(TanhNode, 1)
REGISTER_UNARY_TEST(TanhNode, 2)
REGISTER_UNARY_TEST(TanhNode, 3)
REGISTER_UNARY_TEST(TanhNode, 4)

REGISTER_UNARY_TEST(ErfNode, 1)
REGISTER_UNARY_TEST(ErfNode, 2)
REGISTER_UNARY_TEST(ErfNode, 3)
REGISTER_UNARY_TEST(ErfNode, 4)

REGISTER_UNARY_TEST(ExpNode, 1)
REGISTER_UNARY_TEST(ExpNode, 2)
REGISTER_UNARY_TEST(ExpNode, 3)
REGISTER_UNARY_TEST(ExpNode, 4)

REGISTER_UNARY_TEST(ReLUNode, 1)
REGISTER_UNARY_TEST(ReLUNode, 2)
REGISTER_UNARY_TEST(ReLUNode, 3)
REGISTER_UNARY_TEST(ReLUNode, 4)

REGISTER_UNARY_TEST(SigmoidNode, 1)
REGISTER_UNARY_TEST(SigmoidNode, 2)
REGISTER_UNARY_TEST(SigmoidNode, 3)
REGISTER_UNARY_TEST(SigmoidNode, 4)

REGISTER_UNARY_TEST(EluNode, 1)
REGISTER_UNARY_TEST(EluNode, 2)
REGISTER_UNARY_TEST(EluNode, 3)
REGISTER_UNARY_TEST(EluNode, 4)

REGISTER_UNARY_TEST(HardSigmoidNode, 1)
REGISTER_UNARY_TEST(HardSigmoidNode, 2)
REGISTER_UNARY_TEST(HardSigmoidNode, 3)
REGISTER_UNARY_TEST(HardSigmoidNode, 4)

REGISTER_UNARY_TEST(LeakyReLUNode, 1)
REGISTER_UNARY_TEST(LeakyReLUNode, 2)
REGISTER_UNARY_TEST(LeakyReLUNode, 3)
REGISTER_UNARY_TEST(LeakyReLUNode, 4)

// Binary Tests
REGISTER_BINARY_TEST(AddNode, 1)
REGISTER_BINARY_TEST(AddNode, 2)
REGISTER_BINARY_TEST(AddNode, 3)
REGISTER_BINARY_TEST(AddNode, 4)

REGISTER_BINARY_TEST(SubNode, 1)
REGISTER_BINARY_TEST(SubNode, 2)
REGISTER_BINARY_TEST(SubNode, 3)
REGISTER_BINARY_TEST(SubNode, 4)

REGISTER_BINARY_TEST(MulNode, 1)
REGISTER_BINARY_TEST(MulNode, 2)
REGISTER_BINARY_TEST(MulNode, 3)
REGISTER_BINARY_TEST(MulNode, 4)

REGISTER_BINARY_TEST(DivNode, 1)
REGISTER_BINARY_TEST(DivNode, 2)
REGISTER_BINARY_TEST(DivNode, 3)
REGISTER_BINARY_TEST(DivNode, 4)

REGISTER_BINARY_TEST(PowNode, 1)
REGISTER_BINARY_TEST(PowNode, 2)
REGISTER_BINARY_TEST(PowNode, 3)
REGISTER_BINARY_TEST(PowNode, 4)

// Cast tests - specialized template for CastNode
template<types::PrimitiveType SourceType, types::PrimitiveType TargetType>
void TestCast(std::vector<size_t> shape_dims) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();

    types::Scalar source_desc(SourceType);
    types::Scalar target_desc(TargetType);
    types::Pointer source_ptr(source_desc);
    types::Pointer target_ptr(target_desc);

    builder.add_container("a", source_ptr);
    builder.add_container("b", target_ptr);

    auto& block = builder.add_block(sdfg.root());

    auto& a_node = builder.add_access(block, "a");
    auto& b_node = builder.add_access(block, "b");

    std::vector<symbolic::Expression> shape;
    for (auto d : shape_dims) {
        shape.push_back(symbolic::integer(d));
    }
    types::Tensor tensor_type_source(SourceType, shape);
    types::Tensor tensor_type_target(TargetType, shape);

    auto& node = static_cast<math::tensor::CastNode&>(builder.add_library_node<
                                                      math::tensor::CastNode>(block, DebugInfo(), shape, TargetType));

    builder.add_computational_memlet(block, a_node, node, "X", {}, tensor_type_source, block.debug_info());
    builder.add_computational_memlet(block, node, "Y", b_node, {}, tensor_type_target, block.debug_info());

    sdfg.validate();
    analysis::AnalysisManager analysis_manager(sdfg);
    EXPECT_TRUE(node.expand(builder, analysis_manager));

    auto& new_sequence = dynamic_cast<structured_control_flow::Sequence&>(sdfg.root().at(0).first);

    // Navigate to the innermost map
    structured_control_flow::Sequence* current_scope = &new_sequence;
    for (size_t i = 0; i < shape_dims.size(); ++i) {
        auto map_loop = dynamic_cast<structured_control_flow::Map*>(&current_scope->at(0).first);
        ASSERT_NE(map_loop, nullptr);
        current_scope = &map_loop->root();
    }

    auto code_block = dynamic_cast<structured_control_flow::Block*>(&current_scope->at(0).first);
    ASSERT_NE(code_block, nullptr);

    // Check that the block is not empty (contains either tasklets or library nodes)
    bool has_content = !code_block->dataflow().tasklets().empty() || !code_block->dataflow().library_nodes().empty();
    EXPECT_TRUE(has_content) << "Inner block is empty for CastNode";

    // Check subsets of the first node's edges
    data_flow::DataFlowNode* inner_node = nullptr;
    if (!code_block->dataflow().library_nodes().empty()) {
        inner_node = *code_block->dataflow().library_nodes().begin();
    } else if (!code_block->dataflow().tasklets().empty()) {
        inner_node = *code_block->dataflow().tasklets().begin();
    }
    ASSERT_NE(inner_node, nullptr);

    auto& dataflow = inner_node->get_parent();

    // Check input edges
    for (auto& edge : dataflow.in_edges(*inner_node)) {
        if (edge.subset().size() != shape_dims.size()) {
            EXPECT_EQ(edge.subset().size(), shape_dims.size())
                << "Input subset size is not " << shape_dims.size() << " for CastNode";
        }
        // Check that input type is the source type
        EXPECT_EQ(edge.result_type(sdfg)->primitive_type(), SourceType);
    }

    // Check output edges
    for (auto& edge : dataflow.out_edges(*inner_node)) {
        if (edge.subset().size() != shape_dims.size()) {
            EXPECT_EQ(edge.subset().size(), shape_dims.size())
                << "Output subset size is not " << shape_dims.size() << " for CastNode";
        }
        // Check that output type is the target type
        EXPECT_EQ(edge.result_type(sdfg)->primitive_type(), TargetType);
    }
}

#define REGISTER_CAST_TEST(SourceType, TargetType, Dim)                                     \
    TEST(ElementWiseTest, CastNode_##SourceType##_to_##TargetType##_##Dim##D) {             \
        std::vector<size_t> dims;                                                           \
        for (int i = 0; i < Dim; ++i) dims.push_back(32);                                   \
        TestCast<types::PrimitiveType::SourceType, types::PrimitiveType::TargetType>(dims); \
    }

// Register cast tests for various type conversions
REGISTER_CAST_TEST(Int32, Float, 1)
REGISTER_CAST_TEST(Int32, Float, 2)
REGISTER_CAST_TEST(Int32, Float, 3)
REGISTER_CAST_TEST(Int32, Float, 4)

REGISTER_CAST_TEST(Float, Int32, 1)
REGISTER_CAST_TEST(Float, Int32, 2)
REGISTER_CAST_TEST(Float, Int32, 3)
REGISTER_CAST_TEST(Float, Int32, 4)

REGISTER_CAST_TEST(Float, Double, 1)
REGISTER_CAST_TEST(Float, Double, 2)
REGISTER_CAST_TEST(Float, Double, 3)
REGISTER_CAST_TEST(Float, Double, 4)

REGISTER_CAST_TEST(Double, Float, 1)
REGISTER_CAST_TEST(Double, Float, 2)
REGISTER_CAST_TEST(Double, Float, 3)
REGISTER_CAST_TEST(Double, Float, 4)

REGISTER_CAST_TEST(Int32, Int64, 1)
REGISTER_CAST_TEST(Int32, Int64, 2)
REGISTER_CAST_TEST(Int32, Int64, 3)
REGISTER_CAST_TEST(Int32, Int64, 4)

REGISTER_CAST_TEST(Int64, Int32, 1)
REGISTER_CAST_TEST(Int64, Int32, 2)
REGISTER_CAST_TEST(Int64, Int32, 3)
REGISTER_CAST_TEST(Int64, Int32, 4)
