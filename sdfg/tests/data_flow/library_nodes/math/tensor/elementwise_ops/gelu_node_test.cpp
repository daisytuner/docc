#include "sdfg/data_flow/library_nodes/math/tensor/elementwise_ops/gelu_node.h"

#include <memory>

#include <gtest/gtest.h>
#include <nlohmann/json_fwd.hpp>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_nodes/math/cmath/cmath_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/tensor_layout.h"
#include "sdfg/element.h"
#include "sdfg/function.h"
#include "sdfg/passes/expansion/library_node_expansion_pass.h"
#include "sdfg/serializer/json_serializer.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/structured_sdfg.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/pointer.h"
#include "sdfg/types/tensor.h"
#include "sdfg/types/type.h"
#include "sdfg_debug_dump.h"

using namespace sdfg;

TEST(GELUNodeTest, expansion_precise) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("A", desc, true);
    builder.add_container("B", desc, true);

    symbolic::MultiExpression shape = {symbolic::integer(32)};
    math::tensor::TensorLayout A_layout(shape);
    types::Tensor A_tensor(base_desc, A_layout);
    math::tensor::TensorLayout B_layout(shape);
    types::Tensor B_tensor(base_desc, B_layout);

    auto& block = builder.add_block(root);
    auto& A_access = builder.add_access(block, "A");
    auto& B_access = builder.add_access(block, "B");
    auto& libnode = builder.add_library_node<math::tensor::GELUNode>(block, DebugInfo(), shape, false);
    builder.add_computational_memlet(block, A_access, libnode, "X", {}, A_tensor);
    builder.add_computational_memlet(block, B_access, libnode, "Y", {}, B_tensor);

    ASSERT_NO_THROW(sdfg.validate());
    dump_sdfg(sdfg, "0.before");

    passes::LibNodeExpansionVisitor::NodeOutcome outcome =
        passes::expansion::expand_single_math_node(builder, block, libnode);
    EXPECT_TRUE(outcome.expanded);
    EXPECT_TRUE(outcome.block_removed);

    ASSERT_NO_THROW(sdfg.validate());
    dump_sdfg(sdfg, "1.after");

    ASSERT_EQ(root.size(), 1);
    auto* new_seq = dyn_cast<structured_control_flow::Sequence*>(&root.at(0));
    ASSERT_NE(new_seq, nullptr);

    ASSERT_EQ(new_seq->size(), 1);
    auto* map = dyn_cast<structured_control_flow::Map*>(&new_seq->at(0));
    ASSERT_NE(map, nullptr);
    EXPECT_TRUE(symbolic::eq(map->num_iterations(), shape[0]));

    ASSERT_EQ(map->root().size(), 1);
    auto* new_block = dyn_cast<structured_control_flow::Block*>(&map->root().at(0));
    ASSERT_NE(new_block, nullptr);
    auto& dfg = new_block->dataflow();
    ASSERT_EQ(dfg.library_nodes().size(), 1);
    auto* new_libnode = *dfg.library_nodes().begin();
    auto* cmath_node = dynamic_cast<math::cmath::CMathNode*>(new_libnode);
    ASSERT_NE(cmath_node, nullptr);
    EXPECT_EQ(cmath_node->function(), math::cmath::CMathFunction::erf);
}

TEST(GELUNodeTest, expansion_tanh_approx) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("A", desc, true);
    builder.add_container("B", desc, true);

    symbolic::MultiExpression shape = {symbolic::integer(32)};
    math::tensor::TensorLayout A_layout(shape);
    types::Tensor A_tensor(base_desc, A_layout);
    math::tensor::TensorLayout B_layout(shape);
    types::Tensor B_tensor(base_desc, B_layout);

    auto& block = builder.add_block(root);
    auto& A_access = builder.add_access(block, "A");
    auto& B_access = builder.add_access(block, "B");
    auto& libnode = builder.add_library_node<math::tensor::GELUNode>(block, DebugInfo(), shape, true);
    builder.add_computational_memlet(block, A_access, libnode, "X", {}, A_tensor);
    builder.add_computational_memlet(block, B_access, libnode, "Y", {}, B_tensor);

    ASSERT_NO_THROW(sdfg.validate());
    dump_sdfg(sdfg, "0.before");

    passes::LibNodeExpansionVisitor::NodeOutcome outcome =
        passes::expansion::expand_single_math_node(builder, block, libnode);
    EXPECT_TRUE(outcome.expanded);
    EXPECT_TRUE(outcome.block_removed);

    ASSERT_NO_THROW(sdfg.validate());
    dump_sdfg(sdfg, "1.after");

    ASSERT_EQ(root.size(), 1);
    auto* new_seq = dyn_cast<structured_control_flow::Sequence*>(&root.at(0));
    ASSERT_NE(new_seq, nullptr);

    ASSERT_EQ(new_seq->size(), 1);
    auto* map = dyn_cast<structured_control_flow::Map*>(&new_seq->at(0));
    ASSERT_NE(map, nullptr);
    EXPECT_TRUE(symbolic::eq(map->num_iterations(), shape[0]));

    ASSERT_EQ(map->root().size(), 1);
    auto* new_block = dyn_cast<structured_control_flow::Block*>(&map->root().at(0));
    ASSERT_NE(new_block, nullptr);
    auto& dfg = new_block->dataflow();
    ASSERT_EQ(dfg.library_nodes().size(), 1);
    auto* new_libnode = *dfg.library_nodes().begin();
    auto* cmath_node = dynamic_cast<math::cmath::CMathNode*>(new_libnode);
    ASSERT_NE(cmath_node, nullptr);
    EXPECT_EQ(cmath_node->function(), math::cmath::CMathFunction::tanh);
}

TEST(GELUNodeTest, serialization_precise) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("A", desc, true);
    builder.add_container("B", desc, true);

    symbolic::MultiExpression shape = {symbolic::integer(32)};
    math::tensor::TensorLayout A_layout(shape);
    types::Tensor A_tensor(base_desc, A_layout);
    math::tensor::TensorLayout B_layout(shape);
    types::Tensor B_tensor(base_desc, B_layout);

    auto& block = builder.add_block(root);
    auto& A_access = builder.add_access(block, "A");
    auto& B_access = builder.add_access(block, "B");
    auto& libnode = builder.add_library_node<math::tensor::GELUNode>(block, DebugInfo(), shape, false);
    builder.add_computational_memlet(block, A_access, libnode, "X", {}, A_tensor);
    builder.add_computational_memlet(block, B_access, libnode, "Y", {}, B_tensor);

    ASSERT_NO_THROW(sdfg.validate());
    dump_sdfg(sdfg, "0.before");

    serializer::JSONSerializer serializer;
    nlohmann::json j;
    ASSERT_NO_THROW(j = serializer.serialize(sdfg));

    std::unique_ptr<StructuredSDFG> new_sdfg;
    ASSERT_NO_THROW(new_sdfg = serializer.deserialize(j));

    ASSERT_NO_THROW(new_sdfg->validate());
    dump_sdfg(*new_sdfg, "1.after");

    ASSERT_EQ(new_sdfg->root().size(), 1);
    auto* new_block = dyn_cast<structured_control_flow::Block*>(&new_sdfg->root().at(0));
    ASSERT_NE(new_block, nullptr);
    auto& dfg = new_block->dataflow();
    ASSERT_EQ(dfg.library_nodes().size(), 1);
    auto* new_libnode = *dfg.library_nodes().begin();
    auto* new_gelu_node = dynamic_cast<math::tensor::GELUNode*>(new_libnode);
    ASSERT_NE(new_gelu_node, nullptr);
    EXPECT_FALSE(new_gelu_node->tanh_approx());
}

TEST(GELUNodeTest, serialization_tanh_approx) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("A", desc, true);
    builder.add_container("B", desc, true);

    symbolic::MultiExpression shape = {symbolic::integer(32)};
    math::tensor::TensorLayout A_layout(shape);
    types::Tensor A_tensor(base_desc, A_layout);
    math::tensor::TensorLayout B_layout(shape);
    types::Tensor B_tensor(base_desc, B_layout);

    auto& block = builder.add_block(root);
    auto& A_access = builder.add_access(block, "A");
    auto& B_access = builder.add_access(block, "B");
    auto& libnode = builder.add_library_node<math::tensor::GELUNode>(block, DebugInfo(), shape, true);
    builder.add_computational_memlet(block, A_access, libnode, "X", {}, A_tensor);
    builder.add_computational_memlet(block, B_access, libnode, "Y", {}, B_tensor);

    ASSERT_NO_THROW(sdfg.validate());
    dump_sdfg(sdfg, "0.before");

    serializer::JSONSerializer serializer;
    nlohmann::json j;
    ASSERT_NO_THROW(j = serializer.serialize(sdfg));

    std::unique_ptr<StructuredSDFG> new_sdfg;
    ASSERT_NO_THROW(new_sdfg = serializer.deserialize(j));

    ASSERT_NO_THROW(new_sdfg->validate());
    dump_sdfg(*new_sdfg, "1.after");

    ASSERT_EQ(new_sdfg->root().size(), 1);
    auto* new_block = dyn_cast<structured_control_flow::Block*>(&new_sdfg->root().at(0));
    ASSERT_NE(new_block, nullptr);
    auto& dfg = new_block->dataflow();
    ASSERT_EQ(dfg.library_nodes().size(), 1);
    auto* new_libnode = *dfg.library_nodes().begin();
    auto* new_gelu_node = dynamic_cast<math::tensor::GELUNode*>(new_libnode);
    ASSERT_NE(new_gelu_node, nullptr);
    EXPECT_TRUE(new_gelu_node->tanh_approx());
}
