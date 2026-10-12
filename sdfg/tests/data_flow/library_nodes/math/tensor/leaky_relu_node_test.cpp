#include "sdfg/data_flow/library_nodes/math/tensor/leaky_relu_node.h"

#include <memory>

#include <gtest/gtest.h>
#include <nlohmann/json_fwd.hpp>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_nodes/math/tensor/tensor_layout.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/element.h"
#include "sdfg/function.h"
#include "sdfg/passes/expansion/library_node_expansion_pass.h"
#include "sdfg/serializer/json_serializer.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/if_else.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/structured_sdfg.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/pointer.h"
#include "sdfg/types/scalar.h"
#include "sdfg/types/tensor.h"
#include "sdfg_debug_dump.h"

using namespace sdfg;

namespace {

enum class AlphaKind { Constant, Container, Tensor, Missing };

struct LeakyReLUGraph {
    structured_control_flow::Block& block;
    data_flow::LibraryNode& libnode;
};

LeakyReLUGraph add_leaky_relu(
    builder::StructuredSDFGBuilder& builder,
    const symbolic::MultiExpression& shape,
    const symbolic::MultiExpression& x_shape,
    types::PrimitiveType prim,
    AlphaKind alpha_kind
) {
    auto& root = builder.subject().root();

    types::Scalar base_desc(prim);
    types::Pointer desc(base_desc);
    builder.add_container("X", desc, true);
    builder.add_container("Y", desc, true);
    types::Tensor x_tensor(base_desc, math::tensor::TensorLayout(x_shape));
    types::Tensor y_tensor(base_desc, math::tensor::TensorLayout(shape));

    auto& block = builder.add_block(root);
    auto& x_access = builder.add_access(block, "X");
    auto& y_access = builder.add_access(block, "Y");
    auto& libnode = builder.add_library_node<math::tensor::LeakyReLUNode>(block, DebugInfo(), shape);
    builder.add_computational_memlet(block, x_access, libnode, "X", {}, x_tensor);
    builder.add_computational_memlet(block, y_access, libnode, "Y", {}, y_tensor);

    switch (alpha_kind) {
        case AlphaKind::Constant: {
            auto& alpha = builder.add_constant(block, "0.01", base_desc);
            builder.add_computational_memlet(block, alpha, libnode, "alpha", {}, base_desc);
            break;
        }
        case AlphaKind::Container: {
            builder.add_container("alpha", base_desc, true);
            auto& alpha = builder.add_access(block, "alpha");
            builder.add_computational_memlet(block, alpha, libnode, "alpha", {}, base_desc);
            break;
        }
        case AlphaKind::Tensor: {
            builder.add_container("alpha", desc, true);
            auto& alpha = builder.add_access(block, "alpha");
            builder.add_computational_memlet(block, alpha, libnode, "alpha", {}, y_tensor);
            break;
        }
        case AlphaKind::Missing:
            break;
    }

    return {block, libnode};
}

void check_expansion(AlphaKind alpha_kind) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    symbolic::MultiExpression shape({symbolic::integer(4), symbolic::integer(8)});
    auto graph = add_leaky_relu(builder, shape, shape, types::PrimitiveType::Float, alpha_kind);

    ASSERT_NO_THROW(sdfg.validate());
    dump_sdfg(sdfg, "0.before");

    passes::expansion::NodeOutcome outcome =
        passes::expansion::expand_single_math_node(builder, graph.block, graph.libnode);
    EXPECT_TRUE(outcome.expanded);
    EXPECT_TRUE(outcome.block_removed);

    ASSERT_NO_THROW(sdfg.validate());
    dump_sdfg(sdfg, "1.after");

    ASSERT_EQ(root.size(), 1);
    auto* new_seq = dyn_cast<structured_control_flow::Sequence*>(&root.at(0));
    ASSERT_NE(new_seq, nullptr);
    ASSERT_EQ(new_seq->size(), 1);
    auto* outer_map = dyn_cast<structured_control_flow::Map*>(&new_seq->at(0));
    ASSERT_NE(outer_map, nullptr);
    EXPECT_TRUE(symbolic::eq(outer_map->num_iterations(), shape[0]));
    ASSERT_EQ(outer_map->root().size(), 1);
    auto* inner_map = dyn_cast<structured_control_flow::Map*>(&outer_map->root().at(0));
    ASSERT_NE(inner_map, nullptr);
    EXPECT_TRUE(symbolic::eq(inner_map->num_iterations(), shape[1]));

    auto& body = inner_map->root();
    ASSERT_EQ(body.size(), 2);
    auto* load_block = dyn_cast<structured_control_flow::Block*>(&body.at(0));
    ASSERT_NE(load_block, nullptr);
    bool has_compare = false;
    for (auto* tasklet : load_block->dataflow().tasklets()) {
        has_compare |= tasklet->code() == data_flow::TaskletCode::fp_ogt;
    }
    EXPECT_TRUE(has_compare);

    auto* if_else = dyn_cast<structured_control_flow::IfElse*>(&body.at(1));
    ASSERT_NE(if_else, nullptr);
    ASSERT_EQ(if_else->size(), 2);

    auto expect_single_tasklet = [](structured_control_flow::Sequence& branch, data_flow::TaskletCode code) {
        ASSERT_EQ(branch.size(), 1);
        auto* branch_block = dyn_cast<structured_control_flow::Block*>(&branch.at(0));
        ASSERT_NE(branch_block, nullptr);
        auto tasklets = branch_block->dataflow().tasklets();
        ASSERT_EQ(tasklets.size(), 1);
        EXPECT_EQ((*tasklets.begin())->code(), code);
    };
    expect_single_tasklet(if_else->at(0).first, data_flow::TaskletCode::assign);
    expect_single_tasklet(if_else->at(1).first, data_flow::TaskletCode::fp_mul);
}

} // namespace

TEST(LeakyReLUNodeTest, symbolic) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();

    types::Scalar sym_desc(types::PrimitiveType::Int64);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);
    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");

    symbolic::MultiExpression shape({i, j});
    auto graph = add_leaky_relu(builder, shape, shape, types::PrimitiveType::Float, AlphaKind::Constant);
    ASSERT_NO_THROW(sdfg.validate());

    auto& node = static_cast<math::tensor::LeakyReLUNode&>(graph.libnode);
    EXPECT_FALSE(node.supports_integer_types());
    auto symbols = node.symbols();
    EXPECT_EQ(symbols.size(), 2);
    EXPECT_TRUE(symbols.contains(i));
    EXPECT_TRUE(symbols.contains(j));

    builder.add_container("k", sym_desc);
    auto k = symbolic::symbol("k");
    node.replace(i, k);
    symbols = node.symbols();
    EXPECT_EQ(symbols.size(), 2);
    EXPECT_TRUE(symbols.contains(k));
    EXPECT_TRUE(symbols.contains(j));
}

TEST(LeakyReLUNodeTest, expansion_constant_alpha) {
    check_expansion(AlphaKind::Constant);
}

TEST(LeakyReLUNodeTest, expansion_container_alpha) {
    check_expansion(AlphaKind::Container);
}

TEST(LeakyReLUNodeTest, serialization) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();

    symbolic::MultiExpression shape({symbolic::integer(4), symbolic::integer(8)});
    add_leaky_relu(builder, shape, shape, types::PrimitiveType::Double, AlphaKind::Constant);
    ASSERT_NO_THROW(sdfg.validate());

    serializer::JSONSerializer serializer;
    nlohmann::json j;
    ASSERT_NO_THROW(j = serializer.serialize(sdfg));

    std::unique_ptr<StructuredSDFG> new_sdfg;
    ASSERT_NO_THROW(new_sdfg = serializer.deserialize(j));
    ASSERT_NO_THROW(new_sdfg->validate());

    ASSERT_EQ(new_sdfg->root().size(), 1);
    auto* new_block = dyn_cast<structured_control_flow::Block*>(&new_sdfg->root().at(0));
    ASSERT_NE(new_block, nullptr);
    auto& dfg = new_block->dataflow();
    ASSERT_EQ(dfg.library_nodes().size(), 1);
    auto* new_node = dynamic_cast<math::tensor::LeakyReLUNode*>(*dfg.library_nodes().begin());
    ASSERT_NE(new_node, nullptr);
    ASSERT_EQ(new_node->shape().size(), 2);
    EXPECT_TRUE(symbolic::eq(new_node->shape()[0], shape[0]));
    EXPECT_TRUE(symbolic::eq(new_node->shape()[1], shape[1]));
}

TEST(LeakyReLUNodeTest, validate_missing_alpha) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    symbolic::MultiExpression shape({symbolic::integer(4)});
    add_leaky_relu(builder, shape, shape, types::PrimitiveType::Float, AlphaKind::Missing);
    EXPECT_THROW(builder.subject().validate(), InvalidSDFGException);
}

TEST(LeakyReLUNodeTest, validate_tensor_alpha) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    symbolic::MultiExpression shape({symbolic::integer(4)});
    add_leaky_relu(builder, shape, shape, types::PrimitiveType::Float, AlphaKind::Tensor);
    EXPECT_THROW(builder.subject().validate(), InvalidSDFGException);
}

TEST(LeakyReLUNodeTest, validate_shape_mismatch) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    symbolic::MultiExpression shape({symbolic::integer(4)});
    symbolic::MultiExpression x_shape({symbolic::integer(5)});
    add_leaky_relu(builder, shape, x_shape, types::PrimitiveType::Float, AlphaKind::Constant);
    EXPECT_THROW(builder.subject().validate(), InvalidSDFGException);
}

TEST(LeakyReLUNodeTest, validate_integer_type) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    symbolic::MultiExpression shape({symbolic::integer(4)});
    add_leaky_relu(builder, shape, shape, types::PrimitiveType::Int32, AlphaKind::Constant);
    EXPECT_THROW(builder.subject().validate(), InvalidSDFGException);
}
