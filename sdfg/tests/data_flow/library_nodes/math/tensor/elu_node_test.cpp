#include "sdfg/data_flow/library_nodes/math/tensor/elu_node.h"

#include <memory>

#include <gtest/gtest.h>
#include <nlohmann/json_fwd.hpp>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_nodes/math/cmath/cmath_node.h"
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

enum class ParamKind { Constant, Container, Tensor, Missing };

struct EluGraph {
    structured_control_flow::Block& block;
    data_flow::LibraryNode& libnode;
};

EluGraph add_elu(
    builder::StructuredSDFGBuilder& builder,
    const symbolic::MultiExpression& shape,
    const symbolic::MultiExpression& x_shape,
    types::PrimitiveType prim,
    ParamKind scale_kind
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
    auto& libnode = builder.add_library_node<math::tensor::EluNode>(block, DebugInfo(), shape);
    builder.add_computational_memlet(block, x_access, libnode, "X", {}, x_tensor);
    builder.add_computational_memlet(block, y_access, libnode, "Y", {}, y_tensor);

    auto& alpha = builder.add_constant(block, "1.0", base_desc);
    builder.add_computational_memlet(block, alpha, libnode, "alpha", {}, base_desc);
    auto& input_scale = builder.add_constant(block, "1.0", base_desc);
    builder.add_computational_memlet(block, input_scale, libnode, "input_scale", {}, base_desc);

    switch (scale_kind) {
        case ParamKind::Constant: {
            auto& scale = builder.add_constant(block, "1.0507009873554805", base_desc);
            builder.add_computational_memlet(block, scale, libnode, "scale", {}, base_desc);
            break;
        }
        case ParamKind::Container: {
            builder.add_container("scale", base_desc, true);
            auto& scale = builder.add_access(block, "scale");
            builder.add_computational_memlet(block, scale, libnode, "scale", {}, base_desc);
            break;
        }
        case ParamKind::Tensor: {
            builder.add_container("scale", desc, true);
            auto& scale = builder.add_access(block, "scale");
            builder.add_computational_memlet(block, scale, libnode, "scale", {}, y_tensor);
            break;
        }
        case ParamKind::Missing:
            break;
    }

    return {block, libnode};
}

void check_expansion(ParamKind scale_kind) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    symbolic::MultiExpression shape({symbolic::integer(4), symbolic::integer(8)});
    auto graph = add_elu(builder, shape, shape, types::PrimitiveType::Float, scale_kind);

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
        has_compare |= tasklet->code() == data_flow::TaskletCode::fp_olt;
    }
    EXPECT_TRUE(has_compare);

    auto* if_else = dyn_cast<structured_control_flow::IfElse*>(&body.at(1));
    ASSERT_NE(if_else, nullptr);
    ASSERT_EQ(if_else->size(), 2);

    auto& neg_branch = if_else->at(0).first;
    ASSERT_EQ(neg_branch.size(), 1);
    auto* neg_block = dyn_cast<structured_control_flow::Block*>(&neg_branch.at(0));
    ASSERT_NE(neg_block, nullptr);
    auto neg_tasklets = neg_block->dataflow().tasklets();
    EXPECT_EQ(neg_tasklets.size(), 3);
    for (auto* tasklet : neg_tasklets) {
        EXPECT_EQ(tasklet->code(), data_flow::TaskletCode::fp_mul);
    }
    auto neg_libnodes = neg_block->dataflow().library_nodes();
    ASSERT_EQ(neg_libnodes.size(), 1);
    auto* expm1 = dynamic_cast<math::cmath::CMathNode*>(*neg_libnodes.begin());
    ASSERT_NE(expm1, nullptr);
    EXPECT_EQ(expm1->function(), math::cmath::CMathFunction::expm1);

    auto& pos_branch = if_else->at(1).first;
    ASSERT_EQ(pos_branch.size(), 1);
    auto* pos_block = dyn_cast<structured_control_flow::Block*>(&pos_branch.at(0));
    ASSERT_NE(pos_block, nullptr);
    auto pos_tasklets = pos_block->dataflow().tasklets();
    ASSERT_EQ(pos_tasklets.size(), 1);
    EXPECT_EQ((*pos_tasklets.begin())->code(), data_flow::TaskletCode::fp_mul);
}

} // namespace

TEST(EluNodeTest, symbolic) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();

    types::Scalar sym_desc(types::PrimitiveType::Int64);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);
    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");

    symbolic::MultiExpression shape({i, j});
    auto graph = add_elu(builder, shape, shape, types::PrimitiveType::Float, ParamKind::Constant);
    ASSERT_NO_THROW(sdfg.validate());

    auto& node = static_cast<math::tensor::EluNode&>(graph.libnode);
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

TEST(EluNodeTest, expansion_constant_params) {
    check_expansion(ParamKind::Constant);
}

TEST(EluNodeTest, expansion_container_params) {
    check_expansion(ParamKind::Container);
}

TEST(EluNodeTest, serialization) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();

    symbolic::MultiExpression shape({symbolic::integer(4), symbolic::integer(8)});
    add_elu(builder, shape, shape, types::PrimitiveType::Double, ParamKind::Constant);
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
    auto* new_node = dynamic_cast<math::tensor::EluNode*>(*dfg.library_nodes().begin());
    ASSERT_NE(new_node, nullptr);
    ASSERT_EQ(new_node->shape().size(), 2);
    EXPECT_TRUE(symbolic::eq(new_node->shape()[0], shape[0]));
    EXPECT_TRUE(symbolic::eq(new_node->shape()[1], shape[1]));
}

TEST(EluNodeTest, validate_missing_scale) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    symbolic::MultiExpression shape({symbolic::integer(4)});
    add_elu(builder, shape, shape, types::PrimitiveType::Float, ParamKind::Missing);
    EXPECT_THROW(builder.subject().validate(), InvalidSDFGException);
}

TEST(EluNodeTest, validate_tensor_scale) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    symbolic::MultiExpression shape({symbolic::integer(4)});
    add_elu(builder, shape, shape, types::PrimitiveType::Float, ParamKind::Tensor);
    EXPECT_THROW(builder.subject().validate(), InvalidSDFGException);
}

TEST(EluNodeTest, validate_shape_mismatch) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    symbolic::MultiExpression shape({symbolic::integer(4)});
    symbolic::MultiExpression x_shape({symbolic::integer(5)});
    add_elu(builder, shape, x_shape, types::PrimitiveType::Float, ParamKind::Constant);
    EXPECT_THROW(builder.subject().validate(), InvalidSDFGException);
}

TEST(EluNodeTest, validate_integer_type) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    symbolic::MultiExpression shape({symbolic::integer(4)});
    add_elu(builder, shape, shape, types::PrimitiveType::Int32, ParamKind::Constant);
    EXPECT_THROW(builder.subject().validate(), InvalidSDFGException);
}
