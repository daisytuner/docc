#include "sdfg/data_flow/library_nodes/math/tensor/clamp_node.h"

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

struct ClampGraph {
    structured_control_flow::Block& block;
    data_flow::LibraryNode& libnode;
};

ClampGraph add_clamp(
    builder::StructuredSDFGBuilder& builder,
    const symbolic::MultiExpression& shape,
    const symbolic::MultiExpression& x_shape,
    types::PrimitiveType prim,
    bool has_min,
    bool has_max,
    bool tensor_max = false
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
    auto& libnode = builder.add_library_node<math::tensor::ClampNode>(block, DebugInfo(), shape, has_min, has_max);
    builder.add_computational_memlet(block, x_access, libnode, "X", {}, x_tensor);
    builder.add_computational_memlet(block, y_access, libnode, "Y", {}, y_tensor);

    if (has_min) {
        auto& min = builder.add_constant(block, "0", base_desc);
        builder.add_computational_memlet(block, min, libnode, "min", {}, base_desc);
    }
    if (has_max) {
        if (tensor_max) {
            builder.add_container("max", desc, true);
            auto& max = builder.add_access(block, "max");
            builder.add_computational_memlet(block, max, libnode, "max", {}, y_tensor);
        } else {
            builder.add_container("max", base_desc, true);
            auto& max = builder.add_access(block, "max");
            builder.add_computational_memlet(block, max, libnode, "max", {}, base_desc);
        }
    }

    return {block, libnode};
}

// Returns the body of the innermost map of the expanded node.
structured_control_flow::Sequence& expand_and_get_body(
    builder::StructuredSDFGBuilder& builder, ClampGraph& graph, const symbolic::MultiExpression& shape
) {
    auto& sdfg = builder.subject();
    EXPECT_NO_THROW(sdfg.validate());
    dump_sdfg(sdfg, "0.before");

    passes::expansion::NodeOutcome outcome =
        passes::expansion::expand_single_math_node(builder, graph.block, graph.libnode);
    EXPECT_TRUE(outcome.expanded);
    EXPECT_TRUE(outcome.block_removed);

    EXPECT_NO_THROW(sdfg.validate());
    dump_sdfg(sdfg, "1.after");

    auto& root = sdfg.root();
    EXPECT_EQ(root.size(), 1);
    auto* current = dyn_cast<structured_control_flow::Sequence*>(&root.at(0));
    for (const auto& dim : shape) {
        EXPECT_EQ(current->size(), 1);
        auto* map = dyn_cast<structured_control_flow::Map*>(&current->at(0));
        EXPECT_NE(map, nullptr);
        EXPECT_TRUE(symbolic::eq(map->num_iterations(), dim));
        current = &map->root();
    }
    return *current;
}

bool block_has_tasklet(structured_control_flow::ControlFlowNode& node, data_flow::TaskletCode code) {
    auto* block = dyn_cast<structured_control_flow::Block*>(&node);
    if (!block) {
        return false;
    }
    for (auto* tasklet : block->dataflow().tasklets()) {
        if (tasklet->code() == code) {
            return true;
        }
    }
    return false;
}

} // namespace

TEST(ClampNodeTest, symbolic) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();

    types::Scalar sym_desc(types::PrimitiveType::Int64);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);
    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");

    symbolic::MultiExpression shape({i, j});
    auto graph = add_clamp(builder, shape, shape, types::PrimitiveType::Float, true, true);
    ASSERT_NO_THROW(sdfg.validate());

    auto& node = static_cast<math::tensor::ClampNode&>(graph.libnode);
    EXPECT_TRUE(node.supports_integer_types());
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

TEST(ClampNodeTest, expansion_min_max_float) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    symbolic::MultiExpression shape({symbolic::integer(4), symbolic::integer(8)});
    auto graph = add_clamp(builder, shape, shape, types::PrimitiveType::Float, true, true);
    auto& body = expand_and_get_body(builder, graph, shape);

    // load, lower compare, lower branch, upper compare, upper branch, store
    ASSERT_EQ(body.size(), 6);
    EXPECT_TRUE(block_has_tasklet(body.at(1), data_flow::TaskletCode::fp_olt));
    EXPECT_TRUE(block_has_tasklet(body.at(1), data_flow::TaskletCode::fp_uno));
    EXPECT_NE(dyn_cast<structured_control_flow::IfElse*>(&body.at(2)), nullptr);
    EXPECT_TRUE(block_has_tasklet(body.at(3), data_flow::TaskletCode::fp_ogt));
    EXPECT_TRUE(block_has_tasklet(body.at(3), data_flow::TaskletCode::fp_uno));
    EXPECT_NE(dyn_cast<structured_control_flow::IfElse*>(&body.at(4)), nullptr);
    EXPECT_TRUE(block_has_tasklet(body.at(5), data_flow::TaskletCode::assign));
}

TEST(ClampNodeTest, expansion_min_only_signed_int) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    symbolic::MultiExpression shape({symbolic::integer(8)});
    auto graph = add_clamp(builder, shape, shape, types::PrimitiveType::Int32, true, false);
    auto& body = expand_and_get_body(builder, graph, shape);

    ASSERT_EQ(body.size(), 4);
    EXPECT_TRUE(block_has_tasklet(body.at(1), data_flow::TaskletCode::int_slt));
    EXPECT_FALSE(block_has_tasklet(body.at(1), data_flow::TaskletCode::fp_uno));
    EXPECT_NE(dyn_cast<structured_control_flow::IfElse*>(&body.at(2)), nullptr);
}

TEST(ClampNodeTest, expansion_max_only_unsigned_int) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    symbolic::MultiExpression shape({symbolic::integer(8)});
    auto graph = add_clamp(builder, shape, shape, types::PrimitiveType::UInt16, false, true);
    auto& body = expand_and_get_body(builder, graph, shape);

    ASSERT_EQ(body.size(), 4);
    EXPECT_TRUE(block_has_tasklet(body.at(1), data_flow::TaskletCode::int_ugt));
    EXPECT_NE(dyn_cast<structured_control_flow::IfElse*>(&body.at(2)), nullptr);
}

TEST(ClampNodeTest, serialization) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();

    symbolic::MultiExpression shape({symbolic::integer(4), symbolic::integer(8)});
    add_clamp(builder, shape, shape, types::PrimitiveType::Double, false, true);
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
    auto* new_node = dynamic_cast<math::tensor::ClampNode*>(*dfg.library_nodes().begin());
    ASSERT_NE(new_node, nullptr);
    EXPECT_FALSE(new_node->has_min());
    EXPECT_TRUE(new_node->has_max());
    ASSERT_EQ(new_node->shape().size(), 2);
    EXPECT_TRUE(symbolic::eq(new_node->shape()[0], shape[0]));
    EXPECT_TRUE(symbolic::eq(new_node->shape()[1], shape[1]));
}

TEST(ClampNodeTest, validate_no_bounds) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    symbolic::MultiExpression shape({symbolic::integer(4)});
    add_clamp(builder, shape, shape, types::PrimitiveType::Float, false, false);
    EXPECT_THROW(builder.subject().validate(), InvalidSDFGException);
}

TEST(ClampNodeTest, validate_tensor_bound) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    symbolic::MultiExpression shape({symbolic::integer(4)});
    add_clamp(builder, shape, shape, types::PrimitiveType::Float, true, true, true);
    EXPECT_THROW(builder.subject().validate(), InvalidSDFGException);
}

TEST(ClampNodeTest, validate_shape_mismatch) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    symbolic::MultiExpression shape({symbolic::integer(4)});
    symbolic::MultiExpression x_shape({symbolic::integer(5)});
    add_clamp(builder, shape, x_shape, types::PrimitiveType::Float, true, true);
    EXPECT_THROW(builder.subject().validate(), InvalidSDFGException);
}
