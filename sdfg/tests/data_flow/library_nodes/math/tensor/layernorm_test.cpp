#include "sdfg/data_flow/library_nodes/math/tensor/layernorm_node.h"

#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_nodes/math/tensor/tensor_layout.h"
#include "sdfg/passes/expansion/library_node_expansion_pass.h"
#include "sdfg/serializer/json_serializer.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_sdfg.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/pointer.h"
#include "sdfg/types/scalar.h"
#include "sdfg/types/tensor.h"
#include "sdfg_debug_dump.h"

using namespace sdfg;

namespace {

// Builds a StructuredSDFG containing a single LayerNormNode over `shape`, normalizing over the
// trailing `num_normalized_dims` dimensions. Gamma/Beta memlets are wired only when `affine` /
// `has_bias` are set, matching the node's conditional connector layout. Returns the node and the
// block it lives in so callers can expand it.
struct LayerNormFixture {
    math::tensor::LayerNormNode* node;
    structured_control_flow::Block* block;
};

LayerNormFixture build_layernorm(
    builder::StructuredSDFGBuilder& builder,
    const symbolic::MultiExpression& shape,
    size_t num_normalized_dims,
    bool affine,
    bool has_bias
) {
    auto& sdfg = builder.subject();

    types::Scalar elem(types::PrimitiveType::Float);
    types::Pointer ptr(elem);

    builder.add_container("X", ptr, true);
    builder.add_container("Y_out", ptr, true);
    if (affine) {
        builder.add_container("Gamma", ptr, true);
    }
    if (has_bias) {
        builder.add_container("Beta", ptr, true);
    }

    // Trailing (normalized) shape used for Gamma/Beta.
    symbolic::MultiExpression trailing(shape.end() - num_normalized_dims, shape.end());

    types::Tensor x_tensor(elem.primitive_type(), shape);
    types::Tensor y_tensor(elem.primitive_type(), shape);
    types::Tensor affine_tensor(elem.primitive_type(), trailing);

    auto& block = builder.add_block(sdfg.root());

    auto& x_access = builder.add_access(block, "X");
    auto& y_access = builder.add_access(block, "Y_out");
    auto& eps_access = builder.add_constant(block, "0.00001", elem);

    auto& node = dynamic_cast<math::tensor::LayerNormNode&>(builder.add_library_node<math::tensor::LayerNormNode>(
        block, DebugInfo(), math::tensor::TensorLayout(shape), types::Float, num_normalized_dims, affine, has_bias
    ));

    builder.add_computational_memlet(block, x_access, node, "X", {}, x_tensor, block.debug_info());
    if (affine) {
        auto& gamma_access = builder.add_access(block, "Gamma");
        builder.add_computational_memlet(block, gamma_access, node, "Gamma", {}, affine_tensor, block.debug_info());
    }
    if (has_bias) {
        auto& beta_access = builder.add_access(block, "Beta");
        builder.add_computational_memlet(block, beta_access, node, "Beta", {}, affine_tensor, block.debug_info());
    }
    builder.add_computational_memlet(block, eps_access, node, "epsilon", {}, elem, block.debug_info());
    builder.add_computational_memlet(block, y_access, node, "Y_out", {}, y_tensor, block.debug_info());

    return {&node, &block};
}

} // namespace

// --- API / construction ---

TEST(LayerNormTest, Connectors_AffineBias) {
    builder::StructuredSDFGBuilder builder("sdfg_ln_conn_affine_bias", FunctionType_CPU);
    symbolic::MultiExpression shape = {symbolic::integer(2), symbolic::integer(16)};
    auto fx = build_layernorm(builder, shape, 1, /*affine=*/true, /*has_bias=*/true);

    EXPECT_TRUE(fx.node->affine());
    EXPECT_TRUE(fx.node->has_bias());
    EXPECT_EQ(fx.node->num_normalized_dims(), 1u);
    EXPECT_EQ(fx.node->quantization(), types::PrimitiveType::Float);

    std::vector<std::string> expected = {"X", "Gamma", "Beta", "epsilon", "Y_out"};
    EXPECT_EQ(fx.node->inputs(), expected);
}

TEST(LayerNormTest, Connectors_NoAffine) {
    builder::StructuredSDFGBuilder builder("sdfg_ln_conn_no_affine", FunctionType_CPU);
    symbolic::MultiExpression shape = {symbolic::integer(2), symbolic::integer(16)};
    auto fx = build_layernorm(builder, shape, 1, /*affine=*/false, /*has_bias=*/false);

    EXPECT_FALSE(fx.node->affine());
    EXPECT_FALSE(fx.node->has_bias());

    std::vector<std::string> expected = {"X", "epsilon", "Y_out"};
    EXPECT_EQ(fx.node->inputs(), expected);
}

TEST(LayerNormTest, Connectors_AffineNoBias) {
    builder::StructuredSDFGBuilder builder("sdfg_ln_conn_affine_no_bias", FunctionType_CPU);
    symbolic::MultiExpression shape = {symbolic::integer(2), symbolic::integer(16)};
    auto fx = build_layernorm(builder, shape, 1, /*affine=*/true, /*has_bias=*/false);

    EXPECT_TRUE(fx.node->affine());
    EXPECT_FALSE(fx.node->has_bias());

    std::vector<std::string> expected = {"X", "Gamma", "epsilon", "Y_out"};
    EXPECT_EQ(fx.node->inputs(), expected);
}

TEST(LayerNormTest, ToStr) {
    builder::StructuredSDFGBuilder builder("sdfg_ln_tostr", FunctionType_CPU);
    symbolic::MultiExpression shape = {symbolic::integer(2), symbolic::integer(16)};
    auto fx = build_layernorm(builder, shape, 1, true, true);

    EXPECT_NE(fx.node->toStr().find("LayerNorm"), std::string::npos);
}

TEST(LayerNormTest, Clone_PreservesProperties) {
    builder::StructuredSDFGBuilder builder("sdfg_ln_clone", FunctionType_CPU);
    symbolic::MultiExpression shape = {symbolic::integer(2), symbolic::integer(3), symbolic::integer(16)};
    auto fx = build_layernorm(builder, shape, 2, /*affine=*/true, /*has_bias=*/false);

    auto& dataflow = fx.node->get_parent();
    auto clone = fx.node->clone(fx.node->element_id(), fx.node->vertex(), dataflow);
    auto* cloned = dynamic_cast<math::tensor::LayerNormNode*>(clone.get());
    ASSERT_NE(cloned, nullptr);

    EXPECT_EQ(cloned->num_normalized_dims(), fx.node->num_normalized_dims());
    EXPECT_EQ(cloned->affine(), fx.node->affine());
    EXPECT_EQ(cloned->has_bias(), fx.node->has_bias());
    EXPECT_EQ(cloned->quantization(), fx.node->quantization());
    EXPECT_EQ(cloned->inputs(), fx.node->inputs());
}

TEST(LayerNormTest, SerializeDeserialize_RoundTrip) {
    builder::StructuredSDFGBuilder builder("sdfg_ln_serialize", FunctionType_CPU);
    symbolic::MultiExpression shape = {symbolic::integer(2), symbolic::integer(3), symbolic::integer(16)};
    build_layernorm(builder, shape, 2, /*affine=*/true, /*has_bias=*/true);

    auto& sdfg = builder.subject();
    ASSERT_NO_THROW(sdfg.validate());

    serializer::JSONSerializer serializer;
    nlohmann::json j;
    ASSERT_NO_THROW(j = serializer.serialize(sdfg));

    std::unique_ptr<StructuredSDFG> new_sdfg;
    ASSERT_NO_THROW(new_sdfg = serializer.deserialize(j));
    ASSERT_NE(new_sdfg, nullptr);

    // Locate the deserialized LayerNorm node and confirm its properties survived.
    const math::tensor::LayerNormNode* found = nullptr;
    auto& new_root = new_sdfg->root();
    ASSERT_EQ(new_root.size(), 1);
    auto* block = dyn_cast<structured_control_flow::Block*>(&new_root.at(0));
    ASSERT_NE(block, nullptr);
    for (auto& node : block->dataflow().nodes()) {
        if (auto* ln = dynamic_cast<const math::tensor::LayerNormNode*>(&node)) {
            found = ln;
            break;
        }
    }
    ASSERT_NE(found, nullptr);
    EXPECT_EQ(found->num_normalized_dims(), 2u);
    EXPECT_TRUE(found->affine());
    EXPECT_TRUE(found->has_bias());
    EXPECT_EQ(found->quantization(), types::PrimitiveType::Float);
}

// --- Expansions ---

TEST(LayerNormTest, Expansion_AffineBias_2D) {
    builder::StructuredSDFGBuilder builder("sdfg_ln_exp_affine_bias", FunctionType_CPU);
    symbolic::MultiExpression shape = {symbolic::integer(2), symbolic::integer(16)};
    auto fx = build_layernorm(builder, shape, 1, /*affine=*/true, /*has_bias=*/true);

    auto& sdfg = builder.subject();
    ASSERT_NO_THROW(sdfg.validate());
    dump_sdfg(sdfg, "0.pre-expand");

    auto outcome = passes::expansion::expand_single_math_node(builder, *fx.block, *fx.node);
    EXPECT_TRUE(outcome.expanded);

    dump_sdfg(sdfg, "1.post-expand");
    ASSERT_NO_THROW(sdfg.validate());
    EXPECT_EQ(sdfg.root().size(), 1);
}

TEST(LayerNormTest, Expansion_NoAffine_2D) {
    builder::StructuredSDFGBuilder builder("sdfg_ln_exp_no_affine", FunctionType_CPU);
    symbolic::MultiExpression shape = {symbolic::integer(2), symbolic::integer(16)};
    auto fx = build_layernorm(builder, shape, 1, /*affine=*/false, /*has_bias=*/false);

    auto& sdfg = builder.subject();
    ASSERT_NO_THROW(sdfg.validate());

    auto outcome = passes::expansion::expand_single_math_node(builder, *fx.block, *fx.node);
    EXPECT_TRUE(outcome.expanded);

    ASSERT_NO_THROW(sdfg.validate());
    EXPECT_EQ(sdfg.root().size(), 1);
}

TEST(LayerNormTest, Expansion_AffineNoBias_2D) {
    builder::StructuredSDFGBuilder builder("sdfg_ln_exp_affine_no_bias", FunctionType_CPU);
    symbolic::MultiExpression shape = {symbolic::integer(2), symbolic::integer(16)};
    auto fx = build_layernorm(builder, shape, 1, /*affine=*/true, /*has_bias=*/false);

    auto& sdfg = builder.subject();
    ASSERT_NO_THROW(sdfg.validate());

    auto outcome = passes::expansion::expand_single_math_node(builder, *fx.block, *fx.node);
    EXPECT_TRUE(outcome.expanded);

    ASSERT_NO_THROW(sdfg.validate());
    EXPECT_EQ(sdfg.root().size(), 1);
}

TEST(LayerNormTest, Expansion_MultiNormalizedDims) {
    builder::StructuredSDFGBuilder builder("sdfg_ln_exp_multi_dim", FunctionType_CPU);
    // Normalize over the trailing two dims [3, 16], leading row dim [2].
    symbolic::MultiExpression shape = {symbolic::integer(2), symbolic::integer(3), symbolic::integer(16)};
    auto fx = build_layernorm(builder, shape, 2, /*affine=*/true, /*has_bias=*/true);

    auto& sdfg = builder.subject();
    ASSERT_NO_THROW(sdfg.validate());

    auto outcome = passes::expansion::expand_single_math_node(builder, *fx.block, *fx.node);
    EXPECT_TRUE(outcome.expanded);

    ASSERT_NO_THROW(sdfg.validate());
    EXPECT_EQ(sdfg.root().size(), 1);
}

TEST(LayerNormTest, Expansion_AllDimsNormalized) {
    builder::StructuredSDFGBuilder builder("sdfg_ln_exp_all_dims", FunctionType_CPU);
    // No leading dimensions: normalize over the entire tensor.
    symbolic::MultiExpression shape = {symbolic::integer(2), symbolic::integer(16)};
    auto fx = build_layernorm(builder, shape, 2, /*affine=*/true, /*has_bias=*/true);

    auto& sdfg = builder.subject();
    ASSERT_NO_THROW(sdfg.validate());

    auto outcome = passes::expansion::expand_single_math_node(builder, *fx.block, *fx.node);
    EXPECT_TRUE(outcome.expanded);

    ASSERT_NO_THROW(sdfg.validate());
    EXPECT_EQ(sdfg.root().size(), 1);
}
