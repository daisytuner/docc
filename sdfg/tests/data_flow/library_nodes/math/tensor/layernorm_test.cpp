#include "sdfg/data_flow/library_nodes/math/tensor/layernorm_node.h"

#include <memory>

#include <gtest/gtest.h>
#include <nlohmann/json_fwd.hpp>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_nodes/math/tensor/tensor_layout.h"
#include "sdfg/data_flow/pointer_metadata.h"
#include "sdfg/element.h"
#include "sdfg/function.h"
#include "sdfg/passes/expansion/library_node_expansion_pass.h"
#include "sdfg/serializer/json_serializer.h"
#include "sdfg/structured_sdfg.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/pointer.h"
#include "sdfg/types/scalar.h"
#include "sdfg/types/tensor.h"
#include "sdfg/types/type.h"
#include "sdfg_debug_dump.h"
#include "symengine/mul.h"

using namespace sdfg;

TEST(LayerNormNodeTest, symbolic) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("x", desc, true);
    builder.add_container("eps", base_desc, true);
    builder.add_container("gamma", desc, true);
    builder.add_container("beta", desc, true);
    builder.add_container("y", desc, true);
    builder.add_container("mean", desc, true);
    builder.add_container("rstd", desc, true);

    types::Scalar sym_desc(types::PrimitiveType::Int64);
    builder.add_container("b", sym_desc, true);
    builder.add_container("h", sym_desc, true);
    builder.add_container("w", sym_desc, true);
    auto batch = symbolic::symbol("b");
    auto height = symbolic::symbol("h");
    auto width = symbolic::symbol("w");
    symbolic::MultiExpression non_normalized_shape({batch});
    symbolic::MultiExpression normalized_shape({height, width});
    symbolic::MultiExpression full_shape({batch, height, width});
    math::tensor::TensorLayout x_layout(full_shape);
    types::Tensor x_tensor(base_desc, x_layout);
    math::tensor::TensorLayout gamma_layout(normalized_shape);
    types::Tensor gamma_tensor(base_desc, gamma_layout);
    math::tensor::TensorLayout beta_layout(normalized_shape);
    types::Tensor beta_tensor(base_desc, gamma_layout);
    math::tensor::TensorLayout y_layout(full_shape);
    types::Tensor y_tensor(base_desc, y_layout);
    math::tensor::TensorLayout mean_layout(non_normalized_shape);
    types::Tensor mean_tensor(base_desc, mean_layout);
    math::tensor::TensorLayout rstd_layout(non_normalized_shape);
    types::Tensor rstd_tensor(base_desc, rstd_layout);

    auto& block = builder.add_block(root);
    auto& x_access = builder.add_access(block, "x");
    auto& eps_access = builder.add_access(block, "eps");
    auto& gamma_access = builder.add_access(block, "gamma");
    auto& beta_access = builder.add_access(block, "beta");
    auto& y_access = builder.add_access(block, "y");
    auto& mean_access = builder.add_access(block, "mean");
    auto& rstd_access = builder.add_access(block, "rstd");
    auto& libnode = builder.add_library_node<math::tensor::LayerNormNode>(
        block, DebugInfo(), normalized_shape, y_layout, mean_layout, rstd_layout, x_layout, gamma_layout, beta_layout
    );
    builder.add_computational_memlet(block, x_access, libnode, "_x", {}, x_tensor);
    builder.add_computational_memlet(block, eps_access, libnode, "_eps", {}, base_desc);
    builder.add_computational_memlet(block, gamma_access, libnode, "_gamma", {}, gamma_tensor);
    builder.add_computational_memlet(block, beta_access, libnode, "_beta", {}, gamma_tensor);
    builder.add_computational_memlet(block, y_access, libnode, "_y", {}, y_tensor);
    builder.add_computational_memlet(block, mean_access, libnode, "_mean", {}, mean_tensor);
    builder.add_computational_memlet(block, rstd_access, libnode, "_rstd", {}, rstd_tensor);

    ASSERT_NO_THROW(sdfg.validate());
    auto& layernorm_node = static_cast<math::tensor::LayerNormNode&>(libnode);
    EXPECT_FALSE(layernorm_node.supports_integer_types());
    EXPECT_TRUE(symbolic::eq(
        layernorm_node.flop(),
        symbolic::mul(batch, symbolic::add(SymEngine::mul({symbolic::integer(8), height, width}), symbolic::integer(14)))
    ));

    auto symbols = layernorm_node.symbols();
    EXPECT_EQ(symbols.size(), 3);
    EXPECT_TRUE(symbols.contains(batch));
    EXPECT_TRUE(symbols.contains(height));
    EXPECT_TRUE(symbols.contains(width));

    builder.add_container("c", sym_desc);
    builder.add_container("u", sym_desc);
    builder.add_container("v", sym_desc);
    auto new_batch = symbolic::symbol("c");
    auto new_height = symbolic::symbol("u");
    auto new_width = symbolic::symbol("v");

    builder.replace_symbols(batch, new_batch);

    symbolic::ExpressionMapping replacements({{height, new_height}, {width, new_width}});
    builder.replace_symbols(replacements);

    ASSERT_NO_THROW(sdfg.validate());

    symbols = layernorm_node.symbols();
    EXPECT_EQ(symbols.size(), 3);
    EXPECT_TRUE(symbols.contains(new_batch));
    EXPECT_TRUE(symbols.contains(new_height));
    EXPECT_TRUE(symbols.contains(new_width));

    auto check_pointer_access_meta =
        [](data_flow::PointerAccessType pam, bool no_capture, bool reads, bool writes, bool invalidate) {
            EXPECT_EQ(pam->no_capture(), no_capture);
            EXPECT_EQ(pam->may_contain_reads(), reads);
            EXPECT_EQ(pam->may_contain_writes(), writes);
            EXPECT_EQ(pam->invalidated_after(), invalidate);
        };
    check_pointer_access_meta(
        layernorm_node.pointer_access_type(math::tensor::LayerNormNode::Y_INPUT_IDX), true, false, true, false
    );
    check_pointer_access_meta(
        layernorm_node.pointer_access_type(math::tensor::LayerNormNode::MEAN_INPUT_IDX), true, false, true, false
    );
    check_pointer_access_meta(
        layernorm_node.pointer_access_type(math::tensor::LayerNormNode::RSTD_INPUT_IDX), true, false, true, false
    );
    check_pointer_access_meta(
        layernorm_node.pointer_access_type(math::tensor::LayerNormNode::X_INPUT_IDX), true, true, false, false
    );
    check_pointer_access_meta(
        layernorm_node.pointer_access_type(math::tensor::LayerNormNode::EPS_INPUT_IDX), true, true, false, false
    );
    check_pointer_access_meta(
        layernorm_node.pointer_access_type(math::tensor::LayerNormNode::GAMMA_INPUT_IDX), true, true, false, false
    );
    check_pointer_access_meta(
        layernorm_node.pointer_access_type(math::tensor::LayerNormNode::BETA_INPUT_IDX), true, true, false, false
    );
}

TEST(LayerNormNodeTest, expansion) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("x", desc, true);
    builder.add_container("eps", base_desc, true);
    builder.add_container("y", desc, true);
    builder.add_container("mean", desc, true);
    builder.add_container("rstd", desc, true);

    auto batch = symbolic::integer(32);
    auto height = symbolic::integer(16);
    auto width = symbolic::integer(16);
    symbolic::MultiExpression non_normalized_shape({batch});
    symbolic::MultiExpression normalized_shape({height, width});
    symbolic::MultiExpression full_shape({batch, height, width});
    math::tensor::TensorLayout x_layout(full_shape);
    types::Tensor x_tensor(base_desc, x_layout);
    math::tensor::TensorLayout y_layout(full_shape);
    types::Tensor y_tensor(base_desc, y_layout);
    math::tensor::TensorLayout mean_layout(non_normalized_shape);
    types::Tensor mean_tensor(base_desc, mean_layout);
    math::tensor::TensorLayout rstd_layout(non_normalized_shape);
    types::Tensor rstd_tensor(base_desc, rstd_layout);

    auto& block = builder.add_block(root);
    auto& x_access = builder.add_access(block, "x");
    auto& eps_access = builder.add_access(block, "eps");
    auto& y_access = builder.add_access(block, "y");
    auto& mean_access = builder.add_access(block, "mean");
    auto& rstd_access = builder.add_access(block, "rstd");
    auto& libnode = builder.add_library_node<
        math::tensor::LayerNormNode>(block, DebugInfo(), normalized_shape, y_layout, mean_layout, rstd_layout, x_layout);
    builder.add_computational_memlet(block, x_access, libnode, "_x", {}, x_tensor);
    builder.add_computational_memlet(block, eps_access, libnode, "_eps", {}, base_desc);
    builder.add_computational_memlet(block, y_access, libnode, "_y", {}, y_tensor);
    builder.add_computational_memlet(block, mean_access, libnode, "_mean", {}, mean_tensor);
    builder.add_computational_memlet(block, rstd_access, libnode, "_rstd", {}, rstd_tensor);

    ASSERT_NO_THROW(sdfg.validate());
    dump_sdfg(sdfg, "0.before");

    analysis::AnalysisManager analysis_manager(sdfg);
    passes::LibraryNodeExpansionPass expansion;
    ASSERT_TRUE(expansion.run(builder, analysis_manager));
    ASSERT_NO_THROW(sdfg.validate());
    dump_sdfg(sdfg, "1.after");
}

TEST(LayerNormNodeTest, expansion_affine) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("x", desc, true);
    builder.add_container("eps", base_desc, true);
    builder.add_container("gamma", desc, true);
    builder.add_container("y", desc, true);
    builder.add_container("mean", desc, true);
    builder.add_container("rstd", desc, true);

    auto batch = symbolic::integer(32);
    auto height = symbolic::integer(16);
    auto width = symbolic::integer(16);
    symbolic::MultiExpression non_normalized_shape({batch});
    symbolic::MultiExpression normalized_shape({height, width});
    symbolic::MultiExpression full_shape({batch, height, width});
    math::tensor::TensorLayout x_layout(full_shape);
    types::Tensor x_tensor(base_desc, x_layout);
    math::tensor::TensorLayout gamma_layout(normalized_shape);
    types::Tensor gamma_tensor(base_desc, gamma_layout);
    math::tensor::TensorLayout y_layout(full_shape);
    types::Tensor y_tensor(base_desc, y_layout);
    math::tensor::TensorLayout mean_layout(non_normalized_shape);
    types::Tensor mean_tensor(base_desc, mean_layout);
    math::tensor::TensorLayout rstd_layout(non_normalized_shape);
    types::Tensor rstd_tensor(base_desc, rstd_layout);

    auto& block = builder.add_block(root);
    auto& x_access = builder.add_access(block, "x");
    auto& eps_access = builder.add_access(block, "eps");
    auto& gamma_access = builder.add_access(block, "gamma");
    auto& y_access = builder.add_access(block, "y");
    auto& mean_access = builder.add_access(block, "mean");
    auto& rstd_access = builder.add_access(block, "rstd");
    auto& libnode = builder.add_library_node<math::tensor::LayerNormNode>(
        block, DebugInfo(), normalized_shape, y_layout, mean_layout, rstd_layout, x_layout, gamma_layout
    );
    builder.add_computational_memlet(block, x_access, libnode, "_x", {}, x_tensor);
    builder.add_computational_memlet(block, eps_access, libnode, "_eps", {}, base_desc);
    builder.add_computational_memlet(block, gamma_access, libnode, "_gamma", {}, gamma_tensor);
    builder.add_computational_memlet(block, y_access, libnode, "_y", {}, y_tensor);
    builder.add_computational_memlet(block, mean_access, libnode, "_mean", {}, mean_tensor);
    builder.add_computational_memlet(block, rstd_access, libnode, "_rstd", {}, rstd_tensor);

    ASSERT_NO_THROW(sdfg.validate());
    dump_sdfg(sdfg, "0.before");

    analysis::AnalysisManager analysis_manager(sdfg);
    passes::LibraryNodeExpansionPass expansion;
    ASSERT_TRUE(expansion.run(builder, analysis_manager));
    ASSERT_NO_THROW(sdfg.validate());
    dump_sdfg(sdfg, "1.after");
}

TEST(LayerNormNodeTest, expansion_affine_with_bias) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("x", desc, true);
    builder.add_container("eps", base_desc, true);
    builder.add_container("gamma", desc, true);
    builder.add_container("beta", desc, true);
    builder.add_container("y", desc, true);
    builder.add_container("mean", desc, true);
    builder.add_container("rstd", desc, true);

    auto batch = symbolic::integer(32);
    auto height = symbolic::integer(16);
    auto width = symbolic::integer(16);
    symbolic::MultiExpression non_normalized_shape({batch});
    symbolic::MultiExpression normalized_shape({height, width});
    symbolic::MultiExpression full_shape({batch, height, width});
    math::tensor::TensorLayout x_layout(full_shape);
    types::Tensor x_tensor(base_desc, x_layout);
    math::tensor::TensorLayout gamma_layout(normalized_shape);
    types::Tensor gamma_tensor(base_desc, gamma_layout);
    math::tensor::TensorLayout beta_layout(normalized_shape);
    types::Tensor beta_tensor(base_desc, gamma_layout);
    math::tensor::TensorLayout y_layout(full_shape);
    types::Tensor y_tensor(base_desc, y_layout);
    math::tensor::TensorLayout mean_layout(non_normalized_shape);
    types::Tensor mean_tensor(base_desc, mean_layout);
    math::tensor::TensorLayout rstd_layout(non_normalized_shape);
    types::Tensor rstd_tensor(base_desc, rstd_layout);

    auto& block = builder.add_block(root);
    auto& x_access = builder.add_access(block, "x");
    auto& eps_access = builder.add_access(block, "eps");
    auto& gamma_access = builder.add_access(block, "gamma");
    auto& beta_access = builder.add_access(block, "beta");
    auto& y_access = builder.add_access(block, "y");
    auto& mean_access = builder.add_access(block, "mean");
    auto& rstd_access = builder.add_access(block, "rstd");
    auto& libnode = builder.add_library_node<math::tensor::LayerNormNode>(
        block, DebugInfo(), normalized_shape, y_layout, mean_layout, rstd_layout, x_layout, gamma_layout, beta_layout
    );
    builder.add_computational_memlet(block, x_access, libnode, "_x", {}, x_tensor);
    builder.add_computational_memlet(block, eps_access, libnode, "_eps", {}, base_desc);
    builder.add_computational_memlet(block, gamma_access, libnode, "_gamma", {}, gamma_tensor);
    builder.add_computational_memlet(block, beta_access, libnode, "_beta", {}, gamma_tensor);
    builder.add_computational_memlet(block, y_access, libnode, "_y", {}, y_tensor);
    builder.add_computational_memlet(block, mean_access, libnode, "_mean", {}, mean_tensor);
    builder.add_computational_memlet(block, rstd_access, libnode, "_rstd", {}, rstd_tensor);

    ASSERT_NO_THROW(sdfg.validate());
    dump_sdfg(sdfg, "0.before");

    analysis::AnalysisManager analysis_manager(sdfg);
    passes::LibraryNodeExpansionPass expansion;
    ASSERT_TRUE(expansion.run(builder, analysis_manager));
    ASSERT_NO_THROW(sdfg.validate());
    dump_sdfg(sdfg, "1.after");
}

TEST(LayerNormNodeTest, expansion_full_shape) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("x", desc, true);
    builder.add_container("eps", base_desc, true);
    builder.add_container("y", desc, true);
    builder.add_container("mean", desc, true);
    builder.add_container("rstd", desc, true);

    auto batch = symbolic::integer(32);
    auto height = symbolic::integer(16);
    auto width = symbolic::integer(16);
    symbolic::MultiExpression non_normalized_shape({symbolic::one()});
    symbolic::MultiExpression normalized_shape({batch, height, width});
    symbolic::MultiExpression full_shape({batch, height, width});
    math::tensor::TensorLayout x_layout(full_shape);
    types::Tensor x_tensor(base_desc, x_layout);
    math::tensor::TensorLayout y_layout(full_shape);
    types::Tensor y_tensor(base_desc, y_layout);
    math::tensor::TensorLayout mean_layout(non_normalized_shape);
    types::Tensor mean_tensor(base_desc, mean_layout);
    math::tensor::TensorLayout rstd_layout(non_normalized_shape);
    types::Tensor rstd_tensor(base_desc, rstd_layout);

    auto& block = builder.add_block(root);
    auto& x_access = builder.add_access(block, "x");
    auto& eps_access = builder.add_access(block, "eps");
    auto& y_access = builder.add_access(block, "y");
    auto& mean_access = builder.add_access(block, "mean");
    auto& rstd_access = builder.add_access(block, "rstd");
    auto& libnode = builder.add_library_node<
        math::tensor::LayerNormNode>(block, DebugInfo(), normalized_shape, y_layout, mean_layout, rstd_layout, x_layout);
    builder.add_computational_memlet(block, x_access, libnode, "_x", {}, x_tensor);
    builder.add_computational_memlet(block, eps_access, libnode, "_eps", {}, base_desc);
    builder.add_computational_memlet(block, y_access, libnode, "_y", {}, y_tensor);
    builder.add_computational_memlet(block, mean_access, libnode, "_mean", {}, mean_tensor);
    builder.add_computational_memlet(block, rstd_access, libnode, "_rstd", {}, rstd_tensor);

    ASSERT_NO_THROW(sdfg.validate());
    dump_sdfg(sdfg, "0.before");

    analysis::AnalysisManager analysis_manager(sdfg);
    passes::LibraryNodeExpansionPass expansion;
    ASSERT_TRUE(expansion.run(builder, analysis_manager));
    ASSERT_NO_THROW(sdfg.validate());
    dump_sdfg(sdfg, "1.after");
}

TEST(LayerNormNodeTest, serialization) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("x", desc, true);
    builder.add_container("eps", base_desc, true);
    builder.add_container("y", desc, true);
    builder.add_container("mean", desc, true);
    builder.add_container("rstd", desc, true);

    auto batch = symbolic::integer(32);
    auto height = symbolic::integer(16);
    auto width = symbolic::integer(16);
    symbolic::MultiExpression non_normalized_shape({batch});
    symbolic::MultiExpression normalized_shape({height, width});
    symbolic::MultiExpression full_shape({batch, height, width});
    math::tensor::TensorLayout x_layout(full_shape);
    types::Tensor x_tensor(base_desc, x_layout);
    math::tensor::TensorLayout y_layout(full_shape);
    types::Tensor y_tensor(base_desc, y_layout);
    math::tensor::TensorLayout mean_layout(non_normalized_shape);
    types::Tensor mean_tensor(base_desc, mean_layout);
    math::tensor::TensorLayout rstd_layout(non_normalized_shape);
    types::Tensor rstd_tensor(base_desc, rstd_layout);

    auto& block = builder.add_block(root);
    auto& x_access = builder.add_access(block, "x");
    auto& eps_access = builder.add_access(block, "eps");
    auto& y_access = builder.add_access(block, "y");
    auto& mean_access = builder.add_access(block, "mean");
    auto& rstd_access = builder.add_access(block, "rstd");
    auto& libnode = builder.add_library_node<
        math::tensor::LayerNormNode>(block, DebugInfo(), normalized_shape, y_layout, mean_layout, rstd_layout, x_layout);
    builder.add_computational_memlet(block, x_access, libnode, "_x", {}, x_tensor);
    builder.add_computational_memlet(block, eps_access, libnode, "_eps", {}, base_desc);
    builder.add_computational_memlet(block, y_access, libnode, "_y", {}, y_tensor);
    builder.add_computational_memlet(block, mean_access, libnode, "_mean", {}, mean_tensor);
    builder.add_computational_memlet(block, rstd_access, libnode, "_rstd", {}, rstd_tensor);

    ASSERT_NO_THROW(sdfg.validate());

    serializer::JSONSerializer serializer;
    nlohmann::json j;
    ASSERT_NO_THROW(j = serializer.serialize(sdfg));

    std::unique_ptr<StructuredSDFG> new_sdfg;
    ASSERT_NO_THROW(new_sdfg = serializer.deserialize(j));
    ASSERT_NO_THROW(new_sdfg->validate());
}

TEST(LayerNormNodeTest, serialization_affine) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("x", desc, true);
    builder.add_container("eps", base_desc, true);
    builder.add_container("gamma", desc, true);
    builder.add_container("y", desc, true);
    builder.add_container("mean", desc, true);
    builder.add_container("rstd", desc, true);

    auto batch = symbolic::integer(32);
    auto height = symbolic::integer(16);
    auto width = symbolic::integer(16);
    symbolic::MultiExpression non_normalized_shape({batch});
    symbolic::MultiExpression normalized_shape({height, width});
    symbolic::MultiExpression full_shape({batch, height, width});
    math::tensor::TensorLayout x_layout(full_shape);
    types::Tensor x_tensor(base_desc, x_layout);
    math::tensor::TensorLayout gamma_layout(normalized_shape);
    types::Tensor gamma_tensor(base_desc, gamma_layout);
    math::tensor::TensorLayout y_layout(full_shape);
    types::Tensor y_tensor(base_desc, y_layout);
    math::tensor::TensorLayout mean_layout(non_normalized_shape);
    types::Tensor mean_tensor(base_desc, mean_layout);
    math::tensor::TensorLayout rstd_layout(non_normalized_shape);
    types::Tensor rstd_tensor(base_desc, rstd_layout);

    auto& block = builder.add_block(root);
    auto& x_access = builder.add_access(block, "x");
    auto& eps_access = builder.add_access(block, "eps");
    auto& gamma_access = builder.add_access(block, "gamma");
    auto& y_access = builder.add_access(block, "y");
    auto& mean_access = builder.add_access(block, "mean");
    auto& rstd_access = builder.add_access(block, "rstd");
    auto& libnode = builder.add_library_node<math::tensor::LayerNormNode>(
        block, DebugInfo(), normalized_shape, y_layout, mean_layout, rstd_layout, x_layout, gamma_layout
    );
    builder.add_computational_memlet(block, x_access, libnode, "_x", {}, x_tensor);
    builder.add_computational_memlet(block, eps_access, libnode, "_eps", {}, base_desc);
    builder.add_computational_memlet(block, gamma_access, libnode, "_gamma", {}, gamma_tensor);
    builder.add_computational_memlet(block, y_access, libnode, "_y", {}, y_tensor);
    builder.add_computational_memlet(block, mean_access, libnode, "_mean", {}, mean_tensor);
    builder.add_computational_memlet(block, rstd_access, libnode, "_rstd", {}, rstd_tensor);

    ASSERT_NO_THROW(sdfg.validate());

    serializer::JSONSerializer serializer;
    nlohmann::json j;
    ASSERT_NO_THROW(j = serializer.serialize(sdfg));

    std::unique_ptr<StructuredSDFG> new_sdfg;
    ASSERT_NO_THROW(new_sdfg = serializer.deserialize(j));
    ASSERT_NO_THROW(new_sdfg->validate());
}

TEST(LayerNormNodeTest, serialization_affine_with_bias) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("x", desc, true);
    builder.add_container("eps", base_desc, true);
    builder.add_container("gamma", desc, true);
    builder.add_container("beta", desc, true);
    builder.add_container("y", desc, true);
    builder.add_container("mean", desc, true);
    builder.add_container("rstd", desc, true);

    auto batch = symbolic::integer(32);
    auto height = symbolic::integer(16);
    auto width = symbolic::integer(16);
    symbolic::MultiExpression non_normalized_shape({batch});
    symbolic::MultiExpression normalized_shape({height, width});
    symbolic::MultiExpression full_shape({batch, height, width});
    math::tensor::TensorLayout x_layout(full_shape);
    types::Tensor x_tensor(base_desc, x_layout);
    math::tensor::TensorLayout gamma_layout(normalized_shape);
    types::Tensor gamma_tensor(base_desc, gamma_layout);
    math::tensor::TensorLayout beta_layout(normalized_shape);
    types::Tensor beta_tensor(base_desc, gamma_layout);
    math::tensor::TensorLayout y_layout(full_shape);
    types::Tensor y_tensor(base_desc, y_layout);
    math::tensor::TensorLayout mean_layout(non_normalized_shape);
    types::Tensor mean_tensor(base_desc, mean_layout);
    math::tensor::TensorLayout rstd_layout(non_normalized_shape);
    types::Tensor rstd_tensor(base_desc, rstd_layout);

    auto& block = builder.add_block(root);
    auto& x_access = builder.add_access(block, "x");
    auto& eps_access = builder.add_access(block, "eps");
    auto& gamma_access = builder.add_access(block, "gamma");
    auto& beta_access = builder.add_access(block, "beta");
    auto& y_access = builder.add_access(block, "y");
    auto& mean_access = builder.add_access(block, "mean");
    auto& rstd_access = builder.add_access(block, "rstd");
    auto& libnode = builder.add_library_node<math::tensor::LayerNormNode>(
        block, DebugInfo(), normalized_shape, y_layout, mean_layout, rstd_layout, x_layout, gamma_layout, beta_layout
    );
    builder.add_computational_memlet(block, x_access, libnode, "_x", {}, x_tensor);
    builder.add_computational_memlet(block, eps_access, libnode, "_eps", {}, base_desc);
    builder.add_computational_memlet(block, gamma_access, libnode, "_gamma", {}, gamma_tensor);
    builder.add_computational_memlet(block, beta_access, libnode, "_beta", {}, gamma_tensor);
    builder.add_computational_memlet(block, y_access, libnode, "_y", {}, y_tensor);
    builder.add_computational_memlet(block, mean_access, libnode, "_mean", {}, mean_tensor);
    builder.add_computational_memlet(block, rstd_access, libnode, "_rstd", {}, rstd_tensor);

    ASSERT_NO_THROW(sdfg.validate());

    serializer::JSONSerializer serializer;
    nlohmann::json j;
    ASSERT_NO_THROW(j = serializer.serialize(sdfg));

    std::unique_ptr<StructuredSDFG> new_sdfg;
    ASSERT_NO_THROW(new_sdfg = serializer.deserialize(j));
    ASSERT_NO_THROW(new_sdfg->validate());
}
