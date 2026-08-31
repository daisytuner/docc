#include "sdfg/data_flow/library_nodes/math/tensor/const_padding_node.h"

#include <memory>
#include <sstream>

#include <gtest/gtest.h>
#include <nlohmann/json_fwd.hpp>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_nodes/math/tensor/tensor_layout.h"
#include "sdfg/data_flow/pointer_metadata.h"
#include "sdfg/deepcopy/structured_sdfg_deep_copy.h"
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
#include "symengine/add.h"

using namespace sdfg;

TEST(ConstPaddingNodeTest, symbolic) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("y", desc, true);
    builder.add_container("x", desc, true);
    builder.add_container("val", base_desc, true);

    types::Scalar sym_desc(types::PrimitiveType::Int64);
    builder.add_container("n", sym_desc);
    builder.add_container("c", sym_desc);
    builder.add_container("h", sym_desc);
    builder.add_container("w", sym_desc);
    builder.add_container("a", sym_desc);
    builder.add_container("b", sym_desc);
    auto n = symbolic::symbol("n");
    auto c = symbolic::symbol("c");
    auto h = symbolic::symbol("h");
    auto w = symbolic::symbol("w");
    auto a = symbolic::symbol("a");
    auto b = symbolic::symbol("b");
    symbolic::MultiExpression pads = {a, b, a, b};
    math::tensor::TensorLayout
        y_layout({n, c, SymEngine::add({h, pads[0], pads[1]}), SymEngine::add({w, pads[2], pads[3]})});
    types::Tensor y_tensor(base_desc, y_layout);
    math::tensor::TensorLayout x_layout({n, c, h, w});
    types::Tensor x_tensor(base_desc, x_layout);

    auto& block = builder.add_block(root);
    auto& y_access = builder.add_access(block, "y");
    auto& x_access = builder.add_access(block, "x");
    auto& val_access = builder.add_access(block, "val");
    auto& libnode =
        builder.add_library_node<math::tensor::ConstPaddingNode>(block, DebugInfo(), pads, y_layout, x_layout);
    builder.add_computational_memlet(block, y_access, libnode, "_y", {}, y_tensor);
    builder.add_computational_memlet(block, x_access, libnode, "_x", {}, x_tensor);
    builder.add_computational_memlet(block, val_access, libnode, "_val", {}, base_desc);

    ASSERT_NO_THROW(sdfg.validate());
    auto& const_padding_node = static_cast<math::tensor::ConstPaddingNode&>(libnode);
    EXPECT_TRUE(const_padding_node.supports_integer_types());
    EXPECT_TRUE(symbolic::eq(const_padding_node.flop(), symbolic::zero()));
    std::stringstream repr;
    repr << "ConstPadding(pads: [a,b,a,b], y_layout: " << y_layout << ", x_layout: " << x_layout << ")";
    EXPECT_EQ(const_padding_node.toStr(), repr.str());

    auto symbols = const_padding_node.symbols();
    EXPECT_EQ(symbols.size(), 6);
    EXPECT_TRUE(symbols.contains(n));
    EXPECT_TRUE(symbols.contains(c));
    EXPECT_TRUE(symbols.contains(h));
    EXPECT_TRUE(symbols.contains(w));
    EXPECT_TRUE(symbols.contains(a));
    EXPECT_TRUE(symbols.contains(b));

    builder.add_container("m", sym_desc);
    builder.add_container("d", sym_desc);
    builder.add_container("e", sym_desc);
    auto m = symbolic::symbol("m");
    auto d = symbolic::symbol("d");
    auto e = symbolic::symbol("e");

    builder.replace_symbols(n, m);

    symbolic::ExpressionMapping replacements({{h, d}, {b, e}});
    builder.replace_symbols(replacements);

    ASSERT_NO_THROW(sdfg.validate());

    symbols = const_padding_node.symbols();
    EXPECT_EQ(symbols.size(), 6);
    EXPECT_TRUE(symbols.contains(m));
    EXPECT_TRUE(symbols.contains(c));
    EXPECT_TRUE(symbols.contains(d));
    EXPECT_TRUE(symbols.contains(w));
    EXPECT_TRUE(symbols.contains(a));
    EXPECT_TRUE(symbols.contains(e));
}

TEST(ConstPaddingNodeTest, pointer_access_type) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("y", desc, true);
    builder.add_container("x", desc, true);
    builder.add_container("val", base_desc, true);

    auto n = symbolic::integer(4);
    auto c = symbolic::integer(3);
    auto h = symbolic::integer(8);
    auto w = symbolic::integer(8);
    symbolic::MultiExpression pads = {symbolic::one(), symbolic::zero(), symbolic::zero(), symbolic::one()};
    math::tensor::TensorLayout
        y_layout({n, c, SymEngine::add({h, pads[0], pads[1]}), SymEngine::add({w, pads[2], pads[3]})});
    types::Tensor y_tensor(base_desc, y_layout);
    math::tensor::TensorLayout x_layout({n, c, h, w});
    types::Tensor x_tensor(base_desc, x_layout);

    auto& block = builder.add_block(root);
    auto& y_access = builder.add_access(block, "y");
    auto& x_access = builder.add_access(block, "x");
    auto& val_access = builder.add_access(block, "val");
    auto& libnode =
        builder.add_library_node<math::tensor::ConstPaddingNode>(block, DebugInfo(), pads, y_layout, x_layout);
    builder.add_computational_memlet(block, y_access, libnode, "_y", {}, y_tensor);
    builder.add_computational_memlet(block, x_access, libnode, "_x", {}, x_tensor);
    builder.add_computational_memlet(block, val_access, libnode, "_val", {}, base_desc);

    ASSERT_NO_THROW(sdfg.validate());
    auto& const_padding_node = static_cast<math::tensor::ConstPaddingNode&>(libnode);

    auto y_pam = const_padding_node.pointer_access_type(math::tensor::ConstPaddingNode::Y_INPUT_IDX);
    EXPECT_TRUE(y_pam->no_capture());
    EXPECT_FALSE(y_pam->may_contain_reads());
    EXPECT_TRUE(y_pam->may_contain_writes());
    EXPECT_FALSE(y_pam->invalidated_after());
    auto y_map = y_pam->access_write_pattern();
    auto* y_cap = dynamic_cast<data_flow::ConvexAccessPattern*>(y_map.get());
    ASSERT_NE(y_cap, nullptr);
    EXPECT_TRUE(symbolic::eq(y_cap->size(), y_layout.total_elements()));

    auto x_pam = const_padding_node.pointer_access_type(math::tensor::ConstPaddingNode::X_INPUT_IDX);
    EXPECT_TRUE(x_pam->no_capture());
    EXPECT_TRUE(x_pam->may_contain_reads());
    EXPECT_FALSE(x_pam->may_contain_writes());
    EXPECT_FALSE(x_pam->invalidated_after());
    auto x_map = x_pam->access_read_pattern();
    auto* x_cap = dynamic_cast<data_flow::ConvexAccessPattern*>(x_map.get());
    ASSERT_NE(x_cap, nullptr);
    EXPECT_TRUE(symbolic::eq(x_cap->size(), x_layout.total_elements()));

    EXPECT_EQ(const_padding_node.pointer_access_type(math::tensor::ConstPaddingNode::VAL_INPUT_IDX), nullptr);
}

TEST(ConstPaddingNodeTest, cloning) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("y", desc, true);
    builder.add_container("x", desc, true);
    builder.add_container("val", base_desc, true);

    auto n = symbolic::integer(4);
    auto c = symbolic::integer(3);
    auto h = symbolic::integer(8);
    auto w = symbolic::integer(8);
    symbolic::MultiExpression pads = {symbolic::one(), symbolic::zero(), symbolic::zero(), symbolic::one()};
    math::tensor::TensorLayout
        y_layout({n, c, SymEngine::add({h, pads[0], pads[1]}), SymEngine::add({w, pads[2], pads[3]})});
    types::Tensor y_tensor(base_desc, y_layout);
    math::tensor::TensorLayout x_layout({n, c, h, w});
    types::Tensor x_tensor(base_desc, x_layout);

    auto& block = builder.add_block(root);
    auto& y_access = builder.add_access(block, "y");
    auto& x_access = builder.add_access(block, "x");
    auto& val_access = builder.add_access(block, "val");
    auto& libnode =
        builder.add_library_node<math::tensor::ConstPaddingNode>(block, DebugInfo(), pads, y_layout, x_layout);
    builder.add_computational_memlet(block, y_access, libnode, "_y", {}, y_tensor);
    builder.add_computational_memlet(block, x_access, libnode, "_x", {}, x_tensor);
    builder.add_computational_memlet(block, val_access, libnode, "_val", {}, base_desc);

    ASSERT_NO_THROW(sdfg.validate());
    builder::StructuredSDFGBuilder new_builder("sdfg_2", FunctionType_CPU);
    new_builder.add_container("y", desc, true);
    new_builder.add_container("x", desc, true);
    new_builder.add_container("val", base_desc, true);
    deepcopy::StructuredSDFGDeepCopy deep_copy(new_builder, new_builder.subject().root(), root);
    deep_copy.copy();
    ASSERT_NO_THROW(new_builder.subject().validate());
}

TEST(ConstPaddingNodeTest, expansion) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("y", desc, true);
    builder.add_container("x", desc, true);
    builder.add_container("val", base_desc, true);

    auto n = symbolic::integer(4);
    auto c = symbolic::integer(3);
    auto h = symbolic::integer(8);
    auto w = symbolic::integer(8);
    symbolic::MultiExpression pads = {symbolic::one(), symbolic::zero(), symbolic::zero(), symbolic::one()};
    math::tensor::TensorLayout
        y_layout({n, c, SymEngine::add({h, pads[0], pads[1]}), SymEngine::add({w, pads[2], pads[3]})});
    types::Tensor y_tensor(base_desc, y_layout);
    math::tensor::TensorLayout x_layout({n, c, h, w});
    types::Tensor x_tensor(base_desc, x_layout);

    auto& block = builder.add_block(root);
    auto& y_access = builder.add_access(block, "y");
    auto& x_access = builder.add_access(block, "x");
    auto& val_access = builder.add_access(block, "val");
    auto& libnode =
        builder.add_library_node<math::tensor::ConstPaddingNode>(block, DebugInfo(), pads, y_layout, x_layout);
    builder.add_computational_memlet(block, y_access, libnode, "_y", {}, y_tensor);
    builder.add_computational_memlet(block, x_access, libnode, "_x", {}, x_tensor);
    builder.add_computational_memlet(block, val_access, libnode, "_val", {}, base_desc);

    ASSERT_NO_THROW(sdfg.validate());
    dump_sdfg(sdfg, "0.before");

    analysis::AnalysisManager analysis_manager(sdfg);
    passes::LibraryNodeExpansionPass expansion;
    ASSERT_TRUE(expansion.run(builder, analysis_manager));
    ASSERT_NO_THROW(sdfg.validate());
    dump_sdfg(sdfg, "1.after");
}

TEST(ConstPaddingNodeTest, serialization) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("y", desc, true);
    builder.add_container("x", desc, true);
    builder.add_container("val", base_desc, true);

    auto n = symbolic::integer(4);
    auto c = symbolic::integer(3);
    auto h = symbolic::integer(8);
    auto w = symbolic::integer(8);
    symbolic::MultiExpression pads = {symbolic::one(), symbolic::zero(), symbolic::zero(), symbolic::one()};
    math::tensor::TensorLayout
        y_layout({n, c, SymEngine::add({h, pads[0], pads[1]}), SymEngine::add({w, pads[2], pads[3]})});
    types::Tensor y_tensor(base_desc, y_layout);
    math::tensor::TensorLayout x_layout({n, c, h, w});
    types::Tensor x_tensor(base_desc, x_layout);

    auto& block = builder.add_block(root);
    auto& y_access = builder.add_access(block, "y");
    auto& x_access = builder.add_access(block, "x");
    auto& val_access = builder.add_access(block, "val");
    auto& libnode =
        builder.add_library_node<math::tensor::ConstPaddingNode>(block, DebugInfo(), pads, y_layout, x_layout);
    builder.add_computational_memlet(block, y_access, libnode, "_y", {}, y_tensor);
    builder.add_computational_memlet(block, x_access, libnode, "_x", {}, x_tensor);
    builder.add_computational_memlet(block, val_access, libnode, "_val", {}, base_desc);

    ASSERT_NO_THROW(sdfg.validate());

    serializer::JSONSerializer serializer;
    nlohmann::json j;
    ASSERT_NO_THROW(j = serializer.serialize(sdfg));

    std::unique_ptr<StructuredSDFG> new_sdfg;
    ASSERT_NO_THROW(new_sdfg = serializer.deserialize(j));
    ASSERT_NO_THROW(new_sdfg->validate());
}

TEST(ConstPaddingNodeTest, validate_inputs) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("y", desc, true);
    builder.add_container("x", desc, true);
    builder.add_container("val", base_desc, true);

    auto n = symbolic::integer(4);
    auto c = symbolic::integer(3);
    auto h = symbolic::integer(8);
    auto w = symbolic::integer(8);
    symbolic::MultiExpression pads = {symbolic::one(), symbolic::zero(), symbolic::zero(), symbolic::one()};
    math::tensor::TensorLayout
        y_layout({n, c, SymEngine::add({h, pads[0], pads[1]}), SymEngine::add({w, pads[2], pads[3]})});
    types::Tensor y_tensor(base_desc, y_layout);
    math::tensor::TensorLayout x_layout({n, c, h, w});
    types::Tensor x_tensor(base_desc, x_layout);

    auto& block = builder.add_block(root);
    auto& y_access = builder.add_access(block, "y");
    auto& x_access = builder.add_access(block, "x");
    auto& val_access = builder.add_access(block, "val");
    auto& libnode =
        builder.add_library_node<math::tensor::ConstPaddingNode>(block, DebugInfo(), pads, y_layout, x_layout);
    builder.add_computational_memlet(block, y_access, libnode, "_y", {}, y_tensor);
    builder.add_computational_memlet(block, x_access, libnode, "_x", {}, x_tensor);

    ASSERT_THROW(sdfg.validate(), InvalidSDFGException);
}

TEST(ConstPaddingNodeTest, validate_input_y) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("y", desc, true);
    builder.add_container("x", desc, true);
    builder.add_container("val", base_desc, true);

    auto n = symbolic::integer(4);
    auto c = symbolic::integer(3);
    auto h = symbolic::integer(8);
    auto w = symbolic::integer(8);
    symbolic::MultiExpression pads = {symbolic::one(), symbolic::zero(), symbolic::zero(), symbolic::one()};
    math::tensor::TensorLayout
        y_layout({n, c, SymEngine::add({h, pads[0], pads[1]}), SymEngine::add({w, pads[2], pads[3]})});
    types::Tensor y_tensor(base_desc, y_layout);
    math::tensor::TensorLayout x_layout({n, c, h, w});
    types::Tensor x_tensor(base_desc, x_layout);

    auto& block = builder.add_block(root);
    auto& y_access = builder.add_access(block, "y");
    auto& x_access = builder.add_access(block, "x");
    auto& val_access = builder.add_access(block, "val");
    auto& libnode =
        builder.add_library_node<math::tensor::ConstPaddingNode>(block, DebugInfo(), pads, y_layout, x_layout);
    builder.add_computational_memlet(block, y_access, libnode, "_x", {}, y_tensor);
    builder.add_computational_memlet(block, x_access, libnode, "_x", {}, x_tensor);
    builder.add_computational_memlet(block, val_access, libnode, "_val", {}, base_desc);

    ASSERT_THROW(sdfg.validate(), InvalidSDFGException);
}

TEST(ConstPaddingNodeTest, validate_input_x) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("y", desc, true);
    builder.add_container("x", desc, true);
    builder.add_container("val", base_desc, true);

    auto n = symbolic::integer(4);
    auto c = symbolic::integer(3);
    auto h = symbolic::integer(8);
    auto w = symbolic::integer(8);
    symbolic::MultiExpression pads = {symbolic::one(), symbolic::zero(), symbolic::zero(), symbolic::one()};
    math::tensor::TensorLayout
        y_layout({n, c, SymEngine::add({h, pads[0], pads[1]}), SymEngine::add({w, pads[2], pads[3]})});
    types::Tensor y_tensor(base_desc, y_layout);
    math::tensor::TensorLayout x_layout({n, c, h, w});
    types::Tensor x_tensor(base_desc, x_layout);

    auto& block = builder.add_block(root);
    auto& y_access = builder.add_access(block, "y");
    auto& x_access = builder.add_access(block, "x");
    auto& val_access = builder.add_access(block, "val");
    auto& libnode =
        builder.add_library_node<math::tensor::ConstPaddingNode>(block, DebugInfo(), pads, y_layout, x_layout);
    builder.add_computational_memlet(block, y_access, libnode, "_y", {}, y_tensor);
    builder.add_computational_memlet(block, x_access, libnode, "_y", {}, x_tensor);
    builder.add_computational_memlet(block, val_access, libnode, "_val", {}, base_desc);

    ASSERT_THROW(sdfg.validate(), InvalidSDFGException);
}

TEST(ConstPaddingNodeTest, validate_input_val) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("y", desc, true);
    builder.add_container("x", desc, true);
    builder.add_container("val", base_desc, true);

    auto n = symbolic::integer(4);
    auto c = symbolic::integer(3);
    auto h = symbolic::integer(8);
    auto w = symbolic::integer(8);
    symbolic::MultiExpression pads = {symbolic::one(), symbolic::zero(), symbolic::zero(), symbolic::one()};
    math::tensor::TensorLayout
        y_layout({n, c, SymEngine::add({h, pads[0], pads[1]}), SymEngine::add({w, pads[2], pads[3]})});
    types::Tensor y_tensor(base_desc, y_layout);
    math::tensor::TensorLayout x_layout({n, c, h, w});
    types::Tensor x_tensor(base_desc, x_layout);

    auto& block = builder.add_block(root);
    auto& y_access = builder.add_access(block, "y");
    auto& x_access = builder.add_access(block, "x");
    auto& val_access = builder.add_access(block, "val");
    auto& libnode =
        builder.add_library_node<math::tensor::ConstPaddingNode>(block, DebugInfo(), pads, y_layout, x_layout);
    builder.add_computational_memlet(block, y_access, libnode, "_y", {}, y_tensor);
    builder.add_computational_memlet(block, x_access, libnode, "_x", {}, x_tensor);
    builder.add_computational_memlet(block, val_access, libnode, "_x", {}, base_desc);

    ASSERT_THROW(sdfg.validate(), InvalidSDFGException);
}

TEST(ConstPaddingNodeTest, validate_memlet_tensor_y) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("y", desc, true);
    builder.add_container("x", desc, true);
    builder.add_container("val", base_desc, true);

    auto n = symbolic::integer(4);
    auto c = symbolic::integer(3);
    auto h = symbolic::integer(8);
    auto w = symbolic::integer(8);
    symbolic::MultiExpression pads = {symbolic::one(), symbolic::zero(), symbolic::zero(), symbolic::one()};
    math::tensor::TensorLayout
        y_layout({n, c, SymEngine::add({h, pads[0], pads[1]}), SymEngine::add({w, pads[2], pads[3]})});
    types::Tensor y_tensor(base_desc, y_layout);
    math::tensor::TensorLayout x_layout({n, c, h, w});
    types::Tensor x_tensor(base_desc, x_layout);

    auto& block = builder.add_block(root);
    auto& y_access = builder.add_access(block, "y");
    auto& x_access = builder.add_access(block, "x");
    auto& val_access = builder.add_access(block, "val");
    auto& libnode =
        builder.add_library_node<math::tensor::ConstPaddingNode>(block, DebugInfo(), pads, y_layout, x_layout);
    builder.add_computational_memlet(block, y_access, libnode, "_y", {}, desc);
    builder.add_computational_memlet(block, x_access, libnode, "_x", {}, x_tensor);
    builder.add_computational_memlet(block, val_access, libnode, "_val", {}, base_desc);

    ASSERT_THROW(sdfg.validate(), InvalidSDFGException);
}

TEST(ConstPaddingNodeTest, validate_memlet_tensor_x) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("y", desc, true);
    builder.add_container("x", desc, true);
    builder.add_container("val", base_desc, true);

    auto n = symbolic::integer(4);
    auto c = symbolic::integer(3);
    auto h = symbolic::integer(8);
    auto w = symbolic::integer(8);
    symbolic::MultiExpression pads = {symbolic::one(), symbolic::zero(), symbolic::zero(), symbolic::one()};
    math::tensor::TensorLayout
        y_layout({n, c, SymEngine::add({h, pads[0], pads[1]}), SymEngine::add({w, pads[2], pads[3]})});
    types::Tensor y_tensor(base_desc, y_layout);
    math::tensor::TensorLayout x_layout({n, c, h, w});
    types::Tensor x_tensor(base_desc, x_layout);

    auto& block = builder.add_block(root);
    auto& y_access = builder.add_access(block, "y");
    auto& x_access = builder.add_access(block, "x");
    auto& val_access = builder.add_access(block, "val");
    auto& libnode =
        builder.add_library_node<math::tensor::ConstPaddingNode>(block, DebugInfo(), pads, y_layout, x_layout);
    builder.add_computational_memlet(block, y_access, libnode, "_y", {}, y_tensor);
    builder.add_computational_memlet(block, x_access, libnode, "_x", {}, desc);
    builder.add_computational_memlet(block, val_access, libnode, "_val", {}, base_desc);

    ASSERT_THROW(sdfg.validate(), InvalidSDFGException);
}

TEST(ConstPaddingNodeTest, validate_tensor_layout_y) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("y", desc, true);
    builder.add_container("x", desc, true);
    builder.add_container("val", base_desc, true);

    auto n = symbolic::integer(4);
    auto c = symbolic::integer(3);
    auto h = symbolic::integer(8);
    auto w = symbolic::integer(8);
    symbolic::MultiExpression pads = {symbolic::one(), symbolic::zero(), symbolic::zero(), symbolic::one()};
    math::tensor::TensorLayout
        y_layout({n, c, SymEngine::add({h, pads[0], pads[1]}), SymEngine::add({w, pads[2], pads[3]})});
    types::Tensor y_tensor(base_desc, y_layout);
    math::tensor::TensorLayout x_layout({n, c, h, w});
    types::Tensor x_tensor(base_desc, x_layout);

    auto& block = builder.add_block(root);
    auto& y_access = builder.add_access(block, "y");
    auto& x_access = builder.add_access(block, "x");
    auto& val_access = builder.add_access(block, "val");
    auto& libnode =
        builder.add_library_node<math::tensor::ConstPaddingNode>(block, DebugInfo(), pads, y_layout, x_layout);
    builder.add_computational_memlet(block, y_access, libnode, "_y", {}, x_tensor);
    builder.add_computational_memlet(block, x_access, libnode, "_x", {}, x_tensor);
    builder.add_computational_memlet(block, val_access, libnode, "_val", {}, base_desc);

    ASSERT_THROW(sdfg.validate(), InvalidSDFGException);
}

TEST(ConstPaddingNodeTest, validate_tensor_layout_x) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("y", desc, true);
    builder.add_container("x", desc, true);
    builder.add_container("val", base_desc, true);

    auto n = symbolic::integer(4);
    auto c = symbolic::integer(3);
    auto h = symbolic::integer(8);
    auto w = symbolic::integer(8);
    symbolic::MultiExpression pads = {symbolic::one(), symbolic::zero(), symbolic::zero(), symbolic::one()};
    math::tensor::TensorLayout
        y_layout({n, c, SymEngine::add({h, pads[0], pads[1]}), SymEngine::add({w, pads[2], pads[3]})});
    types::Tensor y_tensor(base_desc, y_layout);
    math::tensor::TensorLayout x_layout({n, c, h, w});
    types::Tensor x_tensor(base_desc, x_layout);

    auto& block = builder.add_block(root);
    auto& y_access = builder.add_access(block, "y");
    auto& x_access = builder.add_access(block, "x");
    auto& val_access = builder.add_access(block, "val");
    auto& libnode =
        builder.add_library_node<math::tensor::ConstPaddingNode>(block, DebugInfo(), pads, y_layout, x_layout);
    builder.add_computational_memlet(block, y_access, libnode, "_y", {}, y_tensor);
    builder.add_computational_memlet(block, x_access, libnode, "_x", {}, y_tensor);
    builder.add_computational_memlet(block, val_access, libnode, "_val", {}, base_desc);

    ASSERT_THROW(sdfg.validate(), InvalidSDFGException);
}

TEST(ConstPaddingNodeTest, validate_layout_shapes) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("y", desc, true);
    builder.add_container("x", desc, true);
    builder.add_container("val", base_desc, true);

    auto n = symbolic::integer(4);
    auto c = symbolic::integer(3);
    auto h = symbolic::integer(8);
    auto w = symbolic::integer(8);
    symbolic::MultiExpression pads = {symbolic::one(), symbolic::zero(), symbolic::zero(), symbolic::one()};
    math::tensor::TensorLayout
        y_layout({n, c, SymEngine::add({h, pads[0], pads[1]}), SymEngine::add({w, pads[2], pads[3]})});
    types::Tensor y_tensor(base_desc, y_layout);
    math::tensor::TensorLayout x_layout({c, h, w});
    types::Tensor x_tensor(base_desc, x_layout);

    auto& block = builder.add_block(root);
    auto& y_access = builder.add_access(block, "y");
    auto& x_access = builder.add_access(block, "x");
    auto& val_access = builder.add_access(block, "val");
    auto& libnode =
        builder.add_library_node<math::tensor::ConstPaddingNode>(block, DebugInfo(), pads, y_layout, x_layout);
    builder.add_computational_memlet(block, y_access, libnode, "_y", {}, y_tensor);
    builder.add_computational_memlet(block, x_access, libnode, "_x", {}, x_tensor);
    builder.add_computational_memlet(block, val_access, libnode, "_val", {}, base_desc);

    ASSERT_THROW(sdfg.validate(), InvalidSDFGException);
}

TEST(ConstPaddingNodeTest, validate_pads_size_even) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("y", desc, true);
    builder.add_container("x", desc, true);
    builder.add_container("val", base_desc, true);

    auto n = symbolic::integer(4);
    auto c = symbolic::integer(3);
    auto h = symbolic::integer(8);
    auto w = symbolic::integer(8);
    symbolic::MultiExpression pads = {
        symbolic::one(), symbolic::zero(), symbolic::zero(), symbolic::one(), symbolic::one()
    };
    math::tensor::TensorLayout
        y_layout({n, c, SymEngine::add({h, pads[0], pads[1]}), SymEngine::add({w, pads[2], pads[3]})});
    types::Tensor y_tensor(base_desc, y_layout);
    math::tensor::TensorLayout x_layout({n, c, h, w});
    types::Tensor x_tensor(base_desc, x_layout);

    auto& block = builder.add_block(root);
    auto& y_access = builder.add_access(block, "y");
    auto& x_access = builder.add_access(block, "x");
    auto& val_access = builder.add_access(block, "val");
    auto& libnode =
        builder.add_library_node<math::tensor::ConstPaddingNode>(block, DebugInfo(), pads, y_layout, x_layout);
    builder.add_computational_memlet(block, y_access, libnode, "_y", {}, y_tensor);
    builder.add_computational_memlet(block, x_access, libnode, "_x", {}, x_tensor);
    builder.add_computational_memlet(block, val_access, libnode, "_val", {}, base_desc);

    ASSERT_THROW(sdfg.validate(), InvalidSDFGException);
}

TEST(ConstPaddingNodeTest, validate_pads) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("y", desc, true);
    builder.add_container("x", desc, true);
    builder.add_container("val", base_desc, true);

    auto n = symbolic::integer(4);
    auto c = symbolic::integer(3);
    auto h = symbolic::integer(8);
    auto w = symbolic::integer(8);
    symbolic::MultiExpression pads = {symbolic::one(), symbolic::zero(), symbolic::zero(), symbolic::one()};
    math::tensor::TensorLayout y_layout({n, c, h, w});
    types::Tensor y_tensor(base_desc, y_layout);
    math::tensor::TensorLayout x_layout({n, c, h, w});
    types::Tensor x_tensor(base_desc, x_layout);

    auto& block = builder.add_block(root);
    auto& y_access = builder.add_access(block, "y");
    auto& x_access = builder.add_access(block, "x");
    auto& val_access = builder.add_access(block, "val");
    auto& libnode =
        builder.add_library_node<math::tensor::ConstPaddingNode>(block, DebugInfo(), pads, y_layout, x_layout);
    builder.add_computational_memlet(block, y_access, libnode, "_y", {}, y_tensor);
    builder.add_computational_memlet(block, x_access, libnode, "_x", {}, x_tensor);
    builder.add_computational_memlet(block, val_access, libnode, "_val", {}, base_desc);

    ASSERT_THROW(sdfg.validate(), InvalidSDFGException);
}
