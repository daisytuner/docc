#include "sdfg/data_flow/library_nodes/math/tensor/conditional_copy_node.h"

#include <memory>

#include <gtest/gtest.h>
#include <nlohmann/json_fwd.hpp>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_nodes/math/tensor/tensor_layout.h"
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

using namespace sdfg;

TEST(ConditionalTensorCopyNodeTest, symbolic) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    types::Scalar bool_scalar(types::PrimitiveType::Bool);
    types::Pointer bool_pointer(bool_scalar);
    builder.add_container("A", bool_pointer, true);
    builder.add_container("B", desc, true);
    builder.add_container("C", desc, true);
    builder.add_container("D", desc, true);

    types::Scalar sym_desc(types::PrimitiveType::Int64);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);
    builder.add_container("k", sym_desc);
    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");
    auto k = symbolic::symbol("k");

    symbolic::MultiExpression shape({i, j, k});
    math::tensor::TensorLayout A_layout(shape);
    types::Tensor A_tensor(bool_scalar, A_layout);
    math::tensor::TensorLayout B_layout(shape);
    types::Tensor B_tensor(base_desc, B_layout);
    math::tensor::TensorLayout C_layout(shape);
    types::Tensor C_tensor(base_desc, C_layout);
    math::tensor::TensorLayout D_layout(shape);
    types::Tensor D_tensor(base_desc, D_layout);

    auto& block = builder.add_block(root);
    auto& A_access = builder.add_access(block, "A");
    auto& B_access = builder.add_access(block, "B");
    auto& C_access = builder.add_access(block, "C");
    auto& D_access = builder.add_access(block, "D");
    auto& libnode = builder.add_library_node<
        math::tensor::ConditionalTensorCopyNode>(block, DebugInfo(), A_layout, B_layout, C_layout, D_layout);
    builder.add_computational_memlet(block, A_access, libnode, "Mask", {}, A_tensor);
    builder.add_computational_memlet(block, B_access, libnode, "X1", {}, B_tensor);
    builder.add_computational_memlet(block, C_access, libnode, "X2", {}, C_tensor);
    builder.add_computational_memlet(block, D_access, libnode, "Y", {}, D_tensor);

    ASSERT_NO_THROW(sdfg.validate());
    auto& conditional_copy_node = static_cast<math::tensor::ConditionalTensorCopyNode&>(libnode);
    EXPECT_TRUE(conditional_copy_node.supports_integer_types());
    EXPECT_TRUE(symbolic::eq(conditional_copy_node.flop(), symbolic::zero()));

    auto symbols = conditional_copy_node.symbols();
    EXPECT_EQ(symbols.size(), 3);
    EXPECT_TRUE(symbols.contains(i));
    EXPECT_TRUE(symbols.contains(j));
    EXPECT_TRUE(symbols.contains(k));

    builder.add_container("l", sym_desc);
    builder.add_container("m", sym_desc);
    builder.add_container("n", sym_desc);
    auto l = symbolic::symbol("l");
    auto m = symbolic::symbol("m");
    auto n = symbolic::symbol("n");

    builder.replace_symbols(i, l);

    symbolic::ExpressionMapping replacements({{j, m}, {k, n}});
    builder.replace_symbols(replacements);

    ASSERT_NO_THROW(sdfg.validate());

    symbols = conditional_copy_node.symbols();
    EXPECT_EQ(symbols.size(), 3);
    EXPECT_TRUE(symbols.contains(l));
    EXPECT_TRUE(symbols.contains(m));
    EXPECT_TRUE(symbols.contains(n));
}

TEST(ConditionalTensorCopyNodeTest, expansion) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    types::Scalar bool_scalar(types::PrimitiveType::Bool);
    types::Pointer bool_pointer(bool_scalar);
    builder.add_container("A", bool_pointer, true);
    builder.add_container("B", desc, true);
    builder.add_container("C", desc, true);
    builder.add_container("D", desc, true);

    symbolic::MultiExpression shape({symbolic::integer(32), symbolic::integer(32)});
    math::tensor::TensorLayout A_layout(shape);
    types::Tensor A_tensor(bool_scalar, A_layout);
    math::tensor::TensorLayout B_layout(shape);
    types::Tensor B_tensor(base_desc, B_layout);
    math::tensor::TensorLayout C_layout(shape);
    types::Tensor C_tensor(base_desc, C_layout);
    math::tensor::TensorLayout D_layout(shape);
    types::Tensor D_tensor(base_desc, D_layout);

    auto& block = builder.add_block(root);
    auto& A_access = builder.add_access(block, "A");
    auto& B_access = builder.add_access(block, "B");
    auto& C_access = builder.add_access(block, "C");
    auto& D_access = builder.add_access(block, "D");
    auto& libnode = builder.add_library_node<
        math::tensor::ConditionalTensorCopyNode>(block, DebugInfo(), A_layout, B_layout, C_layout, D_layout);
    builder.add_computational_memlet(block, A_access, libnode, "Mask", {}, A_tensor);
    builder.add_computational_memlet(block, B_access, libnode, "X1", {}, B_tensor);
    builder.add_computational_memlet(block, C_access, libnode, "X2", {}, C_tensor);
    builder.add_computational_memlet(block, D_access, libnode, "Y", {}, D_tensor);

    ASSERT_NO_THROW(sdfg.validate());
    dump_sdfg(sdfg, "0.before");

    analysis::AnalysisManager analysis_manager(sdfg);
    passes::LibraryNodeExpansionPass expansion;
    ASSERT_TRUE(expansion.run(builder, analysis_manager));
    ASSERT_NO_THROW(sdfg.validate());
    dump_sdfg(sdfg, "1.after");
}

TEST(ConditionalTensorCopyNodeTest, serialization) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    types::Scalar bool_scalar(types::PrimitiveType::Bool);
    types::Pointer bool_pointer(bool_scalar);
    builder.add_container("A", bool_pointer, true);
    builder.add_container("B", desc, true);
    builder.add_container("C", desc, true);
    builder.add_container("D", desc, true);

    symbolic::MultiExpression shape({symbolic::integer(32), symbolic::integer(32)});
    math::tensor::TensorLayout A_layout(shape);
    types::Tensor A_tensor(bool_scalar, A_layout);
    math::tensor::TensorLayout B_layout(shape);
    types::Tensor B_tensor(base_desc, B_layout);
    math::tensor::TensorLayout C_layout(shape);
    types::Tensor C_tensor(base_desc, C_layout);
    math::tensor::TensorLayout D_layout(shape);
    types::Tensor D_tensor(base_desc, D_layout);

    auto& block = builder.add_block(root);
    auto& A_access = builder.add_access(block, "A");
    auto& B_access = builder.add_access(block, "B");
    auto& C_access = builder.add_access(block, "C");
    auto& D_access = builder.add_access(block, "D");
    auto& libnode = builder.add_library_node<
        math::tensor::ConditionalTensorCopyNode>(block, DebugInfo(), A_layout, B_layout, C_layout, D_layout);
    builder.add_computational_memlet(block, A_access, libnode, "Mask", {}, A_tensor);
    builder.add_computational_memlet(block, B_access, libnode, "X1", {}, B_tensor);
    builder.add_computational_memlet(block, C_access, libnode, "X2", {}, C_tensor);
    builder.add_computational_memlet(block, D_access, libnode, "Y", {}, D_tensor);

    ASSERT_NO_THROW(sdfg.validate());

    serializer::JSONSerializer serializer;
    nlohmann::json j;
    ASSERT_NO_THROW(j = serializer.serialize(sdfg));

    std::unique_ptr<StructuredSDFG> new_sdfg;
    ASSERT_NO_THROW(new_sdfg = serializer.deserialize(j));
    ASSERT_NO_THROW(new_sdfg->validate());
}

TEST(ConditionalTensorCopyNodeTest, validate_tensor_type_mask) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    types::Scalar bool_scalar(types::PrimitiveType::Bool);
    types::Pointer bool_pointer(bool_scalar);
    builder.add_container("A", bool_pointer, true);
    builder.add_container("B", desc, true);
    builder.add_container("C", desc, true);
    builder.add_container("D", desc, true);

    symbolic::MultiExpression shape({symbolic::integer(32), symbolic::integer(32)});
    math::tensor::TensorLayout A_layout(shape);
    types::Tensor A_tensor(bool_scalar, A_layout);
    math::tensor::TensorLayout B_layout(shape);
    types::Tensor B_tensor(base_desc, B_layout);
    math::tensor::TensorLayout C_layout(shape);
    types::Tensor C_tensor(base_desc, C_layout);
    math::tensor::TensorLayout D_layout(shape);
    types::Tensor D_tensor(base_desc, D_layout);

    auto& block = builder.add_block(root);
    auto& A_access = builder.add_access(block, "A");
    auto& B_access = builder.add_access(block, "B");
    auto& C_access = builder.add_access(block, "C");
    auto& D_access = builder.add_access(block, "D");
    auto& libnode = builder.add_library_node<
        math::tensor::ConditionalTensorCopyNode>(block, DebugInfo(), A_layout, B_layout, C_layout, D_layout);
    builder.add_computational_memlet(block, A_access, libnode, "Mask", {}, bool_pointer);
    builder.add_computational_memlet(block, B_access, libnode, "X1", {}, B_tensor);
    builder.add_computational_memlet(block, C_access, libnode, "X2", {}, C_tensor);
    builder.add_computational_memlet(block, D_access, libnode, "Y", {}, D_tensor);

    EXPECT_THROW(sdfg.validate(), InvalidSDFGException);
}

TEST(ConditionalTensorCopyNodeTest, validate_tensor_type_x1) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    types::Scalar bool_scalar(types::PrimitiveType::Bool);
    types::Pointer bool_pointer(bool_scalar);
    builder.add_container("A", bool_pointer, true);
    builder.add_container("B", desc, true);
    builder.add_container("C", desc, true);
    builder.add_container("D", desc, true);

    symbolic::MultiExpression shape({symbolic::integer(32), symbolic::integer(32)});
    math::tensor::TensorLayout A_layout(shape);
    types::Tensor A_tensor(bool_scalar, A_layout);
    math::tensor::TensorLayout B_layout(shape);
    types::Tensor B_tensor(base_desc, B_layout);
    math::tensor::TensorLayout C_layout(shape);
    types::Tensor C_tensor(base_desc, C_layout);
    math::tensor::TensorLayout D_layout(shape);
    types::Tensor D_tensor(base_desc, D_layout);

    auto& block = builder.add_block(root);
    auto& A_access = builder.add_access(block, "A");
    auto& B_access = builder.add_access(block, "B");
    auto& C_access = builder.add_access(block, "C");
    auto& D_access = builder.add_access(block, "D");
    auto& libnode = builder.add_library_node<
        math::tensor::ConditionalTensorCopyNode>(block, DebugInfo(), A_layout, B_layout, C_layout, D_layout);
    builder.add_computational_memlet(block, A_access, libnode, "Mask", {}, A_tensor);
    builder.add_computational_memlet(block, B_access, libnode, "X1", {}, desc);
    builder.add_computational_memlet(block, C_access, libnode, "X2", {}, C_tensor);
    builder.add_computational_memlet(block, D_access, libnode, "Y", {}, D_tensor);

    EXPECT_THROW(sdfg.validate(), InvalidSDFGException);
}

TEST(ConditionalTensorCopyNodeTest, validate_tensor_type_x2) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    types::Scalar bool_scalar(types::PrimitiveType::Bool);
    types::Pointer bool_pointer(bool_scalar);
    builder.add_container("A", bool_pointer, true);
    builder.add_container("B", desc, true);
    builder.add_container("C", desc, true);
    builder.add_container("D", desc, true);

    symbolic::MultiExpression shape({symbolic::integer(32), symbolic::integer(32)});
    math::tensor::TensorLayout A_layout(shape);
    types::Tensor A_tensor(bool_scalar, A_layout);
    math::tensor::TensorLayout B_layout(shape);
    types::Tensor B_tensor(base_desc, B_layout);
    math::tensor::TensorLayout C_layout(shape);
    types::Tensor C_tensor(base_desc, C_layout);
    math::tensor::TensorLayout D_layout(shape);
    types::Tensor D_tensor(base_desc, D_layout);

    auto& block = builder.add_block(root);
    auto& A_access = builder.add_access(block, "A");
    auto& B_access = builder.add_access(block, "B");
    auto& C_access = builder.add_access(block, "C");
    auto& D_access = builder.add_access(block, "D");
    auto& libnode = builder.add_library_node<
        math::tensor::ConditionalTensorCopyNode>(block, DebugInfo(), A_layout, B_layout, C_layout, D_layout);
    builder.add_computational_memlet(block, A_access, libnode, "Mask", {}, A_tensor);
    builder.add_computational_memlet(block, B_access, libnode, "X1", {}, B_tensor);
    builder.add_computational_memlet(block, C_access, libnode, "X2", {}, desc);
    builder.add_computational_memlet(block, D_access, libnode, "Y", {}, D_tensor);

    EXPECT_THROW(sdfg.validate(), InvalidSDFGException);
}

TEST(ConditionalTensorCopyNodeTest, validate_tensor_type_y) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    types::Scalar bool_scalar(types::PrimitiveType::Bool);
    types::Pointer bool_pointer(bool_scalar);
    builder.add_container("A", bool_pointer, true);
    builder.add_container("B", desc, true);
    builder.add_container("C", desc, true);
    builder.add_container("D", desc, true);

    symbolic::MultiExpression shape({symbolic::integer(32), symbolic::integer(32)});
    math::tensor::TensorLayout A_layout(shape);
    types::Tensor A_tensor(bool_scalar, A_layout);
    math::tensor::TensorLayout B_layout(shape);
    types::Tensor B_tensor(base_desc, B_layout);
    math::tensor::TensorLayout C_layout(shape);
    types::Tensor C_tensor(base_desc, C_layout);
    math::tensor::TensorLayout D_layout(shape);
    types::Tensor D_tensor(base_desc, D_layout);

    auto& block = builder.add_block(root);
    auto& A_access = builder.add_access(block, "A");
    auto& B_access = builder.add_access(block, "B");
    auto& C_access = builder.add_access(block, "C");
    auto& D_access = builder.add_access(block, "D");
    auto& libnode = builder.add_library_node<
        math::tensor::ConditionalTensorCopyNode>(block, DebugInfo(), A_layout, B_layout, C_layout, D_layout);
    builder.add_computational_memlet(block, A_access, libnode, "Mask", {}, A_tensor);
    builder.add_computational_memlet(block, B_access, libnode, "X1", {}, B_tensor);
    builder.add_computational_memlet(block, C_access, libnode, "X2", {}, C_tensor);
    builder.add_computational_memlet(block, D_access, libnode, "Y", {}, desc);

    EXPECT_THROW(sdfg.validate(), InvalidSDFGException);
}

TEST(ConditionalTensorCopyNodeTest, validate_tensor_layout_mismatch_mask) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    types::Scalar bool_scalar(types::PrimitiveType::Bool);
    types::Pointer bool_pointer(bool_scalar);
    builder.add_container("A", bool_pointer, true);
    builder.add_container("B", desc, true);
    builder.add_container("C", desc, true);
    builder.add_container("D", desc, true);

    math::tensor::TensorLayout A_layout({symbolic::integer(32), symbolic::integer(32)});
    types::Tensor A_tensor(bool_scalar, math::tensor::TensorLayout({symbolic::integer(32)}));
    math::tensor::TensorLayout B_layout({symbolic::integer(32), symbolic::integer(32)});
    types::Tensor B_tensor(base_desc, B_layout);
    math::tensor::TensorLayout C_layout({symbolic::integer(32), symbolic::integer(32)});
    types::Tensor C_tensor(base_desc, C_layout);
    math::tensor::TensorLayout D_layout({symbolic::integer(32), symbolic::integer(32)});
    types::Tensor D_tensor(base_desc, D_layout);

    auto& block = builder.add_block(root);
    auto& A_access = builder.add_access(block, "A");
    auto& B_access = builder.add_access(block, "B");
    auto& C_access = builder.add_access(block, "C");
    auto& D_access = builder.add_access(block, "D");
    auto& libnode = builder.add_library_node<
        math::tensor::ConditionalTensorCopyNode>(block, DebugInfo(), A_layout, B_layout, C_layout, D_layout);
    builder.add_computational_memlet(block, A_access, libnode, "Mask", {}, A_tensor);
    builder.add_computational_memlet(block, B_access, libnode, "X1", {}, B_tensor);
    builder.add_computational_memlet(block, C_access, libnode, "X2", {}, C_tensor);
    builder.add_computational_memlet(block, D_access, libnode, "Y", {}, D_tensor);

    EXPECT_THROW(sdfg.validate(), InvalidSDFGException);
}

TEST(ConditionalTensorCopyNodeTest, validate_tensor_layout_mismatch_x1) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    types::Scalar bool_scalar(types::PrimitiveType::Bool);
    types::Pointer bool_pointer(bool_scalar);
    builder.add_container("A", bool_pointer, true);
    builder.add_container("B", desc, true);
    builder.add_container("C", desc, true);
    builder.add_container("D", desc, true);

    math::tensor::TensorLayout A_layout({symbolic::integer(32), symbolic::integer(32)});
    types::Tensor A_tensor(bool_scalar, A_layout);
    math::tensor::TensorLayout B_layout({symbolic::integer(32), symbolic::integer(32)});
    types::Tensor B_tensor(base_desc, math::tensor::TensorLayout({symbolic::integer(32)}));
    math::tensor::TensorLayout C_layout({symbolic::integer(32), symbolic::integer(32)});
    types::Tensor C_tensor(base_desc, C_layout);
    math::tensor::TensorLayout D_layout({symbolic::integer(32), symbolic::integer(32)});
    types::Tensor D_tensor(base_desc, D_layout);

    auto& block = builder.add_block(root);
    auto& A_access = builder.add_access(block, "A");
    auto& B_access = builder.add_access(block, "B");
    auto& C_access = builder.add_access(block, "C");
    auto& D_access = builder.add_access(block, "D");
    auto& libnode = builder.add_library_node<
        math::tensor::ConditionalTensorCopyNode>(block, DebugInfo(), A_layout, B_layout, C_layout, D_layout);
    builder.add_computational_memlet(block, A_access, libnode, "Mask", {}, A_tensor);
    builder.add_computational_memlet(block, B_access, libnode, "X1", {}, B_tensor);
    builder.add_computational_memlet(block, C_access, libnode, "X2", {}, C_tensor);
    builder.add_computational_memlet(block, D_access, libnode, "Y", {}, D_tensor);

    EXPECT_THROW(sdfg.validate(), InvalidSDFGException);
}

TEST(ConditionalTensorCopyNodeTest, validate_tensor_layout_mismatch_x2) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    types::Scalar bool_scalar(types::PrimitiveType::Bool);
    types::Pointer bool_pointer(bool_scalar);
    builder.add_container("A", bool_pointer, true);
    builder.add_container("B", desc, true);
    builder.add_container("C", desc, true);
    builder.add_container("D", desc, true);

    math::tensor::TensorLayout A_layout({symbolic::integer(32), symbolic::integer(32)});
    types::Tensor A_tensor(bool_scalar, A_layout);
    math::tensor::TensorLayout B_layout({symbolic::integer(32), symbolic::integer(32)});
    types::Tensor B_tensor(base_desc, B_layout);
    math::tensor::TensorLayout C_layout({symbolic::integer(32), symbolic::integer(32)});
    types::Tensor C_tensor(base_desc, math::tensor::TensorLayout({symbolic::integer(32)}));
    math::tensor::TensorLayout D_layout({symbolic::integer(32), symbolic::integer(32)});
    types::Tensor D_tensor(base_desc, D_layout);

    auto& block = builder.add_block(root);
    auto& A_access = builder.add_access(block, "A");
    auto& B_access = builder.add_access(block, "B");
    auto& C_access = builder.add_access(block, "C");
    auto& D_access = builder.add_access(block, "D");
    auto& libnode = builder.add_library_node<
        math::tensor::ConditionalTensorCopyNode>(block, DebugInfo(), A_layout, B_layout, C_layout, D_layout);
    builder.add_computational_memlet(block, A_access, libnode, "Mask", {}, A_tensor);
    builder.add_computational_memlet(block, B_access, libnode, "X1", {}, B_tensor);
    builder.add_computational_memlet(block, C_access, libnode, "X2", {}, C_tensor);
    builder.add_computational_memlet(block, D_access, libnode, "Y", {}, D_tensor);

    EXPECT_THROW(sdfg.validate(), InvalidSDFGException);
}

TEST(ConditionalTensorCopyNodeTest, validate_tensor_layout_mismatch_y) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    types::Scalar bool_scalar(types::PrimitiveType::Bool);
    types::Pointer bool_pointer(bool_scalar);
    builder.add_container("A", bool_pointer, true);
    builder.add_container("B", desc, true);
    builder.add_container("C", desc, true);
    builder.add_container("D", desc, true);

    math::tensor::TensorLayout A_layout({symbolic::integer(32), symbolic::integer(32)});
    types::Tensor A_tensor(bool_scalar, A_layout);
    math::tensor::TensorLayout B_layout({symbolic::integer(32), symbolic::integer(32)});
    types::Tensor B_tensor(base_desc, B_layout);
    math::tensor::TensorLayout C_layout({symbolic::integer(32), symbolic::integer(32)});
    types::Tensor C_tensor(base_desc, C_layout);
    math::tensor::TensorLayout D_layout({symbolic::integer(32), symbolic::integer(32)});
    types::Tensor D_tensor(base_desc, math::tensor::TensorLayout({symbolic::integer(32)}));

    auto& block = builder.add_block(root);
    auto& A_access = builder.add_access(block, "A");
    auto& B_access = builder.add_access(block, "B");
    auto& C_access = builder.add_access(block, "C");
    auto& D_access = builder.add_access(block, "D");
    auto& libnode = builder.add_library_node<
        math::tensor::ConditionalTensorCopyNode>(block, DebugInfo(), A_layout, B_layout, C_layout, D_layout);
    builder.add_computational_memlet(block, A_access, libnode, "Mask", {}, A_tensor);
    builder.add_computational_memlet(block, B_access, libnode, "X1", {}, B_tensor);
    builder.add_computational_memlet(block, C_access, libnode, "X2", {}, C_tensor);
    builder.add_computational_memlet(block, D_access, libnode, "Y", {}, D_tensor);

    EXPECT_THROW(sdfg.validate(), InvalidSDFGException);
}

TEST(ConditionalTensorCopyNodeTest, validate_mismatching_shape_dims) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    types::Scalar bool_scalar(types::PrimitiveType::Bool);
    types::Pointer bool_pointer(bool_scalar);
    builder.add_container("A", bool_pointer, true);
    builder.add_container("B", desc, true);
    builder.add_container("C", desc, true);
    builder.add_container("D", desc, true);

    symbolic::MultiExpression shape({symbolic::integer(32), symbolic::integer(32)});
    math::tensor::TensorLayout A_layout(shape);
    types::Tensor A_tensor(bool_scalar, A_layout);
    math::tensor::TensorLayout B_layout({symbolic::integer(32)});
    types::Tensor B_tensor(base_desc, B_layout);
    math::tensor::TensorLayout C_layout(shape);
    types::Tensor C_tensor(base_desc, C_layout);
    math::tensor::TensorLayout D_layout(shape);
    types::Tensor D_tensor(base_desc, D_layout);

    auto& block = builder.add_block(root);
    auto& A_access = builder.add_access(block, "A");
    auto& B_access = builder.add_access(block, "B");
    auto& C_access = builder.add_access(block, "C");
    auto& D_access = builder.add_access(block, "D");
    auto& libnode = builder.add_library_node<
        math::tensor::ConditionalTensorCopyNode>(block, DebugInfo(), A_layout, B_layout, C_layout, D_layout);
    builder.add_computational_memlet(block, A_access, libnode, "Mask", {}, A_tensor);
    builder.add_computational_memlet(block, B_access, libnode, "X1", {}, B_tensor);
    builder.add_computational_memlet(block, C_access, libnode, "X2", {}, C_tensor);
    builder.add_computational_memlet(block, D_access, libnode, "Y", {}, D_tensor);

    EXPECT_THROW(sdfg.validate(), InvalidSDFGException);
}

TEST(ConditionalTensorCopyNodeTest, validate_mismatching_shape_content) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    types::Scalar bool_scalar(types::PrimitiveType::Bool);
    types::Pointer bool_pointer(bool_scalar);
    builder.add_container("A", bool_pointer, true);
    builder.add_container("B", desc, true);
    builder.add_container("C", desc, true);
    builder.add_container("D", desc, true);

    symbolic::MultiExpression shape({symbolic::integer(32), symbolic::integer(32)});
    math::tensor::TensorLayout A_layout(shape);
    types::Tensor A_tensor(bool_scalar, A_layout);
    math::tensor::TensorLayout B_layout({symbolic::integer(32), symbolic::integer(64)});
    types::Tensor B_tensor(base_desc, B_layout);
    math::tensor::TensorLayout C_layout(shape);
    types::Tensor C_tensor(base_desc, C_layout);
    math::tensor::TensorLayout D_layout(shape);
    types::Tensor D_tensor(base_desc, D_layout);

    auto& block = builder.add_block(root);
    auto& A_access = builder.add_access(block, "A");
    auto& B_access = builder.add_access(block, "B");
    auto& C_access = builder.add_access(block, "C");
    auto& D_access = builder.add_access(block, "D");
    auto& libnode = builder.add_library_node<
        math::tensor::ConditionalTensorCopyNode>(block, DebugInfo(), A_layout, B_layout, C_layout, D_layout);
    builder.add_computational_memlet(block, A_access, libnode, "Mask", {}, A_tensor);
    builder.add_computational_memlet(block, B_access, libnode, "X1", {}, B_tensor);
    builder.add_computational_memlet(block, C_access, libnode, "X2", {}, C_tensor);
    builder.add_computational_memlet(block, D_access, libnode, "Y", {}, D_tensor);

    EXPECT_THROW(sdfg.validate(), InvalidSDFGException);
}

TEST(ConditionalTensorCopyNodeTest, validate_mask_bool) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("A", desc, true);
    builder.add_container("B", desc, true);
    builder.add_container("C", desc, true);
    builder.add_container("D", desc, true);

    symbolic::MultiExpression shape({symbolic::integer(32), symbolic::integer(32)});
    math::tensor::TensorLayout A_layout(shape);
    types::Tensor A_tensor(base_desc, A_layout);
    math::tensor::TensorLayout B_layout(shape);
    types::Tensor B_tensor(base_desc, B_layout);
    math::tensor::TensorLayout C_layout(shape);
    types::Tensor C_tensor(base_desc, C_layout);
    math::tensor::TensorLayout D_layout(shape);
    types::Tensor D_tensor(base_desc, D_layout);

    auto& block = builder.add_block(root);
    auto& A_access = builder.add_access(block, "A");
    auto& B_access = builder.add_access(block, "B");
    auto& C_access = builder.add_access(block, "C");
    auto& D_access = builder.add_access(block, "D");
    auto& libnode = builder.add_library_node<
        math::tensor::ConditionalTensorCopyNode>(block, DebugInfo(), A_layout, B_layout, C_layout, D_layout);
    builder.add_computational_memlet(block, A_access, libnode, "Mask", {}, A_tensor);
    builder.add_computational_memlet(block, B_access, libnode, "X1", {}, B_tensor);
    builder.add_computational_memlet(block, C_access, libnode, "X2", {}, C_tensor);
    builder.add_computational_memlet(block, D_access, libnode, "Y", {}, D_tensor);

    EXPECT_THROW(sdfg.validate(), InvalidSDFGException);
}

TEST(ConditionalTensorCopyNodeTest, validate_same_element_type) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    types::Scalar bool_scalar(types::PrimitiveType::Bool);
    types::Pointer bool_pointer(bool_scalar);
    builder.add_container("A", bool_pointer, true);
    builder.add_container("B", desc, true);
    builder.add_container("C", bool_pointer, true);
    builder.add_container("D", desc, true);

    symbolic::MultiExpression shape({symbolic::integer(32), symbolic::integer(32)});
    math::tensor::TensorLayout A_layout(shape);
    types::Tensor A_tensor(bool_scalar, A_layout);
    math::tensor::TensorLayout B_layout(shape);
    types::Tensor B_tensor(base_desc, B_layout);
    math::tensor::TensorLayout C_layout(shape);
    types::Tensor C_tensor(bool_scalar, C_layout);
    math::tensor::TensorLayout D_layout(shape);
    types::Tensor D_tensor(base_desc, D_layout);

    auto& block = builder.add_block(root);
    auto& A_access = builder.add_access(block, "A");
    auto& B_access = builder.add_access(block, "B");
    auto& C_access = builder.add_access(block, "C");
    auto& D_access = builder.add_access(block, "D");
    auto& libnode = builder.add_library_node<
        math::tensor::ConditionalTensorCopyNode>(block, DebugInfo(), A_layout, B_layout, C_layout, D_layout);
    builder.add_computational_memlet(block, A_access, libnode, "Mask", {}, A_tensor);
    builder.add_computational_memlet(block, B_access, libnode, "X1", {}, B_tensor);
    builder.add_computational_memlet(block, C_access, libnode, "X2", {}, C_tensor);
    builder.add_computational_memlet(block, D_access, libnode, "Y", {}, D_tensor);

    EXPECT_THROW(sdfg.validate(), InvalidSDFGException);
}
