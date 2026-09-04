#include "sdfg/data_flow/library_nodes/math/tensor/index_node.h"

#include <memory>
#include <string>
#include <vector>

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

using namespace sdfg;

TEST(IndexNodeTest, symbolic) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    types::Scalar sym_desc(types::PrimitiveType::Int64);
    types::Pointer index_desc(sym_desc);
    builder.add_container("X", desc, true);
    builder.add_container("Idx1", index_desc, true);
    builder.add_container("Idx2", index_desc, true);
    builder.add_container("Y", desc, true);
    builder.add_container("n", sym_desc, true);
    builder.add_container("a", sym_desc, true);
    builder.add_container("b", sym_desc, true);
    auto n = symbolic::symbol("n");
    auto a = symbolic::symbol("a");
    auto b = symbolic::symbol("b");

    math::tensor::TensorLayout X_layout({n, symbolic::integer(4), symbolic::integer(4), a, b});
    types::Tensor X_tensor(base_desc, X_layout);
    std::vector<math::tensor::TensorLayout> Idx_layouts;
    Idx_layouts.push_back(math::tensor::TensorLayout({symbolic::integer(2), symbolic::integer(2)}));
    Idx_layouts.push_back(math::tensor::TensorLayout({symbolic::integer(2), symbolic::integer(2)}));
    types::Tensor Idx1_tensor(sym_desc, Idx_layouts[0]);
    types::Tensor Idx2_tensor(sym_desc, Idx_layouts[1]);
    math::tensor::TensorLayout Y_layout({n, symbolic::integer(2), symbolic::integer(2), a, b});
    types::Tensor Y_tensor(base_desc, Y_layout);
    std::vector<long long> indices({1, 2});

    auto& block = builder.add_block(root);
    auto& X_access = builder.add_access(block, "X");
    auto& Idx1_access = builder.add_access(block, "Idx1");
    auto& Idx2_access = builder.add_access(block, "Idx2");
    auto& Y_access = builder.add_access(block, "Y");
    auto& libnode =
        builder.add_library_node<math::tensor::IndexNode>(block, DebugInfo(), indices, Y_layout, X_layout, Idx_layouts);
    builder.add_computational_memlet(block, X_access, libnode, "X", {}, X_tensor);
    builder.add_computational_memlet(block, Idx1_access, libnode, "I1", {}, Idx1_tensor);
    builder.add_computational_memlet(block, Idx2_access, libnode, "I2", {}, Idx2_tensor);
    builder.add_computational_memlet(block, Y_access, libnode, "Y", {}, Y_tensor);

    ASSERT_NO_THROW(sdfg.validate());
    auto& index_node = static_cast<math::tensor::IndexNode&>(libnode);
    EXPECT_TRUE(index_node.supports_integer_types());
    EXPECT_TRUE(symbolic::eq(index_node.flop(), symbolic::zero()));
    ASSERT_NO_THROW(index_node.toStr());

    auto symbols = index_node.symbols();
    EXPECT_EQ(symbols.size(), 3);
    EXPECT_TRUE(symbols.contains(n));
    EXPECT_TRUE(symbols.contains(a));
    EXPECT_TRUE(symbols.contains(b));

    builder.add_container("m", sym_desc);
    builder.add_container("c", sym_desc);
    builder.add_container("d", sym_desc);
    auto m = symbolic::symbol("m");
    auto c = symbolic::symbol("c");
    auto d = symbolic::symbol("d");

    builder.replace_symbols(n, m);

    symbolic::ExpressionMapping replacements({{a, c}, {b, d}});
    builder.replace_symbols(replacements);

    ASSERT_NO_THROW(sdfg.validate());

    symbols = index_node.symbols();
    EXPECT_EQ(symbols.size(), 3);
    EXPECT_TRUE(symbols.contains(m));
    EXPECT_TRUE(symbols.contains(c));
    EXPECT_TRUE(symbols.contains(d));

    auto check_pointer_access_meta =
        [](data_flow::PointerAccessType pam, bool no_capture, bool reads, bool writes, bool invalidate) -> void {
        EXPECT_EQ(pam->no_capture(), no_capture);
        EXPECT_EQ(pam->may_contain_reads(), reads);
        EXPECT_EQ(pam->may_contain_writes(), writes);
        EXPECT_EQ(pam->invalidated_after(), invalidate);
    };
    check_pointer_access_meta(
        index_node.pointer_access_type(math::tensor::IndexNode::Y_INPUT_IDX), true, false, true, false
    );
    check_pointer_access_meta(
        index_node.pointer_access_type(math::tensor::IndexNode::X_INPUT_IDX), true, true, false, false
    );
    check_pointer_access_meta(
        index_node.pointer_access_type(math::tensor::IndexNode::INDEX_INPUT_OFFSET + 0), true, true, false, false
    );
    check_pointer_access_meta(
        index_node.pointer_access_type(math::tensor::IndexNode::INDEX_INPUT_OFFSET + 1), true, true, false, false
    );
}

TEST(IndexNodeTest, cloning) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    types::Scalar sym_desc(types::PrimitiveType::Int64);
    types::Pointer index_desc(sym_desc);
    builder.add_container("X", desc, true);
    builder.add_container("Idx1", index_desc, true);
    builder.add_container("Idx2", index_desc, true);
    builder.add_container("Y", desc, true);

    math::tensor::TensorLayout X_layout(
        {symbolic::integer(4), symbolic::integer(4), symbolic::integer(4), symbolic::integer(4), symbolic::integer(4)}
    );
    types::Tensor X_tensor(base_desc, X_layout);
    std::vector<math::tensor::TensorLayout> Idx_layouts;
    Idx_layouts.push_back(math::tensor::TensorLayout({symbolic::integer(2), symbolic::integer(2)}));
    Idx_layouts.push_back(math::tensor::TensorLayout({symbolic::integer(2), symbolic::integer(2)}));
    types::Tensor Idx1_tensor(sym_desc, Idx_layouts[0]);
    types::Tensor Idx2_tensor(sym_desc, Idx_layouts[1]);
    math::tensor::TensorLayout Y_layout(
        {symbolic::integer(4), symbolic::integer(2), symbolic::integer(2), symbolic::integer(4), symbolic::integer(4)}
    );
    types::Tensor Y_tensor(base_desc, Y_layout);
    std::vector<long long> indices({1, 2});

    auto& block = builder.add_block(root);
    auto& X_access = builder.add_access(block, "X");
    auto& Idx1_access = builder.add_access(block, "Idx1");
    auto& Idx2_access = builder.add_access(block, "Idx2");
    auto& Y_access = builder.add_access(block, "Y");
    auto& libnode =
        builder.add_library_node<math::tensor::IndexNode>(block, DebugInfo(), indices, Y_layout, X_layout, Idx_layouts);
    builder.add_computational_memlet(block, X_access, libnode, "X", {}, X_tensor);
    builder.add_computational_memlet(block, Idx1_access, libnode, "I1", {}, Idx1_tensor);
    builder.add_computational_memlet(block, Idx2_access, libnode, "I2", {}, Idx2_tensor);
    builder.add_computational_memlet(block, Y_access, libnode, "Y", {}, Y_tensor);

    ASSERT_NO_THROW(sdfg.validate());
    builder::StructuredSDFGBuilder new_builder("sdfg_2", FunctionType_CPU);
    new_builder.add_container("X", desc, true);
    new_builder.add_container("Idx1", index_desc, true);
    new_builder.add_container("Idx2", index_desc, true);
    new_builder.add_container("Y", desc, true);
    deepcopy::StructuredSDFGDeepCopy deep_copy(new_builder, new_builder.subject().root(), root);
    deep_copy.copy();
    ASSERT_NO_THROW(new_builder.subject().validate());
}

TEST(IndexNodeTest, expansion_contiguous) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    types::Scalar sym_desc(types::PrimitiveType::Int64);
    types::Pointer index_desc(sym_desc);
    builder.add_container("X", desc, true);
    builder.add_container("Idx1", index_desc, true);
    builder.add_container("Idx2", index_desc, true);
    builder.add_container("Y", desc, true);

    math::tensor::TensorLayout X_layout(
        {symbolic::integer(4), symbolic::integer(4), symbolic::integer(4), symbolic::integer(4), symbolic::integer(4)}
    );
    types::Tensor X_tensor(base_desc, X_layout);
    std::vector<math::tensor::TensorLayout> Idx_layouts;
    Idx_layouts.push_back(math::tensor::TensorLayout({symbolic::integer(2), symbolic::integer(1)}));
    Idx_layouts.push_back(math::tensor::TensorLayout({symbolic::integer(2)}));
    types::Tensor Idx1_tensor(sym_desc, Idx_layouts[0]);
    types::Tensor Idx2_tensor(sym_desc, Idx_layouts[1]);
    math::tensor::TensorLayout Y_layout(
        {symbolic::integer(4), symbolic::integer(2), symbolic::integer(2), symbolic::integer(4), symbolic::integer(4)}
    );
    types::Tensor Y_tensor(base_desc, Y_layout);
    std::vector<long long> indices({1, 2});

    auto& block = builder.add_block(root);
    auto& X_access = builder.add_access(block, "X");
    auto& Idx1_access = builder.add_access(block, "Idx1");
    auto& Idx2_access = builder.add_access(block, "Idx2");
    auto& Y_access = builder.add_access(block, "Y");
    auto& libnode =
        builder.add_library_node<math::tensor::IndexNode>(block, DebugInfo(), indices, Y_layout, X_layout, Idx_layouts);
    builder.add_computational_memlet(block, X_access, libnode, "X", {}, X_tensor);
    builder.add_computational_memlet(block, Idx1_access, libnode, "I1", {}, Idx1_tensor);
    builder.add_computational_memlet(block, Idx2_access, libnode, "I2", {}, Idx2_tensor);
    builder.add_computational_memlet(block, Y_access, libnode, "Y", {}, Y_tensor);

    ASSERT_NO_THROW(sdfg.validate());
    dump_sdfg(sdfg, "0.before");

    analysis::AnalysisManager analysis_manager(sdfg);
    passes::LibraryNodeExpansionPass expansion;
    ASSERT_TRUE(expansion.run(builder, analysis_manager));
    ASSERT_NO_THROW(sdfg.validate());
    dump_sdfg(sdfg, "1.after");
}

TEST(IndexNodeTest, expansion_non_contiguous) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    types::Scalar sym_desc(types::PrimitiveType::Int64);
    types::Pointer index_desc(sym_desc);
    builder.add_container("X", desc, true);
    builder.add_container("Idx1", index_desc, true);
    builder.add_container("Idx2", index_desc, true);
    builder.add_container("Y", desc, true);

    math::tensor::TensorLayout X_layout(
        {symbolic::integer(4), symbolic::integer(4), symbolic::integer(4), symbolic::integer(4), symbolic::integer(4)}
    );
    types::Tensor X_tensor(base_desc, X_layout);
    std::vector<math::tensor::TensorLayout> Idx_layouts;
    Idx_layouts.push_back(math::tensor::TensorLayout({symbolic::integer(2)}));
    Idx_layouts.push_back(math::tensor::TensorLayout({symbolic::integer(2), symbolic::integer(1)}));
    types::Tensor Idx1_tensor(sym_desc, Idx_layouts[0]);
    types::Tensor Idx2_tensor(sym_desc, Idx_layouts[1]);
    math::tensor::TensorLayout Y_layout(
        {symbolic::integer(2), symbolic::integer(2), symbolic::integer(4), symbolic::integer(4), symbolic::integer(4)}
    );
    types::Tensor Y_tensor(base_desc, Y_layout);
    std::vector<long long> indices({1, 3});

    auto& block = builder.add_block(root);
    auto& X_access = builder.add_access(block, "X");
    auto& Idx1_access = builder.add_access(block, "Idx1");
    auto& Idx2_access = builder.add_access(block, "Idx2");
    auto& Y_access = builder.add_access(block, "Y");
    auto& libnode =
        builder.add_library_node<math::tensor::IndexNode>(block, DebugInfo(), indices, Y_layout, X_layout, Idx_layouts);
    builder.add_computational_memlet(block, X_access, libnode, "X", {}, X_tensor);
    builder.add_computational_memlet(block, Idx1_access, libnode, "I1", {}, Idx1_tensor);
    builder.add_computational_memlet(block, Idx2_access, libnode, "I3", {}, Idx2_tensor);
    builder.add_computational_memlet(block, Y_access, libnode, "Y", {}, Y_tensor);

    ASSERT_NO_THROW(sdfg.validate());
    dump_sdfg(sdfg, "0.before");

    analysis::AnalysisManager analysis_manager(sdfg);
    passes::LibraryNodeExpansionPass expansion;
    ASSERT_TRUE(expansion.run(builder, analysis_manager));
    ASSERT_NO_THROW(sdfg.validate());
    dump_sdfg(sdfg, "1.after");
}

TEST(IndexNodeTest, serialization) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    types::Scalar sym_desc(types::PrimitiveType::Int64);
    types::Pointer index_desc(sym_desc);
    builder.add_container("X", desc, true);
    builder.add_container("Idx1", index_desc, true);
    builder.add_container("Idx2", index_desc, true);
    builder.add_container("Y", desc, true);

    math::tensor::TensorLayout X_layout(
        {symbolic::integer(4), symbolic::integer(4), symbolic::integer(4), symbolic::integer(4), symbolic::integer(4)}
    );
    types::Tensor X_tensor(base_desc, X_layout);
    std::vector<math::tensor::TensorLayout> Idx_layouts;
    Idx_layouts.push_back(math::tensor::TensorLayout({symbolic::integer(2)}));
    Idx_layouts.push_back(math::tensor::TensorLayout({symbolic::integer(2), symbolic::integer(2)}));
    types::Tensor Idx1_tensor(sym_desc, Idx_layouts[0]);
    types::Tensor Idx2_tensor(sym_desc, Idx_layouts[1]);
    math::tensor::TensorLayout Y_layout(
        {symbolic::integer(4), symbolic::integer(2), symbolic::integer(2), symbolic::integer(4), symbolic::integer(4)}
    );
    types::Tensor Y_tensor(base_desc, Y_layout);
    std::vector<long long> indices({1, 2});

    auto& block = builder.add_block(root);
    auto& X_access = builder.add_access(block, "X");
    auto& Idx1_access = builder.add_access(block, "Idx1");
    auto& Idx2_access = builder.add_access(block, "Idx2");
    auto& Y_access = builder.add_access(block, "Y");
    auto& libnode =
        builder.add_library_node<math::tensor::IndexNode>(block, DebugInfo(), indices, Y_layout, X_layout, Idx_layouts);
    builder.add_computational_memlet(block, X_access, libnode, "X", {}, X_tensor);
    builder.add_computational_memlet(block, Idx1_access, libnode, "I1", {}, Idx1_tensor);
    builder.add_computational_memlet(block, Idx2_access, libnode, "I2", {}, Idx2_tensor);
    builder.add_computational_memlet(block, Y_access, libnode, "Y", {}, Y_tensor);

    ASSERT_NO_THROW(sdfg.validate());

    serializer::JSONSerializer serializer;
    nlohmann::json j;
    ASSERT_NO_THROW(j = serializer.serialize(sdfg));

    std::unique_ptr<StructuredSDFG> new_sdfg;
    ASSERT_NO_THROW(new_sdfg = serializer.deserialize(j));
    ASSERT_NO_THROW(new_sdfg->validate());
}

TEST(IndexNodeTest, validate_indices_size) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    types::Scalar sym_desc(types::PrimitiveType::Int64);
    types::Pointer index_desc(sym_desc);
    builder.add_container("X", desc, true);
    builder.add_container("Idx1", index_desc, true);
    builder.add_container("Idx2", index_desc, true);
    builder.add_container("Y", desc, true);

    math::tensor::TensorLayout X_layout(
        {symbolic::integer(4), symbolic::integer(4), symbolic::integer(4), symbolic::integer(4), symbolic::integer(4)}
    );
    types::Tensor X_tensor(base_desc, X_layout);
    std::vector<math::tensor::TensorLayout> Idx_layouts;
    Idx_layouts.push_back(math::tensor::TensorLayout({symbolic::integer(2), symbolic::integer(2)}));
    Idx_layouts.push_back(math::tensor::TensorLayout({symbolic::integer(2), symbolic::integer(2)}));
    types::Tensor Idx1_tensor(sym_desc, Idx_layouts[0]);
    types::Tensor Idx2_tensor(sym_desc, Idx_layouts[1]);
    math::tensor::TensorLayout Y_layout(
        {symbolic::integer(4), symbolic::integer(2), symbolic::integer(2), symbolic::integer(4), symbolic::integer(4)}
    );
    types::Tensor Y_tensor(base_desc, Y_layout);
    std::vector<long long> indices({1, 2});
    Idx_layouts.pop_back();

    auto& block = builder.add_block(root);
    auto& X_access = builder.add_access(block, "X");
    auto& Idx1_access = builder.add_access(block, "Idx1");
    auto& Idx2_access = builder.add_access(block, "Idx2");
    auto& Y_access = builder.add_access(block, "Y");
    auto& libnode =
        builder.add_library_node<math::tensor::IndexNode>(block, DebugInfo(), indices, Y_layout, X_layout, Idx_layouts);
    builder.add_computational_memlet(block, X_access, libnode, "X", {}, X_tensor);
    builder.add_computational_memlet(block, Idx1_access, libnode, "I1", {}, Idx1_tensor);
    builder.add_computational_memlet(block, Idx2_access, libnode, "I2", {}, Idx2_tensor);
    builder.add_computational_memlet(block, Y_access, libnode, "Y", {}, Y_tensor);

    ASSERT_THROW(sdfg.validate(), InvalidSDFGException);
}

TEST(IndexNodeTest, validate_indices_non_empty) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    types::Scalar sym_desc(types::PrimitiveType::Int64);
    types::Pointer index_desc(sym_desc);
    builder.add_container("X", desc, true);
    builder.add_container("Y", desc, true);

    math::tensor::TensorLayout X_layout(
        {symbolic::integer(4), symbolic::integer(4), symbolic::integer(4), symbolic::integer(4), symbolic::integer(4)}
    );
    types::Tensor X_tensor(base_desc, X_layout);
    std::vector<math::tensor::TensorLayout> Idx_layouts;
    math::tensor::TensorLayout Y_layout(
        {symbolic::integer(4), symbolic::integer(2), symbolic::integer(2), symbolic::integer(4), symbolic::integer(4)}
    );
    types::Tensor Y_tensor(base_desc, Y_layout);
    std::vector<long long> indices;

    auto& block = builder.add_block(root);
    auto& X_access = builder.add_access(block, "X");
    auto& Idx1_access = builder.add_access(block, "Idx1");
    auto& Idx2_access = builder.add_access(block, "Idx2");
    auto& Y_access = builder.add_access(block, "Y");
    auto& libnode =
        builder.add_library_node<math::tensor::IndexNode>(block, DebugInfo(), indices, Y_layout, X_layout, Idx_layouts);
    builder.add_computational_memlet(block, X_access, libnode, "X", {}, X_tensor);
    builder.add_computational_memlet(block, Y_access, libnode, "Y", {}, Y_tensor);

    ASSERT_THROW(sdfg.validate(), InvalidSDFGException);
}

TEST(IndexNodeTest, validate_indices_ascending) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    types::Scalar sym_desc(types::PrimitiveType::Int64);
    types::Pointer index_desc(sym_desc);
    builder.add_container("X", desc, true);
    builder.add_container("Idx1", index_desc, true);
    builder.add_container("Idx2", index_desc, true);
    builder.add_container("Y", desc, true);

    math::tensor::TensorLayout X_layout(
        {symbolic::integer(4), symbolic::integer(4), symbolic::integer(4), symbolic::integer(4), symbolic::integer(4)}
    );
    types::Tensor X_tensor(base_desc, X_layout);
    std::vector<math::tensor::TensorLayout> Idx_layouts;
    Idx_layouts.push_back(math::tensor::TensorLayout({symbolic::integer(2), symbolic::integer(2)}));
    Idx_layouts.push_back(math::tensor::TensorLayout({symbolic::integer(2), symbolic::integer(2)}));
    types::Tensor Idx1_tensor(sym_desc, Idx_layouts[0]);
    types::Tensor Idx2_tensor(sym_desc, Idx_layouts[1]);
    math::tensor::TensorLayout Y_layout(
        {symbolic::integer(4), symbolic::integer(2), symbolic::integer(2), symbolic::integer(4), symbolic::integer(4)}
    );
    types::Tensor Y_tensor(base_desc, Y_layout);
    std::vector<long long> indices({2, 1});

    auto& block = builder.add_block(root);
    auto& X_access = builder.add_access(block, "X");
    auto& Idx1_access = builder.add_access(block, "Idx1");
    auto& Idx2_access = builder.add_access(block, "Idx2");
    auto& Y_access = builder.add_access(block, "Y");
    auto& libnode =
        builder.add_library_node<math::tensor::IndexNode>(block, DebugInfo(), indices, Y_layout, X_layout, Idx_layouts);
    builder.add_computational_memlet(block, X_access, libnode, "X", {}, X_tensor);
    builder.add_computational_memlet(block, Idx1_access, libnode, "I1", {}, Idx1_tensor);
    builder.add_computational_memlet(block, Idx2_access, libnode, "I2", {}, Idx2_tensor);
    builder.add_computational_memlet(block, Y_access, libnode, "Y", {}, Y_tensor);

    ASSERT_THROW(sdfg.validate(), InvalidSDFGException);
}

TEST(IndexNodeTest, validate_indices_invalid) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    types::Scalar sym_desc(types::PrimitiveType::Int64);
    types::Pointer index_desc(sym_desc);
    builder.add_container("X", desc, true);
    builder.add_container("Idx1", index_desc, true);
    builder.add_container("Idx2", index_desc, true);
    builder.add_container("Y", desc, true);

    math::tensor::TensorLayout X_layout(
        {symbolic::integer(4), symbolic::integer(4), symbolic::integer(4), symbolic::integer(4), symbolic::integer(4)}
    );
    types::Tensor X_tensor(base_desc, X_layout);
    std::vector<math::tensor::TensorLayout> Idx_layouts;
    Idx_layouts.push_back(math::tensor::TensorLayout({symbolic::integer(2), symbolic::integer(2)}));
    Idx_layouts.push_back(math::tensor::TensorLayout({symbolic::integer(2), symbolic::integer(2)}));
    types::Tensor Idx1_tensor(sym_desc, Idx_layouts[0]);
    types::Tensor Idx2_tensor(sym_desc, Idx_layouts[1]);
    math::tensor::TensorLayout Y_layout(
        {symbolic::integer(4), symbolic::integer(2), symbolic::integer(2), symbolic::integer(4), symbolic::integer(4)}
    );
    types::Tensor Y_tensor(base_desc, Y_layout);
    std::vector<long long> indices({-1, 5});

    auto& block = builder.add_block(root);
    auto& X_access = builder.add_access(block, "X");
    auto& Idx1_access = builder.add_access(block, "Idx1");
    auto& Idx2_access = builder.add_access(block, "Idx2");
    auto& Y_access = builder.add_access(block, "Y");
    auto& libnode =
        builder.add_library_node<math::tensor::IndexNode>(block, DebugInfo(), indices, Y_layout, X_layout, Idx_layouts);
    builder.add_computational_memlet(block, X_access, libnode, "X", {}, X_tensor);
    builder.add_computational_memlet(block, Idx1_access, libnode, "I-1", {}, Idx1_tensor);
    builder.add_computational_memlet(block, Idx2_access, libnode, "I5", {}, Idx2_tensor);
    builder.add_computational_memlet(block, Y_access, libnode, "Y", {}, Y_tensor);

    ASSERT_THROW(sdfg.validate(), InvalidSDFGException);
}

TEST(IndexNodeTest, validate_inputs) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    types::Scalar sym_desc(types::PrimitiveType::Int64);
    types::Pointer index_desc(sym_desc);
    builder.add_container("X", desc, true);
    builder.add_container("Idx1", index_desc, true);
    builder.add_container("Idx2", index_desc, true);
    builder.add_container("Y", desc, true);

    math::tensor::TensorLayout X_layout(
        {symbolic::integer(4), symbolic::integer(4), symbolic::integer(4), symbolic::integer(4), symbolic::integer(4)}
    );
    types::Tensor X_tensor(base_desc, X_layout);
    std::vector<math::tensor::TensorLayout> Idx_layouts;
    Idx_layouts.push_back(math::tensor::TensorLayout({symbolic::integer(2), symbolic::integer(2)}));
    Idx_layouts.push_back(math::tensor::TensorLayout({symbolic::integer(2), symbolic::integer(2)}));
    types::Tensor Idx1_tensor(sym_desc, Idx_layouts[0]);
    types::Tensor Idx2_tensor(sym_desc, Idx_layouts[1]);
    math::tensor::TensorLayout Y_layout(
        {symbolic::integer(4), symbolic::integer(2), symbolic::integer(2), symbolic::integer(4), symbolic::integer(4)}
    );
    types::Tensor Y_tensor(base_desc, Y_layout);
    std::vector<long long> indices({1, 2});

    auto& block = builder.add_block(root);
    auto& X_access = builder.add_access(block, "X");
    auto& Idx1_access = builder.add_access(block, "Idx1");
    auto& Idx2_access = builder.add_access(block, "Idx2");
    auto& Y_access = builder.add_access(block, "Y");
    auto& libnode =
        builder.add_library_node<math::tensor::IndexNode>(block, DebugInfo(), indices, Y_layout, X_layout, Idx_layouts);
    builder.add_computational_memlet(block, X_access, libnode, "X", {}, X_tensor);
    builder.add_computational_memlet(block, Idx1_access, libnode, "I1", {}, Idx1_tensor);
    builder.add_computational_memlet(block, Y_access, libnode, "Y", {}, Y_tensor);

    ASSERT_THROW(sdfg.validate(), InvalidSDFGException);
}

TEST(IndexNodeTest, validate_input_y) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    types::Scalar sym_desc(types::PrimitiveType::Int64);
    types::Pointer index_desc(sym_desc);
    builder.add_container("X", desc, true);
    builder.add_container("Idx1", index_desc, true);
    builder.add_container("Idx2", index_desc, true);
    builder.add_container("Y", desc, true);

    math::tensor::TensorLayout X_layout(
        {symbolic::integer(4), symbolic::integer(4), symbolic::integer(4), symbolic::integer(4), symbolic::integer(4)}
    );
    types::Tensor X_tensor(base_desc, X_layout);
    std::vector<math::tensor::TensorLayout> Idx_layouts;
    Idx_layouts.push_back(math::tensor::TensorLayout({symbolic::integer(2), symbolic::integer(2)}));
    Idx_layouts.push_back(math::tensor::TensorLayout({symbolic::integer(2), symbolic::integer(2)}));
    types::Tensor Idx1_tensor(sym_desc, Idx_layouts[0]);
    types::Tensor Idx2_tensor(sym_desc, Idx_layouts[1]);
    math::tensor::TensorLayout Y_layout(
        {symbolic::integer(4), symbolic::integer(2), symbolic::integer(2), symbolic::integer(4), symbolic::integer(4)}
    );
    types::Tensor Y_tensor(base_desc, Y_layout);
    std::vector<long long> indices({1, 2});

    auto& block = builder.add_block(root);
    auto& X_access = builder.add_access(block, "X");
    auto& Idx1_access = builder.add_access(block, "Idx1");
    auto& Idx2_access = builder.add_access(block, "Idx2");
    auto& Y_access = builder.add_access(block, "Y");
    auto& libnode =
        builder.add_library_node<math::tensor::IndexNode>(block, DebugInfo(), indices, Y_layout, X_layout, Idx_layouts);
    builder.add_computational_memlet(block, X_access, libnode, "X", {}, X_tensor);
    builder.add_computational_memlet(block, Idx1_access, libnode, "I1", {}, Idx1_tensor);
    builder.add_computational_memlet(block, Idx2_access, libnode, "I2", {}, Idx2_tensor);
    builder.add_computational_memlet(block, Y_access, libnode, "X", {}, Y_tensor);

    ASSERT_THROW(sdfg.validate(), InvalidSDFGException);
}

TEST(IndexNodeTest, validate_input_x) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    types::Scalar sym_desc(types::PrimitiveType::Int64);
    types::Pointer index_desc(sym_desc);
    builder.add_container("X", desc, true);
    builder.add_container("Idx1", index_desc, true);
    builder.add_container("Idx2", index_desc, true);
    builder.add_container("Y", desc, true);

    math::tensor::TensorLayout X_layout(
        {symbolic::integer(4), symbolic::integer(4), symbolic::integer(4), symbolic::integer(4), symbolic::integer(4)}
    );
    types::Tensor X_tensor(base_desc, X_layout);
    std::vector<math::tensor::TensorLayout> Idx_layouts;
    Idx_layouts.push_back(math::tensor::TensorLayout({symbolic::integer(2), symbolic::integer(2)}));
    Idx_layouts.push_back(math::tensor::TensorLayout({symbolic::integer(2), symbolic::integer(2)}));
    types::Tensor Idx1_tensor(sym_desc, Idx_layouts[0]);
    types::Tensor Idx2_tensor(sym_desc, Idx_layouts[1]);
    math::tensor::TensorLayout Y_layout(
        {symbolic::integer(4), symbolic::integer(2), symbolic::integer(2), symbolic::integer(4), symbolic::integer(4)}
    );
    types::Tensor Y_tensor(base_desc, Y_layout);
    std::vector<long long> indices({1, 2});

    auto& block = builder.add_block(root);
    auto& X_access = builder.add_access(block, "X");
    auto& Idx1_access = builder.add_access(block, "Idx1");
    auto& Idx2_access = builder.add_access(block, "Idx2");
    auto& Y_access = builder.add_access(block, "Y");
    auto& libnode =
        builder.add_library_node<math::tensor::IndexNode>(block, DebugInfo(), indices, Y_layout, X_layout, Idx_layouts);
    builder.add_computational_memlet(block, X_access, libnode, "Y", {}, X_tensor);
    builder.add_computational_memlet(block, Idx1_access, libnode, "I1", {}, Idx1_tensor);
    builder.add_computational_memlet(block, Idx2_access, libnode, "I2", {}, Idx2_tensor);
    builder.add_computational_memlet(block, Y_access, libnode, "Y", {}, Y_tensor);

    ASSERT_THROW(sdfg.validate(), InvalidSDFGException);
}

TEST(IndexNodeTest, validate_input_index) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    types::Scalar sym_desc(types::PrimitiveType::Int64);
    types::Pointer index_desc(sym_desc);
    builder.add_container("X", desc, true);
    builder.add_container("Idx1", index_desc, true);
    builder.add_container("Idx2", index_desc, true);
    builder.add_container("Y", desc, true);

    math::tensor::TensorLayout X_layout(
        {symbolic::integer(4), symbolic::integer(4), symbolic::integer(4), symbolic::integer(4), symbolic::integer(4)}
    );
    types::Tensor X_tensor(base_desc, X_layout);
    std::vector<math::tensor::TensorLayout> Idx_layouts;
    Idx_layouts.push_back(math::tensor::TensorLayout({symbolic::integer(2), symbolic::integer(2)}));
    Idx_layouts.push_back(math::tensor::TensorLayout({symbolic::integer(2), symbolic::integer(2)}));
    types::Tensor Idx1_tensor(sym_desc, Idx_layouts[0]);
    types::Tensor Idx2_tensor(sym_desc, Idx_layouts[1]);
    math::tensor::TensorLayout Y_layout(
        {symbolic::integer(4), symbolic::integer(2), symbolic::integer(2), symbolic::integer(4), symbolic::integer(4)}
    );
    types::Tensor Y_tensor(base_desc, Y_layout);
    std::vector<long long> indices({1, 2});

    auto& block = builder.add_block(root);
    auto& X_access = builder.add_access(block, "X");
    auto& Idx1_access = builder.add_access(block, "Idx1");
    auto& Idx2_access = builder.add_access(block, "Idx2");
    auto& Y_access = builder.add_access(block, "Y");
    auto& libnode =
        builder.add_library_node<math::tensor::IndexNode>(block, DebugInfo(), indices, Y_layout, X_layout, Idx_layouts);
    builder.add_computational_memlet(block, X_access, libnode, "X", {}, X_tensor);
    builder.add_computational_memlet(block, Idx1_access, libnode, "I1", {}, Idx1_tensor);
    builder.add_computational_memlet(block, Idx2_access, libnode, "I1", {}, Idx2_tensor);
    builder.add_computational_memlet(block, Y_access, libnode, "Y", {}, Y_tensor);

    ASSERT_THROW(sdfg.validate(), InvalidSDFGException);
}

TEST(IndexNodeTest, validate_memlet_tensor_y) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    types::Scalar sym_desc(types::PrimitiveType::Int64);
    types::Pointer index_desc(sym_desc);
    builder.add_container("X", desc, true);
    builder.add_container("Idx1", index_desc, true);
    builder.add_container("Idx2", index_desc, true);
    builder.add_container("Y", desc, true);

    math::tensor::TensorLayout X_layout(
        {symbolic::integer(4), symbolic::integer(4), symbolic::integer(4), symbolic::integer(4), symbolic::integer(4)}
    );
    types::Tensor X_tensor(base_desc, X_layout);
    std::vector<math::tensor::TensorLayout> Idx_layouts;
    Idx_layouts.push_back(math::tensor::TensorLayout({symbolic::integer(2), symbolic::integer(2)}));
    Idx_layouts.push_back(math::tensor::TensorLayout({symbolic::integer(2), symbolic::integer(2)}));
    types::Tensor Idx1_tensor(sym_desc, Idx_layouts[0]);
    types::Tensor Idx2_tensor(sym_desc, Idx_layouts[1]);
    math::tensor::TensorLayout Y_layout(
        {symbolic::integer(4), symbolic::integer(2), symbolic::integer(2), symbolic::integer(4), symbolic::integer(4)}
    );
    types::Tensor Y_tensor(base_desc, Y_layout);
    std::vector<long long> indices({1, 2});

    auto& block = builder.add_block(root);
    auto& X_access = builder.add_access(block, "X");
    auto& Idx1_access = builder.add_access(block, "Idx1");
    auto& Idx2_access = builder.add_access(block, "Idx2");
    auto& Y_access = builder.add_access(block, "Y");
    auto& libnode =
        builder.add_library_node<math::tensor::IndexNode>(block, DebugInfo(), indices, Y_layout, X_layout, Idx_layouts);
    builder.add_computational_memlet(block, X_access, libnode, "X", {}, X_tensor);
    builder.add_computational_memlet(block, Idx1_access, libnode, "I1", {}, Idx1_tensor);
    builder.add_computational_memlet(block, Idx2_access, libnode, "I2", {}, Idx2_tensor);
    builder.add_computational_memlet(block, Y_access, libnode, "Y", {}, desc);

    ASSERT_THROW(sdfg.validate(), InvalidSDFGException);
}

TEST(IndexNodeTest, validate_memlet_tensor_x) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    types::Scalar sym_desc(types::PrimitiveType::Int64);
    types::Pointer index_desc(sym_desc);
    builder.add_container("X", desc, true);
    builder.add_container("Idx1", index_desc, true);
    builder.add_container("Idx2", index_desc, true);
    builder.add_container("Y", desc, true);

    math::tensor::TensorLayout X_layout(
        {symbolic::integer(4), symbolic::integer(4), symbolic::integer(4), symbolic::integer(4), symbolic::integer(4)}
    );
    types::Tensor X_tensor(base_desc, X_layout);
    std::vector<math::tensor::TensorLayout> Idx_layouts;
    Idx_layouts.push_back(math::tensor::TensorLayout({symbolic::integer(2), symbolic::integer(2)}));
    Idx_layouts.push_back(math::tensor::TensorLayout({symbolic::integer(2), symbolic::integer(2)}));
    types::Tensor Idx1_tensor(sym_desc, Idx_layouts[0]);
    types::Tensor Idx2_tensor(sym_desc, Idx_layouts[1]);
    math::tensor::TensorLayout Y_layout(
        {symbolic::integer(4), symbolic::integer(2), symbolic::integer(2), symbolic::integer(4), symbolic::integer(4)}
    );
    types::Tensor Y_tensor(base_desc, Y_layout);
    std::vector<long long> indices({1, 2});

    auto& block = builder.add_block(root);
    auto& X_access = builder.add_access(block, "X");
    auto& Idx1_access = builder.add_access(block, "Idx1");
    auto& Idx2_access = builder.add_access(block, "Idx2");
    auto& Y_access = builder.add_access(block, "Y");
    auto& libnode =
        builder.add_library_node<math::tensor::IndexNode>(block, DebugInfo(), indices, Y_layout, X_layout, Idx_layouts);
    builder.add_computational_memlet(block, X_access, libnode, "X", {}, desc);
    builder.add_computational_memlet(block, Idx1_access, libnode, "I1", {}, Idx1_tensor);
    builder.add_computational_memlet(block, Idx2_access, libnode, "I2", {}, Idx2_tensor);
    builder.add_computational_memlet(block, Y_access, libnode, "Y", {}, Y_tensor);

    ASSERT_THROW(sdfg.validate(), InvalidSDFGException);
}

TEST(IndexNodeTest, validate_memlet_tensor_index) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    types::Scalar sym_desc(types::PrimitiveType::Int64);
    types::Pointer index_desc(sym_desc);
    builder.add_container("X", desc, true);
    builder.add_container("Idx1", index_desc, true);
    builder.add_container("Idx2", index_desc, true);
    builder.add_container("Y", desc, true);

    math::tensor::TensorLayout X_layout(
        {symbolic::integer(4), symbolic::integer(4), symbolic::integer(4), symbolic::integer(4), symbolic::integer(4)}
    );
    types::Tensor X_tensor(base_desc, X_layout);
    std::vector<math::tensor::TensorLayout> Idx_layouts;
    Idx_layouts.push_back(math::tensor::TensorLayout({symbolic::integer(2), symbolic::integer(2)}));
    Idx_layouts.push_back(math::tensor::TensorLayout({symbolic::integer(2), symbolic::integer(2)}));
    types::Tensor Idx1_tensor(sym_desc, Idx_layouts[0]);
    types::Tensor Idx2_tensor(sym_desc, Idx_layouts[1]);
    math::tensor::TensorLayout Y_layout(
        {symbolic::integer(4), symbolic::integer(2), symbolic::integer(2), symbolic::integer(4), symbolic::integer(4)}
    );
    types::Tensor Y_tensor(base_desc, Y_layout);
    std::vector<long long> indices({1, 2});

    auto& block = builder.add_block(root);
    auto& X_access = builder.add_access(block, "X");
    auto& Idx1_access = builder.add_access(block, "Idx1");
    auto& Idx2_access = builder.add_access(block, "Idx2");
    auto& Y_access = builder.add_access(block, "Y");
    auto& libnode =
        builder.add_library_node<math::tensor::IndexNode>(block, DebugInfo(), indices, Y_layout, X_layout, Idx_layouts);
    builder.add_computational_memlet(block, X_access, libnode, "X", {}, X_tensor);
    builder.add_computational_memlet(block, Idx1_access, libnode, "I1", {}, index_desc);
    builder.add_computational_memlet(block, Idx2_access, libnode, "I2", {}, Idx2_tensor);
    builder.add_computational_memlet(block, Y_access, libnode, "Y", {}, Y_tensor);

    ASSERT_THROW(sdfg.validate(), InvalidSDFGException);
}

TEST(IndexNodeTest, validate_tensor_layout_y) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    types::Scalar sym_desc(types::PrimitiveType::Int64);
    types::Pointer index_desc(sym_desc);
    builder.add_container("X", desc, true);
    builder.add_container("Idx1", index_desc, true);
    builder.add_container("Idx2", index_desc, true);
    builder.add_container("Y", desc, true);

    math::tensor::TensorLayout X_layout(
        {symbolic::integer(4), symbolic::integer(4), symbolic::integer(4), symbolic::integer(4), symbolic::integer(4)}
    );
    types::Tensor X_tensor(base_desc, X_layout);
    std::vector<math::tensor::TensorLayout> Idx_layouts;
    Idx_layouts.push_back(math::tensor::TensorLayout({symbolic::integer(2), symbolic::integer(2)}));
    Idx_layouts.push_back(math::tensor::TensorLayout({symbolic::integer(2), symbolic::integer(2)}));
    types::Tensor Idx1_tensor(sym_desc, Idx_layouts[0]);
    types::Tensor Idx2_tensor(sym_desc, Idx_layouts[1]);
    math::tensor::TensorLayout Y_layout(
        {symbolic::integer(4), symbolic::integer(2), symbolic::integer(2), symbolic::integer(4), symbolic::integer(4)}
    );
    types::Tensor Y_tensor(base_desc, Y_layout);
    std::vector<long long> indices({1, 2});

    auto& block = builder.add_block(root);
    auto& X_access = builder.add_access(block, "X");
    auto& Idx1_access = builder.add_access(block, "Idx1");
    auto& Idx2_access = builder.add_access(block, "Idx2");
    auto& Y_access = builder.add_access(block, "Y");
    auto& libnode =
        builder.add_library_node<math::tensor::IndexNode>(block, DebugInfo(), indices, Y_layout, X_layout, Idx_layouts);
    builder.add_computational_memlet(block, X_access, libnode, "X", {}, X_tensor);
    builder.add_computational_memlet(block, Idx1_access, libnode, "I1", {}, Idx1_tensor);
    builder.add_computational_memlet(block, Idx2_access, libnode, "I2", {}, Idx2_tensor);
    builder.add_computational_memlet(block, Y_access, libnode, "Y", {}, X_tensor);

    ASSERT_THROW(sdfg.validate(), InvalidSDFGException);
}

TEST(IndexNodeTest, validate_tensor_layout_x) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    types::Scalar sym_desc(types::PrimitiveType::Int64);
    types::Pointer index_desc(sym_desc);
    builder.add_container("X", desc, true);
    builder.add_container("Idx1", index_desc, true);
    builder.add_container("Idx2", index_desc, true);
    builder.add_container("Y", desc, true);

    math::tensor::TensorLayout X_layout(
        {symbolic::integer(4), symbolic::integer(4), symbolic::integer(4), symbolic::integer(4), symbolic::integer(4)}
    );
    types::Tensor X_tensor(base_desc, X_layout);
    std::vector<math::tensor::TensorLayout> Idx_layouts;
    Idx_layouts.push_back(math::tensor::TensorLayout({symbolic::integer(2), symbolic::integer(2)}));
    Idx_layouts.push_back(math::tensor::TensorLayout({symbolic::integer(2), symbolic::integer(2)}));
    types::Tensor Idx1_tensor(sym_desc, Idx_layouts[0]);
    types::Tensor Idx2_tensor(sym_desc, Idx_layouts[1]);
    math::tensor::TensorLayout Y_layout(
        {symbolic::integer(4), symbolic::integer(2), symbolic::integer(2), symbolic::integer(4), symbolic::integer(4)}
    );
    types::Tensor Y_tensor(base_desc, Y_layout);
    std::vector<long long> indices({1, 2});

    auto& block = builder.add_block(root);
    auto& X_access = builder.add_access(block, "X");
    auto& Idx1_access = builder.add_access(block, "Idx1");
    auto& Idx2_access = builder.add_access(block, "Idx2");
    auto& Y_access = builder.add_access(block, "Y");
    auto& libnode =
        builder.add_library_node<math::tensor::IndexNode>(block, DebugInfo(), indices, Y_layout, X_layout, Idx_layouts);
    builder.add_computational_memlet(block, X_access, libnode, "X", {}, Y_tensor);
    builder.add_computational_memlet(block, Idx1_access, libnode, "I1", {}, Idx1_tensor);
    builder.add_computational_memlet(block, Idx2_access, libnode, "I2", {}, Idx2_tensor);
    builder.add_computational_memlet(block, Y_access, libnode, "Y", {}, Y_tensor);

    ASSERT_THROW(sdfg.validate(), InvalidSDFGException);
}

TEST(IndexNodeTest, validate_tensor_layout_index) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    types::Scalar sym_desc(types::PrimitiveType::Int64);
    types::Pointer index_desc(sym_desc);
    builder.add_container("X", desc, true);
    builder.add_container("Idx1", index_desc, true);
    builder.add_container("Idx2", index_desc, true);
    builder.add_container("Y", desc, true);

    math::tensor::TensorLayout X_layout(
        {symbolic::integer(4), symbolic::integer(4), symbolic::integer(4), symbolic::integer(4), symbolic::integer(4)}
    );
    types::Tensor X_tensor(base_desc, X_layout);
    std::vector<math::tensor::TensorLayout> Idx_layouts;
    Idx_layouts.push_back(math::tensor::TensorLayout({symbolic::integer(2)}));
    Idx_layouts.push_back(math::tensor::TensorLayout({symbolic::integer(2), symbolic::integer(2)}));
    types::Tensor Idx1_tensor(sym_desc, Idx_layouts[0]);
    types::Tensor Idx2_tensor(sym_desc, Idx_layouts[1]);
    math::tensor::TensorLayout Y_layout(
        {symbolic::integer(4), symbolic::integer(2), symbolic::integer(2), symbolic::integer(4), symbolic::integer(4)}
    );
    types::Tensor Y_tensor(base_desc, Y_layout);
    std::vector<long long> indices({1, 2});

    auto& block = builder.add_block(root);
    auto& X_access = builder.add_access(block, "X");
    auto& Idx1_access = builder.add_access(block, "Idx1");
    auto& Idx2_access = builder.add_access(block, "Idx2");
    auto& Y_access = builder.add_access(block, "Y");
    auto& libnode =
        builder.add_library_node<math::tensor::IndexNode>(block, DebugInfo(), indices, Y_layout, X_layout, Idx_layouts);
    builder.add_computational_memlet(block, X_access, libnode, "X", {}, X_tensor);
    builder.add_computational_memlet(block, Idx1_access, libnode, "I1", {}, Idx1_tensor);
    builder.add_computational_memlet(block, Idx2_access, libnode, "I2", {}, Idx1_tensor);
    builder.add_computational_memlet(block, Y_access, libnode, "Y", {}, Y_tensor);

    ASSERT_THROW(sdfg.validate(), InvalidSDFGException);
}

TEST(IndexNodeTest, validate_tensors_x_y_primitive_type) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    types::Scalar sym_desc(types::PrimitiveType::Int64);
    types::Pointer index_desc(sym_desc);
    builder.add_container("X", desc, true);
    builder.add_container("Idx1", index_desc, true);
    builder.add_container("Idx2", index_desc, true);
    builder.add_container("Y", desc, true);

    math::tensor::TensorLayout X_layout(
        {symbolic::integer(4), symbolic::integer(4), symbolic::integer(4), symbolic::integer(4), symbolic::integer(4)}
    );
    types::Tensor X_tensor(base_desc, X_layout);
    std::vector<math::tensor::TensorLayout> Idx_layouts;
    Idx_layouts.push_back(math::tensor::TensorLayout({symbolic::integer(2), symbolic::integer(2)}));
    Idx_layouts.push_back(math::tensor::TensorLayout({symbolic::integer(2), symbolic::integer(2)}));
    types::Tensor Idx1_tensor(sym_desc, Idx_layouts[0]);
    types::Tensor Idx2_tensor(sym_desc, Idx_layouts[1]);
    math::tensor::TensorLayout Y_layout(
        {symbolic::integer(4), symbolic::integer(2), symbolic::integer(2), symbolic::integer(4), symbolic::integer(4)}
    );
    types::Tensor Y_tensor(sym_desc, Y_layout);
    std::vector<long long> indices({1, 2});

    auto& block = builder.add_block(root);
    auto& X_access = builder.add_access(block, "X");
    auto& Idx1_access = builder.add_access(block, "Idx1");
    auto& Idx2_access = builder.add_access(block, "Idx2");
    auto& Y_access = builder.add_access(block, "Y");
    auto& libnode =
        builder.add_library_node<math::tensor::IndexNode>(block, DebugInfo(), indices, Y_layout, X_layout, Idx_layouts);
    builder.add_computational_memlet(block, X_access, libnode, "X", {}, X_tensor);
    builder.add_computational_memlet(block, Idx1_access, libnode, "I1", {}, Idx1_tensor);
    builder.add_computational_memlet(block, Idx2_access, libnode, "I2", {}, Idx2_tensor);
    builder.add_computational_memlet(block, Y_access, libnode, "Y", {}, Y_tensor);

    ASSERT_THROW(sdfg.validate(), InvalidSDFGException);
}

TEST(IndexNodeTest, validate_tensors_index_integer_primitive_type) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    types::Scalar sym_desc(types::PrimitiveType::Int64);
    types::Pointer index_desc(sym_desc);
    builder.add_container("X", desc, true);
    builder.add_container("Idx1", index_desc, true);
    builder.add_container("Idx2", index_desc, true);
    builder.add_container("Y", desc, true);

    math::tensor::TensorLayout X_layout(
        {symbolic::integer(4), symbolic::integer(4), symbolic::integer(4), symbolic::integer(4), symbolic::integer(4)}
    );
    types::Tensor X_tensor(base_desc, X_layout);
    std::vector<math::tensor::TensorLayout> Idx_layouts;
    Idx_layouts.push_back(math::tensor::TensorLayout({symbolic::integer(2), symbolic::integer(2)}));
    Idx_layouts.push_back(math::tensor::TensorLayout({symbolic::integer(2), symbolic::integer(2)}));
    types::Tensor Idx1_tensor(sym_desc, Idx_layouts[0]);
    types::Tensor Idx2_tensor(base_desc, Idx_layouts[1]);
    math::tensor::TensorLayout Y_layout(
        {symbolic::integer(4), symbolic::integer(2), symbolic::integer(2), symbolic::integer(4), symbolic::integer(4)}
    );
    types::Tensor Y_tensor(base_desc, Y_layout);
    std::vector<long long> indices({1, 2});

    auto& block = builder.add_block(root);
    auto& X_access = builder.add_access(block, "X");
    auto& Idx1_access = builder.add_access(block, "Idx1");
    auto& Idx2_access = builder.add_access(block, "Idx2");
    auto& Y_access = builder.add_access(block, "Y");
    auto& libnode =
        builder.add_library_node<math::tensor::IndexNode>(block, DebugInfo(), indices, Y_layout, X_layout, Idx_layouts);
    builder.add_computational_memlet(block, X_access, libnode, "X", {}, X_tensor);
    builder.add_computational_memlet(block, Idx1_access, libnode, "I1", {}, Idx1_tensor);
    builder.add_computational_memlet(block, Idx2_access, libnode, "I2", {}, Idx2_tensor);
    builder.add_computational_memlet(block, Y_access, libnode, "Y", {}, Y_tensor);

    ASSERT_THROW(sdfg.validate(), InvalidSDFGException);
}

TEST(IndexNodeTest, validate_broadcast_shape_1) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    types::Scalar sym_desc(types::PrimitiveType::Int64);
    types::Pointer index_desc(sym_desc);
    builder.add_container("X", desc, true);
    builder.add_container("Idx1", index_desc, true);
    builder.add_container("Idx2", index_desc, true);
    builder.add_container("Y", desc, true);

    math::tensor::TensorLayout X_layout(
        {symbolic::integer(4), symbolic::integer(4), symbolic::integer(4), symbolic::integer(4), symbolic::integer(4)}
    );
    types::Tensor X_tensor(base_desc, X_layout);
    std::vector<math::tensor::TensorLayout> Idx_layouts;
    Idx_layouts.push_back(math::tensor::TensorLayout({symbolic::integer(3)}));
    Idx_layouts.push_back(math::tensor::TensorLayout({symbolic::integer(2), symbolic::integer(2)}));
    types::Tensor Idx1_tensor(sym_desc, Idx_layouts[0]);
    types::Tensor Idx2_tensor(sym_desc, Idx_layouts[1]);
    math::tensor::TensorLayout Y_layout(
        {symbolic::integer(4), symbolic::integer(2), symbolic::integer(2), symbolic::integer(4), symbolic::integer(4)}
    );
    types::Tensor Y_tensor(base_desc, Y_layout);
    std::vector<long long> indices({1, 2});

    auto& block = builder.add_block(root);
    auto& X_access = builder.add_access(block, "X");
    auto& Idx1_access = builder.add_access(block, "Idx1");
    auto& Idx2_access = builder.add_access(block, "Idx2");
    auto& Y_access = builder.add_access(block, "Y");
    auto& libnode =
        builder.add_library_node<math::tensor::IndexNode>(block, DebugInfo(), indices, Y_layout, X_layout, Idx_layouts);
    builder.add_computational_memlet(block, X_access, libnode, "X", {}, X_tensor);
    builder.add_computational_memlet(block, Idx1_access, libnode, "I1", {}, Idx1_tensor);
    builder.add_computational_memlet(block, Idx2_access, libnode, "I2", {}, Idx2_tensor);
    builder.add_computational_memlet(block, Y_access, libnode, "Y", {}, Y_tensor);

    ASSERT_THROW(sdfg.validate(), InvalidSDFGException);
}

TEST(IndexNodeTest, validate_broadcast_shape_2) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    types::Scalar sym_desc(types::PrimitiveType::Int64);
    types::Pointer index_desc(sym_desc);
    builder.add_container("X", desc, true);
    builder.add_container("Idx1", index_desc, true);
    builder.add_container("Idx2", index_desc, true);
    builder.add_container("Y", desc, true);

    math::tensor::TensorLayout X_layout(
        {symbolic::integer(4), symbolic::integer(4), symbolic::integer(4), symbolic::integer(4), symbolic::integer(4)}
    );
    types::Tensor X_tensor(base_desc, X_layout);
    std::vector<math::tensor::TensorLayout> Idx_layouts;
    Idx_layouts.push_back(math::tensor::TensorLayout({symbolic::integer(2), symbolic::integer(2)}));
    Idx_layouts.push_back(math::tensor::TensorLayout({symbolic::integer(3)}));
    types::Tensor Idx1_tensor(sym_desc, Idx_layouts[0]);
    types::Tensor Idx2_tensor(sym_desc, Idx_layouts[1]);
    math::tensor::TensorLayout Y_layout(
        {symbolic::integer(4), symbolic::integer(2), symbolic::integer(2), symbolic::integer(4), symbolic::integer(4)}
    );
    types::Tensor Y_tensor(base_desc, Y_layout);
    std::vector<long long> indices({1, 2});

    auto& block = builder.add_block(root);
    auto& X_access = builder.add_access(block, "X");
    auto& Idx1_access = builder.add_access(block, "Idx1");
    auto& Idx2_access = builder.add_access(block, "Idx2");
    auto& Y_access = builder.add_access(block, "Y");
    auto& libnode =
        builder.add_library_node<math::tensor::IndexNode>(block, DebugInfo(), indices, Y_layout, X_layout, Idx_layouts);
    builder.add_computational_memlet(block, X_access, libnode, "X", {}, X_tensor);
    builder.add_computational_memlet(block, Idx1_access, libnode, "I1", {}, Idx1_tensor);
    builder.add_computational_memlet(block, Idx2_access, libnode, "I2", {}, Idx2_tensor);
    builder.add_computational_memlet(block, Y_access, libnode, "Y", {}, Y_tensor);

    ASSERT_THROW(sdfg.validate(), InvalidSDFGException);
}
