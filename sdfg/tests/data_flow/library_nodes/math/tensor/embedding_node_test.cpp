#include "sdfg/data_flow/library_nodes/math/tensor/embedding_node.h"

#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <nlohmann/json_fwd.hpp>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
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

namespace {

// Wires up an EmbeddingNode with a 2D float weight, an integer index tensor and
// a float output whose shape is index_shape ++ [embedding_dim]. All three
// connectors ("Y", "W", "I") are input connectors following the indirect
// pointer convention shared by the gather-style tensor nodes.
math::tensor::EmbeddingNode& build_embedding(
    builder::StructuredSDFGBuilder& builder,
    const std::vector<symbolic::Expression>& weight_shape,
    const std::vector<symbolic::Expression>& index_shape
) {
    auto& sdfg = builder.subject();

    types::Scalar float_desc(types::PrimitiveType::Float);
    types::Scalar int_desc(types::PrimitiveType::Int64);
    types::Pointer float_ptr(float_desc);
    types::Pointer int_ptr(int_desc);

    builder.add_container("W", float_ptr, true);
    builder.add_container("I", int_ptr, true);
    builder.add_container("Y", float_ptr, true);

    std::vector<symbolic::Expression> output_shape(index_shape.begin(), index_shape.end());
    output_shape.push_back(weight_shape[1]);

    types::Tensor W_tensor(float_desc, weight_shape);
    types::Tensor I_tensor(int_desc, index_shape);
    types::Tensor Y_tensor(float_desc, output_shape);

    auto& block = builder.add_block(sdfg.root());
    auto& W_access = builder.add_access(block, "W");
    auto& I_access = builder.add_access(block, "I");
    auto& Y_access = builder.add_access(block, "Y");

    auto& libnode =
        builder.add_library_node<math::tensor::EmbeddingNode>(block, DebugInfo(), weight_shape, index_shape);

    builder.add_computational_memlet(block, Y_access, libnode, "Y", {}, Y_tensor);
    builder.add_computational_memlet(block, W_access, libnode, "W", {}, W_tensor);
    builder.add_computational_memlet(block, I_access, libnode, "I", {}, I_tensor);

    return static_cast<math::tensor::EmbeddingNode&>(libnode);
}

} // namespace

TEST(EmbeddingNodeTest, symbolic) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();

    types::Scalar sym_desc(types::PrimitiveType::Int64);
    builder.add_container("n", sym_desc);
    builder.add_container("m", sym_desc);
    builder.add_container("k", sym_desc);
    auto n = symbolic::symbol("n");
    auto m = symbolic::symbol("m");
    auto k = symbolic::symbol("k");

    auto& embedding_node = build_embedding(builder, {n, m}, {k});

    ASSERT_NO_THROW(sdfg.validate());

    EXPECT_EQ(embedding_node.weight_shape().size(), 2);
    EXPECT_EQ(embedding_node.index_shape().size(), 1);

    auto symbols = embedding_node.symbols();
    EXPECT_EQ(symbols.size(), 3);
    EXPECT_TRUE(symbols.contains(n));
    EXPECT_TRUE(symbols.contains(m));
    EXPECT_TRUE(symbols.contains(k));

    builder.add_container("p", sym_desc);
    auto p = symbolic::symbol("p");
    builder.replace_symbols(k, p);

    ASSERT_NO_THROW(sdfg.validate());

    symbols = embedding_node.symbols();
    EXPECT_EQ(symbols.size(), 3);
    EXPECT_TRUE(symbols.contains(n));
    EXPECT_TRUE(symbols.contains(m));
    EXPECT_TRUE(symbols.contains(p));
    EXPECT_FALSE(symbols.contains(k));
}

TEST(EmbeddingNodeTest, expand_1d_indices) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();

    build_embedding(builder, {symbolic::integer(10), symbolic::integer(4)}, {symbolic::integer(3)});

    ASSERT_NO_THROW(sdfg.validate());
    dump_sdfg(sdfg, "0.before");

    analysis::AnalysisManager analysis_manager(sdfg);
    passes::LibraryNodeExpansionPass expansion;
    ASSERT_TRUE(expansion.run(builder, analysis_manager));
    ASSERT_NO_THROW(sdfg.validate());
    dump_sdfg(sdfg, "1.after");
}

TEST(EmbeddingNodeTest, expand_2d_indices) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();

    build_embedding(builder, {symbolic::integer(10), symbolic::integer(4)}, {symbolic::integer(2), symbolic::integer(3)});

    ASSERT_NO_THROW(sdfg.validate());
    dump_sdfg(sdfg, "0.before");

    analysis::AnalysisManager analysis_manager(sdfg);
    passes::LibraryNodeExpansionPass expansion;
    ASSERT_TRUE(expansion.run(builder, analysis_manager));
    ASSERT_NO_THROW(sdfg.validate());
    dump_sdfg(sdfg, "1.after");
}

TEST(EmbeddingNodeTest, serialization) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();

    build_embedding(builder, {symbolic::integer(10), symbolic::integer(4)}, {symbolic::integer(3)});

    ASSERT_NO_THROW(sdfg.validate());

    serializer::JSONSerializer serializer;
    nlohmann::json j;
    ASSERT_NO_THROW(j = serializer.serialize(sdfg));

    std::unique_ptr<StructuredSDFG> new_sdfg;
    ASSERT_NO_THROW(new_sdfg = serializer.deserialize(j));
}

TEST(EmbeddingNodeTest, validate_rejects_non_2d_weight) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();

    types::Scalar float_desc(types::PrimitiveType::Float);
    types::Scalar int_desc(types::PrimitiveType::Int64);
    types::Pointer float_ptr(float_desc);
    types::Pointer int_ptr(int_desc);

    builder.add_container("W", float_ptr, true);
    builder.add_container("I", int_ptr, true);
    builder.add_container("Y", float_ptr, true);

    // 1D weight shape is invalid for an embedding lookup.
    std::vector<symbolic::Expression> weight_shape = {symbolic::integer(10)};
    std::vector<symbolic::Expression> index_shape = {symbolic::integer(3)};

    types::Tensor W_tensor(float_desc, weight_shape);
    types::Tensor I_tensor(int_desc, index_shape);
    types::Tensor Y_tensor(float_desc, index_shape);

    auto& block = builder.add_block(sdfg.root());
    auto& W_access = builder.add_access(block, "W");
    auto& I_access = builder.add_access(block, "I");
    auto& Y_access = builder.add_access(block, "Y");

    auto& libnode =
        builder.add_library_node<math::tensor::EmbeddingNode>(block, DebugInfo(), weight_shape, index_shape);
    builder.add_computational_memlet(block, Y_access, libnode, "Y", {}, Y_tensor);
    builder.add_computational_memlet(block, W_access, libnode, "W", {}, W_tensor);
    builder.add_computational_memlet(block, I_access, libnode, "I", {}, I_tensor);

    auto& embedding_node = static_cast<math::tensor::EmbeddingNode&>(libnode);
    EXPECT_THROW(embedding_node.validate(sdfg), InvalidSDFGException);
}
