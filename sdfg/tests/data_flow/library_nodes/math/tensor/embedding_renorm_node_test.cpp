#include "sdfg/data_flow/library_nodes/math/tensor/embedding_renorm_node.h"

#include <limits>
#include <memory>
#include <string>
#include <vector>

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

namespace {

math::tensor::EmbeddingRenormNode& build_renorm(
    builder::StructuredSDFGBuilder& builder,
    const std::vector<symbolic::Expression>& weight_shape,
    const std::vector<symbolic::Expression>& indices_shape,
    const std::string& max_norm,
    const std::string& norm_type
) {
    auto& sdfg = builder.subject();

    types::Scalar float_desc(types::PrimitiveType::Float);
    types::Scalar int_desc(types::PrimitiveType::Int64);
    types::Pointer float_ptr(float_desc);
    types::Pointer int_ptr(int_desc);

    builder.add_container("y", float_ptr, true);
    builder.add_container("weight", float_ptr, true);
    builder.add_container("indices", int_ptr, true);
    builder.add_container("max_norm", float_desc, true);
    builder.add_container("norm_type", float_desc, true);

    math::tensor::TensorLayout y_layout(weight_shape);
    types::Tensor y_tensor(float_desc, y_layout);
    math::tensor::TensorLayout weight_layout(weight_shape);
    types::Tensor weight_tensor(float_desc, y_layout);
    math::tensor::TensorLayout indices_layout(indices_shape);
    types::Tensor indices_tensor(int_desc, indices_layout);

    auto& block = builder.add_block(sdfg.root());
    auto& y_access = builder.add_access(block, "y");
    auto& weight_access = builder.add_access(block, "weight");
    auto& indices_access = builder.add_access(block, "indices");
    auto& max_norm_access = builder.add_access(block, "max_norm");
    auto& norm_type_access = builder.add_access(block, "norm_type");

    auto& libnode = builder.add_library_node<
        math::tensor::EmbeddingRenormNode>(block, DebugInfo(), y_layout, weight_layout, indices_layout);

    builder.add_computational_memlet(block, y_access, libnode, "Y", {}, y_tensor);
    builder.add_computational_memlet(block, weight_access, libnode, "Weight", {}, weight_tensor);
    builder.add_computational_memlet(block, indices_access, libnode, "Indices", {}, indices_tensor);
    builder.add_computational_memlet(block, max_norm_access, libnode, "MaxNorm", {}, float_desc);
    builder.add_computational_memlet(block, norm_type_access, libnode, "NormType", {}, float_desc);

    return static_cast<math::tensor::EmbeddingRenormNode&>(libnode);
}

// Builds a renorm node with concrete shapes, runs the library-node expansion
// pass, and asserts the resulting SDFG re-validates. Exercises one norm_type
// code path per call.
void expand_with_norm_type(const std::string& norm_type) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();

    build_renorm(builder, {symbolic::integer(10), symbolic::integer(4)}, {symbolic::integer(3)}, "1.0", norm_type);

    ASSERT_NO_THROW(sdfg.validate());
    dump_sdfg(sdfg, "0.before");

    analysis::AnalysisManager analysis_manager(sdfg);
    passes::LibraryNodeExpansionPass expansion;
    ASSERT_TRUE(expansion.run(builder, analysis_manager));
    ASSERT_NO_THROW(sdfg.validate());
    dump_sdfg(sdfg, "1.after");
}

} // namespace

TEST(EmbeddingRenormNodeTest, symbolic) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();

    types::Scalar sym_desc(types::PrimitiveType::Int64);
    builder.add_container("n", sym_desc);
    builder.add_container("m", sym_desc);
    builder.add_container("k", sym_desc);
    auto n = symbolic::symbol("n");
    auto m = symbolic::symbol("m");
    auto k = symbolic::symbol("k");

    auto& renorm_node = build_renorm(builder, {n, m}, {k}, "1.5", "2.0");

    ASSERT_NO_THROW(sdfg.validate());

    EXPECT_EQ(renorm_node.y_layout().dims(), 2);
    EXPECT_EQ(renorm_node.weight_layout().dims(), 2);
    EXPECT_EQ(renorm_node.indices_layout().dims(), 1);

    auto symbols = renorm_node.symbols();
    EXPECT_EQ(symbols.size(), 3);
    EXPECT_TRUE(symbols.contains(n));
    EXPECT_TRUE(symbols.contains(m));
    EXPECT_TRUE(symbols.contains(k));

    builder.add_container("p", sym_desc);
    auto p = symbolic::symbol("p");
    builder.replace_symbols(k, p);

    ASSERT_NO_THROW(sdfg.validate());

    symbols = renorm_node.symbols();
    EXPECT_EQ(symbols.size(), 3);
    EXPECT_TRUE(symbols.contains(n));
    EXPECT_TRUE(symbols.contains(m));
    EXPECT_TRUE(symbols.contains(p));
    EXPECT_FALSE(symbols.contains(k));
}

TEST(EmbeddingRenormNodeTest, expand_l2_norm) { expand_with_norm_type("2.0"); }

TEST(EmbeddingRenormNodeTest, expand_l1_norm) { expand_with_norm_type("1.0"); }

TEST(EmbeddingRenormNodeTest, expand_l3_norm) { expand_with_norm_type("3.0"); }

TEST(EmbeddingRenormNodeTest, expand_fractional_norm) { expand_with_norm_type("0.5"); }

TEST(EmbeddingRenormNodeTest, expand_inf_norm) { expand_with_norm_type("INFINITY"); }

TEST(EmbeddingRenormNodeTest, expand_2d_indices) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();

    build_renorm(
        builder,
        {symbolic::integer(10), symbolic::integer(4)},
        {symbolic::integer(2), symbolic::integer(3)},
        "1.0",
        "2.0"
    );

    ASSERT_NO_THROW(sdfg.validate());
    dump_sdfg(sdfg, "0.before");

    analysis::AnalysisManager analysis_manager(sdfg);
    passes::LibraryNodeExpansionPass expansion;
    ASSERT_TRUE(expansion.run(builder, analysis_manager));
    ASSERT_NO_THROW(sdfg.validate());
    dump_sdfg(sdfg, "1.after");
}

TEST(EmbeddingRenormNodeTest, serialization) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();

    build_renorm(builder, {symbolic::integer(10), symbolic::integer(4)}, {symbolic::integer(3)}, "1.5", "2.0");

    ASSERT_NO_THROW(sdfg.validate());

    serializer::JSONSerializer serializer;
    nlohmann::json j;
    ASSERT_NO_THROW(j = serializer.serialize(sdfg));

    std::unique_ptr<StructuredSDFG> new_sdfg;
    ASSERT_NO_THROW(new_sdfg = serializer.deserialize(j));
}

TEST(EmbeddingRenormNodeTest, validate_rejects_non_2d_weight) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();

    auto& renorm_node = build_renorm(builder, {symbolic::integer(10)}, {symbolic::integer(3)}, "1.0", "2.0");

    EXPECT_THROW(renorm_node.validate(sdfg), InvalidSDFGException);
}
