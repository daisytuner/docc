#include "sdfg/data_flow/library_nodes/math/tensor/concat_node.h"

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

TEST(ConcatNodeTest, symbolic) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("A", desc, true);
    builder.add_container("B", desc, true);
    builder.add_container("C", desc, true);

    types::Scalar sym_desc(types::PrimitiveType::Int64);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);
    builder.add_container("m", sym_desc);
    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");
    auto m = symbolic::symbol("m");

    math::tensor::TensorLayout A_layout({symbolic::integer(1), i, m});
    types::Tensor A_tensor(base_desc, A_layout);
    math::tensor::TensorLayout B_layout({symbolic::integer(1), j, m});
    types::Tensor B_tensor(base_desc, B_layout);
    math::tensor::TensorLayout C_layout({symbolic::integer(1), symbolic::add(i, j), m});
    types::Tensor C_tensor(base_desc, C_layout);

    auto& block = builder.add_block(root);
    auto& A_access = builder.add_access(block, "A");
    auto& B_access = builder.add_access(block, "B");
    auto& C_access = builder.add_access(block, "C");
    auto& libnode = builder.add_library_node<math::tensor::ConcatNode>(
        block,
        DebugInfo(),
        "Y",
        C_layout,
        std::vector<std::string>({"X0", "X1"}),
        std::vector<math::tensor::TensorLayout>({A_layout, B_layout}),
        1
    );
    auto& A_edge = builder.add_computational_memlet(block, A_access, libnode, "X0", {}, A_tensor);
    auto& B_edge = builder.add_computational_memlet(block, B_access, libnode, "X1", {}, B_tensor);
    auto& C_edge = builder.add_computational_memlet(block, C_access, libnode, "Y", {}, C_tensor);

    ASSERT_NO_THROW(sdfg.validate());

    auto& concat_node = static_cast<math::tensor::ConcatNode&>(libnode);
    auto symbols = concat_node.symbols();
    EXPECT_EQ(symbols.size(), 3);
    EXPECT_TRUE(symbols.contains(i));
    EXPECT_TRUE(symbols.contains(j));
    EXPECT_TRUE(symbols.contains(m));

    builder.add_container("k", sym_desc);
    builder.add_container("n", sym_desc);
    auto k = symbolic::symbol("k");
    auto n = symbolic::symbol("n");

    builder.replace_symbols(i, k);

    symbolic::ExpressionMapping mapping({{m, n}});
    builder.replace_symbols(mapping);

    ASSERT_NO_THROW(sdfg.validate());

    symbols = concat_node.symbols();
    EXPECT_EQ(symbols.size(), 3);
    EXPECT_TRUE(symbols.contains(k));
    EXPECT_TRUE(symbols.contains(j));
    EXPECT_TRUE(symbols.contains(n));
}

TEST(ConcatNodeTest, expand) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("A", desc, true);
    builder.add_container("B", desc, true);
    builder.add_container("C", desc, true);

    math::tensor::TensorLayout A_layout({symbolic::integer(1), symbolic::integer(2), symbolic::integer(3)});
    types::Tensor A_tensor(base_desc, A_layout);
    math::tensor::TensorLayout B_layout({symbolic::integer(1), symbolic::integer(4), symbolic::integer(3)});
    types::Tensor B_tensor(base_desc, B_layout);
    math::tensor::TensorLayout C_layout({symbolic::integer(1), symbolic::integer(6), symbolic::integer(3)});
    types::Tensor C_tensor(base_desc, C_layout);

    auto& block = builder.add_block(root);
    auto& A_access = builder.add_access(block, "A");
    auto& B_access = builder.add_access(block, "B");
    auto& C_access = builder.add_access(block, "C");
    auto& libnode = builder.add_library_node<math::tensor::ConcatNode>(
        block,
        DebugInfo(),
        "Y",
        C_layout,
        std::vector<std::string>({"X0", "X1"}),
        std::vector<math::tensor::TensorLayout>({A_layout, B_layout}),
        1
    );
    builder.add_computational_memlet(block, A_access, libnode, "X0", {}, A_tensor);
    builder.add_computational_memlet(block, B_access, libnode, "X1", {}, B_tensor);
    builder.add_computational_memlet(block, C_access, libnode, "Y", {}, C_tensor);

    ASSERT_NO_THROW(sdfg.validate());
    dump_sdfg(sdfg, "0.before");

    analysis::AnalysisManager analysis_manager(sdfg);
    passes::LibraryNodeExpansionPass expansion;
    ASSERT_TRUE(expansion.run(builder, analysis_manager));
    ASSERT_NO_THROW(sdfg.validate());
    dump_sdfg(sdfg, "1.after");
}

TEST(ConcatNodeTest, serialization) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("A", desc, true);
    builder.add_container("B", desc, true);
    builder.add_container("C", desc, true);

    math::tensor::TensorLayout A_layout({symbolic::integer(1), symbolic::integer(2), symbolic::integer(3)});
    types::Tensor A_tensor(base_desc, A_layout);
    math::tensor::TensorLayout B_layout({symbolic::integer(1), symbolic::integer(4), symbolic::integer(3)});
    types::Tensor B_tensor(base_desc, B_layout);
    math::tensor::TensorLayout C_layout({symbolic::integer(1), symbolic::integer(6), symbolic::integer(3)});
    types::Tensor C_tensor(base_desc, C_layout);

    auto& block = builder.add_block(root);
    auto& A_access = builder.add_access(block, "A");
    auto& B_access = builder.add_access(block, "B");
    auto& C_access = builder.add_access(block, "C");
    auto& libnode = builder.add_library_node<math::tensor::ConcatNode>(
        block,
        DebugInfo(),
        "Y",
        C_layout,
        std::vector<std::string>({"X0", "X1"}),
        std::vector<math::tensor::TensorLayout>({A_layout, B_layout}),
        1
    );
    builder.add_computational_memlet(block, A_access, libnode, "X0", {}, A_tensor);
    builder.add_computational_memlet(block, B_access, libnode, "X1", {}, B_tensor);
    builder.add_computational_memlet(block, C_access, libnode, "Y", {}, C_tensor);

    ASSERT_NO_THROW(sdfg.validate());

    serializer::JSONSerializer serializer;
    nlohmann::json j;
    ASSERT_NO_THROW(j = serializer.serialize(sdfg));

    std::unique_ptr<StructuredSDFG> new_sdfg;
    ASSERT_NO_THROW(new_sdfg = serializer.deserialize(j));
}

TEST(ConcatNodeTest, validate_tensor_type_result) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("A", desc, true);
    builder.add_container("B", desc, true);
    builder.add_container("C", desc, true);

    math::tensor::TensorLayout A_layout({symbolic::integer(1), symbolic::integer(2), symbolic::integer(3)});
    types::Tensor A_tensor(base_desc, A_layout);
    math::tensor::TensorLayout B_layout({symbolic::integer(1), symbolic::integer(4), symbolic::integer(3)});
    types::Tensor B_tensor(base_desc, B_layout);
    math::tensor::TensorLayout C_layout({symbolic::integer(1), symbolic::integer(6), symbolic::integer(3)});
    types::Tensor C_tensor(base_desc, C_layout);

    auto& block = builder.add_block(root);
    auto& A_access = builder.add_access(block, "A");
    auto& B_access = builder.add_access(block, "B");
    auto& C_access = builder.add_access(block, "C");
    auto& libnode = builder.add_library_node<math::tensor::ConcatNode>(
        block,
        DebugInfo(),
        "Y",
        C_layout,
        std::vector<std::string>({"X0", "X1"}),
        std::vector<math::tensor::TensorLayout>({A_layout, B_layout}),
        1
    );
    builder.add_computational_memlet(block, A_access, libnode, "X0", {}, A_tensor);
    builder.add_computational_memlet(block, B_access, libnode, "X1", {}, B_tensor);
    builder.add_computational_memlet(block, C_access, libnode, "Y", {}, desc);

    EXPECT_THROW(sdfg.validate(), InvalidSDFGException);
}

TEST(ConcatNodeTest, validate_tensor_type_input) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("A", desc, true);
    builder.add_container("B", desc, true);
    builder.add_container("C", desc, true);

    math::tensor::TensorLayout A_layout({symbolic::integer(1), symbolic::integer(2), symbolic::integer(3)});
    types::Tensor A_tensor(base_desc, A_layout);
    math::tensor::TensorLayout B_layout({symbolic::integer(1), symbolic::integer(4), symbolic::integer(3)});
    types::Tensor B_tensor(base_desc, B_layout);
    math::tensor::TensorLayout C_layout({symbolic::integer(1), symbolic::integer(6), symbolic::integer(3)});
    types::Tensor C_tensor(base_desc, C_layout);

    auto& block = builder.add_block(root);
    auto& A_access = builder.add_access(block, "A");
    auto& B_access = builder.add_access(block, "B");
    auto& C_access = builder.add_access(block, "C");
    auto& libnode = builder.add_library_node<math::tensor::ConcatNode>(
        block,
        DebugInfo(),
        "Y",
        C_layout,
        std::vector<std::string>({"X0", "X1"}),
        std::vector<math::tensor::TensorLayout>({A_layout, B_layout}),
        1
    );
    builder.add_computational_memlet(block, A_access, libnode, "X0", {}, desc);
    builder.add_computational_memlet(block, B_access, libnode, "X1", {}, B_tensor);
    builder.add_computational_memlet(block, C_access, libnode, "Y", {}, C_tensor);

    EXPECT_THROW(sdfg.validate(), InvalidSDFGException);
}

TEST(ConcatNodeTest, validate_tensor_layout_mismatch_result) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("A", desc, true);
    builder.add_container("B", desc, true);
    builder.add_container("C", desc, true);

    math::tensor::TensorLayout A_layout({symbolic::integer(1), symbolic::integer(2), symbolic::integer(3)});
    types::Tensor A_tensor(base_desc, A_layout);
    math::tensor::TensorLayout B_layout({symbolic::integer(1), symbolic::integer(4), symbolic::integer(3)});
    types::Tensor B_tensor(base_desc, B_layout);
    math::tensor::TensorLayout C_layout({symbolic::integer(1), symbolic::integer(6), symbolic::integer(3)});
    types::Tensor C_tensor(base_desc, C_layout);

    auto& block = builder.add_block(root);
    auto& A_access = builder.add_access(block, "A");
    auto& B_access = builder.add_access(block, "B");
    auto& C_access = builder.add_access(block, "C");
    auto& libnode = builder.add_library_node<math::tensor::ConcatNode>(
        block,
        DebugInfo(),
        "Y",
        C_layout,
        std::vector<std::string>({"X0", "X1"}),
        std::vector<math::tensor::TensorLayout>({A_layout, B_layout}),
        1
    );
    builder.add_computational_memlet(block, A_access, libnode, "X0", {}, A_tensor);
    builder.add_computational_memlet(block, B_access, libnode, "X1", {}, B_tensor);
    builder.add_computational_memlet(block, C_access, libnode, "Y", {}, A_tensor);

    EXPECT_THROW(sdfg.validate(), InvalidSDFGException);
}

TEST(ConcatNodeTest, validate_tensor_layout_mismatch_input) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("A", desc, true);
    builder.add_container("B", desc, true);
    builder.add_container("C", desc, true);

    math::tensor::TensorLayout A_layout({symbolic::integer(1), symbolic::integer(2), symbolic::integer(3)});
    types::Tensor A_tensor(base_desc, A_layout);
    math::tensor::TensorLayout B_layout({symbolic::integer(1), symbolic::integer(4), symbolic::integer(3)});
    types::Tensor B_tensor(base_desc, B_layout);
    math::tensor::TensorLayout C_layout({symbolic::integer(1), symbolic::integer(6), symbolic::integer(3)});
    types::Tensor C_tensor(base_desc, C_layout);

    auto& block = builder.add_block(root);
    auto& A_access = builder.add_access(block, "A");
    auto& B_access = builder.add_access(block, "B");
    auto& C_access = builder.add_access(block, "C");
    auto& libnode = builder.add_library_node<math::tensor::ConcatNode>(
        block,
        DebugInfo(),
        "Y",
        C_layout,
        std::vector<std::string>({"X0", "X1"}),
        std::vector<math::tensor::TensorLayout>({A_layout, B_layout}),
        1
    );
    builder.add_computational_memlet(block, A_access, libnode, "X0", {}, C_tensor);
    builder.add_computational_memlet(block, B_access, libnode, "X1", {}, B_tensor);
    builder.add_computational_memlet(block, C_access, libnode, "Y", {}, C_tensor);

    EXPECT_THROW(sdfg.validate(), InvalidSDFGException);
}

TEST(ConcatNodeTest, validate_dim) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("A", desc, true);
    builder.add_container("B", desc, true);
    builder.add_container("C", desc, true);

    math::tensor::TensorLayout A_layout({symbolic::integer(1), symbolic::integer(2), symbolic::integer(3)});
    types::Tensor A_tensor(base_desc, A_layout);
    math::tensor::TensorLayout B_layout({symbolic::integer(1), symbolic::integer(4), symbolic::integer(3)});
    types::Tensor B_tensor(base_desc, B_layout);
    math::tensor::TensorLayout C_layout({symbolic::integer(1), symbolic::integer(6), symbolic::integer(3)});
    types::Tensor C_tensor(base_desc, C_layout);

    auto& block = builder.add_block(root);
    auto& A_access = builder.add_access(block, "A");
    auto& B_access = builder.add_access(block, "B");
    auto& C_access = builder.add_access(block, "C");
    auto& libnode = builder.add_library_node<math::tensor::ConcatNode>(
        block,
        DebugInfo(),
        "Y",
        C_layout,
        std::vector<std::string>({"X0", "X1"}),
        std::vector<math::tensor::TensorLayout>({A_layout, B_layout}),
        5
    );
    builder.add_computational_memlet(block, A_access, libnode, "X0", {}, A_tensor);
    builder.add_computational_memlet(block, B_access, libnode, "X1", {}, B_tensor);
    builder.add_computational_memlet(block, C_access, libnode, "Y", {}, C_tensor);

    EXPECT_THROW(sdfg.validate(), InvalidSDFGException);
}

TEST(ConcatNodeTest, validate_wrong_dims) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("A", desc, true);
    builder.add_container("B", desc, true);
    builder.add_container("C", desc, true);

    math::tensor::TensorLayout A_layout({symbolic::integer(2), symbolic::integer(3)});
    types::Tensor A_tensor(base_desc, A_layout);
    math::tensor::TensorLayout B_layout({symbolic::integer(1), symbolic::integer(4), symbolic::integer(3)});
    types::Tensor B_tensor(base_desc, B_layout);
    math::tensor::TensorLayout C_layout({symbolic::integer(1), symbolic::integer(6), symbolic::integer(3)});
    types::Tensor C_tensor(base_desc, C_layout);

    auto& block = builder.add_block(root);
    auto& A_access = builder.add_access(block, "A");
    auto& B_access = builder.add_access(block, "B");
    auto& C_access = builder.add_access(block, "C");
    auto& libnode = builder.add_library_node<math::tensor::ConcatNode>(
        block,
        DebugInfo(),
        "Y",
        C_layout,
        std::vector<std::string>({"X0", "X1"}),
        std::vector<math::tensor::TensorLayout>({A_layout, B_layout}),
        1
    );
    builder.add_computational_memlet(block, A_access, libnode, "X0", {}, A_tensor);
    builder.add_computational_memlet(block, B_access, libnode, "X1", {}, B_tensor);
    builder.add_computational_memlet(block, C_access, libnode, "Y", {}, C_tensor);

    EXPECT_THROW(sdfg.validate(), InvalidSDFGException);
}

TEST(ConcatNodeTest, validate_dim_shape) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("A", desc, true);
    builder.add_container("B", desc, true);
    builder.add_container("C", desc, true);

    math::tensor::TensorLayout A_layout({symbolic::integer(1), symbolic::integer(3), symbolic::integer(3)});
    types::Tensor A_tensor(base_desc, A_layout);
    math::tensor::TensorLayout B_layout({symbolic::integer(1), symbolic::integer(4), symbolic::integer(3)});
    types::Tensor B_tensor(base_desc, B_layout);
    math::tensor::TensorLayout C_layout({symbolic::integer(1), symbolic::integer(6), symbolic::integer(3)});
    types::Tensor C_tensor(base_desc, C_layout);

    auto& block = builder.add_block(root);
    auto& A_access = builder.add_access(block, "A");
    auto& B_access = builder.add_access(block, "B");
    auto& C_access = builder.add_access(block, "C");
    auto& libnode = builder.add_library_node<math::tensor::ConcatNode>(
        block,
        DebugInfo(),
        "Y",
        C_layout,
        std::vector<std::string>({"X0", "X1"}),
        std::vector<math::tensor::TensorLayout>({A_layout, B_layout}),
        1
    );
    builder.add_computational_memlet(block, A_access, libnode, "X0", {}, A_tensor);
    builder.add_computational_memlet(block, B_access, libnode, "X1", {}, B_tensor);
    builder.add_computational_memlet(block, C_access, libnode, "Y", {}, C_tensor);

    EXPECT_THROW(sdfg.validate(), InvalidSDFGException);
}

TEST(ConcatNodeTest, validate_non_dim_shape) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("A", desc, true);
    builder.add_container("B", desc, true);
    builder.add_container("C", desc, true);

    math::tensor::TensorLayout A_layout({symbolic::integer(1), symbolic::integer(2), symbolic::integer(3)});
    types::Tensor A_tensor(base_desc, A_layout);
    math::tensor::TensorLayout B_layout({symbolic::integer(2), symbolic::integer(4), symbolic::integer(3)});
    types::Tensor B_tensor(base_desc, B_layout);
    math::tensor::TensorLayout C_layout({symbolic::integer(1), symbolic::integer(6), symbolic::integer(3)});
    types::Tensor C_tensor(base_desc, C_layout);

    auto& block = builder.add_block(root);
    auto& A_access = builder.add_access(block, "A");
    auto& B_access = builder.add_access(block, "B");
    auto& C_access = builder.add_access(block, "C");
    auto& libnode = builder.add_library_node<math::tensor::ConcatNode>(
        block,
        DebugInfo(),
        "Y",
        C_layout,
        std::vector<std::string>({"X0", "X1"}),
        std::vector<math::tensor::TensorLayout>({A_layout, B_layout}),
        1
    );
    builder.add_computational_memlet(block, A_access, libnode, "X0", {}, A_tensor);
    builder.add_computational_memlet(block, B_access, libnode, "X1", {}, B_tensor);
    builder.add_computational_memlet(block, C_access, libnode, "Y", {}, C_tensor);

    EXPECT_THROW(sdfg.validate(), InvalidSDFGException);
}
