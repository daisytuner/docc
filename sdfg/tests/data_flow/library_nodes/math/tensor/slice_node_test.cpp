#include "sdfg/data_flow/library_nodes/math/tensor/slice_node.h"

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

TEST(SliceNodeTest, symbolic) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("X", desc, true);
    builder.add_container("Y", desc, true);

    types::Scalar sym_desc(types::PrimitiveType::Int64);
    builder.add_container("i", sym_desc);
    builder.add_container("m", sym_desc);
    auto i = symbolic::symbol("i");
    auto m = symbolic::symbol("m");

    std::vector<symbolic::Expression> input_shape({symbolic::integer(1), i, m});
    math::tensor::TensorLayout X_layout(input_shape);
    types::Tensor X_tensor(base_desc, X_layout);
    // Slice dim 0 (size 1) -> output shape unchanged.
    math::tensor::TensorLayout Y_layout({symbolic::integer(1), i, m});
    types::Tensor Y_tensor(base_desc, Y_layout);

    auto& block = builder.add_block(root);
    auto& X_access = builder.add_access(block, "X");
    auto& Y_access = builder.add_access(block, "Y");
    auto& libnode = builder.add_library_node<math::tensor::SliceNode>(block, DebugInfo(), input_shape, 0, 0, 1, 1);
    builder.add_computational_memlet(block, Y_access, libnode, "Y", {}, Y_tensor);
    builder.add_computational_memlet(block, X_access, libnode, "X", {}, X_tensor);

    ASSERT_NO_THROW(sdfg.validate());

    auto& slice_node = static_cast<math::tensor::SliceNode&>(libnode);
    auto symbols = slice_node.symbols();
    EXPECT_EQ(symbols.size(), 2);
    EXPECT_TRUE(symbols.contains(i));
    EXPECT_TRUE(symbols.contains(m));

    builder.add_container("k", sym_desc);
    builder.add_container("n", sym_desc);
    auto k = symbolic::symbol("k");
    auto n = symbolic::symbol("n");

    builder.replace_symbols(i, k);

    symbolic::ExpressionMapping mapping({{m, n}});
    builder.replace_symbols(mapping);

    ASSERT_NO_THROW(sdfg.validate());

    symbols = slice_node.symbols();
    EXPECT_EQ(symbols.size(), 2);
    EXPECT_TRUE(symbols.contains(k));
    EXPECT_TRUE(symbols.contains(n));
}

TEST(SliceNodeTest, getters) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("X", desc, true);
    builder.add_container("Y", desc, true);

    std::vector<symbolic::Expression> input_shape({symbolic::integer(5), symbolic::integer(4)});

    auto& block = builder.add_block(root);
    auto& libnode = builder.add_library_node<math::tensor::SliceNode>(block, DebugInfo(), input_shape, 1, 1, 3, 2);

    auto& slice_node = static_cast<math::tensor::SliceNode&>(libnode);
    EXPECT_EQ(slice_node.dim(), 1);
    EXPECT_EQ(slice_node.start(), 1);
    EXPECT_EQ(slice_node.end(), 3);
    EXPECT_EQ(slice_node.step(), 2);
    ASSERT_EQ(slice_node.input_shape().size(), 2);
    EXPECT_TRUE(symbolic::eq(slice_node.input_shape().at(0), symbolic::integer(5)));
    EXPECT_TRUE(symbolic::eq(slice_node.input_shape().at(1), symbolic::integer(4)));
    EXPECT_TRUE(slice_node.supports_integer_types());
}

TEST(SliceNodeTest, expand) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("X", desc, true);
    builder.add_container("Y", desc, true);

    // Input (5, 4), slice dim 1 [1:3:1] -> output (5, 2).
    std::vector<symbolic::Expression> input_shape({symbolic::integer(5), symbolic::integer(4)});
    math::tensor::TensorLayout X_layout(input_shape);
    types::Tensor X_tensor(base_desc, X_layout);
    math::tensor::TensorLayout Y_layout({symbolic::integer(5), symbolic::integer(2)});
    types::Tensor Y_tensor(base_desc, Y_layout);

    auto& block = builder.add_block(root);
    auto& X_access = builder.add_access(block, "X");
    auto& Y_access = builder.add_access(block, "Y");
    auto& libnode = builder.add_library_node<math::tensor::SliceNode>(block, DebugInfo(), input_shape, 1, 1, 3, 1);
    builder.add_computational_memlet(block, Y_access, libnode, "Y", {}, Y_tensor);
    builder.add_computational_memlet(block, X_access, libnode, "X", {}, X_tensor);

    ASSERT_NO_THROW(sdfg.validate());
    dump_sdfg(sdfg, "0.before");

    analysis::AnalysisManager analysis_manager(sdfg);
    passes::LibraryNodeExpansionPass expansion;
    ASSERT_TRUE(expansion.run(builder, analysis_manager));
    ASSERT_NO_THROW(sdfg.validate());
    dump_sdfg(sdfg, "1.after");
}

TEST(SliceNodeTest, expand_step) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("X", desc, true);
    builder.add_container("Y", desc, true);

    // Input (5, 4), slice dim 0 [0:5:2] -> output (3, 4).
    std::vector<symbolic::Expression> input_shape({symbolic::integer(5), symbolic::integer(4)});
    math::tensor::TensorLayout X_layout(input_shape);
    types::Tensor X_tensor(base_desc, X_layout);
    math::tensor::TensorLayout Y_layout({symbolic::integer(3), symbolic::integer(4)});
    types::Tensor Y_tensor(base_desc, Y_layout);

    auto& block = builder.add_block(root);
    auto& X_access = builder.add_access(block, "X");
    auto& Y_access = builder.add_access(block, "Y");
    auto& libnode = builder.add_library_node<math::tensor::SliceNode>(block, DebugInfo(), input_shape, 0, 0, 5, 2);
    builder.add_computational_memlet(block, Y_access, libnode, "Y", {}, Y_tensor);
    builder.add_computational_memlet(block, X_access, libnode, "X", {}, X_tensor);

    ASSERT_NO_THROW(sdfg.validate());

    analysis::AnalysisManager analysis_manager(sdfg);
    passes::LibraryNodeExpansionPass expansion;
    ASSERT_TRUE(expansion.run(builder, analysis_manager));
    ASSERT_NO_THROW(sdfg.validate());
}

TEST(SliceNodeTest, serialization) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("X", desc, true);
    builder.add_container("Y", desc, true);

    std::vector<symbolic::Expression> input_shape({symbolic::integer(5), symbolic::integer(4)});
    math::tensor::TensorLayout X_layout(input_shape);
    types::Tensor X_tensor(base_desc, X_layout);
    math::tensor::TensorLayout Y_layout({symbolic::integer(5), symbolic::integer(2)});
    types::Tensor Y_tensor(base_desc, Y_layout);

    auto& block = builder.add_block(root);
    auto& X_access = builder.add_access(block, "X");
    auto& Y_access = builder.add_access(block, "Y");
    auto& libnode = builder.add_library_node<math::tensor::SliceNode>(block, DebugInfo(), input_shape, 1, 1, 3, 1);
    builder.add_computational_memlet(block, Y_access, libnode, "Y", {}, Y_tensor);
    builder.add_computational_memlet(block, X_access, libnode, "X", {}, X_tensor);

    ASSERT_NO_THROW(sdfg.validate());

    serializer::JSONSerializer serializer;
    nlohmann::json j;
    ASSERT_NO_THROW(j = serializer.serialize(sdfg));

    std::unique_ptr<StructuredSDFG> new_sdfg;
    ASSERT_NO_THROW(new_sdfg = serializer.deserialize(j));

    // Verify the round-tripped node preserves its parameters.
    auto& new_block = static_cast<structured_control_flow::Block&>(new_sdfg->root().at(0));
    math::tensor::SliceNode* new_node = nullptr;
    for (auto& node : new_block.dataflow().nodes()) {
        if (auto* candidate = dynamic_cast<math::tensor::SliceNode*>(&node)) {
            new_node = candidate;
            break;
        }
    }
    ASSERT_NE(new_node, nullptr);
    EXPECT_EQ(new_node->dim(), 1);
    EXPECT_EQ(new_node->start(), 1);
    EXPECT_EQ(new_node->end(), 3);
    EXPECT_EQ(new_node->step(), 1);
    ASSERT_EQ(new_node->input_shape().size(), 2);
    EXPECT_TRUE(symbolic::eq(new_node->input_shape().at(0), symbolic::integer(5)));
    EXPECT_TRUE(symbolic::eq(new_node->input_shape().at(1), symbolic::integer(4)));
}

TEST(SliceNodeTest, validate_dim_out_of_range) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("X", desc, true);
    builder.add_container("Y", desc, true);

    std::vector<symbolic::Expression> input_shape({symbolic::integer(5), symbolic::integer(4)});
    math::tensor::TensorLayout X_layout(input_shape);
    types::Tensor X_tensor(base_desc, X_layout);
    types::Tensor Y_tensor(base_desc, X_layout);

    auto& block = builder.add_block(root);
    auto& X_access = builder.add_access(block, "X");
    auto& Y_access = builder.add_access(block, "Y");
    // dim 2 is out of range for a rank-2 tensor.
    auto& libnode = builder.add_library_node<math::tensor::SliceNode>(block, DebugInfo(), input_shape, 2, 0, 1, 1);
    builder.add_computational_memlet(block, Y_access, libnode, "Y", {}, Y_tensor);
    builder.add_computational_memlet(block, X_access, libnode, "X", {}, X_tensor);

    EXPECT_THROW(sdfg.validate(), InvalidSDFGException);
}

TEST(SliceNodeTest, validate_negative_dim) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("X", desc, true);
    builder.add_container("Y", desc, true);

    std::vector<symbolic::Expression> input_shape({symbolic::integer(5), symbolic::integer(4)});
    math::tensor::TensorLayout X_layout(input_shape);
    types::Tensor X_tensor(base_desc, X_layout);
    types::Tensor Y_tensor(base_desc, X_layout);

    auto& block = builder.add_block(root);
    auto& X_access = builder.add_access(block, "X");
    auto& Y_access = builder.add_access(block, "Y");
    auto& libnode = builder.add_library_node<math::tensor::SliceNode>(block, DebugInfo(), input_shape, -1, 0, 1, 1);
    builder.add_computational_memlet(block, Y_access, libnode, "Y", {}, Y_tensor);
    builder.add_computational_memlet(block, X_access, libnode, "X", {}, X_tensor);

    EXPECT_THROW(sdfg.validate(), InvalidSDFGException);
}

TEST(SliceNodeTest, validate_step) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("X", desc, true);
    builder.add_container("Y", desc, true);

    std::vector<symbolic::Expression> input_shape({symbolic::integer(5), symbolic::integer(4)});
    math::tensor::TensorLayout X_layout(input_shape);
    types::Tensor X_tensor(base_desc, X_layout);
    types::Tensor Y_tensor(base_desc, X_layout);

    auto& block = builder.add_block(root);
    auto& X_access = builder.add_access(block, "X");
    auto& Y_access = builder.add_access(block, "Y");
    // step must be positive.
    auto& libnode = builder.add_library_node<math::tensor::SliceNode>(block, DebugInfo(), input_shape, 0, 0, 1, 0);
    builder.add_computational_memlet(block, Y_access, libnode, "Y", {}, Y_tensor);
    builder.add_computational_memlet(block, X_access, libnode, "X", {}, X_tensor);

    EXPECT_THROW(sdfg.validate(), InvalidSDFGException);
}

TEST(SliceNodeTest, validate_negative_start) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("X", desc, true);
    builder.add_container("Y", desc, true);

    std::vector<symbolic::Expression> input_shape({symbolic::integer(5), symbolic::integer(4)});
    math::tensor::TensorLayout X_layout(input_shape);
    types::Tensor X_tensor(base_desc, X_layout);
    types::Tensor Y_tensor(base_desc, X_layout);

    auto& block = builder.add_block(root);
    auto& X_access = builder.add_access(block, "X");
    auto& Y_access = builder.add_access(block, "Y");
    auto& libnode = builder.add_library_node<math::tensor::SliceNode>(block, DebugInfo(), input_shape, 0, -1, 1, 1);
    builder.add_computational_memlet(block, Y_access, libnode, "Y", {}, Y_tensor);
    builder.add_computational_memlet(block, X_access, libnode, "X", {}, X_tensor);

    EXPECT_THROW(sdfg.validate(), InvalidSDFGException);
}

TEST(SliceNodeTest, validate_end_before_start) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("X", desc, true);
    builder.add_container("Y", desc, true);

    std::vector<symbolic::Expression> input_shape({symbolic::integer(5), symbolic::integer(4)});
    math::tensor::TensorLayout X_layout(input_shape);
    types::Tensor X_tensor(base_desc, X_layout);
    types::Tensor Y_tensor(base_desc, X_layout);

    auto& block = builder.add_block(root);
    auto& X_access = builder.add_access(block, "X");
    auto& Y_access = builder.add_access(block, "Y");
    // end < start is invalid.
    auto& libnode = builder.add_library_node<math::tensor::SliceNode>(block, DebugInfo(), input_shape, 0, 3, 1, 1);
    builder.add_computational_memlet(block, Y_access, libnode, "Y", {}, Y_tensor);
    builder.add_computational_memlet(block, X_access, libnode, "X", {}, X_tensor);

    EXPECT_THROW(sdfg.validate(), InvalidSDFGException);
}
