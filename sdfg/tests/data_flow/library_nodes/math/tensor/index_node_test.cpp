#include "sdfg/data_flow/library_nodes/math/tensor/index_node.h"

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

TEST(IndexNodeTest, symbolic) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Scalar int_desc(types::PrimitiveType::Int64);
    types::Pointer desc(base_desc);
    types::Pointer int_ptr_desc(int_desc);
    builder.add_container("X", desc, true);
    builder.add_container("I0", int_ptr_desc, true);
    builder.add_container("Y", desc, true);

    types::Scalar sym_desc(types::PrimitiveType::Int64);
    builder.add_container("n", sym_desc);
    builder.add_container("m", sym_desc);
    builder.add_container("k", sym_desc);
    auto n = symbolic::symbol("n");
    auto m = symbolic::symbol("m");
    auto k = symbolic::symbol("k");

    // X: (n, m), I0: (k,), Y: (k, m); dim_offset=0, num_indices=1.
    std::vector<symbolic::Expression> input_shape({n, m});
    std::vector<symbolic::Expression> index_shape({k});
    std::vector<symbolic::Expression> output_shape({k, m});

    math::tensor::TensorLayout X_layout(input_shape);
    types::Tensor X_tensor(base_desc, X_layout);
    math::tensor::TensorLayout I0_layout(index_shape);
    types::Tensor I0_tensor(int_desc, I0_layout);
    math::tensor::TensorLayout Y_layout(output_shape);
    types::Tensor Y_tensor(base_desc, Y_layout);

    auto& block = builder.add_block(root);
    auto& X_access = builder.add_access(block, "X");
    auto& I0_access = builder.add_access(block, "I0");
    auto& Y_access = builder.add_access(block, "Y");
    auto& libnode =
        builder.add_library_node<math::tensor::IndexNode>(block, DebugInfo(), input_shape, index_shape, 0, 1);
    builder.add_computational_memlet(block, Y_access, libnode, "Y", {}, Y_tensor);
    builder.add_computational_memlet(block, X_access, libnode, "X", {}, X_tensor);
    builder.add_computational_memlet(block, I0_access, libnode, "I0", {}, I0_tensor);

    ASSERT_NO_THROW(sdfg.validate());

    auto& index_node = static_cast<math::tensor::IndexNode&>(libnode);
    auto symbols = index_node.symbols();
    EXPECT_EQ(symbols.size(), 3);
    EXPECT_TRUE(symbols.contains(n));
    EXPECT_TRUE(symbols.contains(m));
    EXPECT_TRUE(symbols.contains(k));

    builder.add_container("p", sym_desc);
    builder.add_container("q", sym_desc);
    auto p = symbolic::symbol("p");
    auto q = symbolic::symbol("q");

    builder.replace_symbols(n, p);

    symbolic::ExpressionMapping mapping({{m, q}});
    builder.replace_symbols(mapping);

    ASSERT_NO_THROW(sdfg.validate());

    symbols = index_node.symbols();
    EXPECT_EQ(symbols.size(), 3);
    EXPECT_TRUE(symbols.contains(p));
    EXPECT_TRUE(symbols.contains(q));
    EXPECT_TRUE(symbols.contains(k));
}

TEST(IndexNodeTest, expand) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Scalar int_desc(types::PrimitiveType::Int64);
    types::Pointer desc(base_desc);
    types::Pointer int_ptr_desc(int_desc);
    builder.add_container("X", desc, true);
    builder.add_container("I0", int_ptr_desc, true);
    builder.add_container("Y", desc, true);

    // X: (5, 4), I0: (3,), Y: (3, 4); dim_offset=0, num_indices=1.
    std::vector<symbolic::Expression> input_shape({symbolic::integer(5), symbolic::integer(4)});
    std::vector<symbolic::Expression> index_shape({symbolic::integer(3)});
    std::vector<symbolic::Expression> output_shape({symbolic::integer(3), symbolic::integer(4)});

    math::tensor::TensorLayout X_layout(input_shape);
    types::Tensor X_tensor(base_desc, X_layout);
    math::tensor::TensorLayout I0_layout(index_shape);
    types::Tensor I0_tensor(int_desc, I0_layout);
    math::tensor::TensorLayout Y_layout(output_shape);
    types::Tensor Y_tensor(base_desc, Y_layout);

    auto& block = builder.add_block(root);
    auto& X_access = builder.add_access(block, "X");
    auto& I0_access = builder.add_access(block, "I0");
    auto& Y_access = builder.add_access(block, "Y");
    auto& libnode =
        builder.add_library_node<math::tensor::IndexNode>(block, DebugInfo(), input_shape, index_shape, 0, 1);
    builder.add_computational_memlet(block, Y_access, libnode, "Y", {}, Y_tensor);
    builder.add_computational_memlet(block, X_access, libnode, "X", {}, X_tensor);
    builder.add_computational_memlet(block, I0_access, libnode, "I0", {}, I0_tensor);

    ASSERT_NO_THROW(sdfg.validate());
    dump_sdfg(sdfg, "0.before");

    analysis::AnalysisManager analysis_manager(sdfg);
    passes::LibraryNodeExpansionPass expansion;
    ASSERT_TRUE(expansion.run(builder, analysis_manager));
    ASSERT_NO_THROW(sdfg.validate());
    dump_sdfg(sdfg, "1.after");
}

TEST(IndexNodeTest, expand_two_indices) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Scalar int_desc(types::PrimitiveType::Int64);
    types::Pointer desc(base_desc);
    types::Pointer int_ptr_desc(int_desc);
    builder.add_container("X", desc, true);
    builder.add_container("I0", int_ptr_desc, true);
    builder.add_container("I1", int_ptr_desc, true);
    builder.add_container("Y", desc, true);

    // X: (5, 4), I0/I1: (3,), Y: (3,); dim_offset=0, num_indices=2 (advanced indexing).
    std::vector<symbolic::Expression> input_shape({symbolic::integer(5), symbolic::integer(4)});
    std::vector<symbolic::Expression> index_shape({symbolic::integer(3)});
    std::vector<symbolic::Expression> output_shape({symbolic::integer(3)});

    math::tensor::TensorLayout X_layout(input_shape);
    types::Tensor X_tensor(base_desc, X_layout);
    math::tensor::TensorLayout I_layout(index_shape);
    types::Tensor I_tensor(int_desc, I_layout);
    math::tensor::TensorLayout Y_layout(output_shape);
    types::Tensor Y_tensor(base_desc, Y_layout);

    auto& block = builder.add_block(root);
    auto& X_access = builder.add_access(block, "X");
    auto& I0_access = builder.add_access(block, "I0");
    auto& I1_access = builder.add_access(block, "I1");
    auto& Y_access = builder.add_access(block, "Y");
    auto& libnode =
        builder.add_library_node<math::tensor::IndexNode>(block, DebugInfo(), input_shape, index_shape, 0, 2);
    builder.add_computational_memlet(block, Y_access, libnode, "Y", {}, Y_tensor);
    builder.add_computational_memlet(block, X_access, libnode, "X", {}, X_tensor);
    builder.add_computational_memlet(block, I0_access, libnode, "I0", {}, I_tensor);
    builder.add_computational_memlet(block, I1_access, libnode, "I1", {}, I_tensor);

    ASSERT_NO_THROW(sdfg.validate());
    dump_sdfg(sdfg, "0.before");

    analysis::AnalysisManager analysis_manager(sdfg);
    passes::LibraryNodeExpansionPass expansion;
    ASSERT_TRUE(expansion.run(builder, analysis_manager));
    ASSERT_NO_THROW(sdfg.validate());
    dump_sdfg(sdfg, "1.after");
}

TEST(IndexNodeTest, expand_middle_dim) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Scalar int_desc(types::PrimitiveType::Int64);
    types::Pointer desc(base_desc);
    types::Pointer int_ptr_desc(int_desc);
    builder.add_container("X", desc, true);
    builder.add_container("I0", int_ptr_desc, true);
    builder.add_container("Y", desc, true);

    // X: (5, 4, 6), I0: (3,), Y: (5, 3, 6); dim_offset=1, num_indices=1.
    std::vector<symbolic::Expression> input_shape({symbolic::integer(5), symbolic::integer(4), symbolic::integer(6)});
    std::vector<symbolic::Expression> index_shape({symbolic::integer(3)});
    std::vector<symbolic::Expression> output_shape({symbolic::integer(5), symbolic::integer(3), symbolic::integer(6)});

    math::tensor::TensorLayout X_layout(input_shape);
    types::Tensor X_tensor(base_desc, X_layout);
    math::tensor::TensorLayout I0_layout(index_shape);
    types::Tensor I0_tensor(int_desc, I0_layout);
    math::tensor::TensorLayout Y_layout(output_shape);
    types::Tensor Y_tensor(base_desc, Y_layout);

    auto& block = builder.add_block(root);
    auto& X_access = builder.add_access(block, "X");
    auto& I0_access = builder.add_access(block, "I0");
    auto& Y_access = builder.add_access(block, "Y");
    auto& libnode =
        builder.add_library_node<math::tensor::IndexNode>(block, DebugInfo(), input_shape, index_shape, 1, 1);
    builder.add_computational_memlet(block, Y_access, libnode, "Y", {}, Y_tensor);
    builder.add_computational_memlet(block, X_access, libnode, "X", {}, X_tensor);
    builder.add_computational_memlet(block, I0_access, libnode, "I0", {}, I0_tensor);

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
    types::Scalar int_desc(types::PrimitiveType::Int64);
    types::Pointer desc(base_desc);
    types::Pointer int_ptr_desc(int_desc);
    builder.add_container("X", desc, true);
    builder.add_container("I0", int_ptr_desc, true);
    builder.add_container("Y", desc, true);

    std::vector<symbolic::Expression> input_shape({symbolic::integer(5), symbolic::integer(4)});
    std::vector<symbolic::Expression> index_shape({symbolic::integer(3)});
    std::vector<symbolic::Expression> output_shape({symbolic::integer(3), symbolic::integer(4)});

    math::tensor::TensorLayout X_layout(input_shape);
    types::Tensor X_tensor(base_desc, X_layout);
    math::tensor::TensorLayout I0_layout(index_shape);
    types::Tensor I0_tensor(int_desc, I0_layout);
    math::tensor::TensorLayout Y_layout(output_shape);
    types::Tensor Y_tensor(base_desc, Y_layout);

    auto& block = builder.add_block(root);
    auto& X_access = builder.add_access(block, "X");
    auto& I0_access = builder.add_access(block, "I0");
    auto& Y_access = builder.add_access(block, "Y");
    auto& libnode =
        builder.add_library_node<math::tensor::IndexNode>(block, DebugInfo(), input_shape, index_shape, 0, 1);
    builder.add_computational_memlet(block, Y_access, libnode, "Y", {}, Y_tensor);
    builder.add_computational_memlet(block, X_access, libnode, "X", {}, X_tensor);
    builder.add_computational_memlet(block, I0_access, libnode, "I0", {}, I0_tensor);

    ASSERT_NO_THROW(sdfg.validate());

    serializer::JSONSerializer serializer;
    nlohmann::json j;
    ASSERT_NO_THROW(j = serializer.serialize(sdfg));

    std::unique_ptr<StructuredSDFG> new_sdfg;
    ASSERT_NO_THROW(new_sdfg = serializer.deserialize(j));
}

TEST(IndexNodeTest, validate_dim_offset_out_of_range) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Scalar int_desc(types::PrimitiveType::Int64);
    types::Pointer desc(base_desc);
    types::Pointer int_ptr_desc(int_desc);
    builder.add_container("X", desc, true);
    builder.add_container("I0", int_ptr_desc, true);
    builder.add_container("Y", desc, true);

    std::vector<symbolic::Expression> input_shape({symbolic::integer(5), symbolic::integer(4)});
    std::vector<symbolic::Expression> index_shape({symbolic::integer(3)});
    std::vector<symbolic::Expression> output_shape({symbolic::integer(3), symbolic::integer(4)});

    math::tensor::TensorLayout X_layout(input_shape);
    types::Tensor X_tensor(base_desc, X_layout);
    math::tensor::TensorLayout I0_layout(index_shape);
    types::Tensor I0_tensor(int_desc, I0_layout);
    math::tensor::TensorLayout Y_layout(output_shape);
    types::Tensor Y_tensor(base_desc, Y_layout);

    auto& block = builder.add_block(root);
    auto& X_access = builder.add_access(block, "X");
    auto& I0_access = builder.add_access(block, "I0");
    auto& Y_access = builder.add_access(block, "Y");
    // dim_offset=2 with num_indices=1 on a rank-2 input is out of range.
    auto& libnode =
        builder.add_library_node<math::tensor::IndexNode>(block, DebugInfo(), input_shape, index_shape, 2, 1);
    builder.add_computational_memlet(block, Y_access, libnode, "Y", {}, Y_tensor);
    builder.add_computational_memlet(block, X_access, libnode, "X", {}, X_tensor);
    builder.add_computational_memlet(block, I0_access, libnode, "I0", {}, I0_tensor);

    EXPECT_THROW(sdfg.validate(), InvalidSDFGException);
}
