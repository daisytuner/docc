#include "sdfg/data_flow/library_nodes/math/tensor/arange_node.h"

#include <gtest/gtest.h>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/element.h"
#include "sdfg/passes/expansion/library_node_expansion_pass.h"
#include "sdfg/types/scalar.h"
#include "sdfg/types/tensor.h"
#include "sdfg/types/type.h"

using namespace sdfg;

TEST(ArangeNodeTest, Expansion) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar start_type(types::PrimitiveType::Int64);
    types::Pointer start_ptr(start_type);
    types::Scalar end_type(types::PrimitiveType::Int64);
    types::Pointer end_ptr(end_type);
    types::Scalar step_type(types::PrimitiveType::Int64);
    types::Pointer step_ptr(step_type);

    std::vector<symbolic::Expression> shape = {symbolic::integer(10)};
    math::tensor::TensorLayout out_layout(shape);
    types::Tensor out_type(types::PrimitiveType::Int64, out_layout);
    types::Pointer out_ptr(start_type);

    builder.add_container("start", start_ptr, true);
    builder.add_container("end", end_ptr, true);
    builder.add_container("step", step_ptr, true);
    builder.add_container("out", out_ptr, true);

    auto& block = builder.add_block(root);
    auto& start_acc = builder.add_access(block, "start");
    auto& end_acc = builder.add_access(block, "end");
    auto& step_acc = builder.add_access(block, "step");
    auto& out_acc = builder.add_access(block, "out");

    auto& libnode = builder.add_library_node<math::tensor::ArangeNode>(block, DebugInfo(), shape);

    builder.add_computational_memlet(block, start_acc, libnode, "_start", {}, start_type);
    builder.add_computational_memlet(block, end_acc, libnode, "_end", {}, end_type);
    builder.add_computational_memlet(block, step_acc, libnode, "_step", {}, step_type);
    builder.add_computational_memlet(block, out_acc, libnode, "_out", {}, out_type);

    ASSERT_NO_THROW(sdfg.validate());

    analysis::AnalysisManager analysis_manager(sdfg);
    passes::LibraryNodeExpansionPass pass;
    ASSERT_TRUE(pass.run(builder, analysis_manager));
    ASSERT_NO_THROW(sdfg.validate());
}

TEST(ArangeNodeTest, SerializeDeserialize_RoundTrip) {
    builder::StructuredSDFGBuilder builder("sdfg_arange_serialize", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar start_type(types::PrimitiveType::Int64);
    types::Pointer start_ptr(start_type);
    types::Scalar end_type(types::PrimitiveType::Int64);
    types::Pointer end_ptr(end_type);
    types::Scalar step_type(types::PrimitiveType::Int64);
    types::Pointer step_ptr(step_type);
    std::vector<symbolic::Expression> shape = {symbolic::integer(10)};
    math::tensor::TensorLayout out_layout(shape);
    types::Tensor out_type(types::PrimitiveType::Int64, out_layout);
    types::Pointer out_ptr(start_type);

    builder.add_container("start", start_ptr, true);
    builder.add_container("end", end_ptr, true);
    builder.add_container("step", step_ptr, true);
    builder.add_container("out", out_ptr, true);

    auto& block = builder.add_block(root);
    auto& start_acc = builder.add_access(block, "start");
    auto& end_acc = builder.add_access(block, "end");
    auto& step_acc = builder.add_access(block, "step");
    auto& out_acc = builder.add_access(block, "out");

    auto& libnode = builder.add_library_node<math::tensor::ArangeNode>(block, DebugInfo(), shape);
    builder.add_computational_memlet(block, start_acc, libnode, "_start", {}, start_type);
    builder.add_computational_memlet(block, end_acc, libnode, "_end", {}, end_type);
    builder.add_computational_memlet(block, step_acc, libnode, "_step", {}, step_type);
    builder.add_computational_memlet(block, out_acc, libnode, "_out", {}, out_type);

    nlohmann::json j;
    serializer::JSONSerializer serializer;
    ASSERT_NO_THROW(j = serializer.serialize(sdfg));

    std::unique_ptr<StructuredSDFG> new_sdfg;
    ASSERT_NO_THROW(new_sdfg = serializer.deserialize(j));

    auto& new_root = new_sdfg->root();
    auto* deserialized_block = dynamic_cast<structured_control_flow::Block*>(&new_root.at(0));
    ASSERT_NE(deserialized_block, nullptr);

    bool found_arange = false;
    for (auto& n : deserialized_block->dataflow().nodes()) {
        if (auto* arange_node = dynamic_cast<const math::tensor::ArangeNode*>(&n)) {
            found_arange = true;
            EXPECT_EQ(arange_node->shape().size(), 1);
            EXPECT_TRUE(symbolic::null_safe_eq(arange_node->shape()[0], symbolic::integer(10)));
        }
    }
    EXPECT_TRUE(found_arange);
}

TEST(ArangeNodeTest, Expansion2D) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar start_type(types::PrimitiveType::Int64);
    types::Pointer start_ptr(start_type);
    types::Scalar end_type(types::PrimitiveType::Int64);
    types::Pointer end_ptr(end_type);
    types::Scalar step_type(types::PrimitiveType::Int64);
    types::Pointer step_ptr(step_type);

    std::vector<symbolic::Expression> shape = {symbolic::integer(10), symbolic::integer(5)};
    math::tensor::TensorLayout out_layout(shape);
    types::Tensor out_type(types::PrimitiveType::Int64, out_layout);
    types::Pointer out_ptr(start_type);

    builder.add_container("start", start_ptr, true);
    builder.add_container("end", end_ptr, true);
    builder.add_container("step", step_ptr, true);
    builder.add_container("out", out_ptr, true);

    auto& block = builder.add_block(root);
    auto& start_acc = builder.add_access(block, "start");
    auto& end_acc = builder.add_access(block, "end");
    auto& step_acc = builder.add_access(block, "step");
    auto& out_acc = builder.add_access(block, "out");

    auto& libnode = builder.add_library_node<math::tensor::ArangeNode>(block, DebugInfo(), shape);

    builder.add_computational_memlet(block, start_acc, libnode, "_start", {}, start_type);
    builder.add_computational_memlet(block, end_acc, libnode, "_end", {}, end_type);
    builder.add_computational_memlet(block, step_acc, libnode, "_step", {}, step_type);
    builder.add_computational_memlet(block, out_acc, libnode, "_out", {}, out_type);

    ASSERT_NO_THROW(sdfg.validate());

    analysis::AnalysisManager analysis_manager(sdfg);
    passes::LibraryNodeExpansionPass pass;
    ASSERT_TRUE(pass.run(builder, analysis_manager));
    ASSERT_NO_THROW(sdfg.validate());
}

TEST(ArangeNodeTest, ExpansionFloat) {
    builder::StructuredSDFGBuilder builder("sdfg_float", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar start_type(types::PrimitiveType::Float);
    types::Pointer start_ptr(start_type);
    types::Scalar end_type(types::PrimitiveType::Float);
    types::Pointer end_ptr(end_type);
    types::Scalar step_type(types::PrimitiveType::Float);
    types::Pointer step_ptr(step_type);

    std::vector<symbolic::Expression> shape = {symbolic::integer(10)};
    math::tensor::TensorLayout out_layout(shape);
    types::Tensor out_type(types::PrimitiveType::Float, out_layout);
    types::Pointer out_ptr(start_type);

    builder.add_container("start", start_ptr, true);
    builder.add_container("end", end_ptr, true);
    builder.add_container("step", step_ptr, true);
    builder.add_container("out", out_ptr, true);

    auto& block = builder.add_block(root);
    auto& start_acc = builder.add_access(block, "start");
    auto& end_acc = builder.add_access(block, "end");
    auto& step_acc = builder.add_access(block, "step");
    auto& out_acc = builder.add_access(block, "out");

    auto& libnode = builder.add_library_node<math::tensor::ArangeNode>(block, DebugInfo(), shape);

    builder.add_computational_memlet(block, start_acc, libnode, "_start", {}, start_type);
    builder.add_computational_memlet(block, end_acc, libnode, "_end", {}, end_type);
    builder.add_computational_memlet(block, step_acc, libnode, "_step", {}, step_type);
    builder.add_computational_memlet(block, out_acc, libnode, "_out", {}, out_type);

    ASSERT_NO_THROW(sdfg.validate());

    analysis::AnalysisManager analysis_manager(sdfg);
    passes::LibraryNodeExpansionPass pass;
    ASSERT_TRUE(pass.run(builder, analysis_manager));
    ASSERT_NO_THROW(sdfg.validate());
}

TEST(ArangeNodeTest, CloneAndPointerAccess) {
    builder::StructuredSDFGBuilder builder("sdfg_clone", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    std::vector<symbolic::Expression> shape = {symbolic::integer(10)};
    auto& block = builder.add_block(root);
    auto& libnode = builder.add_library_node<math::tensor::ArangeNode>(block, DebugInfo(), shape);

    // Test clone
    auto cloned_node = libnode.clone(libnode.element_id(), libnode.vertex(), block.dataflow());
    ASSERT_NE(cloned_node, nullptr);
    auto* typed_clone = dynamic_cast<math::tensor::ArangeNode*>(cloned_node.get());
    ASSERT_NE(typed_clone, nullptr);

    // Test pointer_access_type
    auto out_access = libnode.pointer_access_type(math::tensor::ArangeNode::RESULT_PTR_IDX);
    EXPECT_TRUE(out_access->may_contain_writes());
    EXPECT_FALSE(out_access->may_contain_reads());

    auto start_access = libnode.pointer_access_type(math::tensor::ArangeNode::START_IDX);
    EXPECT_FALSE(start_access->may_contain_writes());
    EXPECT_TRUE(start_access->may_contain_reads());

    auto end_access = libnode.pointer_access_type(math::tensor::ArangeNode::END_IDX);
    EXPECT_FALSE(end_access->may_contain_writes());
    EXPECT_TRUE(end_access->may_contain_reads());

    auto step_access = libnode.pointer_access_type(math::tensor::ArangeNode::STEP_IDX);
    EXPECT_FALSE(step_access->may_contain_writes());
    EXPECT_TRUE(step_access->may_contain_reads());
}

TEST(ArangeNodeTest, SymbolsAndReplace) {
    builder::StructuredSDFGBuilder builder("sdfg_symbols", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    auto N = symbolic::symbol("N");
    std::vector<symbolic::Expression> shape = {N};
    auto& block = builder.add_block(root);
    auto& libnode = builder.add_library_node<math::tensor::ArangeNode>(block, DebugInfo(), shape);

    // Test symbols()
    auto syms = libnode.symbols();
    EXPECT_EQ(syms.size(), 1);
    EXPECT_TRUE(syms.find(N) != syms.end());

    // Test replace(old_expr, new_expr)
    auto M = symbolic::symbol("M");
    libnode.replace(N, M);
    syms = libnode.symbols();
    EXPECT_EQ(syms.size(), 1);
    EXPECT_TRUE(syms.find(M) != syms.end());
    EXPECT_TRUE(syms.find(N) == syms.end());

    // Test replace(replacements)
    auto K = symbolic::symbol("K");
    symbolic::ExpressionMapping replacements;
    replacements[M] = K;
    libnode.replace(replacements);
    syms = libnode.symbols();
    EXPECT_EQ(syms.size(), 1);
    EXPECT_TRUE(syms.find(K) != syms.end());
    EXPECT_TRUE(syms.find(M) == syms.end());
}
