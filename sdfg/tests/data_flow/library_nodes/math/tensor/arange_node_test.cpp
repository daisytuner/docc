#include "sdfg/data_flow/library_nodes/math/tensor/arange_node.h"

#include <memory>

#include <gtest/gtest.h>
#include <nlohmann/json_fwd.hpp>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/deepcopy/structured_sdfg_deep_copy.h"
#include "sdfg/element.h"
#include "sdfg/passes/expansion/library_node_expansion_pass.h"
#include "sdfg/serializer/json_serializer.h"
#include "sdfg/structured_sdfg.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/scalar.h"
#include "sdfg/types/tensor.h"
#include "sdfg/types/type.h"
#include "sdfg_debug_dump.h"
#include "symengine/mul.h"

using namespace sdfg;

TEST(ArangeNodeTest, symbolic) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Int64);
    types::Pointer desc(base_desc);
    builder.add_container("start", base_desc, true);
    builder.add_container("end", base_desc, true);
    builder.add_container("step", base_desc, true);
    builder.add_container("result", desc, true);
    builder.add_container("n", base_desc);
    builder.add_container("m", base_desc);
    auto n = symbolic::symbol("n");
    auto m = symbolic::symbol("m");

    symbolic::MultiExpression shape = {n, m};
    math::tensor::TensorLayout result_layout(shape);
    types::Tensor result_tensor(base_desc, result_layout);

    auto& block = builder.add_block(root);
    auto& start_access = builder.add_access(block, "start");
    auto& end_access = builder.add_access(block, "end");
    auto& step_access = builder.add_access(block, "step");
    auto& out_access = builder.add_access(block, "result");
    auto& libnode = builder.add_library_node<math::tensor::ArangeNode>(block, DebugInfo(), shape);
    builder.add_computational_memlet(block, start_access, libnode, "_start", {}, base_desc);
    builder.add_computational_memlet(block, end_access, libnode, "_end", {}, base_desc);
    builder.add_computational_memlet(block, step_access, libnode, "_step", {}, base_desc);
    builder.add_computational_memlet(block, out_access, libnode, "_out", {}, result_tensor);

    ASSERT_NO_THROW(sdfg.validate());
    auto& arange_node = static_cast<math::tensor::ArangeNode&>(libnode);
    EXPECT_TRUE(arange_node.supports_integer_types());
    EXPECT_TRUE(symbolic::eq(arange_node.flop(), symbolic::mul(SymEngine::mul(shape), symbolic::integer(2))));
    ASSERT_NO_THROW(arange_node.toStr());

    auto symbols = arange_node.symbols();
    EXPECT_EQ(symbols.size(), 2);
    EXPECT_TRUE(symbols.contains(n));
    EXPECT_TRUE(symbols.contains(m));

    builder.add_container("a", base_desc);
    builder.add_container("b", base_desc);
    auto a = symbolic::symbol("a");
    auto b = symbolic::symbol("b");

    builder.replace_symbols(n, a);

    symbolic::ExpressionMapping replacements({{m, b}});
    builder.replace_symbols(replacements);

    ASSERT_NO_THROW(sdfg.validate());

    symbols = arange_node.symbols();
    EXPECT_EQ(symbols.size(), 2);
    EXPECT_TRUE(symbols.contains(a));
    EXPECT_TRUE(symbols.contains(b));

    auto check_pointer_access_meta =
        [](data_flow::PointerAccessType pam, bool no_capture, bool reads, bool writes, bool invalidate) -> void {
        EXPECT_EQ(pam->no_capture(), no_capture);
        EXPECT_EQ(pam->may_contain_reads(), reads);
        EXPECT_EQ(pam->may_contain_writes(), writes);
        EXPECT_EQ(pam->invalidated_after(), invalidate);
    };
    check_pointer_access_meta(
        arange_node.pointer_access_type(math::tensor::ArangeNode::RESULT_PTR_IDX), true, false, true, false
    );
    check_pointer_access_meta(
        arange_node.pointer_access_type(math::tensor::ArangeNode::START_IDX), true, true, false, false
    );
    check_pointer_access_meta(arange_node.pointer_access_type(math::tensor::ArangeNode::END_IDX), true, true, false, false);
    check_pointer_access_meta(
        arange_node.pointer_access_type(math::tensor::ArangeNode::STEP_IDX), true, true, false, false
    );
}

TEST(ArangeNodeTest, cloning) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Int64);
    types::Pointer desc(base_desc);
    builder.add_container("start", base_desc, true);
    builder.add_container("end", base_desc, true);
    builder.add_container("step", base_desc, true);
    builder.add_container("result", desc, true);

    symbolic::MultiExpression shape = {symbolic::integer(10)};
    math::tensor::TensorLayout result_layout(shape);
    types::Tensor result_tensor(base_desc, result_layout);

    auto& block = builder.add_block(root);
    auto& start_access = builder.add_access(block, "start");
    auto& end_access = builder.add_access(block, "end");
    auto& step_access = builder.add_access(block, "step");
    auto& out_access = builder.add_access(block, "result");
    auto& libnode = builder.add_library_node<math::tensor::ArangeNode>(block, DebugInfo(), shape);
    builder.add_computational_memlet(block, start_access, libnode, "_start", {}, base_desc);
    builder.add_computational_memlet(block, end_access, libnode, "_end", {}, base_desc);
    builder.add_computational_memlet(block, step_access, libnode, "_step", {}, base_desc);
    builder.add_computational_memlet(block, out_access, libnode, "_out", {}, result_tensor);

    ASSERT_NO_THROW(sdfg.validate());
    builder::StructuredSDFGBuilder new_builder("sdfg_2", FunctionType_CPU);
    new_builder.add_container("start", base_desc, true);
    new_builder.add_container("end", base_desc, true);
    new_builder.add_container("step", base_desc, true);
    new_builder.add_container("result", desc, true);
    deepcopy::StructuredSDFGDeepCopy deep_copy(new_builder, new_builder.subject().root(), root);
    deep_copy.copy();
    ASSERT_NO_THROW(new_builder.subject().validate());
}

TEST(ArangeNodeTest, expansion) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Int64);
    types::Pointer desc(base_desc);
    builder.add_container("start", base_desc, true);
    builder.add_container("end", base_desc, true);
    builder.add_container("step", base_desc, true);
    builder.add_container("result", desc, true);

    symbolic::MultiExpression shape = {symbolic::integer(10)};
    math::tensor::TensorLayout result_layout(shape);
    types::Tensor result_tensor(base_desc, result_layout);

    auto& block = builder.add_block(root);
    auto& start_access = builder.add_access(block, "start");
    auto& end_access = builder.add_access(block, "end");
    auto& step_access = builder.add_access(block, "step");
    auto& out_access = builder.add_access(block, "result");
    auto& libnode = builder.add_library_node<math::tensor::ArangeNode>(block, DebugInfo(), shape);
    builder.add_computational_memlet(block, start_access, libnode, "_start", {}, base_desc);
    builder.add_computational_memlet(block, end_access, libnode, "_end", {}, base_desc);
    builder.add_computational_memlet(block, step_access, libnode, "_step", {}, base_desc);
    builder.add_computational_memlet(block, out_access, libnode, "_out", {}, result_tensor);

    ASSERT_NO_THROW(sdfg.validate());
    dump_sdfg(sdfg, "0.before");

    analysis::AnalysisManager analysis_manager(sdfg);
    passes::LibraryNodeExpansionPass pass;
    ASSERT_TRUE(pass.run(builder, analysis_manager));

    ASSERT_NO_THROW(sdfg.validate());
    dump_sdfg(sdfg, "1.after");
}

TEST(ArangeNodeTest, expansion_2d) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Int64);
    types::Pointer desc(base_desc);
    builder.add_container("start", base_desc, true);
    builder.add_container("end", base_desc, true);
    builder.add_container("step", base_desc, true);
    builder.add_container("result", desc, true);

    symbolic::MultiExpression shape = {symbolic::integer(10), symbolic::integer(5)};
    math::tensor::TensorLayout result_layout(shape);
    types::Tensor result_tensor(base_desc, result_layout);

    auto& block = builder.add_block(root);
    auto& start_access = builder.add_access(block, "start");
    auto& end_access = builder.add_access(block, "end");
    auto& step_access = builder.add_access(block, "step");
    auto& out_access = builder.add_access(block, "result");
    auto& libnode = builder.add_library_node<math::tensor::ArangeNode>(block, DebugInfo(), shape);
    builder.add_computational_memlet(block, start_access, libnode, "_start", {}, base_desc);
    builder.add_computational_memlet(block, end_access, libnode, "_end", {}, base_desc);
    builder.add_computational_memlet(block, step_access, libnode, "_step", {}, base_desc);
    builder.add_computational_memlet(block, out_access, libnode, "_out", {}, result_tensor);

    ASSERT_NO_THROW(sdfg.validate());
    dump_sdfg(sdfg, "0.before");

    analysis::AnalysisManager analysis_manager(sdfg);
    passes::LibraryNodeExpansionPass pass;
    ASSERT_TRUE(pass.run(builder, analysis_manager));

    ASSERT_NO_THROW(sdfg.validate());
    dump_sdfg(sdfg, "1.after");
}

TEST(ArangeNodeTest, expansion_float_fma) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("start", base_desc, true);
    builder.add_container("end", base_desc, true);
    builder.add_container("step", base_desc, true);
    builder.add_container("result", desc, true);

    symbolic::MultiExpression shape = {symbolic::integer(10)};
    math::tensor::TensorLayout result_layout(shape);
    types::Tensor result_tensor(base_desc, result_layout);

    auto& block = builder.add_block(root);
    auto& start_access = builder.add_access(block, "start");
    auto& end_access = builder.add_access(block, "end");
    auto& step_access = builder.add_access(block, "step");
    auto& out_access = builder.add_access(block, "result");
    auto& libnode = builder.add_library_node<math::tensor::ArangeNode>(block, DebugInfo(), shape);
    builder.add_computational_memlet(block, start_access, libnode, "_start", {}, base_desc);
    builder.add_computational_memlet(block, end_access, libnode, "_end", {}, base_desc);
    builder.add_computational_memlet(block, step_access, libnode, "_step", {}, base_desc);
    builder.add_computational_memlet(block, out_access, libnode, "_out", {}, result_tensor);

    ASSERT_NO_THROW(sdfg.validate());
    dump_sdfg(sdfg, "0.before");

    analysis::AnalysisManager analysis_manager(sdfg);
    passes::LibraryNodeExpansionPass pass;
    ASSERT_TRUE(pass.run(builder, analysis_manager));

    ASSERT_NO_THROW(sdfg.validate());
    dump_sdfg(sdfg, "1.after");
}

TEST(ArangeNodeTest, expansion_assign) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Int64);
    types::Pointer desc(base_desc);
    builder.add_container("end", base_desc, true);
    builder.add_container("result", desc, true);

    symbolic::MultiExpression shape = {symbolic::integer(10)};
    math::tensor::TensorLayout result_layout(shape);
    types::Tensor result_tensor(base_desc, result_layout);

    auto& block = builder.add_block(root);
    auto& start_access = builder.add_constant(block, "0", base_desc);
    auto& end_access = builder.add_access(block, "end");
    auto& step_access = builder.add_constant(block, "1", base_desc);
    auto& out_access = builder.add_access(block, "result");
    auto& libnode = builder.add_library_node<math::tensor::ArangeNode>(block, DebugInfo(), shape);
    builder.add_computational_memlet(block, start_access, libnode, "_start", {}, base_desc);
    builder.add_computational_memlet(block, end_access, libnode, "_end", {}, base_desc);
    builder.add_computational_memlet(block, step_access, libnode, "_step", {}, base_desc);
    builder.add_computational_memlet(block, out_access, libnode, "_out", {}, result_tensor);

    ASSERT_NO_THROW(sdfg.validate());
    dump_sdfg(sdfg, "0.before");

    analysis::AnalysisManager analysis_manager(sdfg);
    passes::LibraryNodeExpansionPass pass;
    ASSERT_TRUE(pass.run(builder, analysis_manager));

    ASSERT_NO_THROW(sdfg.validate());
    dump_sdfg(sdfg, "1.after");
}

TEST(ArangeNodeTest, expansion_start_zero) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Int64);
    types::Pointer desc(base_desc);
    builder.add_container("end", base_desc, true);
    builder.add_container("step", base_desc, true);
    builder.add_container("result", desc, true);

    symbolic::MultiExpression shape = {symbolic::integer(10)};
    math::tensor::TensorLayout result_layout(shape);
    types::Tensor result_tensor(base_desc, result_layout);

    auto& block = builder.add_block(root);
    auto& start_access = builder.add_constant(block, "0", base_desc);
    auto& end_access = builder.add_access(block, "end");
    auto& step_access = builder.add_access(block, "step");
    auto& out_access = builder.add_access(block, "result");
    auto& libnode = builder.add_library_node<math::tensor::ArangeNode>(block, DebugInfo(), shape);
    builder.add_computational_memlet(block, start_access, libnode, "_start", {}, base_desc);
    builder.add_computational_memlet(block, end_access, libnode, "_end", {}, base_desc);
    builder.add_computational_memlet(block, step_access, libnode, "_step", {}, base_desc);
    builder.add_computational_memlet(block, out_access, libnode, "_out", {}, result_tensor);

    ASSERT_NO_THROW(sdfg.validate());
    dump_sdfg(sdfg, "0.before");

    analysis::AnalysisManager analysis_manager(sdfg);
    passes::LibraryNodeExpansionPass pass;
    ASSERT_TRUE(pass.run(builder, analysis_manager));

    ASSERT_NO_THROW(sdfg.validate());
    dump_sdfg(sdfg, "1.after");
}

TEST(ArangeNodeTest, expansion_step_one) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Int64);
    types::Pointer desc(base_desc);
    builder.add_container("start", base_desc, true);
    builder.add_container("end", base_desc, true);
    builder.add_container("result", desc, true);

    symbolic::MultiExpression shape = {symbolic::integer(10)};
    math::tensor::TensorLayout result_layout(shape);
    types::Tensor result_tensor(base_desc, result_layout);

    auto& block = builder.add_block(root);
    auto& start_access = builder.add_access(block, "start");
    auto& end_access = builder.add_access(block, "end");
    auto& step_access = builder.add_constant(block, "1", base_desc);
    auto& out_access = builder.add_access(block, "result");
    auto& libnode = builder.add_library_node<math::tensor::ArangeNode>(block, DebugInfo(), shape);
    builder.add_computational_memlet(block, start_access, libnode, "_start", {}, base_desc);
    builder.add_computational_memlet(block, end_access, libnode, "_end", {}, base_desc);
    builder.add_computational_memlet(block, step_access, libnode, "_step", {}, base_desc);
    builder.add_computational_memlet(block, out_access, libnode, "_out", {}, result_tensor);

    ASSERT_NO_THROW(sdfg.validate());
    dump_sdfg(sdfg, "0.before");

    analysis::AnalysisManager analysis_manager(sdfg);
    passes::LibraryNodeExpansionPass pass;
    ASSERT_TRUE(pass.run(builder, analysis_manager));

    ASSERT_NO_THROW(sdfg.validate());
    dump_sdfg(sdfg, "1.after");
}

TEST(ArangeNodeTest, serialization) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Int64);
    types::Pointer desc(base_desc);
    builder.add_container("start", base_desc, true);
    builder.add_container("end", base_desc, true);
    builder.add_container("step", base_desc, true);
    builder.add_container("result", desc, true);

    symbolic::MultiExpression shape = {symbolic::integer(10)};
    math::tensor::TensorLayout result_layout(shape);
    types::Tensor result_tensor(base_desc, result_layout);

    auto& block = builder.add_block(root);
    auto& start_access = builder.add_access(block, "start");
    auto& end_access = builder.add_access(block, "end");
    auto& step_access = builder.add_access(block, "step");
    auto& out_access = builder.add_access(block, "result");
    auto& libnode = builder.add_library_node<math::tensor::ArangeNode>(block, DebugInfo(), shape);
    builder.add_computational_memlet(block, start_access, libnode, "_start", {}, base_desc);
    builder.add_computational_memlet(block, end_access, libnode, "_end", {}, base_desc);
    builder.add_computational_memlet(block, step_access, libnode, "_step", {}, base_desc);
    builder.add_computational_memlet(block, out_access, libnode, "_out", {}, result_tensor);

    ASSERT_NO_THROW(sdfg.validate());

    serializer::JSONSerializer serializer;
    nlohmann::json j;
    ASSERT_NO_THROW(j = serializer.serialize(sdfg));

    std::unique_ptr<StructuredSDFG> new_sdfg;
    ASSERT_NO_THROW(new_sdfg = serializer.deserialize(j));
    ASSERT_NO_THROW(new_sdfg->validate());
}

TEST(ArangeNodeTest, validate_result_tensor) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Int64);
    types::Pointer desc(base_desc);
    builder.add_container("start", base_desc, true);
    builder.add_container("end", base_desc, true);
    builder.add_container("step", base_desc, true);
    builder.add_container("result", desc, true);

    symbolic::MultiExpression shape = {symbolic::integer(10)};

    auto& block = builder.add_block(root);
    auto& start_access = builder.add_access(block, "start");
    auto& end_access = builder.add_access(block, "end");
    auto& step_access = builder.add_access(block, "step");
    auto& out_access = builder.add_access(block, "result");
    auto& libnode = builder.add_library_node<math::tensor::ArangeNode>(block, DebugInfo(), shape);
    builder.add_computational_memlet(block, start_access, libnode, "_start", {}, base_desc);
    builder.add_computational_memlet(block, end_access, libnode, "_end", {}, base_desc);
    builder.add_computational_memlet(block, step_access, libnode, "_step", {}, base_desc);
    builder.add_computational_memlet(block, out_access, libnode, "_out", {}, desc);

    ASSERT_THROW(sdfg.validate(), InvalidSDFGException);
}

TEST(ArangeNodeTest, validate_start_scalar) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Int64);
    types::Pointer desc(base_desc);
    builder.add_container("start", desc, true);
    builder.add_container("end", base_desc, true);
    builder.add_container("step", base_desc, true);
    builder.add_container("result", desc, true);

    symbolic::MultiExpression shape = {symbolic::integer(10)};
    math::tensor::TensorLayout result_layout(shape);
    types::Tensor result_tensor(base_desc, result_layout);

    auto& block = builder.add_block(root);
    auto& start_access = builder.add_access(block, "start");
    auto& end_access = builder.add_access(block, "end");
    auto& step_access = builder.add_access(block, "step");
    auto& out_access = builder.add_access(block, "result");
    auto& libnode = builder.add_library_node<math::tensor::ArangeNode>(block, DebugInfo(), shape);
    builder.add_computational_memlet(block, start_access, libnode, "_start", {}, desc);
    builder.add_computational_memlet(block, end_access, libnode, "_end", {}, base_desc);
    builder.add_computational_memlet(block, step_access, libnode, "_step", {}, base_desc);
    builder.add_computational_memlet(block, out_access, libnode, "_out", {}, result_tensor);

    ASSERT_THROW(sdfg.validate(), InvalidSDFGException);
}

TEST(ArangeNodeTest, validate_end_scalar) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Int64);
    types::Pointer desc(base_desc);
    builder.add_container("start", base_desc, true);
    builder.add_container("end", desc, true);
    builder.add_container("step", base_desc, true);
    builder.add_container("result", desc, true);

    symbolic::MultiExpression shape = {symbolic::integer(10)};
    math::tensor::TensorLayout result_layout(shape);
    types::Tensor result_tensor(base_desc, result_layout);

    auto& block = builder.add_block(root);
    auto& start_access = builder.add_access(block, "start");
    auto& end_access = builder.add_access(block, "end");
    auto& step_access = builder.add_access(block, "step");
    auto& out_access = builder.add_access(block, "result");
    auto& libnode = builder.add_library_node<math::tensor::ArangeNode>(block, DebugInfo(), shape);
    builder.add_computational_memlet(block, start_access, libnode, "_start", {}, base_desc);
    builder.add_computational_memlet(block, end_access, libnode, "_end", {}, desc);
    builder.add_computational_memlet(block, step_access, libnode, "_step", {}, base_desc);
    builder.add_computational_memlet(block, out_access, libnode, "_out", {}, result_tensor);

    ASSERT_THROW(sdfg.validate(), InvalidSDFGException);
}

TEST(ArangeNodeTest, validate_step_scalar) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Int64);
    types::Pointer desc(base_desc);
    builder.add_container("start", base_desc, true);
    builder.add_container("end", base_desc, true);
    builder.add_container("step", desc, true);
    builder.add_container("result", desc, true);

    symbolic::MultiExpression shape = {symbolic::integer(10)};
    math::tensor::TensorLayout result_layout(shape);
    types::Tensor result_tensor(base_desc, result_layout);

    auto& block = builder.add_block(root);
    auto& start_access = builder.add_access(block, "start");
    auto& end_access = builder.add_access(block, "end");
    auto& step_access = builder.add_access(block, "step");
    auto& out_access = builder.add_access(block, "result");
    auto& libnode = builder.add_library_node<math::tensor::ArangeNode>(block, DebugInfo(), shape);
    builder.add_computational_memlet(block, start_access, libnode, "_start", {}, base_desc);
    builder.add_computational_memlet(block, end_access, libnode, "_end", {}, base_desc);
    builder.add_computational_memlet(block, step_access, libnode, "_step", {}, desc);
    builder.add_computational_memlet(block, out_access, libnode, "_out", {}, result_tensor);

    ASSERT_THROW(sdfg.validate(), InvalidSDFGException);
}
