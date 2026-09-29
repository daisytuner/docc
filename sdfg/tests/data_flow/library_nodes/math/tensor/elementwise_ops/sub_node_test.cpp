#include "sdfg/data_flow/library_nodes/math/tensor/elementwise_ops/sub_node.h"

#include <gtest/gtest.h>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_nodes/math/tensor/tensor_layout.h"
#include "sdfg/deepcopy/structured_sdfg_deep_copy.h"
#include "sdfg/element.h"
#include "sdfg/function.h"
#include "sdfg/passes/expansion/library_node_expansion_pass.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/pointer.h"
#include "sdfg/types/scalar.h"
#include "sdfg/types/type.h"
#include "sdfg_debug_dump.h"

using namespace sdfg;

builder::StructuredSDFGBuilder build_sub_node(types::PrimitiveType prim) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& root = builder.subject().root();

    types::Scalar base_desc(prim);
    types::Pointer desc(base_desc);
    builder.add_container("a", desc, true);
    builder.add_container("b", desc, true);
    builder.add_container("c", desc, true);

    symbolic::MultiExpression shape = {symbolic::integer(20), symbolic::integer(30)};
    math::tensor::TensorLayout a_layout(shape);
    types::Tensor a_tensor(base_desc, a_layout);
    math::tensor::TensorLayout b_layout(shape);
    types::Tensor b_tensor(base_desc, b_layout);
    math::tensor::TensorLayout c_layout(shape);
    types::Tensor c_tensor(base_desc, c_layout);

    auto& block = builder.add_block(root);
    auto& a_access = builder.add_access(block, "a");
    auto& b_access = builder.add_access(block, "b");
    auto& c_access = builder.add_access(block, "c");
    auto& libnode = builder.add_library_node<math::tensor::SubNode>(block, DebugInfo(), shape);
    builder.add_computational_memlet(block, a_access, libnode, "A", {}, a_tensor);
    builder.add_computational_memlet(block, b_access, libnode, "B", {}, b_tensor);
    builder.add_computational_memlet(block, c_access, libnode, "C", {}, c_tensor);

    return builder;
}

TEST(SubNodeTest, expansion_float) {
    auto builder = build_sub_node(types::PrimitiveType::Float);
    auto& sdfg = builder.subject();

    ASSERT_NO_THROW(sdfg.validate());
    dump_sdfg(sdfg, "0.before");

    analysis::AnalysisManager analysis_manager(sdfg);
    passes::LibraryNodeExpansionPass expansion;
    ASSERT_TRUE(expansion.run(builder, analysis_manager));

    ASSERT_NO_THROW(sdfg.validate());
    dump_sdfg(sdfg, "1.after");
}

TEST(SubNodeTest, expansion_int) {
    auto builder = build_sub_node(types::PrimitiveType::Int32);
    auto& sdfg = builder.subject();

    ASSERT_NO_THROW(sdfg.validate());
    dump_sdfg(sdfg, "0.before");

    analysis::AnalysisManager analysis_manager(sdfg);
    passes::LibraryNodeExpansionPass expansion;
    ASSERT_TRUE(expansion.run(builder, analysis_manager));

    ASSERT_NO_THROW(sdfg.validate());
    dump_sdfg(sdfg, "1.after");
}

TEST(SubNodeTest, cloning) {
    auto builder = build_sub_node(types::PrimitiveType::Float);
    auto& sdfg = builder.subject();

    ASSERT_NO_THROW(sdfg.validate());
    builder::StructuredSDFGBuilder new_builder("sdfg_2", FunctionType_CPU);
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    new_builder.add_container("a", desc, true);
    new_builder.add_container("b", desc, true);
    new_builder.add_container("c", desc, true);
    deepcopy::StructuredSDFGDeepCopy deep_copy(new_builder, new_builder.subject().root(), sdfg.root());
    deep_copy.copy();
    ASSERT_NO_THROW(new_builder.subject().validate());
}
