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
    types::Scalar step_type(types::PrimitiveType::Int64);
    types::Pointer step_ptr(step_type);

    std::vector<symbolic::Expression> shape = {symbolic::integer(10)};
    math::tensor::TensorLayout out_layout(shape);
    types::Tensor out_type(types::PrimitiveType::Int64, out_layout);
    types::Pointer out_ptr(start_type);

    builder.add_container("start", start_ptr, true);
    builder.add_container("step", step_ptr, true);
    builder.add_container("out", out_ptr, true);

    auto& block = builder.add_block(root);
    auto& start_acc = builder.add_access(block, "start");
    auto& step_acc = builder.add_access(block, "step");
    auto& out_acc = builder.add_access(block, "out");

    auto& libnode = builder.add_library_node<math::tensor::ArangeNode>(block, DebugInfo(), shape);

    builder.add_computational_memlet(block, start_acc, libnode, "_start", {}, start_type);
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
    types::Scalar step_type(types::PrimitiveType::Int64);
    types::Pointer step_ptr(step_type);
    std::vector<symbolic::Expression> shape = {symbolic::integer(10)};
    math::tensor::TensorLayout out_layout(shape);
    types::Tensor out_type(types::PrimitiveType::Int64, out_layout);
    types::Pointer out_ptr(start_type);

    builder.add_container("start", start_ptr, true);
    builder.add_container("step", step_ptr, true);
    builder.add_container("out", out_ptr, true);

    auto& block = builder.add_block(root);
    auto& start_acc = builder.add_access(block, "start");
    auto& step_acc = builder.add_access(block, "step");
    auto& out_acc = builder.add_access(block, "out");

    auto& libnode = builder.add_library_node<math::tensor::ArangeNode>(block, DebugInfo(), shape);
    builder.add_computational_memlet(block, start_acc, libnode, "_start", {}, start_type);
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
