#include "sdfg/passes/schedules/expansion_pass.h"

#include <gtest/gtest.h>

#include "sdfg/data_flow/library_nodes/math/math.h"

using namespace sdfg;

TEST(ExpansionPassTest, MeanNode_2D) {
    builder::StructuredSDFGBuilder builder("sdfg_sum", FunctionType_CPU);

    auto& sdfg = builder.subject();

    types::Scalar desc(types::PrimitiveType::Double);
    types::Pointer desc_ptr(desc);

    types::Pointer opaque_desc;
    builder.add_container("a", opaque_desc);
    builder.add_container("b", opaque_desc);

    auto& block = builder.add_block(sdfg.root());

    auto& a_node = builder.add_access(block, "a");
    auto& b_node = builder.add_access(block, "b");

    std::vector<symbolic::Expression> shape = {symbolic::integer(32), symbolic::integer(16)};
    std::vector<int64_t> axes = {-1};
    bool keepdims = false;
    types::Tensor tensor_desc_input(types::PrimitiveType::Double, shape);
    types::Tensor tensor_desc_output(types::PrimitiveType::Double, {symbolic::integer(32)});

    auto& mean_node =
        static_cast<math::tensor::MeanNode&>(builder.add_library_node<
                                             math::tensor::MeanNode>(block, DebugInfo(), shape, axes, keepdims));

    builder.add_computational_memlet(block, a_node, mean_node, "X", {}, tensor_desc_input, block.debug_info());
    builder.add_computational_memlet(block, mean_node, "Y", b_node, {}, tensor_desc_output, block.debug_info());

    // Check inputs and outputs
    EXPECT_EQ(mean_node.inputs().size(), 1);
    EXPECT_EQ(mean_node.inputs()[0], "X");
    EXPECT_EQ(mean_node.outputs().size(), 1);
    EXPECT_EQ(mean_node.outputs()[0], "Y");

    EXPECT_EQ(block.dataflow().nodes().size(), 3);

    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::ExpansionPass expansion_pass;
    EXPECT_TRUE(expansion_pass.run(builder, analysis_manager));
}

TEST(ExpansionPassTest, StdNode_1D) {
    builder::StructuredSDFGBuilder builder("sdfg_std", FunctionType_CPU);

    auto& sdfg = builder.subject();

    types::Scalar desc(types::PrimitiveType::Double);
    types::Pointer desc_ptr(desc);

    types::Pointer opaque_desc;
    builder.add_container("a", opaque_desc);
    builder.add_container("b", desc);

    auto& block = builder.add_block(sdfg.root());

    auto& a_node = builder.add_access(block, "a");
    auto& b_node = builder.add_access(block, "b");

    std::vector<symbolic::Expression> shape = {symbolic::integer(32)};
    std::vector<int64_t> axes = {0};
    bool keepdims = false;
    types::Tensor tensor_desc_input(types::PrimitiveType::Double, shape);
    types::Tensor tensor_desc_output(types::PrimitiveType::Double, {});

    auto& std_node =
        static_cast<math::tensor::StdNode&>(builder.add_library_node<
                                            math::tensor::StdNode>(block, DebugInfo(), shape, axes, keepdims));

    builder.add_computational_memlet(block, a_node, std_node, "X", {}, tensor_desc_input, block.debug_info());
    builder.add_computational_memlet(block, std_node, "Y", b_node, {}, tensor_desc_output, block.debug_info());

    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::ExpansionPass expansion_pass;
    EXPECT_TRUE(expansion_pass.run(builder, analysis_manager));
}
