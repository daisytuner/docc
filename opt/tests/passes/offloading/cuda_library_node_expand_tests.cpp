#include <gtest/gtest.h>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_nodes/math/tensor/conv_node.h"
#include "sdfg/targets/cuda/math/tensor/conv_expander.h"

using namespace sdfg;

// Test that CudaConvExpander successfully expands a valid 2D ConvNode with group==1
TEST(CudaConvExpanderTest, ExpandsValidConv2D_Group1) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();

    types::Scalar desc(types::PrimitiveType::Float);
    types::Pointer desc_ptr(desc);

    builder.add_container("input", desc_ptr);
    builder.add_container("weights", desc_ptr);
    builder.add_container("output", desc_ptr);

    auto& block = builder.add_block(sdfg.root());

    auto& input_node = builder.add_access(block, "input");
    auto& weights_node = builder.add_access(block, "weights");
    auto& output_node = builder.add_access(block, "output");

    std::vector<symbolic::Expression> kernel_shape = {symbolic::integer(3), symbolic::integer(3)};
    std::vector<symbolic::Expression> strides = {symbolic::integer(1), symbolic::integer(1)};
    std::vector<symbolic::Expression> pads = {
        symbolic::integer(0), symbolic::integer(0), symbolic::integer(0), symbolic::integer(0)
    };
    std::vector<symbolic::Expression> dilations = {symbolic::integer(1), symbolic::integer(1)};
    auto group = symbolic::integer(1);

    // X shape: [N, C_in, H, W] = [1, 1, 4, 4]
    std::vector<symbolic::Expression> shape = {
        symbolic::integer(1), symbolic::integer(1), symbolic::integer(4), symbolic::integer(4)
    };
    types::Tensor desc_tensor_input(desc, shape);
    types::Tensor desc_tensor_weights(
        desc, {symbolic::integer(1), symbolic::integer(1), symbolic::integer(3), symbolic::integer(3)}
    );
    types::Tensor
        desc_tensor_output(desc, {symbolic::integer(1), symbolic::integer(1), symbolic::integer(2), symbolic::integer(2)});

    auto& conv_node = static_cast<math::tensor::ConvNode&>(builder.add_library_node<math::tensor::ConvNode>(
        block, DebugInfo(), shape, kernel_shape, strides, pads, dilations, symbolic::one(), group
    ));

    builder.add_computational_memlet(block, input_node, conv_node, "X", {}, desc_tensor_input, block.debug_info());
    builder.add_computational_memlet(block, weights_node, conv_node, "W", {}, desc_tensor_weights, block.debug_info());
    builder.add_computational_memlet(block, output_node, conv_node, "Y", {}, desc_tensor_output, block.debug_info());

    EXPECT_NO_THROW(sdfg.validate());

    analysis::AnalysisManager analysis_manager(sdfg);
    offloading::CudaConvExpander expander(conv_node);
    bool expanded = expander.expand(builder, analysis_manager);

    EXPECT_TRUE(expanded);
}

// Test that CudaConvExpander declines when group != 1
TEST(CudaConvExpanderTest, DeclinesConv2D_GroupNotOne) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();

    types::Scalar desc(types::PrimitiveType::Float);
    types::Pointer desc_ptr(desc);

    builder.add_container("input", desc_ptr);
    builder.add_container("weights", desc_ptr);
    builder.add_container("output", desc_ptr);

    auto& block = builder.add_block(sdfg.root());

    auto& input_node = builder.add_access(block, "input");
    auto& weights_node = builder.add_access(block, "weights");
    auto& output_node = builder.add_access(block, "output");

    std::vector<symbolic::Expression> kernel_shape = {symbolic::integer(3), symbolic::integer(3)};
    std::vector<symbolic::Expression> strides = {symbolic::integer(1), symbolic::integer(1)};
    std::vector<symbolic::Expression> pads = {
        symbolic::integer(1), symbolic::integer(1), symbolic::integer(1), symbolic::integer(1)
    };
    std::vector<symbolic::Expression> dilations = {symbolic::integer(1), symbolic::integer(1)};
    auto group = symbolic::integer(2); // group != 1

    // X shape: [N, C_in, H, W] = [1, 4, 8, 8]
    std::vector<symbolic::Expression> shape = {
        symbolic::integer(1), symbolic::integer(4), symbolic::integer(8), symbolic::integer(8)
    };
    types::Tensor desc_tensor_input(desc, shape);
    types::Tensor desc_tensor_weights(
        desc, {symbolic::integer(4), symbolic::integer(2), symbolic::integer(3), symbolic::integer(3)}
    );
    types::Tensor
        desc_tensor_output(desc, {symbolic::integer(1), symbolic::integer(4), symbolic::integer(8), symbolic::integer(8)});

    auto& conv_node = static_cast<math::tensor::ConvNode&>(builder.add_library_node<math::tensor::ConvNode>(
        block, DebugInfo(), shape, kernel_shape, strides, pads, dilations, symbolic::integer(4), group, false
    ));

    builder.add_computational_memlet(block, input_node, conv_node, "X", {}, desc_tensor_input, block.debug_info());
    builder.add_computational_memlet(block, weights_node, conv_node, "W", {}, desc_tensor_weights, block.debug_info());
    builder.add_computational_memlet(block, output_node, conv_node, "Y", {}, desc_tensor_output, block.debug_info());

    EXPECT_NO_THROW(sdfg.validate());

    analysis::AnalysisManager analysis_manager(sdfg);
    offloading::CudaConvExpander expander(conv_node);
    bool expanded = expander.expand(builder, analysis_manager);

    // Should decline because group != 1
    EXPECT_FALSE(expanded);
}

// Test that CudaConvExpander declines when the dataflow graph has extra nodes
// (i.e., check_expandable fails due to unexpected graph structure)
TEST(CudaConvExpanderTest, DeclinesWhenNotExpandable_ExtraNodes) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();

    types::Scalar desc(types::PrimitiveType::Float);
    types::Pointer desc_ptr(desc);

    builder.add_container("input", desc_ptr);
    builder.add_container("weights", desc_ptr);
    builder.add_container("output", desc_ptr);
    builder.add_container("extra", desc_ptr);

    auto& block = builder.add_block(sdfg.root());

    auto& input_node = builder.add_access(block, "input");
    auto& weights_node = builder.add_access(block, "weights");
    auto& output_node = builder.add_access(block, "output");
    // Add an extra access node that makes the DFG have more nodes than expected
    auto& extra_node = builder.add_access(block, "extra");
    (void) extra_node;

    std::vector<symbolic::Expression> kernel_shape = {symbolic::integer(3), symbolic::integer(3)};
    std::vector<symbolic::Expression> strides = {symbolic::integer(1), symbolic::integer(1)};
    std::vector<symbolic::Expression> pads = {
        symbolic::integer(0), symbolic::integer(0), symbolic::integer(0), symbolic::integer(0)
    };
    std::vector<symbolic::Expression> dilations = {symbolic::integer(1), symbolic::integer(1)};
    auto group = symbolic::integer(1);

    std::vector<symbolic::Expression> shape = {
        symbolic::integer(1), symbolic::integer(1), symbolic::integer(4), symbolic::integer(4)
    };
    types::Tensor desc_tensor_input(desc, shape);
    types::Tensor desc_tensor_weights(
        desc, {symbolic::integer(1), symbolic::integer(1), symbolic::integer(3), symbolic::integer(3)}
    );
    types::Tensor
        desc_tensor_output(desc, {symbolic::integer(1), symbolic::integer(1), symbolic::integer(2), symbolic::integer(2)});

    auto& conv_node = static_cast<math::tensor::ConvNode&>(builder.add_library_node<math::tensor::ConvNode>(
        block, DebugInfo(), shape, kernel_shape, strides, pads, dilations, symbolic::one(), group, false
    ));

    builder.add_computational_memlet(block, input_node, conv_node, "X", {}, desc_tensor_input, block.debug_info());
    builder.add_computational_memlet(block, weights_node, conv_node, "W", {}, desc_tensor_weights, block.debug_info());
    builder.add_computational_memlet(block, output_node, conv_node, "Y", {}, desc_tensor_output, block.debug_info());

    analysis::AnalysisManager analysis_manager(sdfg);
    offloading::CudaConvExpander expander(conv_node);
    bool expanded = expander.expand(builder, analysis_manager);

    // Should decline because check_expandable fails (extra node in DFG)
    EXPECT_FALSE(expanded);
}

// Test that CudaConvExpander successfully expands a valid 1D ConvNode with group==1
TEST(CudaConvExpanderTest, ExpandsValidConv1D_Group1) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();

    types::Scalar desc(types::PrimitiveType::Float);
    types::Pointer desc_ptr(desc);

    builder.add_container("input", desc_ptr);
    builder.add_container("weights", desc_ptr);
    builder.add_container("output", desc_ptr);

    auto& block = builder.add_block(sdfg.root());

    auto& input_node = builder.add_access(block, "input");
    auto& weights_node = builder.add_access(block, "weights");
    auto& output_node = builder.add_access(block, "output");

    std::vector<symbolic::Expression> kernel_shape = {symbolic::integer(5)};
    std::vector<symbolic::Expression> strides = {symbolic::integer(1)};
    std::vector<symbolic::Expression> pads = {symbolic::integer(2), symbolic::integer(2)};
    std::vector<symbolic::Expression> dilations = {symbolic::integer(1)};
    auto group = symbolic::integer(1);

    // X shape: [N, C_in, L] = [1, 1, 10]
    std::vector<symbolic::Expression> shape = {symbolic::integer(1), symbolic::integer(1), symbolic::integer(10)};
    types::Tensor desc_tensor_input(desc, shape);
    types::Tensor desc_tensor_weights(desc, {symbolic::integer(1), symbolic::integer(1), symbolic::integer(5)});
    types::Tensor desc_tensor_output(desc, {symbolic::integer(1), symbolic::integer(1), symbolic::integer(10)});

    auto& conv_node = static_cast<math::tensor::ConvNode&>(builder.add_library_node<math::tensor::ConvNode>(
        block, DebugInfo(), shape, kernel_shape, strides, pads, dilations, symbolic::one(), group, false
    ));

    builder.add_computational_memlet(block, input_node, conv_node, "X", {}, desc_tensor_input, block.debug_info());
    builder.add_computational_memlet(block, weights_node, conv_node, "W", {}, desc_tensor_weights, block.debug_info());
    builder.add_computational_memlet(block, output_node, conv_node, "Y", {}, desc_tensor_output, block.debug_info());

    EXPECT_NO_THROW(sdfg.validate());

    analysis::AnalysisManager analysis_manager(sdfg);
    offloading::CudaConvExpander expander(conv_node);
    bool expanded = expander.expand(builder, analysis_manager);

    EXPECT_TRUE(expanded);
}
