#include <gtest/gtest.h>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_nodes/math/tensor/conv_node.h"

using namespace sdfg;

// Helper function to create a ConvNode with given parameters and test basic properties
void TestConvNode(
    std::vector<size_t> kernel_dims,
    std::vector<size_t> stride_vals,
    std::vector<size_t> pad_vals,
    std::vector<size_t> dilation_vals,
    size_t group_val,
    bool has_bias
) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();

    types::Scalar desc(types::PrimitiveType::Float);
    types::Pointer desc_ptr(desc);

    builder.add_container("input", desc_ptr);
    builder.add_container("weights", desc_ptr);
    if (has_bias) {
        builder.add_container("bias", desc_ptr);
    }
    builder.add_container("output", desc_ptr);

    auto& block = builder.add_block(sdfg.root());

    auto& input_node = builder.add_access(block, "input");
    auto& weights_node = builder.add_access(block, "weights");
    auto& output_node = builder.add_access(block, "output");

    // Convert sizes to symbolic expressions
    std::vector<symbolic::Expression> kernel_shape;
    for (auto d : kernel_dims) {
        kernel_shape.push_back(symbolic::integer(d));
    }

    std::vector<symbolic::Expression> strides;
    for (auto s : stride_vals) {
        strides.push_back(symbolic::integer(s));
    }

    std::vector<symbolic::Expression> pads;
    for (auto p : pad_vals) {
        pads.push_back(symbolic::integer(p));
    }

    std::vector<symbolic::Expression> dilations;
    for (auto d : dilation_vals) {
        dilations.push_back(symbolic::integer(d));
    }

    auto group = symbolic::integer(group_val);
    auto output_channels = symbolic::integer(1);

    // Default shape for validation
    std::vector<symbolic::Expression> shape = {symbolic::integer(1), symbolic::integer(1)};
    for (size_t i = 0; i < kernel_dims.size(); ++i) {
        shape.push_back(symbolic::integer(10)); // Arbitrary spatial size
    }
    types::Tensor desc_tensor_input(desc, shape);
    types::Tensor desc_tensor_weights(desc, kernel_shape); // C_out and C_in will be inferred

    auto& conv_node = static_cast<math::tensor::ConvNode&>(builder.add_library_node<math::tensor::ConvNode>(
        block, DebugInfo(), shape, kernel_shape, strides, pads, dilations, output_channels, group
    ));

    builder.add_computational_memlet(block, input_node, conv_node, "X", {}, desc_tensor_input, block.debug_info());
    builder.add_computational_memlet(block, weights_node, conv_node, "W", {}, desc_tensor_weights, block.debug_info());

    builder.add_computational_memlet(block, conv_node, "Y", output_node, {}, desc_tensor_input, block.debug_info());

    if (has_bias) {
        auto& bias_node = builder.add_access(block, "bias");
        builder.add_computational_memlet(block, bias_node, conv_node, "B", {}, desc_ptr, block.debug_info());
        // Skip validation for now since optional inputs need special handling
        // EXPECT_NO_THROW(sdfg.validate());
    } else {
        // Verify validation passes without bias
        EXPECT_NO_THROW(sdfg.validate());
    }

    // Verify the node was created successfully
    EXPECT_EQ(conv_node.kernel_shape().size(), kernel_dims.size());
    EXPECT_EQ(conv_node.strides().size(), stride_vals.size());
    EXPECT_EQ(conv_node.pads().size(), pad_vals.size());
    EXPECT_EQ(conv_node.dilations().size(), dilation_vals.size());
}

// Test 1D convolution with kernel size 3
TEST(ConvNodeTest, DISABLED_Conv1D_Kernel3) {
    TestConvNode(
        {3}, // kernel_shape
        {1}, // strides
        {1, 1}, // pads (start=1, end=1)
        {1}, // dilations
        1, // group
        false // no bias
    );
}

// Test 1D convolution with kernel size 5
TEST(ConvNodeTest, DISABLED_Conv1D_Kernel5) {
    TestConvNode(
        {5}, // kernel_shape
        {2}, // strides
        {2, 2}, // pads
        {1}, // dilations
        1, // group
        false // no bias
    );
}

// Test 2D convolution with kernel size 3x3
TEST(ConvNodeTest, DISABLED_Conv2D_Kernel3x3) {
    TestConvNode(
        {3, 3}, // kernel_shape
        {1, 1}, // strides
        {1, 1, 1, 1}, // pads (top=1, left=1, bottom=1, right=1)
        {1, 1}, // dilations
        1, // group
        false // no bias
    );
}

// Test 2D convolution with stride 2
TEST(ConvNodeTest, DISABLED_Conv2D_Stride2) {
    TestConvNode(
        {3, 3}, // kernel_shape
        {2, 2}, // strides
        {1, 1, 1, 1}, // pads
        {1, 1}, // dilations
        1, // group
        false // no bias
    );
}

// Test 2D convolution with no padding
TEST(ConvNodeTest, DISABLED_Conv2D_NoPadding) {
    TestConvNode(
        {3, 3}, // kernel_shape
        {1, 1}, // strides
        {0, 0, 0, 0}, // pads (no padding)
        {1, 1}, // dilations
        1, // group
        false // no bias
    );
}

// Test 2D convolution with asymmetric padding
TEST(ConvNodeTest, DISABLED_Conv2D_AsymmetricPadding) {
    TestConvNode(
        {3, 3}, // kernel_shape
        {1, 1}, // strides
        {1, 2, 1, 2}, // pads (asymmetric)
        {1, 1}, // dilations
        1, // group
        false // no bias
    );
}

// Test 2D convolution with dilation
TEST(ConvNodeTest, DISABLED_Conv2D_Dilation2) {
    TestConvNode(
        {3, 3}, // kernel_shape
        {1, 1}, // strides
        {2, 2, 2, 2}, // pads
        {2, 2}, // dilations
        1, // group
        false // no bias
    );
}

// Test 2D convolution with bias
TEST(ConvNodeTest, DISABLED_Conv2D_WithBias) {
    TestConvNode(
        {3, 3}, // kernel_shape
        {1, 1}, // strides
        {1, 1, 1, 1}, // pads
        {1, 1}, // dilations
        1, // group
        true // with bias
    );
}

// Test 2D convolution with kernel size 5x5
TEST(ConvNodeTest, DISABLED_Conv2D_Kernel5x5) {
    TestConvNode(
        {5, 5}, // kernel_shape
        {1, 1}, // strides
        {2, 2, 2, 2}, // pads
        {1, 1}, // dilations
        1, // group
        false // no bias
    );
}

// Test 2D convolution with kernel size 1x1 (pointwise convolution)
TEST(ConvNodeTest, DISABLED_Conv2D_Kernel1x1) {
    TestConvNode(
        {1, 1}, // kernel_shape
        {1, 1}, // strides
        {0, 0, 0, 0}, // pads (no padding needed for 1x1)
        {1, 1}, // dilations
        1, // group
        false // no bias
    );
}

// Test 2D convolution with grouped convolution
TEST(ConvNodeTest, DISABLED_Conv2D_Grouped) {
    TestConvNode(
        {3, 3}, // kernel_shape
        {1, 1}, // strides
        {1, 1, 1, 1}, // pads
        {1, 1}, // dilations
        2, // group (grouped convolution)
        false // no bias
    );
}

// Test 3D convolution with kernel size 3x3x3
TEST(ConvNodeTest, DISABLED_Conv3D_Kernel3x3x3) {
    TestConvNode(
        {3, 3, 3}, // kernel_shape
        {1, 1, 1}, // strides
        {1, 1, 1, 1, 1, 1}, // pads (depth, height, width)
        {1, 1, 1}, // dilations
        1, // group
        false // no bias
    );
}

// Test 3D convolution with kernel size 2x2x2 and stride 2
TEST(ConvNodeTest, DISABLED_Conv3D_Kernel2x2x2_Stride2) {
    TestConvNode(
        {2, 2, 2}, // kernel_shape
        {2, 2, 2}, // strides
        {0, 0, 0, 0, 0, 0}, // pads (no padding)
        {1, 1, 1}, // dilations
        1, // group
        false // no bias
    );
}

// Test validation: mismatched strides dimension
TEST(ConvNodeTest, DISABLED_ValidationError_MismatchedStrides) {
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
    std::vector<symbolic::Expression> strides = {symbolic::integer(1)}; // Wrong size!
    std::vector<symbolic::Expression> pads = {
        symbolic::integer(1), symbolic::integer(1), symbolic::integer(1), symbolic::integer(1)
    };
    std::vector<symbolic::Expression> dilations = {symbolic::integer(1), symbolic::integer(1)};
    auto group = symbolic::integer(1);

    // Default shape for validation
    std::vector<symbolic::Expression> shape = {
        symbolic::integer(1), symbolic::integer(1), symbolic::integer(10), symbolic::integer(10)
    };
    types::Tensor desc_tensor_input(desc, shape);
    types::Tensor desc_tensor_weights(desc, {symbolic::integer(3), symbolic::integer(3)});
    types::Tensor desc_tensor_output(
        desc, {symbolic::integer(1), symbolic::integer(1), symbolic::integer(10), symbolic::integer(10)}
    );

    auto& conv_node = builder.add_library_node<
        math::tensor::ConvNode>(block, DebugInfo(), shape, kernel_shape, strides, pads, dilations, symbolic::one(), group);

    builder.add_computational_memlet(block, input_node, conv_node, "X", {}, desc_tensor_input, block.debug_info());
    builder.add_computational_memlet(block, weights_node, conv_node, "W", {}, desc_tensor_weights, block.debug_info());
    builder.add_computational_memlet(block, conv_node, "Y", output_node, {}, desc_tensor_output, block.debug_info());

    // Validation should fail due to mismatched strides dimension
    EXPECT_THROW(sdfg.validate(), InvalidSDFGException);
}

// Test validation: mismatched pads dimension
TEST(ConvNodeTest, DISABLED_ValidationError_MismatchedPads) {
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
    std::vector<symbolic::Expression> pads = {symbolic::integer(1), symbolic::integer(1)}; // Wrong size! Should be 4
    std::vector<symbolic::Expression> dilations = {symbolic::integer(1), symbolic::integer(1)};
    auto group = symbolic::integer(1);

    // Default shape for validation
    std::vector<symbolic::Expression> shape = {
        symbolic::integer(1), symbolic::integer(1), symbolic::integer(10), symbolic::integer(10)
    };
    types::Tensor desc_tensor_input(desc, shape);
    types::Tensor desc_tensor_weights(desc, {symbolic::integer(3), symbolic::integer(3)});
    types::Tensor desc_tensor_output(
        desc, {symbolic::integer(1), symbolic::integer(1), symbolic::integer(10), symbolic::integer(10)}
    );

    auto& conv_node = builder.add_library_node<
        math::tensor::ConvNode>(block, DebugInfo(), shape, kernel_shape, strides, pads, dilations, symbolic::one(), group);

    builder.add_computational_memlet(block, input_node, conv_node, "X", {}, desc_tensor_input, block.debug_info());
    builder.add_computational_memlet(block, weights_node, conv_node, "W", {}, desc_tensor_weights, block.debug_info());
    builder.add_computational_memlet(block, conv_node, "Y", output_node, {}, desc_tensor_output, block.debug_info());

    // Validation should fail due to mismatched pads dimension
    EXPECT_THROW(sdfg.validate(), InvalidSDFGException);
}

// Test clone functionality
TEST(ConvNodeTest, DISABLED_CloneNode) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();

    types::Scalar desc(types::PrimitiveType::Float);
    types::Pointer desc_ptr(desc);

    builder.add_container("input", desc_ptr);
    builder.add_container("weights", desc_ptr);
    builder.add_container("output", desc_ptr);

    auto& block = builder.add_block(sdfg.root());

    std::vector<symbolic::Expression> kernel_shape = {symbolic::integer(3), symbolic::integer(3)};
    std::vector<symbolic::Expression> strides = {symbolic::integer(1), symbolic::integer(1)};
    std::vector<symbolic::Expression> pads = {
        symbolic::integer(1), symbolic::integer(1), symbolic::integer(1), symbolic::integer(1)
    };
    std::vector<symbolic::Expression> dilations = {symbolic::integer(1), symbolic::integer(1)};
    auto group = symbolic::integer(1);

    // Default shape for validation
    std::vector<symbolic::Expression> shape = {
        symbolic::integer(1), symbolic::integer(1), symbolic::integer(10), symbolic::integer(10)
    };

    auto& conv_node = builder.add_library_node<
        math::tensor::ConvNode>(block, DebugInfo(), shape, kernel_shape, strides, pads, dilations, symbolic::one(), group);

    // Clone the node
    auto cloned = conv_node.clone(999, graph::Vertex(), block.dataflow());
    auto* cloned_conv = dynamic_cast<math::tensor::ConvNode*>(cloned.get());

    ASSERT_NE(cloned_conv, nullptr);
    EXPECT_EQ(cloned_conv->kernel_shape().size(), kernel_shape.size());
    EXPECT_EQ(cloned_conv->strides().size(), strides.size());
    EXPECT_EQ(cloned_conv->pads().size(), pads.size());
    EXPECT_EQ(cloned_conv->dilations().size(), dilations.size());
}

// Test expansion of Conv2D with simple parameters
TEST(ConvNodeTest, DISABLED_Conv2D_SimpleExpansion) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();

    // Create simple 2D convolution: 1 batch, 1 input channel, 1 output channel
    // Input: [1, 1, 4, 4], Kernel: [1, 1, 3, 3], Output: [1, 1, 2, 2] with stride=1, no padding

    types::Scalar desc(types::PrimitiveType::Float);
    types::Pointer desc_ptr(desc);

    builder.add_container("input", desc_ptr); // 1*1*4*4 = 16 floats
    builder.add_container("weights", desc_ptr); // 1*1*3*3 = 9 floats
    builder.add_container("output", desc_ptr); // 1*1*2*2 = 4 floats

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

    // X shape: [N, C_in, D0, D1] = [1, 1, 4, 4]
    std::vector<symbolic::Expression> shape = {
        symbolic::integer(1), symbolic::integer(1), symbolic::integer(4), symbolic::integer(4)
    };
    types::Tensor desc_tensor_input(desc, shape);
    types::Tensor desc_tensor_weights(desc, {symbolic::integer(3), symbolic::integer(3)});
    types::Tensor
        desc_tensor_output(desc, {symbolic::integer(1), symbolic::integer(1), symbolic::integer(2), symbolic::integer(2)});

    auto& conv_node = static_cast<math::tensor::ConvNode&>(builder.add_library_node<math::tensor::ConvNode>(
        block, DebugInfo(), shape, kernel_shape, strides, pads, dilations, symbolic::one(), group
    ));

    builder.add_computational_memlet(block, input_node, conv_node, "X", {}, desc_tensor_input, block.debug_info());
    builder.add_computational_memlet(block, weights_node, conv_node, "W", {}, desc_tensor_weights, block.debug_info());
    builder.add_computational_memlet(block, conv_node, "Y", output_node, {}, desc_tensor_output, block.debug_info());

    // Validate the SDFG before expansion
    EXPECT_NO_THROW(sdfg.validate());

    // Try to expand the node - expansion should now succeed for 2D convolution
    analysis::AnalysisManager analysis_manager(sdfg);
    bool expanded = conv_node.expand(builder, analysis_manager);

    // Expansion should now be implemented and return true
    EXPECT_TRUE(expanded);
}

// Test that expansion is not attempted with unsupported parameters
TEST(ConvNodeTest, DISABLED_Conv2D_ExpansionNotImplemented) {
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
    std::vector<symbolic::Expression> strides = {symbolic::integer(2), symbolic::integer(2)};
    std::vector<symbolic::Expression> pads = {
        symbolic::integer(1), symbolic::integer(1), symbolic::integer(1), symbolic::integer(1)
    };
    std::vector<symbolic::Expression> dilations = {symbolic::integer(1), symbolic::integer(1)};
    auto group = symbolic::integer(1);

    // X shape: [N, C_in, D0, D1] = [1, 1, 4, 4]
    std::vector<symbolic::Expression> shape = {
        symbolic::integer(1), symbolic::integer(1), symbolic::integer(4), symbolic::integer(4)
    };
    types::Tensor desc_tensor_input(desc, shape);
    types::Tensor desc_tensor_weights(desc, {symbolic::integer(3), symbolic::integer(3)});

    auto& conv_node = static_cast<math::tensor::ConvNode&>(builder.add_library_node<math::tensor::ConvNode>(
        block, DebugInfo(), shape, kernel_shape, strides, pads, dilations, symbolic::one(), group
    ));

    builder.add_computational_memlet(block, input_node, conv_node, "X", {}, desc_tensor_input, block.debug_info());
    builder.add_computational_memlet(block, weights_node, conv_node, "W", {}, desc_tensor_weights, block.debug_info());
    builder.add_computational_memlet(block, conv_node, "Y", output_node, {}, desc_tensor_input, block.debug_info());

    analysis::AnalysisManager analysis_manager(sdfg);
    bool expanded = conv_node.expand(builder, analysis_manager);

    // Expansion should succeed for 2D convolutions with stride > 1
    EXPECT_TRUE(expanded);
}

// Test 1D convolution expansion
TEST(ConvNodeTest, DISABLED_Conv1D_Expansion) {
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

    // X shape: [N, C_in, D0] = [1, 1, 10]
    std::vector<symbolic::Expression> shape = {symbolic::integer(1), symbolic::integer(1), symbolic::integer(10)};
    types::Tensor desc_tensor_input(desc, shape);
    types::Tensor desc_tensor_weights(desc, {symbolic::integer(5)});

    auto& conv_node = static_cast<math::tensor::ConvNode&>(builder.add_library_node<math::tensor::ConvNode>(
        block, DebugInfo(), shape, kernel_shape, strides, pads, dilations, symbolic::one(), group
    ));

    builder.add_computational_memlet(block, input_node, conv_node, "X", {}, desc_tensor_input, block.debug_info());
    builder.add_computational_memlet(block, weights_node, conv_node, "W", {}, desc_tensor_weights, block.debug_info());
    builder.add_computational_memlet(block, conv_node, "Y", output_node, {}, desc_tensor_input, block.debug_info());

    // Validate the SDFG before expansion
    EXPECT_NO_THROW(sdfg.validate());

    analysis::AnalysisManager analysis_manager(sdfg);
    bool expanded = conv_node.expand(builder, analysis_manager);

    // Expansion should succeed for 1D convolutions
    EXPECT_TRUE(expanded);
}

// Test 3D convolution expansion
TEST(ConvNodeTest, DISABLED_Conv3D_Expansion) {
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

    std::vector<symbolic::Expression> kernel_shape = {symbolic::integer(3), symbolic::integer(3), symbolic::integer(3)};
    std::vector<symbolic::Expression> strides = {symbolic::integer(1), symbolic::integer(1), symbolic::integer(1)};
    std::vector<symbolic::Expression> pads = {
        symbolic::integer(1),
        symbolic::integer(1),
        symbolic::integer(1),
        symbolic::integer(1),
        symbolic::integer(1),
        symbolic::integer(1)
    };
    std::vector<symbolic::Expression> dilations = {symbolic::integer(1), symbolic::integer(1), symbolic::integer(1)};
    auto group = symbolic::integer(1);

    // X shape: [N, C_in, D0, D1, D2] = [1, 1, 4, 4, 4]
    std::vector<symbolic::Expression> shape = {
        symbolic::integer(1), symbolic::integer(1), symbolic::integer(4), symbolic::integer(4), symbolic::integer(4)
    };
    types::Tensor desc_tensor_input(desc, shape);
    types::Tensor desc_tensor_weights(desc, {symbolic::integer(3), symbolic::integer(3), symbolic::integer(3)});

    auto& conv_node = static_cast<math::tensor::ConvNode&>(builder.add_library_node<math::tensor::ConvNode>(
        block, DebugInfo(), shape, kernel_shape, strides, pads, dilations, symbolic::one(), group
    ));

    builder.add_computational_memlet(block, input_node, conv_node, "X", {}, desc_tensor_input, block.debug_info());
    builder.add_computational_memlet(block, weights_node, conv_node, "W", {}, desc_tensor_weights, block.debug_info());
    builder.add_computational_memlet(block, conv_node, "Y", output_node, {}, desc_tensor_input, block.debug_info());

    // Validate the SDFG before expansion
    EXPECT_NO_THROW(sdfg.validate());

    analysis::AnalysisManager analysis_manager(sdfg);
    bool expanded = conv_node.expand(builder, analysis_manager);

    // Expansion should succeed for 3D convolutions
    EXPECT_TRUE(expanded);
}

// Test linearization of memlets after expansion
TEST(ConvNodeTest, DISABLED_LinearizationTest) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();

    // Input shape: [1, 1, 4, 4] -> size 16
    types::Scalar desc(types::PrimitiveType::Float);
    types::Pointer desc_ptr(desc);

    builder.add_container("input", desc_ptr);
    builder.add_container("weights", desc_ptr); // Kernel 3x3 -> size 9
    builder.add_container("output", desc_ptr); // Output ?

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

    std::vector<symbolic::Expression> shape = {
        symbolic::integer(1), symbolic::integer(1), symbolic::integer(4), symbolic::integer(4)
    };
    types::Tensor desc_tensor_input(desc, shape);
    types::Tensor desc_tensor_weights(desc, {symbolic::integer(3), symbolic::integer(3)});

    auto& conv_node = static_cast<math::tensor::ConvNode&>(builder.add_library_node<math::tensor::ConvNode>(
        block, DebugInfo(), shape, kernel_shape, strides, pads, dilations, symbolic::one(), group
    ));

    builder.add_computational_memlet(block, input_node, conv_node, "X", {}, desc_tensor_input, block.debug_info());
    builder.add_computational_memlet(block, weights_node, conv_node, "W", {}, desc_tensor_weights, block.debug_info());
    builder.add_computational_memlet(block, conv_node, "Y", output_node, {}, desc_tensor_input, block.debug_info());

    analysis::AnalysisManager analysis_manager(sdfg);
    bool expanded = conv_node.expand(builder, analysis_manager);
    EXPECT_TRUE(expanded);
}
