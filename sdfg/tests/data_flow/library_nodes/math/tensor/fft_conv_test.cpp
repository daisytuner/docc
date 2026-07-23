#include <gtest/gtest.h>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_nodes/math/tensor/fft_conv_node.h"
#include "sdfg/serializer/json_serializer.h"
#include "sdfg/structured_control_flow/block.h"

using namespace sdfg;

namespace {

// Build an SDFG with a single FFTConvNode wired to Y (output ptr), X, W (and optional B).
math::tensor::FFTConvNode& build_fft_conv_sdfg(builder::StructuredSDFGBuilder& builder, bool with_bias) {
    auto& sdfg = builder.subject();

    types::Scalar f_scalar(types::PrimitiveType::Float);
    types::Pointer f_ptr(f_scalar);
    builder.add_container("Y", f_ptr, true);
    builder.add_container("X", f_ptr, true);
    builder.add_container("W", f_ptr, true);
    if (with_bias) {
        builder.add_container("B", f_ptr, true);
    }

    // Shape [N, C, H, W] = [2, 3, 8, 8], kernel [3, 3], pads [1, 1, 1, 1].
    std::vector<symbolic::Expression> shape = {
        symbolic::integer(2), symbolic::integer(3), symbolic::integer(8), symbolic::integer(8)
    };
    std::vector<symbolic::Expression> kernel_shape = {symbolic::integer(3), symbolic::integer(3)};
    std::vector<symbolic::Expression> pads = {
        symbolic::integer(1), symbolic::integer(1), symbolic::integer(1), symbolic::integer(1)
    };

    auto& block = builder.add_block(sdfg.root());
    auto& y_node = builder.add_access(block, "Y");
    auto& x_node = builder.add_access(block, "X");
    auto& w_node = builder.add_access(block, "W");

    auto& node = static_cast<math::tensor::FFTConvNode&>(builder.add_library_node<math::tensor::FFTConvNode>(
        block,
        DebugInfo(),
        data_flow::ImplementationType_NONE,
        shape,
        kernel_shape,
        pads,
        types::PrimitiveType::Float,
        with_bias
    ));

    builder.add_computational_memlet(block, y_node, node, "Y", {}, f_ptr, block.debug_info());
    builder.add_computational_memlet(block, x_node, node, "X", {}, f_ptr, block.debug_info());
    builder.add_computational_memlet(block, w_node, node, "W", {}, f_ptr, block.debug_info());
    if (with_bias) {
        auto& b_node = builder.add_access(block, "B");
        builder.add_computational_memlet(block, b_node, node, "B", {}, f_ptr, block.debug_info());
    }

    return node;
}

} // namespace

TEST(FFTConvNodeTest, Accessors) {
    builder::StructuredSDFGBuilder builder("fftconv", FunctionType_CPU);
    auto& node = build_fft_conv_sdfg(builder, /*with_bias=*/true);

    EXPECT_EQ(node.code().value(), math::tensor::LibraryNodeType_FFTConv.value());
    EXPECT_EQ(node.shape().size(), 4u);
    EXPECT_EQ(node.kernel_shape().size(), 2u);
    EXPECT_EQ(node.pads().size(), 4u);
    EXPECT_EQ(node.real_primitive(), types::PrimitiveType::Float);
    EXPECT_EQ(node.complex_primitive(), types::PrimitiveType::CFloat);
    EXPECT_TRUE(node.with_bias());
}

TEST(FFTConvNodeTest, SerializationRoundTripWithBias) {
    builder::StructuredSDFGBuilder builder("fftconv_ser", FunctionType_CPU);
    build_fft_conv_sdfg(builder, /*with_bias=*/true);
    auto& sdfg = builder.subject();

    serializer::JSONSerializer serializer;
    auto j = serializer.serialize(sdfg);
    auto deserialized = serializer.deserialize(j);
    ASSERT_NE(deserialized, nullptr);

    auto& child = deserialized->root().at(0).first;
    auto& block = dynamic_cast<structured_control_flow::Block&>(child);
    const math::tensor::FFTConvNode* conv = nullptr;
    for (auto& node : block.dataflow().nodes()) {
        if (auto* f = dynamic_cast<const math::tensor::FFTConvNode*>(&node)) {
            conv = f;
        }
    }
    ASSERT_NE(conv, nullptr);
    EXPECT_EQ(conv->code().value(), math::tensor::LibraryNodeType_FFTConv.value());
    EXPECT_EQ(conv->shape().size(), 4u);
    EXPECT_EQ(conv->shape()[0]->__str__(), "2");
    EXPECT_EQ(conv->shape()[3]->__str__(), "8");
    EXPECT_EQ(conv->kernel_shape()[0]->__str__(), "3");
    EXPECT_EQ(conv->pads()[0]->__str__(), "1");
    EXPECT_EQ(conv->real_primitive(), types::PrimitiveType::Float);
    EXPECT_TRUE(conv->with_bias());
}

TEST(FFTConvNodeTest, SerializationRoundTripNoBias) {
    builder::StructuredSDFGBuilder builder("fftconv_ser_nb", FunctionType_CPU);
    build_fft_conv_sdfg(builder, /*with_bias=*/false);
    auto& sdfg = builder.subject();

    serializer::JSONSerializer serializer;
    auto j = serializer.serialize(sdfg);
    auto deserialized = serializer.deserialize(j);
    ASSERT_NE(deserialized, nullptr);

    auto& child = deserialized->root().at(0).first;
    auto& block = dynamic_cast<structured_control_flow::Block&>(child);
    const math::tensor::FFTConvNode* conv = nullptr;
    for (auto& node : block.dataflow().nodes()) {
        if (auto* f = dynamic_cast<const math::tensor::FFTConvNode*>(&node)) {
            conv = f;
        }
    }
    ASSERT_NE(conv, nullptr);
    EXPECT_FALSE(conv->with_bias());
}
