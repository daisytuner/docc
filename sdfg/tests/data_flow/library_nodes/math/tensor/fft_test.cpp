#include <gtest/gtest.h>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_nodes/math/tensor/fft_node.h"
#include "sdfg/serializer/json_serializer.h"
#include "sdfg/structured_control_flow/block.h"

using namespace sdfg;

namespace {

// Build an SDFG containing a single FFT or IFFT node wired to X (input) and Y (output).
math::tensor::FFTNodeBase& build_fft_sdfg(
    builder::StructuredSDFGBuilder& builder,
    bool inverse,
    const std::vector<symbolic::Expression>& shape,
    symbolic::Expression batch,
    types::PrimitiveType precision
) {
    auto& sdfg = builder.subject();

    types::PrimitiveType cplx = precision == types::PrimitiveType::Double ? types::PrimitiveType::CDouble
                                                                          : types::PrimitiveType::CFloat;
    types::Pointer x_ptr(types::Scalar(inverse ? cplx : precision));
    types::Pointer y_ptr(types::Scalar(inverse ? precision : cplx));

    builder.add_container("X", x_ptr, true);
    builder.add_container("Y", y_ptr, true);

    auto& block = builder.add_block(sdfg.root());
    auto& x_node = builder.add_access(block, "X");
    auto& y_node = builder.add_access(block, "Y");

    data_flow::LibraryNode* node = nullptr;
    if (inverse) {
        node = &builder.add_library_node<
            math::tensor::IFFTNode>(block, DebugInfo(), data_flow::ImplementationType_NONE, shape, batch, precision);
    } else {
        node = &builder.add_library_node<
            math::tensor::FFTNode>(block, DebugInfo(), data_flow::ImplementationType_NONE, shape, batch, precision);
    }

    builder.add_computational_memlet(block, y_node, *node, "__Y", {}, y_ptr, block.debug_info());
    builder.add_computational_memlet(block, x_node, *node, "__X", {}, x_ptr, block.debug_info());

    return static_cast<math::tensor::FFTNodeBase&>(*node);
}

} // namespace

TEST(FFTNodeTest, ForwardHermitianExtents) {
    builder::StructuredSDFGBuilder builder("fft", FunctionType_CPU);
    std::vector<symbolic::Expression> shape = {symbolic::integer(8), symbolic::integer(8)};
    auto& node = build_fft_sdfg(builder, /*inverse=*/false, shape, symbolic::integer(4), types::PrimitiveType::Float);

    EXPECT_EQ(node.direction(), math::tensor::FFTDirection::Forward);
    EXPECT_EQ(node.rank(), 2u);
    EXPECT_EQ(node.real_primitive(), types::PrimitiveType::Float);
    EXPECT_EQ(node.complex_primitive(), types::PrimitiveType::CFloat);

    // Hermitian: last dim 8 -> 8/2 + 1 = 5.
    EXPECT_EQ(node.complex_last_dim()->__str__(), "5");
    // real_extent = 4 * 8 * 8 = 256.
    EXPECT_EQ(node.real_extent()->__str__(), "256");
    // complex_extent = 4 * 8 * 5 = 160.
    EXPECT_EQ(node.complex_extent()->__str__(), "160");

    EXPECT_NO_THROW(builder.subject().validate());
}

TEST(FFTNodeTest, InverseComplexPrimitiveDouble) {
    builder::StructuredSDFGBuilder builder("ifft", FunctionType_CPU);
    std::vector<symbolic::Expression> shape = {symbolic::integer(16), symbolic::integer(16)};
    auto& node = build_fft_sdfg(builder, /*inverse=*/true, shape, symbolic::integer(2), types::PrimitiveType::Double);

    EXPECT_EQ(node.direction(), math::tensor::FFTDirection::Inverse);
    EXPECT_EQ(node.real_primitive(), types::PrimitiveType::Double);
    EXPECT_EQ(node.complex_primitive(), types::PrimitiveType::CDouble);

    // Hermitian last dim: 16/2 + 1 = 9; complex_extent = 2 * 16 * 9 = 288.
    EXPECT_EQ(node.complex_last_dim()->__str__(), "9");
    EXPECT_EQ(node.complex_extent()->__str__(), "288");
    EXPECT_EQ(node.real_extent()->__str__(), "512");
}

TEST(FFTNodeTest, SymbolicExtents) {
    builder::StructuredSDFGBuilder builder("fft_sym", FunctionType_CPU);
    builder.add_container("H", types::Scalar(types::PrimitiveType::UInt64), true);
    builder.add_container("W", types::Scalar(types::PrimitiveType::UInt64), true);
    std::vector<symbolic::Expression> shape = {symbolic::symbol("H"), symbolic::symbol("W")};
    auto& node = build_fft_sdfg(builder, /*inverse=*/false, shape, symbolic::symbol("N"), types::PrimitiveType::Float);

    auto syms = node.symbols();
    EXPECT_TRUE(syms.find(symbolic::symbol("H")) != syms.end());
    EXPECT_TRUE(syms.find(symbolic::symbol("W")) != syms.end());
    EXPECT_TRUE(syms.find(symbolic::symbol("N")) != syms.end());

    // complex_last_dim = W/2 + 1 (symbolic): depends on W.
    auto cld = node.complex_last_dim()->__str__();
    EXPECT_NE(cld.find("W"), std::string::npos);
}

TEST(FFTNodeTest, SerializationRoundTripForward) {
    builder::StructuredSDFGBuilder builder("fft_ser", FunctionType_CPU);
    std::vector<symbolic::Expression> shape = {symbolic::integer(8), symbolic::integer(8)};
    build_fft_sdfg(builder, /*inverse=*/false, shape, symbolic::integer(4), types::PrimitiveType::Float);
    auto& sdfg = builder.subject();

    serializer::JSONSerializer serializer;
    auto j = serializer.serialize(sdfg);
    auto deserialized = serializer.deserialize(j);
    ASSERT_NE(deserialized, nullptr);

    // Locate the FFT node in the deserialized SDFG.
    auto& child = deserialized->root().at(0).first;
    auto& block = dynamic_cast<structured_control_flow::Block&>(child);
    const math::tensor::FFTNode* fft = nullptr;
    for (auto& node : block.dataflow().nodes()) {
        if (auto* f = dynamic_cast<const math::tensor::FFTNode*>(&node)) {
            fft = f;
        }
    }
    ASSERT_NE(fft, nullptr);
    EXPECT_EQ(fft->code().value(), math::tensor::LibraryNodeType_FFT.value());
    EXPECT_EQ(fft->rank(), 2u);
    EXPECT_EQ(fft->real_primitive(), types::PrimitiveType::Float);
    EXPECT_EQ(fft->batch()->__str__(), "4");
    EXPECT_EQ(fft->complex_extent()->__str__(), "160");
}

TEST(FFTNodeTest, SerializationRoundTripInverseDouble) {
    builder::StructuredSDFGBuilder builder("ifft_ser", FunctionType_CPU);
    std::vector<symbolic::Expression> shape = {symbolic::integer(16), symbolic::integer(16)};
    build_fft_sdfg(builder, /*inverse=*/true, shape, symbolic::integer(2), types::PrimitiveType::Double);
    auto& sdfg = builder.subject();

    serializer::JSONSerializer serializer;
    auto j = serializer.serialize(sdfg);
    auto deserialized = serializer.deserialize(j);
    ASSERT_NE(deserialized, nullptr);

    const math::tensor::IFFTNode* ifft = nullptr;
    auto& ichild = deserialized->root().at(0).first;
    auto& iblock = dynamic_cast<structured_control_flow::Block&>(ichild);
    for (auto& node : iblock.dataflow().nodes()) {
        if (auto* f = dynamic_cast<const math::tensor::IFFTNode*>(&node)) {
            ifft = f;
        }
    }
    ASSERT_NE(ifft, nullptr);
    EXPECT_EQ(ifft->code().value(), math::tensor::LibraryNodeType_IFFT.value());
    EXPECT_EQ(ifft->direction(), math::tensor::FFTDirection::Inverse);
    EXPECT_EQ(ifft->real_primitive(), types::PrimitiveType::Double);
    EXPECT_EQ(ifft->complex_extent()->__str__(), "288");
}
