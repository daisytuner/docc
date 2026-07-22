#include <gtest/gtest.h>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_nodes/math/tensor/fft_node.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/targets/cuda/cuda.h"
#include "sdfg/targets/rocm/rocm.h"
#include "sdfg/transformations/offloading/cufft_data_transfer_extraction.h"
#include "sdfg/transformations/offloading/rocfft_data_transfer_extraction.h"

using namespace sdfg;

namespace {

// Build an SDFG with a single FFT node in its own block; return node + the X/Y access nodes.
struct FFTFixture {
    data_flow::AccessNode* x_access;
    data_flow::AccessNode* y_access;
    math::tensor::FFTNodeBase* node;
};

FFTFixture build_fft(builder::StructuredSDFGBuilder& builder, bool inverse, const data_flow::ImplementationType& impl_type) {
    auto& sdfg = builder.subject();
    types::Pointer x_ptr(types::Scalar(inverse ? types::PrimitiveType::CFloat : types::PrimitiveType::Float));
    types::Pointer y_ptr(types::Scalar(inverse ? types::PrimitiveType::Float : types::PrimitiveType::CFloat));
    builder.add_container("X", x_ptr, true);
    builder.add_container("Y", y_ptr, true);

    auto& block = builder.add_block(sdfg.root());
    auto& x_node = builder.add_access(block, "X");
    auto& y_node = builder.add_access(block, "Y");

    std::vector<symbolic::Expression> shape = {symbolic::integer(8), symbolic::integer(8)};
    data_flow::LibraryNode* node = nullptr;
    if (inverse) {
        node = &builder.add_library_node<math::tensor::IFFTNode>(
            block, DebugInfo(), impl_type, shape, symbolic::integer(4), types::PrimitiveType::Float
        );
    } else {
        node = &builder.add_library_node<math::tensor::FFTNode>(
            block, DebugInfo(), impl_type, shape, symbolic::integer(4), types::PrimitiveType::Float
        );
    }
    builder.add_computational_memlet(block, y_node, *node, "__Y", {}, y_ptr, block.debug_info());
    builder.add_computational_memlet(block, x_node, *node, "__X", {}, x_ptr, block.debug_info());
    return {&x_node, &y_node, static_cast<math::tensor::FFTNodeBase*>(node)};
}

} // namespace

TEST(FFTDataTransferExtractionTest, CudaForwardCanBeAppliedAndApply) {
    builder::StructuredSDFGBuilder builder("fft", FunctionType_CPU);
    auto fx = build_fft(builder, /*inverse=*/false, cuda::ImplementationType_CUDAWithTransfers);
    auto& sdfg = builder.subject();

    analysis::AnalysisManager analysis_manager(sdfg);
    cuda::CUFFTDataTransferExtraction extraction(*fx.node);
    ASSERT_TRUE(extraction.can_be_applied(builder, analysis_manager));
    extraction.apply(builder, analysis_manager);

    // Node is now on the without-transfers path.
    EXPECT_EQ(fx.node->implementation_type().value(), cuda::ImplementationType_CUDAWithoutTransfers.value());

    // Surrounding alloc/copy/free blocks were inserted (X: copy+alloc, free; Y: alloc, copy+free).
    EXPECT_EQ(sdfg.root().size(), 5u);

    // Operands now reference device containers.
    EXPECT_NE(fx.x_access->data().find(cuda::CUDA_DEVICE_PREFIX), std::string::npos);
    EXPECT_NE(fx.y_access->data().find(cuda::CUDA_DEVICE_PREFIX), std::string::npos);
}

TEST(FFTDataTransferExtractionTest, CudaWrongImplTypeNotApplied) {
    builder::StructuredSDFGBuilder builder("fft", FunctionType_CPU);
    auto fx = build_fft(builder, /*inverse=*/false, cuda::ImplementationType_CUDAWithoutTransfers);
    auto& sdfg = builder.subject();

    analysis::AnalysisManager analysis_manager(sdfg);
    cuda::CUFFTDataTransferExtraction extraction(*fx.node);
    EXPECT_FALSE(extraction.can_be_applied(builder, analysis_manager));
}

TEST(FFTDataTransferExtractionTest, CudaInverseApply) {
    builder::StructuredSDFGBuilder builder("ifft", FunctionType_CPU);
    auto fx = build_fft(builder, /*inverse=*/true, cuda::ImplementationType_CUDAWithTransfers);
    auto& sdfg = builder.subject();

    analysis::AnalysisManager analysis_manager(sdfg);
    cuda::CUFFTDataTransferExtraction extraction(*fx.node);
    ASSERT_TRUE(extraction.can_be_applied(builder, analysis_manager));
    extraction.apply(builder, analysis_manager);

    EXPECT_EQ(fx.node->implementation_type().value(), cuda::ImplementationType_CUDAWithoutTransfers.value());
    EXPECT_NE(fx.x_access->data().find(cuda::CUDA_DEVICE_PREFIX), std::string::npos);
    EXPECT_NE(fx.y_access->data().find(cuda::CUDA_DEVICE_PREFIX), std::string::npos);
}

TEST(FFTDataTransferExtractionTest, RocmForwardApply) {
    builder::StructuredSDFGBuilder builder("fft", FunctionType_CPU);
    auto fx = build_fft(builder, /*inverse=*/false, rocm::ImplementationType_ROCMWithTransfers);
    auto& sdfg = builder.subject();

    analysis::AnalysisManager analysis_manager(sdfg);
    rocm::ROCFFTDataTransferExtraction extraction(*fx.node);
    ASSERT_TRUE(extraction.can_be_applied(builder, analysis_manager));
    extraction.apply(builder, analysis_manager);

    EXPECT_EQ(fx.node->implementation_type().value(), rocm::ImplementationType_ROCMWithoutTransfers.value());
    EXPECT_EQ(sdfg.root().size(), 5u);
    EXPECT_NE(fx.x_access->data().find(rocm::ROCM_DEVICE_PREFIX), std::string::npos);
    EXPECT_NE(fx.y_access->data().find(rocm::ROCM_DEVICE_PREFIX), std::string::npos);
}

TEST(FFTDataTransferExtractionTest, RocmWrongImplTypeNotApplied) {
    builder::StructuredSDFGBuilder builder("fft", FunctionType_CPU);
    auto fx = build_fft(builder, /*inverse=*/false, rocm::ImplementationType_ROCMWithoutTransfers);
    auto& sdfg = builder.subject();

    analysis::AnalysisManager analysis_manager(sdfg);
    rocm::ROCFFTDataTransferExtraction extraction(*fx.node);
    EXPECT_FALSE(extraction.can_be_applied(builder, analysis_manager));
}
