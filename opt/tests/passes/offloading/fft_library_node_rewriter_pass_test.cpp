#include <gtest/gtest.h>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_nodes/math/tensor/fft_node.h"
#include "sdfg/passes/offloading/cuda_library_node_rewriter_pass.h"
#include "sdfg/passes/offloading/rocm_library_node_rewriter_pass.h"
#include "sdfg/targets/cuda/cuda.h"
#include "sdfg/targets/rocm/rocm.h"

using namespace sdfg;

namespace {

math::tensor::FFTNodeBase& build_fft(builder::StructuredSDFGBuilder& builder, bool inverse, types::PrimitiveType precision) {
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

    std::vector<symbolic::Expression> shape = {symbolic::integer(8), symbolic::integer(8)};
    data_flow::LibraryNode* node = nullptr;
    if (inverse) {
        node = &builder.add_library_node<math::tensor::IFFTNode>(
            block, DebugInfo(), data_flow::ImplementationType_NONE, shape, symbolic::integer(4), precision
        );
    } else {
        node = &builder.add_library_node<math::tensor::FFTNode>(
            block, DebugInfo(), data_flow::ImplementationType_NONE, shape, symbolic::integer(4), precision
        );
    }
    builder.add_computational_memlet(block, y_node, *node, "__Y", {}, y_ptr, block.debug_info());
    builder.add_computational_memlet(block, x_node, *node, "__X", {}, x_ptr, block.debug_info());
    return static_cast<math::tensor::FFTNodeBase&>(*node);
}

} // namespace

TEST(FFTRewriterPassTest, CudaForwardFloat) {
    builder::StructuredSDFGBuilder builder("fft", FunctionType_CPU);
    auto& node = build_fft(builder, /*inverse=*/false, types::PrimitiveType::Float);
    EXPECT_EQ(node.implementation_type().value(), data_flow::ImplementationType_NONE.value());

    analysis::AnalysisManager analysis_manager(builder.subject());
    cuda::CudaLibraryNodeRewriterPass pass;
    pass.run(builder, analysis_manager);

    EXPECT_EQ(node.implementation_type().value(), cuda::ImplementationType_CUDAWithTransfers.value());
}

TEST(FFTRewriterPassTest, CudaInverseDouble) {
    builder::StructuredSDFGBuilder builder("ifft", FunctionType_CPU);
    auto& node = build_fft(builder, /*inverse=*/true, types::PrimitiveType::Double);

    analysis::AnalysisManager analysis_manager(builder.subject());
    cuda::CudaLibraryNodeRewriterPass pass;
    pass.run(builder, analysis_manager);

    EXPECT_EQ(node.implementation_type().value(), cuda::ImplementationType_CUDAWithTransfers.value());
}

TEST(FFTRewriterPassTest, RocmForwardFloat) {
    builder::StructuredSDFGBuilder builder("fft", FunctionType_CPU);
    auto& node = build_fft(builder, /*inverse=*/false, types::PrimitiveType::Float);

    analysis::AnalysisManager analysis_manager(builder.subject());
    rocm::RocmLibraryNodeRewriterPass pass;
    pass.run(builder, analysis_manager);

    EXPECT_EQ(node.implementation_type().value(), rocm::ImplementationType_ROCMWithTransfers.value());
}

TEST(FFTRewriterPassTest, RocmInverseDouble) {
    builder::StructuredSDFGBuilder builder("ifft", FunctionType_CPU);
    auto& node = build_fft(builder, /*inverse=*/true, types::PrimitiveType::Double);

    analysis::AnalysisManager analysis_manager(builder.subject());
    rocm::RocmLibraryNodeRewriterPass pass;
    pass.run(builder, analysis_manager);

    EXPECT_EQ(node.implementation_type().value(), rocm::ImplementationType_ROCMWithTransfers.value());
}
