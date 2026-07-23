#include <gtest/gtest.h>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/codegen/language_extensions/c_language_extension.h"
#include "sdfg/codegen/utils.h"
#include "sdfg/data_flow/library_nodes/math/tensor/fft_node.h"
#include "sdfg/targets/cuda/cuda.h"
#include "sdfg/targets/cuda/math/tensor/fft.h"
#include "sdfg/targets/cuda/plugin.h"

namespace sdfg::cuda {

// Helper: build an SDFG containing an FFT/IFFT node and return the dispatched code.
static std::string dispatch_fft(
    bool inverse,
    const std::vector<symbolic::Expression>& shape,
    symbolic::Expression batch,
    types::PrimitiveType precision,
    const data_flow::ImplementationType& impl_type
) {
    builder::StructuredSDFGBuilder builder("fft_test", FunctionType_CPU);
    auto& sdfg = builder.subject();

    types::PrimitiveType cplx = precision == types::PrimitiveType::Double ? types::PrimitiveType::CDouble
                                                                          : types::PrimitiveType::CFloat;
    // Forward: X real, Y complex. Inverse: X complex, Y real.
    types::Pointer x_ptr(types::Scalar(inverse ? cplx : precision));
    types::Pointer y_ptr(types::Scalar(inverse ? precision : cplx));

    builder.add_container("X", x_ptr, true);
    builder.add_container("Y", y_ptr, true);

    auto& block = builder.add_block(sdfg.root());
    auto& x_node = builder.add_access(block, "X");
    auto& y_node = builder.add_access(block, "Y");

    data_flow::LibraryNode* node = nullptr;
    if (inverse) {
        node =
            &builder.add_library_node<math::tensor::IFFTNode>(block, DebugInfo(), impl_type, shape, batch, precision);
    } else {
        node = &builder.add_library_node<math::tensor::FFTNode>(block, DebugInfo(), impl_type, shape, batch, precision);
    }

    builder.add_computational_memlet(block, y_node, *node, "__Y", {}, y_ptr, block.debug_info());
    builder.add_computational_memlet(block, x_node, *node, "__X", {}, x_ptr, block.debug_info());

    codegen::LibraryNodeDispatcherRegistry local_registry;
    plugins::Context ctx{
        serializer::LibraryNodeSerializerRegistry::instance(),
        codegen::NodeDispatcherRegistry::instance(),
        codegen::MapDispatcherRegistry::instance(),
        codegen::ReduceDispatcherRegistry::instance(),
        local_registry,
        passes::scheduler::SchedulerRegistry::instance()
    };
    cuda::register_cuda_plugin(ctx);

    const auto code = inverse ? math::tensor::LibraryNodeType_IFFT.value() : math::tensor::LibraryNodeType_FFT.value();
    auto dispatcher_fn = local_registry.get_library_node_dispatcher(code + "::" + impl_type.value());
    EXPECT_NE(dispatcher_fn, nullptr);
    if (!dispatcher_fn) return "";

    codegen::CLanguageExtension language_extension(sdfg);
    auto dispatcher = dispatcher_fn(language_extension, sdfg, block.dataflow(), *node);

    codegen::PrettyPrinter stream;
    codegen::PrettyPrinter globals_stream;
    codegen::CodeSnippetFactory snippet_factory;
    EXPECT_NO_THROW(dispatcher->dispatch(stream, globals_stream, snippet_factory));
    std::string result = stream.str();
    for (const auto& s : snippet_factory.setup_snippets()) {
        result += s;
    }
    return result;
}

TEST(FFTDispatcherTest, ForwardR2C_Float_WithTransfers) {
    std::vector<symbolic::Expression> shape = {symbolic::integer(8), symbolic::integer(8)};
    std::string code =
        dispatch_fft(false, shape, symbolic::integer(4), types::PrimitiveType::Float, ImplementationType_CUDAWithTransfers);

    EXPECT_NE(code.find("cufftPlanMany"), std::string::npos);
    EXPECT_NE(code.find("cufftExecR2C"), std::string::npos);
    EXPECT_NE(code.find("CUFFT_R2C"), std::string::npos);
    EXPECT_NE(code.find("cudaMalloc"), std::string::npos);
    EXPECT_NE(code.find("cudaMemcpy"), std::string::npos);
    EXPECT_NE(code.find("cudaFree"), std::string::npos);
    // Hermitian odist for 8x8: 8 * (8/2+1) = 40.
    EXPECT_NE(code.find("40"), std::string::npos);
}

TEST(FFTDispatcherTest, ForwardR2C_Float_WithoutTransfers) {
    std::vector<symbolic::Expression> shape = {symbolic::integer(8), symbolic::integer(8)};
    std::string code = dispatch_fft(
        false, shape, symbolic::integer(4), types::PrimitiveType::Float, ImplementationType_CUDAWithoutTransfers
    );

    EXPECT_NE(code.find("cufftPlanMany"), std::string::npos);
    EXPECT_NE(code.find("cufftExecR2C"), std::string::npos);
    // No host<->device transfers on the without-transfers path.
    EXPECT_EQ(code.find("cudaMalloc"), std::string::npos);
}

TEST(FFTDispatcherTest, ForwardR2C_Double_WithTransfers) {
    std::vector<symbolic::Expression> shape = {symbolic::integer(16), symbolic::integer(16)};
    std::string code = dispatch_fft(
        false, shape, symbolic::integer(2), types::PrimitiveType::Double, ImplementationType_CUDAWithTransfers
    );

    EXPECT_NE(code.find("cufftExecD2Z"), std::string::npos);
    EXPECT_NE(code.find("CUFFT_D2Z"), std::string::npos);
    EXPECT_NE(code.find("cufftDoubleComplex"), std::string::npos);
}

TEST(FFTDispatcherTest, InverseC2R_Float_WithTransfers) {
    std::vector<symbolic::Expression> shape = {symbolic::integer(8), symbolic::integer(8)};
    std::string code =
        dispatch_fft(true, shape, symbolic::integer(4), types::PrimitiveType::Float, ImplementationType_CUDAWithTransfers);

    EXPECT_NE(code.find("cufftExecC2R"), std::string::npos);
    EXPECT_NE(code.find("CUFFT_C2R"), std::string::npos);
}

TEST(FFTDispatcherTest, InverseC2R_Double_WithoutTransfers) {
    std::vector<symbolic::Expression> shape = {symbolic::integer(8), symbolic::integer(8)};
    std::string code = dispatch_fft(
        true, shape, symbolic::integer(1), types::PrimitiveType::Double, ImplementationType_CUDAWithoutTransfers
    );

    EXPECT_NE(code.find("cufftExecZ2D"), std::string::npos);
    EXPECT_EQ(code.find("cudaMalloc"), std::string::npos);
}

} // namespace sdfg::cuda
