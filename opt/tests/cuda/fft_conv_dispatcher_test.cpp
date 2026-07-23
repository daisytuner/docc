#include <gtest/gtest.h>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/codegen/language_extensions/c_language_extension.h"
#include "sdfg/codegen/utils.h"
#include "sdfg/data_flow/library_nodes/math/tensor/fft_conv_node.h"
#include "sdfg/targets/cuda/cuda.h"
#include "sdfg/targets/cuda/math/tensor/fft_conv.h"
#include "sdfg/targets/cuda/plugin.h"

namespace sdfg::cuda {

// Build an SDFG containing a single FFTConvNode and return the concatenation of the
// host launch stream plus every emitted code snippet (i.e. the hand-tuned kernels).
static std::string dispatch_fft_conv(
    const std::vector<symbolic::Expression>& shape,
    const std::vector<symbolic::Expression>& kernel_shape,
    const std::vector<symbolic::Expression>& pads,
    types::PrimitiveType precision,
    bool with_bias,
    const data_flow::ImplementationType& impl_type
) {
    builder::StructuredSDFGBuilder builder("fft_conv_test", FunctionType_CPU);
    auto& sdfg = builder.subject();

    types::Scalar f_scalar(precision);
    types::Pointer f_ptr(f_scalar);
    builder.add_container("Y", f_ptr, true);
    builder.add_container("X", f_ptr, true);
    builder.add_container("W", f_ptr, true);
    if (with_bias) {
        builder.add_container("B", f_ptr, true);
    }

    auto& block = builder.add_block(sdfg.root());
    auto& y_node = builder.add_access(block, "Y");
    auto& x_node = builder.add_access(block, "X");
    auto& w_node = builder.add_access(block, "W");

    auto& node = static_cast<math::tensor::FFTConvNode&>(builder.add_library_node<math::tensor::FFTConvNode>(
        block, DebugInfo(), impl_type, shape, kernel_shape, pads, precision, with_bias
    ));

    builder.add_computational_memlet(block, y_node, node, "Y", {}, f_ptr, block.debug_info());
    builder.add_computational_memlet(block, x_node, node, "X", {}, f_ptr, block.debug_info());
    builder.add_computational_memlet(block, w_node, node, "W", {}, f_ptr, block.debug_info());
    if (with_bias) {
        auto& b_node = builder.add_access(block, "B");
        builder.add_computational_memlet(block, b_node, node, "B", {}, f_ptr, block.debug_info());
    }

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

    auto dispatcher_fn =
        local_registry
            .get_library_node_dispatcher(math::tensor::LibraryNodeType_FFTConv.value() + "::" + impl_type.value());
    EXPECT_NE(dispatcher_fn, nullptr);
    if (!dispatcher_fn) return "";

    codegen::CLanguageExtension language_extension(sdfg);
    auto dispatcher = dispatcher_fn(language_extension, sdfg, block.dataflow(), node);

    codegen::PrettyPrinter stream;
    codegen::PrettyPrinter globals_stream;
    codegen::CodeSnippetFactory snippet_factory;
    EXPECT_NO_THROW(dispatcher->dispatch(stream, globals_stream, snippet_factory));

    std::string result = stream.str() + globals_stream.str();
    for (const auto& [name, snippet] : snippet_factory.snippets()) {
        result += snippet.stream().str();
    }
    return result;
}

TEST(FFTConvDispatcherTest, Float_WithTransfers_EmitsTunedKernels) {
    std::vector<symbolic::Expression> shape = {
        symbolic::integer(2), symbolic::integer(3), symbolic::integer(8), symbolic::integer(8)
    };
    std::vector<symbolic::Expression> kernel_shape = {symbolic::integer(3), symbolic::integer(3)};
    std::vector<symbolic::Expression> pads = {
        symbolic::integer(1), symbolic::integer(1), symbolic::integer(1), symbolic::integer(1)
    };

    std::string code = dispatch_fft_conv(
        shape, kernel_shape, pads, types::PrimitiveType::Float, /*with_bias=*/true, ImplementationType_CUDAWithTransfers
    );

    // Hand-tuned Stockham kernels on native complex buffers.
    EXPECT_NE(code.find("_fftRows"), std::string::npos);
    EXPECT_NE(code.find("_fftCols"), std::string::npos);
    EXPECT_NE(code.find("_stockham"), std::string::npos);
    EXPECT_NE(code.find("_complexMul"), std::string::npos);
    EXPECT_NE(code.find("_pad"), std::string::npos);
    EXPECT_NE(code.find("_crop"), std::string::npos);
    // Uses native complex type, not a separate real/imag SoA like the reference.
    EXPECT_NE(code.find("float2"), std::string::npos);
    EXPECT_EQ(code.find("Imag"), std::string::npos);
    // Host-managed device buffers + transfers.
    EXPECT_NE(code.find("cudaMalloc"), std::string::npos);
    EXPECT_NE(code.find("cudaMemcpy"), std::string::npos);
    EXPECT_NE(code.find("cudaFree"), std::string::npos);
    // Kernel launch present.
    EXPECT_NE(code.find("<<<"), std::string::npos);
}

TEST(FFTConvDispatcherTest, Float_NoBias) {
    std::vector<symbolic::Expression> shape = {
        symbolic::integer(1), symbolic::integer(4), symbolic::integer(16), symbolic::integer(16)
    };
    std::vector<symbolic::Expression> kernel_shape = {symbolic::integer(3), symbolic::integer(3)};
    std::vector<symbolic::Expression> pads = {
        symbolic::integer(1), symbolic::integer(1), symbolic::integer(1), symbolic::integer(1)
    };

    std::string code = dispatch_fft_conv(
        shape, kernel_shape, pads, types::PrimitiveType::Float, /*with_bias=*/false, ImplementationType_CUDAWithTransfers
    );

    EXPECT_NE(code.find("_fftRows"), std::string::npos);
    EXPECT_NE(code.find("float2"), std::string::npos);
    // Bias pointer is null when absent.
    EXPECT_NE(code.find("nullptr"), std::string::npos);
}

TEST(FFTConvDispatcherTest, Double_Throws) {
    std::vector<symbolic::Expression> shape = {
        symbolic::integer(1), symbolic::integer(2), symbolic::integer(8), symbolic::integer(8)
    };
    std::vector<symbolic::Expression> kernel_shape = {symbolic::integer(3), symbolic::integer(3)};
    std::vector<symbolic::Expression> pads = {
        symbolic::integer(1), symbolic::integer(1), symbolic::integer(1), symbolic::integer(1)
    };

    builder::StructuredSDFGBuilder builder("fft_conv_dbl", FunctionType_CPU);
    auto& sdfg = builder.subject();
    types::Scalar d_scalar(types::PrimitiveType::Double);
    types::Pointer d_ptr(d_scalar);
    builder.add_container("Y", d_ptr, true);
    builder.add_container("X", d_ptr, true);
    builder.add_container("W", d_ptr, true);
    auto& block = builder.add_block(sdfg.root());
    auto& y_node = builder.add_access(block, "Y");
    auto& x_node = builder.add_access(block, "X");
    auto& w_node = builder.add_access(block, "W");
    auto& node = static_cast<math::tensor::FFTConvNode&>(builder.add_library_node<math::tensor::FFTConvNode>(
        block,
        DebugInfo(),
        ImplementationType_CUDAWithTransfers,
        shape,
        kernel_shape,
        pads,
        types::PrimitiveType::Double,
        false
    ));
    builder.add_computational_memlet(block, y_node, node, "Y", {}, d_ptr, block.debug_info());
    builder.add_computational_memlet(block, x_node, node, "X", {}, d_ptr, block.debug_info());
    builder.add_computational_memlet(block, w_node, node, "W", {}, d_ptr, block.debug_info());

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
    auto dispatcher_fn = local_registry.get_library_node_dispatcher(
        math::tensor::LibraryNodeType_FFTConv.value() + "::" + ImplementationType_CUDAWithTransfers.value()
    );
    ASSERT_NE(dispatcher_fn, nullptr);
    codegen::CLanguageExtension language_extension(sdfg);
    auto dispatcher = dispatcher_fn(language_extension, sdfg, block.dataflow(), node);
    codegen::PrettyPrinter stream;
    codegen::PrettyPrinter globals_stream;
    codegen::CodeSnippetFactory snippet_factory;
    EXPECT_ANY_THROW(dispatcher->dispatch(stream, globals_stream, snippet_factory));
}

} // namespace sdfg::cuda
