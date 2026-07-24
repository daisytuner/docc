#include <gtest/gtest.h>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/codegen/language_extensions/c_language_extension.h"
#include "sdfg/codegen/utils.h"
#include "sdfg/data_flow/library_nodes/math/tensor/c2r_fft2d_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/r2c_fft2d_node.h"
#include "sdfg/targets/cuda/cuda.h"
#include "sdfg/targets/cuda/plugin.h"

namespace sdfg::cuda {

// Build an SDFG with a single FFT2D node (Y, X pointer inputs) and return the
// concatenation of the host launch stream plus every emitted kernel snippet.
template<typename NodeT>
static std::string dispatch_fft2d(
    const std::vector<symbolic::Expression>& shape,
    types::PrimitiveType real_precision,
    types::PrimitiveType y_precision,
    const std::string& code_key,
    const data_flow::ImplementationType& impl_type
) {
    builder::StructuredSDFGBuilder builder("fft2d_test", FunctionType_CPU);
    auto& sdfg = builder.subject();

    types::Scalar y_scalar(y_precision);
    types::Scalar x_scalar(real_precision);
    types::Pointer y_ptr(y_scalar);
    types::Pointer x_ptr(x_scalar);
    builder.add_container("Y", y_ptr, true);
    builder.add_container("X", x_ptr, true);

    auto& block = builder.add_block(sdfg.root());
    auto& y_node = builder.add_access(block, "Y");
    auto& x_node = builder.add_access(block, "X");

    auto& node =
        static_cast<NodeT&>(builder.add_library_node<NodeT>(block, DebugInfo(), impl_type, shape, real_precision));

    builder.add_computational_memlet(block, y_node, node, "Y", {}, y_ptr, block.debug_info());
    builder.add_computational_memlet(block, x_node, node, "X", {}, x_ptr, block.debug_info());

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

    auto dispatcher_fn = local_registry.get_library_node_dispatcher(code_key + "::" + impl_type.value());
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

TEST(FFT2DDispatcherTest, R2C_Float_WithTransfers_EmitsTunedKernels) {
    // [matrices, fftH, fftW]
    std::vector<symbolic::Expression> shape = {symbolic::integer(6), symbolic::integer(8), symbolic::integer(8)};
    std::string code = dispatch_fft2d<math::tensor::R2CFFT2DNode>(
        shape,
        types::PrimitiveType::Float,
        types::PrimitiveType::CFloat,
        math::tensor::LibraryNodeType_R2CFFT2D.value(),
        ImplementationType_CUDAWithTransfers
    );

    EXPECT_NE(code.find("_fftRowsR2C"), std::string::npos);
    EXPECT_NE(code.find("_fftCols"), std::string::npos);
    EXPECT_NE(code.find("_stockham"), std::string::npos);
    EXPECT_NE(code.find("float2"), std::string::npos);
    EXPECT_NE(code.find("cudaMalloc"), std::string::npos);
    EXPECT_NE(code.find("cudaMemcpy"), std::string::npos);
    EXPECT_NE(code.find("<<<"), std::string::npos);
}

TEST(FFT2DDispatcherTest, C2R_Float_WithTransfers_EmitsTunedKernels) {
    std::vector<symbolic::Expression> shape = {symbolic::integer(6), symbolic::integer(8), symbolic::integer(8)};
    std::string code = dispatch_fft2d<math::tensor::C2RFFT2DNode>(
        shape,
        types::PrimitiveType::Float,
        types::PrimitiveType::CFloat,
        math::tensor::LibraryNodeType_C2RFFT2D.value(),
        ImplementationType_CUDAWithTransfers
    );

    EXPECT_NE(code.find("_fftRowsC2R"), std::string::npos);
    EXPECT_NE(code.find("_fftCols"), std::string::npos);
    EXPECT_NE(code.find("_stockham"), std::string::npos);
    EXPECT_NE(code.find("float2"), std::string::npos);
    EXPECT_NE(code.find("<<<"), std::string::npos);
}

TEST(FFT2DDispatcherTest, R2C_WithoutTransfers_UsesDeviceOperands) {
    std::vector<symbolic::Expression> shape = {symbolic::integer(2), symbolic::integer(9), symbolic::integer(9)};
    std::string code = dispatch_fft2d<math::tensor::R2CFFT2DNode>(
        shape,
        types::PrimitiveType::Float,
        types::PrimitiveType::CFloat,
        math::tensor::LibraryNodeType_R2CFFT2D.value(),
        ImplementationType_CUDAWithoutTransfers
    );

    // Device-resident variant reinterpret-casts the operands rather than copying X/Y.
    EXPECT_NE(code.find("reinterpret_cast<float2*>"), std::string::npos);
    EXPECT_NE(code.find("_fftRowsR2C"), std::string::npos);
}

} // namespace sdfg::cuda
