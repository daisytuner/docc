#include <gtest/gtest.h>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/codegen/language_extensions/c_language_extension.h"
#include "sdfg/codegen/utils.h"
#include "sdfg/data_flow/library_nodes/math/tensor/reduce_ops/softmax_node.h"
#include "sdfg/passes/dataflow/tensor_to_pointer_conversion.h"
#include "sdfg/passes/offloading/cuda_library_node_expansion_pass.h"
#include "sdfg/passes/pipeline.h"
#include "sdfg/targets/cuda/cuda.h"
#include "sdfg/targets/cuda/math/tensor/softmax.h"
#include "sdfg/targets/cuda/plugin.h"

namespace sdfg::cuda {

// Helper: create an SDFG with a SoftmaxNode and dispatch it
static std::string dispatch_softmax(
    const std::vector<symbolic::Expression>& shape,
    const std::vector<int64_t>& axes,
    const data_flow::ImplementationType& impl_type
) {
    builder::StructuredSDFGBuilder builder("softmax_test", FunctionType_CPU);
    auto& sdfg = builder.subject();

    types::Scalar desc(types::PrimitiveType::Float);
    types::Pointer desc_ptr(desc);

    builder.add_container("X", desc_ptr, true);
    builder.add_container("Y", desc_ptr, true);

    auto& block = builder.add_block(sdfg.root());

    auto& x_node = builder.add_access(block, "X");
    auto& y_node = builder.add_access(block, "Y");

    auto& softmax_node =
        dynamic_cast<math::tensor::SoftmaxNode&>(builder.add_library_node<
                                                 math::tensor::SoftmaxNode>(block, DebugInfo(), shape, axes, false));

    // Set the implementation type to CUDA
    softmax_node.implementation_type() = impl_type;

    // Connectors: inputs_={"Y", "X"}
    // Y is the output buffer passed as input, X is the input data
    symbolic::MultiExpression tensor_shape;
    for (auto& s : shape) {
        tensor_shape.push_back(s);
    }
    types::Tensor tensor_type(desc, tensor_shape);

    builder.add_computational_memlet(block, y_node, softmax_node, "Y", {}, tensor_type, block.debug_info());
    builder.add_computational_memlet(block, x_node, softmax_node, "X", {}, tensor_type, block.debug_info());

    sdfg.validate();

    // Get the dispatcher
    codegen::LibraryNodeDispatcherRegistry local_registry;
    plugins::Context ctx{
        serializer::LibraryNodeSerializerRegistry::instance(),
        codegen::NodeDispatcherRegistry::instance(),
        codegen::MapDispatcherRegistry::instance(),
        local_registry,
        passes::scheduler::SchedulerRegistry::instance()
    };
    cuda::register_cuda_plugin(ctx);

    auto dispatcher_fn =
        local_registry
            .get_library_node_dispatcher(math::tensor::LibraryNodeType_Softmax.value() + "::" + impl_type.value());
    EXPECT_NE(dispatcher_fn, nullptr);
    if (!dispatcher_fn) return "";

    codegen::CLanguageExtension language_extension(sdfg);
    auto dispatcher = dispatcher_fn(language_extension, sdfg, block.dataflow(), softmax_node);

    codegen::PrettyPrinter stream;
    codegen::PrettyPrinter globals_stream;
    codegen::CodeSnippetFactory snippet_factory;

    EXPECT_NO_THROW(dispatcher->dispatch(stream, globals_stream, snippet_factory));

    return stream.str();
}

TEST(SoftmaxDispatcherTest, WithoutTransfers_2D_LastAxis) {
    // Shape: (16384, 256), axis=-1 → num_rows=16384, row_size=256
    std::vector<symbolic::Expression> shape = {symbolic::integer(16384), symbolic::integer(256)};
    std::vector<int64_t> axes = {-1};

    std::string code = dispatch_softmax(shape, axes, ImplementationType_CUDAWithoutTransfers);

    // Should contain the kernel launch
    EXPECT_NE(code.find("softmax_kernel_"), std::string::npos);
    EXPECT_NE(code.find("<<<"), std::string::npos);
    // Should compute num_rows and row_size
    EXPECT_NE(code.find("__softmax_num_rows"), std::string::npos);
    EXPECT_NE(code.find("__softmax_row_size"), std::string::npos);
}

TEST(SoftmaxDispatcherTest, WithoutTransfers_4D_LastAxis) {
    // Shape: (1, 8, 256, 256), axis=-1 → num_rows=1*8*256=2048, row_size=256
    std::vector<symbolic::Expression> shape = {
        symbolic::integer(1), symbolic::integer(8), symbolic::integer(256), symbolic::integer(256)
    };
    std::vector<int64_t> axes = {-1};

    std::string code = dispatch_softmax(shape, axes, ImplementationType_CUDAWithoutTransfers);

    EXPECT_NE(code.find("softmax_kernel_"), std::string::npos);
    EXPECT_NE(code.find("<<<"), std::string::npos);
}

TEST(SoftmaxDispatcherTest, WithTransfers_2D_LastAxis) {
    // Shape: (8192, 256), axis=-1
    std::vector<symbolic::Expression> shape = {symbolic::integer(8192), symbolic::integer(256)};
    std::vector<int64_t> axes = {-1};

    std::string code = dispatch_softmax(shape, axes, ImplementationType_CUDAWithTransfers);

    // Should contain cudaMalloc, cudaMemcpy, kernel launch, cudaFree
    EXPECT_NE(code.find("cudaMalloc"), std::string::npos);
    EXPECT_NE(code.find("cudaMemcpy"), std::string::npos);
    EXPECT_NE(code.find("softmax_kernel_"), std::string::npos);
    EXPECT_NE(code.find("cudaFree"), std::string::npos);
}

TEST(SoftmaxDispatcherTest, WithTransfers_4D_LastAxis) {
    // Shape: (16, 2, 4096, 256), axis=-1 → segformer block1 batch=16
    std::vector<symbolic::Expression> shape = {
        symbolic::integer(16), symbolic::integer(2), symbolic::integer(4096), symbolic::integer(256)
    };
    std::vector<int64_t> axes = {-1};

    std::string code = dispatch_softmax(shape, axes, ImplementationType_CUDAWithTransfers);

    EXPECT_NE(code.find("cudaMalloc"), std::string::npos);
    EXPECT_NE(code.find("softmax_kernel_"), std::string::npos);
    EXPECT_NE(code.find("cudaFree"), std::string::npos);
}

TEST(SoftmaxDispatcherTest, WithoutTransfers_KernelFileGenerated) {
    // Verify the kernel .cu snippet is created
    builder::StructuredSDFGBuilder builder("softmax_kernel_file_test", FunctionType_CPU);
    auto& sdfg = builder.subject();

    types::Scalar desc(types::PrimitiveType::Float);
    types::Pointer desc_ptr(desc);

    builder.add_container("X", desc_ptr, true);
    builder.add_container("Y", desc_ptr, true);

    auto& block = builder.add_block(sdfg.root());

    auto& x_node = builder.add_access(block, "X");
    auto& y_node = builder.add_access(block, "Y");

    std::vector<symbolic::Expression> shape = {symbolic::integer(1024), symbolic::integer(256)};
    std::vector<int64_t> axes = {-1};

    auto& softmax_node =
        dynamic_cast<math::tensor::SoftmaxNode&>(builder.add_library_node<
                                                 math::tensor::SoftmaxNode>(block, DebugInfo(), shape, axes, false));
    softmax_node.implementation_type() = ImplementationType_CUDAWithoutTransfers;

    symbolic::MultiExpression tensor_shape = {symbolic::integer(1024), symbolic::integer(256)};
    types::Tensor tensor_type(desc, tensor_shape);

    builder.add_computational_memlet(block, y_node, softmax_node, "Y", {}, tensor_type, block.debug_info());
    builder.add_computational_memlet(block, x_node, softmax_node, "X", {}, tensor_type, block.debug_info());

    sdfg.validate();

    codegen::LibraryNodeDispatcherRegistry local_registry;
    plugins::Context ctx{
        serializer::LibraryNodeSerializerRegistry::instance(),
        codegen::NodeDispatcherRegistry::instance(),
        codegen::MapDispatcherRegistry::instance(),
        local_registry,
        passes::scheduler::SchedulerRegistry::instance()
    };
    cuda::register_cuda_plugin(ctx);

    auto dispatcher_fn = local_registry.get_library_node_dispatcher(
        math::tensor::LibraryNodeType_Softmax.value() + "::" + ImplementationType_CUDAWithoutTransfers.value()
    );
    ASSERT_NE(dispatcher_fn, nullptr);

    codegen::CLanguageExtension language_extension(sdfg);
    auto dispatcher = dispatcher_fn(language_extension, sdfg, block.dataflow(), softmax_node);

    codegen::PrettyPrinter stream;
    codegen::PrettyPrinter globals_stream;
    codegen::CodeSnippetFactory snippet_factory;

    dispatcher->dispatch(stream, globals_stream, snippet_factory);

    // A .cu kernel file should have been created in the snippet factory
    EXPECT_EQ(snippet_factory.snippets().size(), 1);

    // The globals should contain a forward declaration of the kernel
    std::string globals = globals_stream.str();
    EXPECT_NE(globals.find("__global__ void softmax_kernel_"), std::string::npos);

    // The kernel snippet should contain the warp-shuffle reduction
    auto it = snippet_factory.snippets().begin();
    std::string kernel_code = it->second.stream().str();
    EXPECT_NE(kernel_code.find("__shfl_xor_sync"), std::string::npos);
    EXPECT_NE(kernel_code.find("row_max"), std::string::npos);
    EXPECT_NE(kernel_code.find("row_sum"), std::string::npos);
    EXPECT_NE(kernel_code.find("expf"), std::string::npos);
}

TEST(SoftmaxDispatcherTest, RegistrationKeys) {
    // Verify that both dispatchers are properly registered
    codegen::LibraryNodeDispatcherRegistry local_registry;
    plugins::Context ctx{
        serializer::LibraryNodeSerializerRegistry::instance(),
        codegen::NodeDispatcherRegistry::instance(),
        codegen::MapDispatcherRegistry::instance(),
        local_registry,
        passes::scheduler::SchedulerRegistry::instance()
    };
    cuda::register_cuda_plugin(ctx);

    auto with_transfers = local_registry.get_library_node_dispatcher(
        math::tensor::LibraryNodeType_Softmax.value() + "::" + ImplementationType_CUDAWithTransfers.value()
    );
    EXPECT_NE(with_transfers, nullptr);

    auto without_transfers = local_registry.get_library_node_dispatcher(
        math::tensor::LibraryNodeType_Softmax.value() + "::" + ImplementationType_CUDAWithoutTransfers.value()
    );
    EXPECT_NE(without_transfers, nullptr);
}

TEST(SoftmaxDispatcherTest, SoftmaxSurvivesCudaPipeline) {
    // Verify that a SoftmaxNode is preserved through the CUDA pipeline:
    // CudaExpansionPass sets impl_type → generic ExpansionPass skips it →
    // TensorToPointerConversionPass leaves it intact.
    builder::StructuredSDFGBuilder builder("softmax_pipeline_test", FunctionType_CPU);
    auto& sdfg = builder.subject();

    types::Scalar desc(types::PrimitiveType::Float);
    types::Pointer desc_ptr(desc);

    builder.add_container("X", desc_ptr, true);
    builder.add_container("Y", desc_ptr, true);

    auto& block = builder.add_block(sdfg.root());

    auto& x_node = builder.add_access(block, "X");
    auto& y_node = builder.add_access(block, "Y");

    std::vector<symbolic::Expression> shape = {symbolic::integer(64), symbolic::integer(128)};
    std::vector<int64_t> axes = {-1};

    auto& softmax_node =
        dynamic_cast<math::tensor::SoftmaxNode&>(builder.add_library_node<
                                                 math::tensor::SoftmaxNode>(block, DebugInfo(), shape, axes, false));

    symbolic::MultiExpression tensor_shape = {symbolic::integer(64), symbolic::integer(128)};
    types::Tensor tensor_type(desc, tensor_shape);

    builder.add_computational_memlet(block, y_node, softmax_node, "Y", {}, tensor_type, block.debug_info());
    builder.add_computational_memlet(block, x_node, softmax_node, "X", {}, tensor_type, block.debug_info());

    sdfg.validate();

    // Before: impl_type is NONE
    EXPECT_EQ(softmax_node.implementation_type().value(), data_flow::ImplementationType_NONE.value());

    analysis::AnalysisManager analysis_manager(sdfg);

    // Step 1: CudaExpansionPass should set impl_type to CUDAWithTransfers
    passes::CudaExpansionPass cuda_expansion_pass;
    cuda_expansion_pass.run(builder, analysis_manager);

    // SoftmaxNode should still exist with CUDAWithTransfers
    auto library_nodes_after_expand = block.dataflow().library_nodes();
    ASSERT_EQ(library_nodes_after_expand.size(), 1);
    auto* node_after_expand = dynamic_cast<math::tensor::SoftmaxNode*>(*library_nodes_after_expand.begin());
    ASSERT_NE(node_after_expand, nullptr);
    EXPECT_EQ(node_after_expand->implementation_type().value(), ImplementationType_CUDAWithTransfers.value());

    // Step 2: Generic ExpansionPass should skip it (impl_type != NONE)
    passes::Pipeline expansion = passes::Pipeline::expansion();
    expansion.run(builder, analysis_manager);

    auto library_nodes_after_generic = block.dataflow().library_nodes();
    ASSERT_EQ(library_nodes_after_generic.size(), 1);
    auto* node_after_generic = dynamic_cast<math::tensor::SoftmaxNode*>(*library_nodes_after_generic.begin());
    ASSERT_NE(node_after_generic, nullptr);
    EXPECT_EQ(node_after_generic->implementation_type().value(), ImplementationType_CUDAWithTransfers.value());

    // Step 3: TensorToPointerConversionPass should not destroy it
    passes::TensorToPointerConversionPass tensor_to_ptr_pass;
    tensor_to_ptr_pass.run(builder, analysis_manager);

    sdfg.validate();

    auto library_nodes_final = block.dataflow().library_nodes();
    ASSERT_EQ(library_nodes_final.size(), 1);
    auto* node_final = dynamic_cast<math::tensor::SoftmaxNode*>(*library_nodes_final.begin());
    ASSERT_NE(node_final, nullptr);
    EXPECT_EQ(node_final->implementation_type().value(), ImplementationType_CUDAWithTransfers.value());
}

} // namespace sdfg::cuda
