#include <gtest/gtest.h>
#include <sdfg/serializer/json_serializer.h>

#include "sdfg/codegen/language_extensions/c_language_extension.h"
#include "sdfg/codegen/utils.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/targets/offloading/data_offloading_node.h"
#include "sdfg/targets/rocm/plugin.h"
#include "sdfg/targets/rocm/rocm.h"
#include "sdfg/targets/rocm/rocm_data_offloading_node.h"
#include "symengine/symengine_rcp.h"

namespace sdfg::rocm {

TEST(ROCMD2HTransferTest, DispatcherNoThrow) {
    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto& root = builder.subject().root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Scalar integer_desc(types::PrimitiveType::Int32);
    types::Pointer pointer_type(base_desc);

    builder.add_container("A_host", pointer_type);
    builder.add_container("A_device", pointer_type);
    builder.add_container("N", integer_desc);

    auto [block, d2h_node] = offloading::add_offloading_block<ROCMDataOffloadingNode>(
        builder,
        root,
        "A_host",
        "A_device",
        offloading::DataTransferDirection::D2H,
        offloading::BufferLifecycle::NO_CHANGE,
        pointer_type,
        DebugInfo("test_file.cpp", 1, 1, 1, 10),
        symbolic::symbol("N"),
        symbolic::integer(0)
    );

    codegen::CLanguageExtension language_extension(builder.subject());
    codegen::PrettyPrinter pretty_printer;
    codegen::PrettyPrinter globals_printer;
    codegen::CodeSnippetFactory snippet_factory;

    ROCMDataOffloadingNodeDispatcher dispatcher(language_extension, builder.subject(), block.dataflow(), d2h_node);
    EXPECT_NO_THROW(dispatcher.dispatch(pretty_printer, globals_printer, snippet_factory));
}

TEST(ROCMH2DTransferTest, DispatcherNoThrow) {
    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto& root = builder.subject().root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Scalar integer_desc(types::PrimitiveType::Int32);
    types::Pointer pointer_type(base_desc);

    builder.add_container("A_host", pointer_type);
    builder.add_container("A_device", pointer_type);
    builder.add_container("N", integer_desc);

    auto [block, h2d_node] = offloading::add_offloading_block<ROCMDataOffloadingNode>(
        builder,
        root,
        "A_host",
        "A_device",
        offloading::DataTransferDirection::H2D,
        offloading::BufferLifecycle::NO_CHANGE,
        pointer_type,
        DebugInfo("test_file.cpp", 1, 1, 1, 10),
        symbolic::symbol("N"),
        symbolic::integer(0)
    );

    codegen::CLanguageExtension language_extension(builder.subject());
    codegen::PrettyPrinter pretty_printer;
    codegen::PrettyPrinter globals_printer;
    codegen::CodeSnippetFactory snippet_factory;

    ROCMDataOffloadingNodeDispatcher dispatcher(language_extension, builder.subject(), block.dataflow(), h2d_node);
    EXPECT_NO_THROW(dispatcher.dispatch(pretty_printer, globals_printer, snippet_factory));
}

TEST(ROCMMallocTest, DispatcherNoThrow) {
    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto& root = builder.subject().root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Scalar integer_desc(types::PrimitiveType::Int32);
    types::Pointer pointer_type(base_desc);

    builder.add_container("A_device", pointer_type);
    builder.add_container("N", integer_desc);

    auto [block, malloc_node] = offloading::add_offloading_block<ROCMDataOffloadingNode>(
        builder,
        root,
        "A_device",
        "A_device",
        offloading::DataTransferDirection::NONE,
        offloading::BufferLifecycle::ALLOC,
        pointer_type,
        DebugInfo("test_file.cpp", 1, 1, 1, 10),
        symbolic::symbol("N"),
        symbolic::integer(0)
    );

    codegen::CLanguageExtension language_extension(builder.subject());
    codegen::PrettyPrinter pretty_printer;
    codegen::PrettyPrinter globals_printer;
    codegen::CodeSnippetFactory snippet_factory;

    ROCMDataOffloadingNodeDispatcher dispatcher(language_extension, builder.subject(), block.dataflow(), malloc_node);
    EXPECT_NO_THROW(dispatcher.dispatch(pretty_printer, globals_printer, snippet_factory));
}

TEST(ROCMFreeTest, DispatcherNoThrow) {
    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto& root = builder.subject().root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer pointer_type(base_desc);

    builder.add_container("A_device", pointer_type);

    auto [block, free_node] = offloading::add_offloading_block<ROCMDataOffloadingNode>(
        builder,
        root,
        "A_host",
        "A_device",
        offloading::DataTransferDirection::NONE,
        offloading::BufferLifecycle::FREE,
        pointer_type,
        DebugInfo("test_file.cpp", 1, 1, 1, 10),
        SymEngine::null,
        symbolic::integer(0)
    );

    codegen::CLanguageExtension language_extension(builder.subject());
    codegen::PrettyPrinter pretty_printer;
    codegen::PrettyPrinter globals_printer;
    codegen::CodeSnippetFactory snippet_factory;

    ROCMDataOffloadingNodeDispatcher dispatcher(language_extension, builder.subject(), block.dataflow(), free_node);
    EXPECT_NO_THROW(dispatcher.dispatch(pretty_printer, globals_printer, snippet_factory));
}


TEST(RocBlasTest, GemmNodeWithoutDataTransfers_DoublePrecisionNoThrow) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();

    int dim_i = 10;
    int dim_j = 20;
    int dim_k = 30;

    types::Scalar desc(types::PrimitiveType::Double);
    types::Array arr_a_type(desc, symbolic::mul(symbolic::integer(dim_k), symbolic::integer(dim_i)));
    types::Array arr_b_type(desc, symbolic::mul(symbolic::integer(dim_j), symbolic::integer(dim_k)));
    types::Array arr_res_type(desc, symbolic::mul(symbolic::integer(dim_j), symbolic::integer(dim_i)));

    builder.add_container("arr_a", arr_a_type);
    builder.add_container("arr_b", arr_b_type);
    builder.add_container("output", arr_res_type);

    auto& block = builder.add_block(sdfg.root());

    auto& input_a_node = builder.add_access(block, "arr_a");
    auto& input_b_node = builder.add_access(block, "arr_b");
    auto& dummy_input_node = builder.add_access(block, "output");
    auto& gemm_node = static_cast<math::blas::GEMMNode&>(builder.add_library_node<math::blas::GEMMNode>(
        block,
        DebugInfo(),
        rocm::ImplementationType_ROCMWithoutTransfers,
        math::blas::BLAS_Precision::d,
        math::blas::BLAS_Layout::RowMajor,
        math::blas::BLAS_Transpose::No,
        math::blas::BLAS_Transpose::No,
        symbolic::integer(dim_i),
        symbolic::integer(dim_j),
        symbolic::integer(dim_k),
        symbolic::integer(dim_j),
        symbolic::integer(dim_k),
        symbolic::integer(dim_j)
    ));

    auto& alpha_node = builder.add_constant(block, "1.0", desc);
    auto& beta_node = builder.add_constant(block, "0.0", desc);

    builder.add_computational_memlet(block, input_a_node, gemm_node, "__A", {symbolic::integer(0)}, arr_a_type);
    builder.add_computational_memlet(block, input_b_node, gemm_node, "__B", {symbolic::integer(0)}, arr_b_type);
    builder.add_computational_memlet(block, dummy_input_node, gemm_node, "__C", {symbolic::integer(0)}, arr_res_type);
    builder.add_computational_memlet(block, alpha_node, gemm_node, "__alpha", {}, desc);
    builder.add_computational_memlet(block, beta_node, gemm_node, "__beta", {}, desc);

    // Use a local registry so the test is isolated from global plugin state.
    codegen::LibraryNodeDispatcherRegistry local_registry;
    plugins::Context ctx{
        serializer::LibraryNodeSerializerRegistry::instance(),
        codegen::NodeDispatcherRegistry::instance(),
        codegen::MapDispatcherRegistry::instance(),
        codegen::ReduceDispatcherRegistry::instance(),
        local_registry,
        passes::scheduler::SchedulerRegistry::instance()
    };
    rocm::register_rocm_plugin(ctx);

    auto dispatcher_fn = local_registry.get_library_node_dispatcher(
        math::blas::LibraryNodeType_GEMM.value() + "::" + rocm::ImplementationType_ROCMWithoutTransfers.value()
    );
    ASSERT_NE(dispatcher_fn, nullptr);

    codegen::CLanguageExtension language_extension(sdfg);
    auto dispatcher = dispatcher_fn(language_extension, sdfg, block.dataflow(), gemm_node);

    codegen::PrettyPrinter stream;
    codegen::PrettyPrinter globals_stream;
    codegen::CodeSnippetFactory snippet_factory;

    EXPECT_NO_THROW(dispatcher->dispatch(stream, globals_stream, snippet_factory));
}

TEST(RocBlasTest, GemmNodeWithoutDataTransfers_SinglePrecisionUsesHandTuned) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();

    int dim_i = 10;
    int dim_j = 20;
    int dim_k = 30;

    types::Scalar desc(types::PrimitiveType::Float);
    types::Array arr_a_type(desc, symbolic::mul(symbolic::integer(dim_k), symbolic::integer(dim_i)));
    types::Array arr_b_type(desc, symbolic::mul(symbolic::integer(dim_j), symbolic::integer(dim_k)));
    types::Array arr_res_type(desc, symbolic::mul(symbolic::integer(dim_j), symbolic::integer(dim_i)));

    builder.add_container("arr_a", arr_a_type);
    builder.add_container("arr_b", arr_b_type);
    builder.add_container("output", arr_res_type);

    auto& block = builder.add_block(sdfg.root());

    auto& input_a_node = builder.add_access(block, "arr_a");
    auto& input_b_node = builder.add_access(block, "arr_b");
    auto& dummy_input_node = builder.add_access(block, "output");
    auto& gemm_node = static_cast<math::blas::GEMMNode&>(builder.add_library_node<math::blas::GEMMNode>(
        block,
        DebugInfo(),
        rocm::ImplementationType_ROCMWithoutTransfers,
        math::blas::BLAS_Precision::s,
        math::blas::BLAS_Layout::RowMajor,
        math::blas::BLAS_Transpose::No,
        math::blas::BLAS_Transpose::No,
        symbolic::integer(dim_i),
        symbolic::integer(dim_j),
        symbolic::integer(dim_k),
        symbolic::integer(dim_j),
        symbolic::integer(dim_k),
        symbolic::integer(dim_j)
    ));

    builder.add_computational_memlet(block, input_a_node, gemm_node, "__A", {symbolic::integer(0)}, arr_a_type);
    builder.add_computational_memlet(block, input_b_node, gemm_node, "__B", {symbolic::integer(0)}, arr_b_type);
    builder.add_computational_memlet(block, dummy_input_node, gemm_node, "__C", {symbolic::integer(0)}, arr_res_type);
    builder.add_constant(block, "1.0", desc);
    builder.add_constant(block, "0.0", desc);

    codegen::LibraryNodeDispatcherRegistry local_registry;
    plugins::Context ctx{
        serializer::LibraryNodeSerializerRegistry::instance(),
        codegen::NodeDispatcherRegistry::instance(),
        codegen::MapDispatcherRegistry::instance(),
        codegen::ReduceDispatcherRegistry::instance(),
        local_registry,
        passes::scheduler::SchedulerRegistry::instance()
    };
    rocm::register_rocm_plugin(ctx);

    auto dispatcher_fn = local_registry.get_library_node_dispatcher(
        math::blas::LibraryNodeType_GEMM.value() + "::" + rocm::ImplementationType_ROCMWithoutTransfers.value()
    );
    ASSERT_NE(dispatcher_fn, nullptr);

    codegen::CLanguageExtension language_extension(sdfg);
    auto dispatcher = dispatcher_fn(language_extension, sdfg, block.dataflow(), gemm_node);

    // Single precision must be routed to the hand-tuned dispatcher.
    auto* hand_tuned = dynamic_cast<rocm::blas::GEMMNodeDispatcher_ROCMHandTuned*>(dispatcher.get());
    EXPECT_NE(hand_tuned, nullptr);
}


} // namespace sdfg::rocm
