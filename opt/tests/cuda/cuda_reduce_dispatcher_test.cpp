#include <gtest/gtest.h>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/codegen/language_extensions/c_language_extension.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/structured_control_flow/reduce.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/targets/cuda/cuda.h"
#include "sdfg/targets/cuda/cuda_reduce_dispatcher.h"

namespace sdfg::cuda {

TEST(CUDAReduceDispatcherTest, AtomicSumKernel) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto& root = builder.subject().root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer pointer_type(types::StorageType::NV_Generic(), 0, "", base_desc);
    types::Scalar int_desc(types::PrimitiveType::Int32);

    builder.add_container("i", int_desc);
    builder.add_container("__daisy_cuda_A", pointer_type);
    builder.add_container("__daisy_cuda_acc", pointer_type);

    auto cuda_schedule = ScheduleType_CUDA::create();
    ScheduleType_CUDA::dimension(cuda_schedule, CUDADimension::X);
    ScheduleType_CUDA::block_size(cuda_schedule, symbolic::integer(32));

    auto condition = symbolic::Lt(symbolic::symbol("i"), symbolic::integer(100));
    auto init = symbolic::integer(0);
    auto update = symbolic::add(symbolic::symbol("i"), symbolic::integer(1));

    auto& reduce = builder.add_reduce(
        root,
        symbolic::symbol("i"),
        condition,
        init,
        update,
        {structured_control_flow::ReductionInfo{structured_control_flow::ReductionOperation::Add, "__daisy_cuda_acc"}},
        cuda_schedule
    );

    // Body: acc[0] = acc[0] + A[i]
    auto& block = builder.add_block(reduce.root());
    auto& a_access = builder.add_access(block, "__daisy_cuda_A");
    auto& acc_in = builder.add_access(block, "__daisy_cuda_acc");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in0", "_in1"});
    auto& acc_out = builder.add_access(block, "__daisy_cuda_acc");

    builder.add_computational_memlet(block, acc_in, tasklet, "_in0", {symbolic::zero()}, pointer_type);
    builder.add_computational_memlet(block, a_access, tasklet, "_in1", {symbolic::symbol("i")}, pointer_type);
    builder.add_computational_memlet(block, tasklet, "_out", acc_out, {symbolic::zero()}, pointer_type);

    codegen::CLanguageExtension language_extension(builder.subject());
    auto instrumentation = codegen::InstrumentationPlan::none(builder.subject());
    auto arg_capture = codegen::ArgCapturePlan::none(builder.subject());
    analysis::AnalysisManager analysis_manager(builder.subject());

    CUDAReduceDispatcher
        dispatcher(language_extension, builder.subject(), analysis_manager, reduce, *instrumentation, *arg_capture);

    codegen::PrettyPrinter main_stream;
    codegen::PrettyPrinter globals_stream;
    codegen::CodeSnippetFactory library_snippet_factory;

    dispatcher.dispatch_node(main_stream, globals_stream, library_snippet_factory);

    std::string kernel_name = "kernel_test_sdfg_1";

    // One kernel snippet produced.
    EXPECT_EQ(library_snippet_factory.snippets().size(), 1u);
    EXPECT_TRUE(library_snippet_factory.snippets().count(kernel_name));

    // Kernel declaration emitted to globals.
    EXPECT_NE(globals_stream.str().find("__global__ void " + kernel_name), std::string::npos);

    // Host launch.
    std::string host = main_stream.str();
    EXPECT_NE(host.find(kernel_name + "<<<"), std::string::npos);

    // Kernel body checks.
    std::string kernel = library_snippet_factory.snippets().at(kernel_name).stream().str();
    // grid-stride parameters on the X dimension
    EXPECT_NE(kernel.find("int __daisy_reduce_tid = threadIdx.x + blockIdx.x*blockDim.x;"), std::string::npos);
    EXPECT_NE(kernel.find("int __daisy_reduce_nthreads = blockDim.x*gridDim.x;"), std::string::npos);
    // thread-private partial initialized to identity
    EXPECT_NE(kernel.find("float __daisy_reduce___daisy_cuda_acc = 0;"), std::string::npos);
    // accumulator shadow redirects body combine to private storage
    EXPECT_NE(kernel.find("float *__daisy_cuda_acc = &__daisy_reduce___daisy_cuda_acc;"), std::string::npos);
    // grid-stride reduce loop
    EXPECT_NE(kernel.find("for (int i = __daisy_reduce_tid; i < 100; i = __daisy_reduce_nthreads + i)"), std::string::npos);
    // native atomicAdd merge into the real device accumulator (float Add fast path)
    EXPECT_NE(
        kernel.find("atomicAdd(&(reinterpret_cast<float *>(__daisy_cuda_acc))[0], __daisy_reduce___daisy_cuda_acc);"),
        std::string::npos
    );

    // Float Add uses the native atomic; no CAS combine helper should be emitted.
    for (auto& g : library_snippet_factory.globals_snippets()) {
        EXPECT_EQ(g.find("__daisy_reduce_combine_add_float"), std::string::npos);
    }
}

TEST(CUDAReduceDispatcherTest, NestedReduceInlinesGridStride) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto& root = builder.subject().root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer pointer_type(types::StorageType::NV_Generic(), 0, "", base_desc);
    types::Scalar int_desc(types::PrimitiveType::Int32);

    builder.add_container("i", int_desc);
    builder.add_container("j", int_desc);
    builder.add_container("__daisy_cuda_A", pointer_type);
    builder.add_container("__daisy_cuda_acc", pointer_type);

    // Enclosing CUDA map over i on the X dimension.
    auto map_schedule = ScheduleType_CUDA::create();
    ScheduleType_CUDA::dimension(map_schedule, CUDADimension::X);
    ScheduleType_CUDA::block_size(map_schedule, symbolic::integer(32));
    auto& map = builder.add_map(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::integer(64)),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1)),
        map_schedule
    );

    // Nested CUDA reduce over j on the Y dimension.
    auto reduce_schedule = ScheduleType_CUDA::create();
    ScheduleType_CUDA::dimension(reduce_schedule, CUDADimension::Y);
    ScheduleType_CUDA::block_size(reduce_schedule, symbolic::integer(32));
    auto& reduce = builder.add_reduce(
        map.root(),
        symbolic::symbol("j"),
        symbolic::Lt(symbolic::symbol("j"), symbolic::integer(100)),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("j"), symbolic::integer(1)),
        {structured_control_flow::ReductionInfo{structured_control_flow::ReductionOperation::Add, "__daisy_cuda_acc"}},
        reduce_schedule
    );

    // Body: acc[0] = acc[0] + A[j]
    auto& block = builder.add_block(reduce.root());
    auto& a_access = builder.add_access(block, "__daisy_cuda_A");
    auto& acc_in = builder.add_access(block, "__daisy_cuda_acc");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in0", "_in1"});
    auto& acc_out = builder.add_access(block, "__daisy_cuda_acc");

    builder.add_computational_memlet(block, acc_in, tasklet, "_in0", {symbolic::zero()}, pointer_type);
    builder.add_computational_memlet(block, a_access, tasklet, "_in1", {symbolic::symbol("j")}, pointer_type);
    builder.add_computational_memlet(block, tasklet, "_out", acc_out, {symbolic::zero()}, pointer_type);

    codegen::CLanguageExtension language_extension(builder.subject());
    auto instrumentation = codegen::InstrumentationPlan::none(builder.subject());
    auto arg_capture = codegen::ArgCapturePlan::none(builder.subject());
    analysis::AnalysisManager analysis_manager(builder.subject());

    CUDAReduceDispatcher
        dispatcher(language_extension, builder.subject(), analysis_manager, reduce, *instrumentation, *arg_capture);

    codegen::PrettyPrinter main_stream;
    codegen::PrettyPrinter globals_stream;
    codegen::CodeSnippetFactory library_snippet_factory;

    dispatcher.dispatch_node(main_stream, globals_stream, library_snippet_factory);

    std::string inlined = main_stream.str();

    // No kernel snippet and no nested launch: the reduce is inlined.
    EXPECT_EQ(library_snippet_factory.snippets().size(), 0u);
    EXPECT_EQ(inlined.find("<<<"), std::string::npos);

    // Grid-stride parameters on the reduce's own Y dimension.
    EXPECT_NE(inlined.find("int __daisy_reduce_tid = threadIdx.y + blockIdx.y*blockDim.y;"), std::string::npos);
    EXPECT_NE(inlined.find("int __daisy_reduce_nthreads = blockDim.y*gridDim.y;"), std::string::npos);
    // Private partial + shadow + grid-stride loop over j.
    EXPECT_NE(inlined.find("float __daisy_reduce___daisy_cuda_acc = 0;"), std::string::npos);
    EXPECT_NE(inlined.find("float *__daisy_cuda_acc = &__daisy_reduce___daisy_cuda_acc;"), std::string::npos);
    EXPECT_NE(
        inlined.find("for (int j = __daisy_reduce_tid; j < 100; j = __daisy_reduce_nthreads + j)"), std::string::npos
    );
    // Native atomicAdd merge.
    EXPECT_NE(
        inlined.find("atomicAdd(&(reinterpret_cast<float *>(__daisy_cuda_acc))[0], __daisy_reduce___daisy_cuda_acc);"),
        std::string::npos
    );
}

TEST(CUDAReduceDispatcherTest, IndexedAccumulatorAtomicAtSlot) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto& root = builder.subject().root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer pointer_type(types::StorageType::NV_Generic(), 0, "", base_desc);
    types::Scalar int_desc(types::PrimitiveType::Int32);

    builder.add_container("i", int_desc);
    builder.add_container("j", int_desc);
    builder.add_container("__daisy_cuda_A", pointer_type);
    builder.add_container("__daisy_cuda_acc", pointer_type);

    // Enclosing data-parallel CUDA map over i on the X dimension.
    auto map_schedule = ScheduleType_CUDA::create();
    ScheduleType_CUDA::dimension(map_schedule, CUDADimension::X);
    ScheduleType_CUDA::block_size(map_schedule, symbolic::integer(32));
    auto& map = builder.add_map(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::integer(64)),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1)),
        map_schedule
    );

    // Nested CUDA reduce over j on the Y dimension, accumulating into acc[i].
    auto reduce_schedule = ScheduleType_CUDA::create();
    ScheduleType_CUDA::dimension(reduce_schedule, CUDADimension::Y);
    ScheduleType_CUDA::block_size(reduce_schedule, symbolic::integer(32));
    auto& reduce = builder.add_reduce(
        map.root(),
        symbolic::symbol("j"),
        symbolic::Lt(symbolic::symbol("j"), symbolic::integer(100)),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("j"), symbolic::integer(1)),
        {structured_control_flow::ReductionInfo{structured_control_flow::ReductionOperation::Add, "__daisy_cuda_acc"}},
        reduce_schedule
    );

    // Body: acc[i] = acc[i] + A[j]
    auto& block = builder.add_block(reduce.root());
    auto& a_access = builder.add_access(block, "__daisy_cuda_A");
    auto& acc_in = builder.add_access(block, "__daisy_cuda_acc");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in0", "_in1"});
    auto& acc_out = builder.add_access(block, "__daisy_cuda_acc");

    builder.add_computational_memlet(block, acc_in, tasklet, "_in0", {symbolic::symbol("i")}, pointer_type);
    builder.add_computational_memlet(block, a_access, tasklet, "_in1", {symbolic::symbol("j")}, pointer_type);
    builder.add_computational_memlet(block, tasklet, "_out", acc_out, {symbolic::symbol("i")}, pointer_type);

    codegen::CLanguageExtension language_extension(builder.subject());
    auto instrumentation = codegen::InstrumentationPlan::none(builder.subject());
    auto arg_capture = codegen::ArgCapturePlan::none(builder.subject());
    analysis::AnalysisManager analysis_manager(builder.subject());

    CUDAReduceDispatcher
        dispatcher(language_extension, builder.subject(), analysis_manager, reduce, *instrumentation, *arg_capture);

    codegen::PrettyPrinter main_stream;
    codegen::PrettyPrinter globals_stream;
    codegen::CodeSnippetFactory library_snippet_factory;

    dispatcher.dispatch_node(main_stream, globals_stream, library_snippet_factory);

    std::string inlined = main_stream.str();

    // Inlined (nested) reduction: no launch / snippet.
    EXPECT_EQ(library_snippet_factory.snippets().size(), 0u);
    EXPECT_EQ(inlined.find("<<<"), std::string::npos);

    // Privatization shadow is offset by the accumulator index so the body's
    // acc[i] resolves to the single private register.
    EXPECT_NE(inlined.find("float *__daisy_cuda_acc = &__daisy_reduce___daisy_cuda_acc - (i);"), std::string::npos);

    // Grid-stride reduce over j on the Y dimension.
    EXPECT_NE(
        inlined.find("for (int j = __daisy_reduce_tid; j < 100; j = __daisy_reduce_nthreads + j)"), std::string::npos
    );

    // Atomic merge into the indexed slot acc[i].
    EXPECT_NE(
        inlined.find("atomicAdd(&(reinterpret_cast<float *>(__daisy_cuda_acc))[i], __daisy_reduce___daisy_cuda_acc);"),
        std::string::npos
    );
}

TEST(CUDAReduceDispatcherTest, MulReductionUsesCASHelper) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto& root = builder.subject().root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer pointer_type(types::StorageType::NV_Generic(), 0, "", base_desc);
    types::Scalar int_desc(types::PrimitiveType::Int32);

    builder.add_container("i", int_desc);
    builder.add_container("__daisy_cuda_A", pointer_type);
    builder.add_container("__daisy_cuda_acc", pointer_type);

    auto cuda_schedule = ScheduleType_CUDA::create();
    ScheduleType_CUDA::dimension(cuda_schedule, CUDADimension::X);
    ScheduleType_CUDA::block_size(cuda_schedule, symbolic::integer(32));

    auto& reduce = builder.add_reduce(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::integer(100)),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1)),
        {structured_control_flow::ReductionInfo{structured_control_flow::ReductionOperation::Mul, "__daisy_cuda_acc"}},
        cuda_schedule
    );

    auto& block = builder.add_block(reduce.root());
    auto& a_access = builder.add_access(block, "__daisy_cuda_A");
    auto& acc_in = builder.add_access(block, "__daisy_cuda_acc");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_mul, "_out", {"_in0", "_in1"});
    auto& acc_out = builder.add_access(block, "__daisy_cuda_acc");

    builder.add_computational_memlet(block, acc_in, tasklet, "_in0", {symbolic::zero()}, pointer_type);
    builder.add_computational_memlet(block, a_access, tasklet, "_in1", {symbolic::symbol("i")}, pointer_type);
    builder.add_computational_memlet(block, tasklet, "_out", acc_out, {symbolic::zero()}, pointer_type);

    codegen::CLanguageExtension language_extension(builder.subject());
    auto instrumentation = codegen::InstrumentationPlan::none(builder.subject());
    auto arg_capture = codegen::ArgCapturePlan::none(builder.subject());
    analysis::AnalysisManager analysis_manager(builder.subject());

    CUDAReduceDispatcher
        dispatcher(language_extension, builder.subject(), analysis_manager, reduce, *instrumentation, *arg_capture);

    codegen::PrettyPrinter main_stream;
    codegen::PrettyPrinter globals_stream;
    codegen::CodeSnippetFactory library_snippet_factory;

    dispatcher.dispatch_node(main_stream, globals_stream, library_snippet_factory);

    std::string kernel = library_snippet_factory.snippets().at("kernel_test_sdfg_1").stream().str();
    // Non-Add operators call the CAS combine helper (defined in daisy_rtl.h).
    EXPECT_NE(kernel.find("__daisy_reduce_combine_mul_float("), std::string::npos);
    EXPECT_EQ(kernel.find("atomicAdd("), std::string::npos);

    // The helper is no longer emitted inline; it lives in daisy_rtl.h, so no
    // global snippet should contain a CAS-based combine definition.
    for (auto& g : library_snippet_factory.globals_snippets()) {
        EXPECT_EQ(g.find("atomicCAS"), std::string::npos);
    }
}

} // namespace sdfg::cuda
