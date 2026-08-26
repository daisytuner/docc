#include <gtest/gtest.h>

#include <optional>
#include <string>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/codegen/language_extensions/c_language_extension.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/structured_control_flow/reduce.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/targets/cuda/cuda.h"
#include "sdfg/targets/cuda/cuda_offload_reduce_dispatcher.h"
#include "sdfg/targets/gpu/gpu_offload_schedule_type.h"
#include "sdfg/types/array.h"

namespace sdfg::cuda {

// Build a standalone block-level (X_BLOCK) offloaded reduction `acc[0] += A[i]`
// and return the generated kernel body. When @p strategy is set it is written to
// the reduce schedule's partial_storage property; otherwise the level default
// (Shared for a block level) applies. When @p partial_container is non-empty it is
// set as the placed partials-buffer name, optionally also declared as an NV_Shared
// container (@p declare_partials).
static std::string dispatch_block_reduce(
    std::optional<gpu::ReduceStrategy> strategy,
    const std::string& partial_container = "",
    bool declare_partials = false
) {
    builder::StructuredSDFGBuilder builder("red", FunctionType_CPU);
    auto& root = builder.subject().root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer pointer_type(types::StorageType::NV_Generic(), 0, "", base_desc);
    types::Scalar int_desc(types::PrimitiveType::Int32);

    builder.add_container("i", int_desc);
    builder.add_container("A", pointer_type);
    builder.add_container("acc", pointer_type);
    if (declare_partials) {
        types::Array partials_type(types::StorageType::NV_Shared(), 0, "", base_desc, symbolic::integer(32));
        builder.add_container(partial_container, partials_type);
    }

    auto schedule = gpu::ScheduleType_GPU_Offload::create<
        ScheduleType_CUDA_Offload>(gpu::TargetLevel::X_BLOCK, symbolic::integer(32));
    if (strategy.has_value()) {
        gpu::ScheduleType_GPU_Offload::partial_storage(schedule, *strategy);
    }
    if (!partial_container.empty()) {
        gpu::ScheduleType_GPU_Offload::partial_container(schedule, partial_container);
    }

    auto& reduce = builder.add_reduce(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::integer(100)),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1)),
        {structured_control_flow::ReductionInfo{structured_control_flow::ReductionOperation::Add, "acc"}},
        schedule
    );

    auto& block = builder.add_block(reduce.root());
    auto& a_access = builder.add_access(block, "A");
    auto& acc_in = builder.add_access(block, "acc");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in0", "_in1"});
    auto& acc_out = builder.add_access(block, "acc");
    builder.add_computational_memlet(block, acc_in, tasklet, "_in0", {symbolic::zero()}, pointer_type);
    builder.add_computational_memlet(block, a_access, tasklet, "_in1", {symbolic::symbol("i")}, pointer_type);
    builder.add_computational_memlet(block, tasklet, "_out", acc_out, {symbolic::zero()}, pointer_type);

    codegen::CLanguageExtension language_extension(builder.subject());
    auto instrumentation = codegen::InstrumentationPlan::none(builder.subject());
    auto arg_capture = codegen::ArgCapturePlan::none(builder.subject());
    analysis::AnalysisManager analysis_manager(builder.subject());

    CUDAOffloadReduceDispatcher
        dispatcher(language_extension, builder.subject(), analysis_manager, reduce, *instrumentation, *arg_capture);

    codegen::PrettyPrinter main_stream;
    codegen::PrettyPrinter globals_stream;
    codegen::CodeSnippetFactory library_snippet_factory;
    dispatcher.dispatch_node(main_stream, globals_stream, library_snippet_factory);

    EXPECT_EQ(library_snippet_factory.snippets().size(), 1u);
    return library_snippet_factory.snippets().begin()->second.stream().str();
}

// Default block level → shared-memory halving tree, no atomics.
TEST(CUDAOffloadReduceDispatcherTest, BlockLevelDefaultsToSharedTree) {
    std::string kernel = dispatch_block_reduce(std::nullopt);

    EXPECT_NE(kernel.find("__shared__ float __daisy_reduce_smem_acc[32]"), std::string::npos);
    EXPECT_NE(kernel.find("__syncthreads()"), std::string::npos);
    // The block result is assigned (single owner), not atomically merged.
    EXPECT_EQ(kernel.find("atomicAdd"), std::string::npos);
}

// Explicit Shared strategy at a block level reproduces the default output.
TEST(CUDAOffloadReduceDispatcherTest, BlockLevelExplicitSharedMatchesDefault) {
    EXPECT_EQ(dispatch_block_reduce(gpu::ReduceStrategy::Shared), dispatch_block_reduce(std::nullopt));
}

// Global override at a block level → per-thread register merged via atomics, no
// shared buffer and no single-committer guard (every thread holds a distinct
// reduce-axis partial, so all of them commit).
TEST(CUDAOffloadReduceDispatcherTest, BlockLevelGlobalUsesAtomicsNoShared) {
    std::string kernel = dispatch_block_reduce(gpu::ReduceStrategy::Global);

    EXPECT_EQ(kernel.find("__shared__"), std::string::npos);
    EXPECT_NE(kernel.find("float __daisy_reduce_reg_acc = 0;"), std::string::npos);
    EXPECT_NE(kernel.find("atomicAdd(&reinterpret_cast<float *>(acc)[0], __daisy_reduce_reg_acc);"), std::string::npos);
    // Every thread commits (each holds a distinct reduce-axis partial): no single-committer guard.
    EXPECT_EQ(kernel.find("threadIdx.x == 0"), std::string::npos);
    // No halving tree (the shared-tree bound variable is never emitted).
    EXPECT_EQ(kernel.find("__daisy_reduce_m_acc"), std::string::npos);
}

// Register strategy is warp-only; a block level must reject it loudly.
TEST(CUDAOffloadReduceDispatcherTest, RegisterAtBlockLevelThrows) {
    EXPECT_THROW(dispatch_block_reduce(gpu::ReduceStrategy::Register), InvalidSDFGException);
}

// A placed partial_container renames the shared buffer; the invented default name
// no longer appears, and the placed name is declared once and addressed by the tree.
TEST(CUDAOffloadReduceDispatcherTest, PlacedPartialContainerRenamesBuffer) {
    std::string kernel = dispatch_block_reduce(std::nullopt, "__daisy_reduce_myBuf", /*declare_partials*/ true);

    EXPECT_EQ(kernel.find("__daisy_reduce_smem_acc"), std::string::npos);
    EXPECT_NE(kernel.find("__shared__ float __daisy_reduce_myBuf[32]"), std::string::npos);
    // Declared exactly once (no clash with a scope-variable declaration path), and
    // referenced again by init + tree.
    size_t first_decl = kernel.find("__shared__ float __daisy_reduce_myBuf[32]");
    EXPECT_EQ(kernel.find("__shared__ float __daisy_reduce_myBuf[32]", first_decl + 1), std::string::npos);
    size_t count = 0;
    for (size_t p = kernel.find("__daisy_reduce_myBuf"); p != std::string::npos;
         p = kernel.find("__daisy_reduce_myBuf", p + 1)) {
        count++;
    }
    EXPECT_GE(count, 3u); // declaration + identity init + halving tree
}

// A metadata-only placed name (no declared container) still just renames the buffer.
TEST(CUDAOffloadReduceDispatcherTest, PlacedPartialContainerWithoutDeclarationRenames) {
    std::string kernel = dispatch_block_reduce(std::nullopt, "__daisy_reduce_myBuf", /*declare_partials*/ false);
    EXPECT_EQ(kernel.find("__daisy_reduce_smem_acc"), std::string::npos);
    EXPECT_NE(kernel.find("__shared__ float __daisy_reduce_myBuf[32]"), std::string::npos);
}

// partial_container is a Shared-only concept: pairing it with Global must throw.
TEST(CUDAOffloadReduceDispatcherTest, PlacedPartialContainerWithGlobalThrows) {
    EXPECT_THROW(dispatch_block_reduce(gpu::ReduceStrategy::Global, "__daisy_reduce_myBuf", false), InvalidSDFGException);
}

// A placed container that exists but is not NV_Shared is rejected.
TEST(CUDAOffloadReduceDispatcherTest, PlacedPartialContainerWrongStorageThrows) {
    builder::StructuredSDFGBuilder builder("red", FunctionType_CPU);
    auto& root = builder.subject().root();
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer pointer_type(types::StorageType::NV_Generic(), 0, "", base_desc);
    types::Scalar int_desc(types::PrimitiveType::Int32);
    builder.add_container("i", int_desc);
    builder.add_container("A", pointer_type);
    builder.add_container("acc", pointer_type);
    // CPU_Stack (not NV_Shared) partials container.
    types::Array partials_type(base_desc, symbolic::integer(32));
    builder.add_container("__daisy_reduce_myBuf", partials_type);

    auto schedule = gpu::ScheduleType_GPU_Offload::create<
        ScheduleType_CUDA_Offload>(gpu::TargetLevel::X_BLOCK, symbolic::integer(32));
    gpu::ScheduleType_GPU_Offload::partial_container(schedule, "__daisy_reduce_myBuf");

    auto& reduce = builder.add_reduce(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::integer(100)),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1)),
        {structured_control_flow::ReductionInfo{structured_control_flow::ReductionOperation::Add, "acc"}},
        schedule
    );
    auto& block = builder.add_block(reduce.root());
    auto& a_access = builder.add_access(block, "A");
    auto& acc_in = builder.add_access(block, "acc");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in0", "_in1"});
    auto& acc_out = builder.add_access(block, "acc");
    builder.add_computational_memlet(block, acc_in, tasklet, "_in0", {symbolic::zero()}, pointer_type);
    builder.add_computational_memlet(block, a_access, tasklet, "_in1", {symbolic::symbol("i")}, pointer_type);
    builder.add_computational_memlet(block, tasklet, "_out", acc_out, {symbolic::zero()}, pointer_type);

    codegen::CLanguageExtension language_extension(builder.subject());
    auto instrumentation = codegen::InstrumentationPlan::none(builder.subject());
    auto arg_capture = codegen::ArgCapturePlan::none(builder.subject());
    analysis::AnalysisManager analysis_manager(builder.subject());
    CUDAOffloadReduceDispatcher
        dispatcher(language_extension, builder.subject(), analysis_manager, reduce, *instrumentation, *arg_capture);
    codegen::PrettyPrinter main_stream, globals_stream;
    codegen::CodeSnippetFactory library_snippet_factory;
    EXPECT_THROW(dispatcher.dispatch_node(main_stream, globals_stream, library_snippet_factory), InvalidSDFGException);
}

} // namespace sdfg::cuda
