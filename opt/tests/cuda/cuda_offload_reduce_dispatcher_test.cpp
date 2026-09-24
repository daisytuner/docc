#include <gtest/gtest.h>

#include <limits>
#include <optional>
#include <string>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/codegen/language_extensions/c_language_extension.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/passes/offloading/reduction_shared_memory_delinearization.h"
#include "sdfg/serializer/json_serializer.h"
#include "sdfg/structured_control_flow/reduce.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/targets/cuda/cuda.h"
#include "sdfg/targets/cuda/cuda_data_offloading_node.h"
#include "sdfg/targets/cuda/cuda_offload_dispatcher_strategy.h"
#include "sdfg/targets/gpu/gpu_offload_reduce_dispatcher.h"
#include "sdfg/targets/gpu/gpu_offload_schedule_type.h"
#include "sdfg/targets/gpu/gpu_reduce_layout.h"
#include "sdfg/tiles/analysis/reduction_buffer_analysis.h"
#include "sdfg/types/array.h"

namespace sdfg::cuda {

TEST(CUDAOffloadReduceDispatcherTest, PackedLayoutRoundTrip) {
    gpu::ReductionLayout layout(symbolic::integer(35), {{16, 2}, {1, 2}});
    EXPECT_EQ(layout.extent, 4);
    const std::vector<int64_t> addresses{35, 36, 51, 52};
    for (size_t slot = 0; slot < addresses.size(); ++slot) {
        EXPECT_TRUE(symbolic::eq(layout.unpack(symbolic::integer(slot)), symbolic::integer(addresses[slot])));
        EXPECT_TRUE(symbolic::eq(layout.pack(symbolic::integer(addresses[slot])), symbolic::integer(slot)));
    }
}

TEST(CUDAOffloadReduceDispatcherTest, PackedLayoutNestedTilesAndStrides) {
    gpu::ReductionLayout layout(symbolic::zero(), {{32, 2}, {1, 2}, {16, 2}, {2, 2}});
    EXPECT_EQ(layout.extent, 16);
    for (int64_t slot = 0; slot < layout.extent; ++slot) {
        auto address = symbolic::integer(16 * (slot / 4) + slot % 4);
        EXPECT_TRUE(symbolic::eq(layout.unpack(symbolic::integer(slot)), address));
        EXPECT_TRUE(symbolic::eq(layout.pack(address), symbolic::integer(slot)));
    }
    gpu::ReductionLayout strided(symbolic::integer(7), {{5, 3}, {2, 2}});
    EXPECT_EQ(strided.extent, 6);
    for (int64_t slot = 0; slot < strided.extent; ++slot) {
        auto address = strided.unpack(symbolic::integer(slot));
        EXPECT_TRUE(symbolic::eq(strided.pack(address), symbolic::integer(slot)));
    }
}

TEST(CUDAOffloadReduceDispatcherTest, PackedLayoutRejectsAliasingAndOverflow) {
    EXPECT_THROW((gpu::ReductionLayout(symbolic::zero(), {{1, 3}, {2, 2}})), InvalidSDFGException);
    EXPECT_THROW(
        (gpu::ReductionLayout(symbolic::zero(), {{std::numeric_limits<int64_t>::max(), 3}})), InvalidSDFGException
    );
}

TEST(CUDAOffloadReduceDispatcherTest, PackedLayoutScalarAndUnitAxes) {
    auto origin = symbolic::symbol("origin");
    gpu::ReductionLayout scalar(origin, {});
    EXPECT_EQ(scalar.extent, 1);
    EXPECT_TRUE(symbolic::eq(scalar.unpack(symbolic::zero()), origin));
    EXPECT_TRUE(symbolic::eq(scalar.pack(origin), symbolic::zero()));
    gpu::ReductionLayout layout(origin, {{16, 2}, {1, 2}, {1, 1}});
    EXPECT_EQ(layout.extent, 4);
    EXPECT_EQ(layout.dimensions.size(), 2);
    for (int64_t slot = 0; slot < layout.extent; ++slot) {
        auto index = layout.unpack(symbolic::integer(slot));
        EXPECT_TRUE(symbolic::eq(layout.pack(index), symbolic::integer(slot)));
    }
}

TEST(CUDAOffloadReduceDispatcherTest, PackedLayoutInvalidDimensions) {
    EXPECT_THROW((gpu::ReductionLayout(SymEngine::null, {})), InvalidSDFGException);
    for (auto dimension : std::vector<gpu::ReductionLayout::Dimension>{{0, 2}, {-1, 2}, {1, 0}, {1, -1}}) {
        EXPECT_THROW((gpu::ReductionLayout(symbolic::zero(), {dimension})), InvalidSDFGException);
    }
}

// Build a standalone block-level (X_BLOCK) offloaded reduction `acc[0] += A[i]`
// and return the generated kernel body. When @p strategy is set it is written to
// the reduce schedule's partial_storage property; otherwise the level default
// (Shared for a block level) applies. When @p partial_container is non-empty it is
// set as the placed partials-buffer name, optionally also declared as an NV_Shared
// container (@p declare_partials). Optionally return the materialized @p buffer_info.
// With @p materialize disabled, verify that dispatch rejects the unchanged graph.
static std::string dispatch_block_reduce(
    std::optional<gpu::ReduceStrategy> strategy,
    const std::string& partial_container = "",
    bool declare_partials = false,
    bool nested = false,
    tiles::ReductionBufferInfo* buffer_info = nullptr,
    bool materialize = true,
    bool opaque_accumulator = false
) {
    builder::StructuredSDFGBuilder builder("red", FunctionType_CPU);
    auto& root = builder.subject().root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer pointer_type(types::StorageType::NV_Generic(), 0, "", base_desc);
    types::Scalar int_desc(types::PrimitiveType::Int32);

    builder.add_container("i", int_desc);
    builder.add_container("A", pointer_type);
    builder.add_container("acc", pointer_type);
    if (opaque_accumulator) {
        builder.change_type("acc", types::Pointer(types::StorageType::NV_Generic(), 0, ""));
        offloading::add_offloading_block<CUDADataOffloadingNode>(
            builder,
            root,
            "acc",
            "acc",
            offloading::DataTransferDirection::NONE,
            offloading::BufferLifecycle::ALLOC,
            builder.subject().type("acc"),
            DebugInfo(),
            symbolic::integer(4),
            symbolic::zero()
        );
    }
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

    auto* body = &reduce.root();
    if (nested) {
        builder.add_container("nested_index", int_desc);
        auto nested_index = symbolic::symbol("nested_index");
        auto& inner = builder.add_reduce(
            *body,
            nested_index,
            symbolic::Lt(nested_index, symbolic::integer(2)),
            symbolic::zero(),
            symbolic::add(nested_index, symbolic::one()),
            {{structured_control_flow::ReductionOperation::Add, "acc"}},
            gpu::ScheduleType_GPU_Offload::create<
                ScheduleType_CUDA_Offload>(gpu::TargetLevel::Y_BLOCK, symbolic::integer(2))
        );
        body = &inner.root();
    }
    auto& block = builder.add_block(*body);
    auto& a_access = builder.add_access(block, "A");
    auto& acc_in = builder.add_access(block, "acc");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in0", "_in1"});
    auto& acc_out = builder.add_access(block, "acc");
    builder.add_computational_memlet(block, acc_in, tasklet, "_in0", {symbolic::zero()}, pointer_type);
    builder.add_computational_memlet(block, a_access, tasklet, "_in1", {symbolic::symbol("i")}, pointer_type);
    builder.add_computational_memlet(block, tasklet, "_out", acc_out, {symbolic::zero()}, pointer_type);

    analysis::AnalysisManager analysis_manager(builder.subject());
    if (materialize) {
        passes::ReductionSharedMemoryDelinearization reduction_buffers;
        reduction_buffers.run(builder, analysis_manager);
        if (opaque_accumulator) {
            EXPECT_FALSE(static_cast<const types::Pointer&>(builder.subject().type("acc")).has_pointee_type());
            EXPECT_FALSE(reduction_buffers.run(builder, analysis_manager));
            auto clone = builder.subject().clone();
            EXPECT_NO_THROW(clone->validate());
            analysis::AnalysisManager cloned_manager(*clone);
            auto& cloned_reduce = static_cast<structured_control_flow::Reduce&>(clone->root().at(1));
            const auto& cloned_info =
                cloned_manager.get<tiles::ReductionBufferAnalysis>().require(cloned_reduce, "acc");
            EXPECT_EQ(cloned_info.primitive, types::PrimitiveType::Float);
        }
        if (nested) {
            EXPECT_NO_THROW(builder.subject().validate());
            auto clone = builder.subject().clone();
            EXPECT_NO_THROW(clone->validate());
            analysis::AnalysisManager cloned_manager(*clone);
            auto& cloned_reduce =
                static_cast<structured_control_flow::Reduce&>(clone->root().at(opaque_accumulator ? 1 : 0));
            const auto& cloned_info =
                cloned_manager.get<tiles::ReductionBufferAnalysis>().require(cloned_reduce, "acc");
            for (const auto& builtin : {symbolic::threadIdx_x(), symbolic::threadIdx_y(), symbolic::threadIdx_z()}) {
                EXPECT_FALSE(clone->exists(builtin->get_name()));
                EXPECT_TRUE(symbolic::uses(cloned_info.linear_thread_index, builtin));
            }
            for (const auto& container : clone->containers()) {
                EXPECT_FALSE(container.starts_with("__daisy_gpu_reduce_thread_"));
            }
        }
    }
    codegen::CLanguageExtension language_extension(builder.subject());
    auto instrumentation = codegen::InstrumentationPlan::none(builder.subject());
    auto arg_capture = codegen::ArgCapturePlan::none(builder.subject());

    gpu::GPUOffloadReduceDispatcher dispatcher(
        language_extension,
        builder.subject(),
        analysis_manager,
        reduce,
        *instrumentation,
        *arg_capture,
        std::make_unique<CUDAOffloadDispatcherStrategy>(builder.subject())
    );

    codegen::PrettyPrinter main_stream;
    codegen::PrettyPrinter globals_stream;
    codegen::CodeSnippetFactory library_snippet_factory;
    if (opaque_accumulator) {
        auto& allocation = static_cast<structured_control_flow::Block&>(root.at(0));
        auto* malloc_node = *allocation.dataflow().library_nodes().begin();
        CUDADataOffloadingNodeDispatcher
            allocation_dispatcher(language_extension, builder.subject(), allocation.dataflow(), *malloc_node);
        codegen::PrettyPrinter allocation_stream;
        allocation_dispatcher.dispatch(allocation_stream, globals_stream, library_snippet_factory);
        EXPECT_EQ(language_extension.declaration("acc", builder.subject().type("acc")), "void* acc");
        EXPECT_NE(allocation_stream.str().find("void* _dev;"), std::string::npos);
        EXPECT_NE(allocation_stream.str().find("acc = _dev;"), std::string::npos);
    }
    if (!materialize) {
        serializer::JSONSerializer serializer;
        const auto before = serializer.serialize(builder.subject());
        EXPECT_THROW(dispatcher.dispatch_node(main_stream, globals_stream, library_snippet_factory), InvalidSDFGException);
        EXPECT_EQ(serializer.serialize(builder.subject()), before);
        return "";
    }
    dispatcher.dispatch_node(main_stream, globals_stream, library_snippet_factory);

    const auto total = analysis_manager.get<tiles::ReductionBufferAnalysis>().kernel(reduce);
    EXPECT_TRUE(total.shared_bytes.has_value()) << total.diagnostic;
    EXPECT_EQ(total.shared_bytes, strategy == gpu::ReduceStrategy::Global ? 0 : (nested ? 256 : 128));
    if (buffer_info) {
        *buffer_info = analysis_manager.get<tiles::ReductionBufferAnalysis>().require(reduce, "acc");
    }

    EXPECT_EQ(library_snippet_factory.snippets().size(), 2u);
    const codegen::CodeSnippet* source_snippet = nullptr;
    const codegen::CodeSnippet* header_snippet = nullptr;
    for (auto& [key, snippet] : library_snippet_factory.snippets()) {
        if (snippet.extension() == "cu.h") {
            header_snippet = &snippet;
        } else if (snippet.extension() == "cu") {
            source_snippet = &snippet;
        }
    }
    EXPECT_NE(source_snippet, nullptr);
    EXPECT_NE(header_snippet, nullptr);
    EXPECT_EQ(globals_stream.str().find("threadIdx."), std::string::npos);
    if (opaque_accumulator) {
        EXPECT_NE(globals_stream.str().find("void* __restrict__ acc"), std::string::npos);
        EXPECT_NE(source_snippet->stream().str().find("reinterpret_cast<float *>(acc)[0]"), std::string::npos);
    }
    return source_snippet->stream().str();
}

// Default block level → shared-memory halving tree, no atomics.
TEST(CUDAOffloadReduceDispatcherTest, UnmaterializedReductionIsRejectedWithoutMutation) {
    dispatch_block_reduce(std::nullopt, "", false, false, nullptr, false);
}

TEST(CUDAOffloadReduceDispatcherTest, MaterializationPreservesOpaqueAccumulator) {
    dispatch_block_reduce(std::nullopt, "", false, false, nullptr, true, true);
}

TEST(CUDAOffloadReduceDispatcherTest, NestedSharedPreservesOpaqueAccumulator) {
    dispatch_block_reduce(std::nullopt, "", false, true, nullptr, true, true);
}

TEST(CUDAOffloadReduceDispatcherTest, GlobalPreservesOpaqueAccumulator) {
    dispatch_block_reduce(gpu::ReduceStrategy::Global, "", false, false, nullptr, true, true);
}

TEST(CUDAOffloadReduceDispatcherTest, BlockLevelDefaultsToSharedTree) {
    tiles::ReductionBufferInfo info;
    std::string kernel = dispatch_block_reduce(std::nullopt, "", false, false, &info);

    ASSERT_FALSE(info.shared_buffer.empty());
    EXPECT_NE(kernel.find("__shared__ float " + info.shared_buffer + "[32]"), std::string::npos);
    EXPECT_NE(kernel.find("__syncthreads()"), std::string::npos);
    // The block result is assigned (single owner), not atomically merged.
    EXPECT_EQ(kernel.find("atomicAdd"), std::string::npos);
}

TEST(CUDAOffloadReduceDispatcherTest, NestedSharedUsesBuiltinThreadIndices) {
    tiles::ReductionBufferInfo info;
    auto kernel = dispatch_block_reduce(std::nullopt, "", false, true, &info);
    ASSERT_FALSE(info.shared_buffer.empty());
    EXPECT_NE(kernel.find("__shared__ float " + info.shared_buffer + "[64]"), std::string::npos);
    EXPECT_NE(kernel.find("threadIdx.x"), std::string::npos);
    EXPECT_NE(kernel.find("threadIdx.y"), std::string::npos);
    EXPECT_EQ(kernel.find("__daisy_gpu_reduce_thread_"), std::string::npos);
}

// Explicit Shared strategy at a block level reproduces the default output.
TEST(CUDAOffloadReduceDispatcherTest, BlockLevelExplicitSharedMatchesDefault) {
    EXPECT_EQ(dispatch_block_reduce(gpu::ReduceStrategy::Shared), dispatch_block_reduce(std::nullopt));
}

// Global override at a block level → per-thread register merged via atomics, no
// shared buffer and no single-committer guard (every thread holds a distinct
// reduce-axis partial, so all of them commit).
TEST(CUDAOffloadReduceDispatcherTest, BlockLevelGlobalUsesAtomicsNoShared) {
    tiles::ReductionBufferInfo info;
    std::string kernel = dispatch_block_reduce(gpu::ReduceStrategy::Global, "", false, false, &info);

    EXPECT_EQ(kernel.find("__shared__"), std::string::npos);
    ASSERT_FALSE(info.private_buffer.empty());
    EXPECT_NE(kernel.find("float " + info.private_buffer + "[1];"), std::string::npos);
    EXPECT_NE(
        kernel.find("atomicAdd(&reinterpret_cast<float *>(acc)[0], " + info.private_buffer + "[0]);"), std::string::npos
    );
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
    gpu::GPUOffloadReduceDispatcher dispatcher(
        language_extension,
        builder.subject(),
        analysis_manager,
        reduce,
        *instrumentation,
        *arg_capture,
        std::make_unique<CUDAOffloadDispatcherStrategy>(builder.subject())
    );
    codegen::PrettyPrinter main_stream, globals_stream;
    codegen::CodeSnippetFactory library_snippet_factory;
    EXPECT_THROW(dispatcher.dispatch_node(main_stream, globals_stream, library_snippet_factory), InvalidSDFGException);
}

} // namespace sdfg::cuda
