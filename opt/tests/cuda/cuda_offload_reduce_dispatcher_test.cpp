#include <gtest/gtest.h>

#include <limits>
#include <optional>
#include <string>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/codegen/language_extensions/c_language_extension.h"
#include "sdfg/data_flow/library_nodes/math/cmath/cmath_node.h"
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
    if (!nested) {
        reduce.original_index("acc", symbolic::integer(7));
        analysis_manager.invalidate_all();
        codegen::PrettyPrinter refreshed_main;
        codegen::PrettyPrinter refreshed_globals;
        codegen::CodeSnippetFactory refreshed_snippets;
        dispatcher.dispatch_node(refreshed_main, refreshed_globals, refreshed_snippets);
        bool found_kernel = false;
        for (auto& [key, snippet] : refreshed_snippets.snippets()) {
            if (snippet.extension() == "cu") {
                found_kernel = true;
                EXPECT_NE(snippet.stream().str().find("reinterpret_cast<float *>(acc)[7]"), std::string::npos);
            }
        }
        EXPECT_TRUE(found_kernel);
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
    // A single-warp block combines entirely by shuffles: no halving tree, no barrier.
    EXPECT_NE(kernel.find("__shfl_xor_sync"), std::string::npos);
    EXPECT_EQ(kernel.find("__daisy_reduce_m_acc"), std::string::npos);
    EXPECT_EQ(kernel.find("__syncthreads()"), std::string::npos);
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
    // Declared exactly once (no clash with a scope-variable declaration path).
    size_t first_decl = kernel.find("__shared__ float __daisy_reduce_myBuf[32]");
    EXPECT_EQ(kernel.find("__shared__ float __daisy_reduce_myBuf[32]", first_decl + 1), std::string::npos);
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

namespace {

void add_accumulate(
    builder::StructuredSDFGBuilder& builder,
    structured_control_flow::Sequence& scope,
    const std::string& acc,
    const symbolic::Expression& acc_index,
    const types::IType& pointer_type
) {
    auto& block = builder.add_block(scope);
    auto& a_access = builder.add_access(block, "A");
    auto& acc_in = builder.add_access(block, acc);
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in0", "_in1"});
    auto& acc_out = builder.add_access(block, acc);
    builder.add_computational_memlet(block, acc_in, tasklet, "_in0", {acc_index}, pointer_type);
    builder.add_computational_memlet(block, a_access, tasklet, "_in1", {symbolic::symbol("i")}, pointer_type);
    builder.add_computational_memlet(block, tasklet, "_out", acc_out, {acc_index}, pointer_type);
}

// Block-level reduce carrying the max anchor `m[0]` and `d[0]` rescaled by exp(m); with
// @p indexed, also an `o[j]` (j < 8) dependent like a FlashAttention output row. With
// @p body_exp, the body also evaluates a CMath exp with that approximate flag.
std::string dispatch_rescale_reduce(
    int64_t block_threads,
    bool indexed,
    gpu::TargetLevel level = gpu::TargetLevel::X_BLOCK,
    std::optional<bool> body_exp = std::nullopt
) {
    builder::StructuredSDFGBuilder builder("rescale", FunctionType_CPU);
    auto& root = builder.subject().root();
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer pointer_type(types::StorageType::NV_Generic(), 0, "", base_desc);
    builder.add_container("i", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("j", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("t", base_desc);
    for (const auto* name : {"A", "m", "d", "o"}) {
        builder.add_container(name, pointer_type);
    }

    std::vector<structured_control_flow::ReductionInfo> reductions{
        {.operation = structured_control_flow::ReductionOperation::Max, .container = "m"},
        {.operation = structured_control_flow::ReductionOperation::Add, .container = "d", .rescale = "m"},
    };
    if (indexed) {
        reductions
            .push_back({.operation = structured_control_flow::ReductionOperation::Add, .container = "o", .rescale = "m"});
    }
    auto& reduce = builder.add_reduce(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::integer(1024)),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1)),
        reductions,
        gpu::ScheduleType_GPU_Offload::create<ScheduleType_CUDA_Offload>(level, symbolic::integer(block_threads))
    );
    add_accumulate(builder, reduce.root(), "m", symbolic::zero(), pointer_type);
    add_accumulate(builder, reduce.root(), "d", symbolic::zero(), pointer_type);
    if (body_exp.has_value()) {
        auto& block = builder.add_block(reduce.root());
        auto& exp_node = static_cast<math::cmath::CMathNode&>(builder.add_library_node<math::cmath::CMathNode>(
            block, DebugInfo(), math::cmath::CMathFunction::exp, types::PrimitiveType::Float
        ));
        exp_node.approximate(*body_exp);
        builder.add_computational_memlet(
            block, builder.add_access(block, "A"), exp_node, "_in1", {symbolic::symbol("i")}, pointer_type
        );
        builder.add_computational_memlet(block, exp_node, "_out", builder.add_access(block, "t"), {}, base_desc);
    }
    if (indexed) {
        auto j = symbolic::symbol("j");
        auto& map = builder.add_map(
            reduce.root(),
            j,
            symbolic::Lt(j, symbolic::integer(8)),
            symbolic::zero(),
            symbolic::add(j, symbolic::one()),
            structured_control_flow::ScheduleType_Sequential::create()
        );
        add_accumulate(builder, map.root(), "o", j, pointer_type);
    }

    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::ReductionSharedMemoryDelinearization reduction_buffers;
    reduction_buffers.run(builder, analysis_manager);
    builder.subject().validate();

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
    codegen::PrettyPrinter main_stream, globals_stream;
    codegen::CodeSnippetFactory library_snippet_factory;
    dispatcher.dispatch_node(main_stream, globals_stream, library_snippet_factory);
    for (auto& [key, snippet] : library_snippet_factory.snippets()) {
        if (snippet.extension() == "cu") {
            return snippet.stream().str();
        }
    }
    return "";
}

size_t count(const std::string& text, const std::string& needle) {
    size_t n = 0;
    for (size_t p = text.find(needle); p != std::string::npos; p = text.find(needle, p + 1)) {
        ++n;
    }
    return n;
}

} // namespace

// Online-softmax pair: the group is shuffled jointly, folded across the two warps of the
// row, and committed by rescaling against the live-in (m, d).
TEST(CUDAOffloadReduceDispatcherTest, RescaleGroup_SharedCombine) {
    std::string kernel = dispatch_rescale_reduce(64, false);
    ASSERT_FALSE(kernel.empty());
    // 1024 iterations over 64 threads: whole waves, so the body is unguarded.
    EXPECT_EQ(kernel.find("if (i < 1024)"), std::string::npos);
    EXPECT_NE(kernel.find("__shfl_xor_sync"), std::string::npos);
    EXPECT_NE(kernel.find("__daisy_rescale_nm_m"), std::string::npos);
    EXPECT_NE(kernel.find("expf("), std::string::npos);
    // Two warps per row: one cross-warp fold.
    EXPECT_NE(kernel.find("__daisy_reduce_group_m"), std::string::npos);
    // No per-container tree or separate max/add combine for the group members.
    EXPECT_EQ(kernel.find("__daisy_reduce_m_m"), std::string::npos);
    EXPECT_EQ(kernel.find("__daisy_reduce_m_d"), std::string::npos);
    EXPECT_EQ(kernel.find("__daisy_reduce_shfl_d"), std::string::npos);
    // The identity guard avoids exp(-inf - -inf).
    EXPECT_NE(kernel.find("== (__daisy_rescale_nm_m) ?"), std::string::npos);
}

TEST(CUDAOffloadReduceDispatcherTest, RescaleGroup_IndexedDependent) {
    std::string kernel = dispatch_rescale_reduce(32, true);
    ASSERT_FALSE(kernel.empty());
    // The indexed dependent is combined per slot, both in the shuffle and at commit.
    EXPECT_GE(count(kernel, "for (int __daisy_reduce_slot_o = 0; __daisy_reduce_slot_o < 8;"), 2u);
    EXPECT_NE(kernel.find("__daisy_rescale_od_o"), std::string::npos);
    // Single warp: no cross-warp fold and no barrier in the combine.
    EXPECT_EQ(kernel.find("__daisy_reduce_group_m"), std::string::npos);
}

TEST(CUDAOffloadReduceDispatcherTest, RescaleGroup_RequiresWarpShuffleWidth) {
    // 48 threads: lane groups of 32 do not tile the row.
    EXPECT_THROW(dispatch_rescale_reduce(48, false), InvalidSDFGException);
}

// The combine's exp follows the precision the body's exp was compiled with.
TEST(CUDAOffloadReduceDispatcherTest, RescaleGroup_ExpFollowsApproximateFlag) {
    std::string precise = dispatch_rescale_reduce(32, false, gpu::TargetLevel::X_BLOCK, false);
    EXPECT_NE(precise.find("expf((float)((__daisy_rescale"), std::string::npos);
    EXPECT_EQ(precise.find("__expf"), std::string::npos);

    std::string approximate = dispatch_rescale_reduce(32, false, gpu::TargetLevel::X_BLOCK, true);
    EXPECT_NE(approximate.find("__expf((float)((__daisy_rescale"), std::string::npos);
}

} // namespace sdfg::cuda
