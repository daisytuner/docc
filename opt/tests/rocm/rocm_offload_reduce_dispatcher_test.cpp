#include <gtest/gtest.h>

#include <string>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/codegen/language_extensions/c_language_extension.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/passes/offloading/reduction_shared_memory_delinearization.h"
#include "sdfg/structured_control_flow/reduce.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/targets/gpu/gpu_offload_map_dispatcher.h"
#include "sdfg/targets/gpu/gpu_offload_schedule_type.h"
#include "sdfg/targets/rocm/rocm.h"
#include "sdfg/targets/rocm/rocm_offload_dispatcher_strategy.h"

namespace sdfg::rocm {

namespace {

// Y_GRID map b -> Z_BLOCK reduce g -> sequential s: acc[out(b, s)] += A[16*b + 4*g + s],
// the split-K GEMM shape (wave groups on Z reduce a multi-output register tile).
std::string dispatch_multi_output_reduce(bool disjoint, bool panel = false) {
    builder::StructuredSDFGBuilder builder("red_multi", FunctionType_CPU);
    auto& root = builder.subject().root();
    types::Scalar base_desc(types::PrimitiveType::Half);
    types::Pointer pointer_type(types::StorageType::NV_Generic(), 0, "", base_desc);
    types::Scalar int_desc(types::PrimitiveType::Int32);
    for (auto name : {"b", "g", "s", "p"}) {
        builder.add_container(name, int_desc);
    }
    builder.add_container("A", pointer_type);
    builder.add_container("acc", pointer_type);
    auto b = symbolic::symbol("b");
    auto g = symbolic::symbol("g");
    auto s = symbolic::symbol("s");
    auto p = symbolic::symbol("p");

    auto& grid = builder.add_map(
        root,
        b,
        symbolic::Lt(b, symbolic::integer(4)),
        symbolic::zero(),
        symbolic::add(b, symbolic::one()),
        gpu::ScheduleType_GPU_Offload::create<ScheduleType_ROCM_Offload>(gpu::TargetLevel::Y_GRID, symbolic::integer(4))
    );
    auto* reduce_parent = &grid.root();
    if (panel) {
        auto& loop_p = builder.add_for(
            grid.root(), p, symbolic::Lt(p, symbolic::integer(8)), symbolic::zero(), symbolic::add(p, symbolic::one())
        );
        reduce_parent = &loop_p.root();
    }
    auto& reduce = builder.add_reduce(
        *reduce_parent,
        g,
        symbolic::Lt(g, symbolic::integer(4)),
        symbolic::zero(),
        symbolic::add(g, symbolic::one()),
        {structured_control_flow::ReductionInfo{structured_control_flow::ReductionOperation::Add, "acc"}},
        gpu::ScheduleType_GPU_Offload::create<ScheduleType_ROCM_Offload>(gpu::TargetLevel::Z_BLOCK, symbolic::integer(4))
    );
    auto& loop_s = builder.add_for(
        reduce.root(), s, symbolic::Lt(s, symbolic::integer(4)), symbolic::zero(), symbolic::add(s, symbolic::one())
    );
    symbolic::Expression out = disjoint ? symbolic::add(symbolic::mul(symbolic::integer(4), b), s)
                                        : symbolic::Expression(s);
    auto& block = builder.add_block(loop_s.root());
    auto& a_access = builder.add_access(block, "A");
    auto& acc_in = builder.add_access(block, "acc");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in0", "_in1"});
    auto& acc_out = builder.add_access(block, "acc");
    auto a_index =
        symbolic::add(symbolic::mul(symbolic::integer(16), b), symbolic::add(symbolic::mul(symbolic::integer(4), g), s));
    if (panel) {
        a_index = symbolic::add(a_index, symbolic::mul(symbolic::integer(64), p));
    }
    builder.add_computational_memlet(block, acc_in, tasklet, "_in0", {out}, pointer_type);
    builder.add_computational_memlet(block, a_access, tasklet, "_in1", {a_index}, pointer_type);
    builder.add_computational_memlet(block, tasklet, "_out", acc_out, {out}, pointer_type);

    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::ReductionSharedMemoryDelinearization reduction_buffers;
    reduction_buffers.run(builder, analysis_manager);

    codegen::CLanguageExtension language_extension(builder.subject());
    auto instrumentation = codegen::InstrumentationPlan::none(builder.subject());
    auto arg_capture = codegen::ArgCapturePlan::none(builder.subject());
    gpu::GPUOffloadMapDispatcher dispatcher(
        language_extension,
        builder.subject(),
        analysis_manager,
        grid,
        *instrumentation,
        *arg_capture,
        std::make_unique<ROCMOffloadDispatcherStrategy>(builder.subject(), grid)
    );
    codegen::PrettyPrinter main_stream, globals_stream;
    codegen::CodeSnippetFactory library_snippet_factory;
    dispatcher.dispatch_node(main_stream, globals_stream, library_snippet_factory);
    for (auto& [key, snippet] : library_snippet_factory.snippets()) {
        if (snippet.extension() == "rocm.cpp") {
            return snippet.stream().str();
        }
    }
    return "";
}

size_t count_of(const std::string& haystack, const std::string& needle) {
    size_t n = 0;
    for (size_t p = haystack.find(needle); p != std::string::npos; p = haystack.find(needle, p + 1)) {
        n++;
    }
    return n;
}

} // namespace

// The tree emits one leading and one per-step barrier (inside its loop), none per slot.
TEST(ROCMOffloadReduceDispatcherTest, ZBlockMultiOutputTreeBarriersIndependentOfSlots) {
    auto kernel = dispatch_multi_output_reduce(/*disjoint=*/true);
    ASSERT_FALSE(kernel.empty());
    EXPECT_EQ(count_of(kernel, "__syncthreads();"), 2u) << kernel;
    EXPECT_NE(kernel.find("for (int __daisy_reduce_slot_acc"), std::string::npos) << kernel;
}

// Each grid block owns its 4 fp16 outputs: no 16-bit CAS combine.
TEST(ROCMOffloadReduceDispatcherTest, ZBlockMultiOutputDisjointCommitsPlainly) {
    auto kernel = dispatch_multi_output_reduce(/*disjoint=*/true);
    EXPECT_EQ(kernel.find("__daisy_reduce_combine_"), std::string::npos) << kernel;
    EXPECT_EQ(kernel.find("atomic"), std::string::npos) << kernel;
}

TEST(ROCMOffloadReduceDispatcherTest, ZBlockMultiOutputCollidingKeepsCAS) {
    auto kernel = dispatch_multi_output_reduce(/*disjoint=*/false);
    EXPECT_NE(kernel.find("__daisy_reduce_combine_add__Float16"), std::string::npos) << kernel;
}

// Split-K over a sequential K-panel loop: the fp16 register tile accumulates across panels and
// the tree + commit runs once after the last panel.
TEST(ROCMOffloadReduceDispatcherTest, ZBlockPanelLoopHoistsPartialAndCombine) {
    auto kernel = dispatch_multi_output_reduce(/*disjoint=*/true, /*panel=*/true);
    ASSERT_FALSE(kernel.empty());
    const auto panel_loop = kernel.find("for(p = 0");
    ASSERT_NE(panel_loop, std::string::npos) << kernel;
    const auto decl = kernel.find("__daisy_reduce_reg_acc");
    ASSERT_NE(decl, std::string::npos) << kernel;
    EXPECT_LT(decl, panel_loop) << kernel;
    EXPECT_NE(kernel.find("if (((0 == p))) for (int __daisy_reduce_slot_acc"), std::string::npos) << kernel;
    EXPECT_NE(kernel.find("if ((8 <= 1 + p)) {"), std::string::npos) << kernel;
    EXPECT_EQ(count_of(kernel, "__syncthreads();"), 2u) << kernel;
}

} // namespace sdfg::rocm
