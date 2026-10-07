#include "sdfg/targets/gpu/gpu_mma_expander.h"

#include <gtest/gtest.h>
#include <sdfg/serializer/json_serializer.h>
#include <strstream>

#include "sdfg/analysis/analysis.h"
#include "sdfg/analysis/loop_analysis.h"
#include "sdfg/codegen/code_generators/cpp_code_generator.h"
#include "sdfg/codegen/language_extensions/c_language_extension.h"
#include "sdfg/codegen/utils.h"
#include "sdfg/data_flow/library_nodes/math/tensor/matmul_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/tensor_layout.h"
#include "sdfg/passes/expansion/library_node_expansion_pass.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/targets/gpu/gpu_mma_fragment_load_node.h"
#include "sdfg/targets/gpu/gpu_offload_schedule_type.h"
#include "sdfg/targets/gpu/gpu_types.h"
#include "sdfg/targets/rocm/rocm.h"
#include "sdfg/targets/rocm/rocm_arch.h"
#include "sdfg/tiles/library_nodes/tile_copy_node.h"
#include "sdfg/tiles/transformations/local_storage.h"
#include "sdfg/types/array.h"
#include "sdfg/types/tensor.h"
#include "sdfg/visitor/for_each.h"
#include "sdfg_debug_dump.h"

namespace test::utils {

struct TestCodeSnippet {
    std::string name;
    bool as_file;
    std::string extension;
    std::string content;
};

struct TestCodegenOut {
    std::string main_function;
    std::unordered_map<std::string, TestCodeSnippet> snippets;

    TestCodegenOut(
        const std::string& main_function, const std::unordered_map<std::string, sdfg::codegen::CodeSnippet>& snippets
    )
        : main_function(main_function) {
        for (const auto& [name, snippet] : snippets) {
            this->snippets[name] = TestCodeSnippet{
                .name = name,
                .as_file = snippet.is_as_file(),
                .extension = snippet.extension(),
                .content = snippet.stream().str()
            };
        }
    }
};

TestCodegenOut test_codegen(sdfg::StructuredSDFG& sdfg, const std::string& group = "code", bool dump = true) {
    sdfg::analysis::AnalysisManager ana(sdfg);
    auto instr_plan = sdfg::codegen::InstrumentationPlan::none(sdfg);
    auto cap_plan = sdfg::codegen::ArgCapturePlan::none(sdfg);
    auto snippetFactory = std::make_shared<sdfg::codegen::CodeSnippetFactory>();
    sdfg::codegen::CPPCodeGenerator codegen(sdfg, ana, *instr_plan, *cap_plan, snippetFactory);
    codegen.generate();
    std::stringstream ss;
    codegen.append_function_source(ss);

    if (dump) {
        if (auto dir = get_test_output_dir()) {
            auto out_dir = dir.value() / group;
            std::filesystem::create_directories(out_dir);
            auto main_file = out_dir / "main.cpp";
            std::ofstream ofs(main_file.c_str(), std::ofstream::out);
            ofs << ss.str();
            ofs.close();
            for (auto& [name, snippet] : codegen.library_snippets()) {
                auto snippet_file = out_dir / (name + "." + snippet.extension());
                std::ofstream ofs_snippet(snippet_file.c_str(), std::ofstream::out);
                ofs_snippet << snippet.stream().str();
                ofs_snippet.close();
            }
        }
    }

    return TestCodegenOut(ss.str(), codegen.library_snippets());
}

} // namespace test::utils

namespace sdfg::rocm {

static std::tuple<Block&, math::tensor::MatMulNode&> build_offloaded_mma_structure(
    builder::StructuredSDFGBuilder& builder,
    int M,
    int N,
    int K,
    int tile_m,
    int tile_n,
    int tile_k,
    const gpu::GpuArch& arch
) {
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Half);
    types::Pointer dev_pointer_type(base_desc);
    dev_pointer_type.storage_type().value("AMD_Generic");
    types::Scalar index_desc(types::PrimitiveType::Int32);
    index_desc.storage_type().value("AMD_Generic");

    // Matrix pointers (device buffers) and the map induction variables.
    builder.add_container("A", dev_pointer_type, true);
    builder.add_container("B", dev_pointer_type, true);
    builder.add_container("C", dev_pointer_type, true);
    builder.add_container("block_row", index_desc);
    builder.add_container("block_col", index_desc);

    auto row = symbolic::symbol("block_row");
    auto col = symbolic::symbol("block_col");

    // Outer map over the rows of C: i = 0, TILE, 2*TILE, ...
    auto& map_row = builder.add_map(
        root,
        row,
        symbolic::Lt(row, symbolic::integer(M)),
        symbolic::integer(0),
        symbolic::add(row, symbolic::integer(tile_m)),
        gpu::ScheduleType_GPU_Offload::create(
            arch,
            gpu::TargetLevel::Y_GRID,
            SymEngine::rcp_dynamic_cast<
                const SymEngine::Integer>(symbolic::ceil_div(symbolic::integer(M), symbolic::integer(tile_m)))
        )
    );

    // Inner map over the columns of C: j = 0, TILE, 2*TILE, ...
    auto& map_col = builder.add_map(
        map_row.root(),
        col,
        symbolic::Lt(col, symbolic::integer(N)),
        symbolic::integer(0),
        symbolic::add(col, symbolic::integer(tile_n)),
        gpu::ScheduleType_GPU_Offload::create(
            arch,
            gpu::TargetLevel::X_GRID,
            SymEngine::rcp_dynamic_cast<
                const SymEngine::Integer>(symbolic::ceil_div(symbolic::integer(N), symbolic::integer(tile_n)))
        )
    );

    auto& block = builder.add_block(map_col.root());

    auto& a_node = builder.add_access(block, "A");
    auto& b_node = builder.add_access(block, "B");
    auto& c_node = builder.add_access(block, "C");

    // Slice layouts for this (i, j) tile. Strides follow the row-major layout of the
    // full matrices, while the offset selects the tile inside the base pointer.
    //   A tile: [TILE, K]  starting at row i           -> offset i * K
    //   B tile: [K, TILE]  starting at column j        -> offset j
    //   C tile: [TILE, TILE] starting at (i, j)        -> offset i * N + j
    math::tensor::TensorLayout a_layout(
        {symbolic::integer(tile_m), symbolic::integer(K)},
        {symbolic::integer(K), symbolic::integer(1)},
        symbolic::mul(row, symbolic::integer(K))
    );
    math::tensor::TensorLayout
        b_layout({symbolic::integer(K), symbolic::integer(tile_n)}, {symbolic::integer(N), symbolic::integer(1)}, col);
    math::tensor::TensorLayout c_layout(
        {symbolic::integer(tile_m), symbolic::integer(tile_n)},
        {symbolic::integer(N), symbolic::integer(1)},
        symbolic::add(symbolic::mul(row, symbolic::integer(N)), col)
    );

    types::Tensor a_tensor(base_desc, a_layout);
    types::Tensor b_tensor(base_desc, b_layout);
    types::Tensor c_tensor(base_desc, c_layout);

    auto& matmul_node = static_cast<math::tensor::MatMulNode&>(builder.add_library_node<math::tensor::MatMulNode>(
        block, DebugInfo(), a_layout, b_layout, types::PrimitiveType::Half, &c_layout
    ));

    builder.add_computational_memlet(block, a_node, matmul_node, "A", {}, a_tensor, block.debug_info());
    builder.add_computational_memlet(block, b_node, matmul_node, "B", {}, b_tensor, block.debug_info());
    builder.add_computational_memlet(block, c_node, matmul_node, "Y", {}, c_tensor, block.debug_info());

    return {block, matmul_node};
}

TEST(ROCMMMATest, 1K_1K_1K_16x16_gfx1201) {
    constexpr int M = 1024; // rows of A / C
    constexpr int N = 1024; // cols of B / C
    constexpr int K = 1024; // contraction dimension
    constexpr int TILE = 16; // MMA-friendly tile width for the outer maps

    auto& arch = gpu::rocm::ROCM_ARCH_GFX1201;

    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto [block, matmul_node] = build_offloaded_mma_structure(builder, M, N, K, TILE, TILE, 16, arch);

    dump_sdfg(builder.subject(), "0.init");

    EXPECT_NO_THROW(builder.subject().validate());

    passes::expansion::expand_single_node(builder, block, matmul_node, gpu::GpuMmaExpander(&arch));

    dump_sdfg(builder.subject(), "1.expanded");

    EXPECT_NO_THROW(builder.subject().validate());

    test::utils::test_codegen(builder.subject(), "result", true);
}

TEST(ROCMMMATest, 1K_1K_1K_32x32_gfx1201) {
    constexpr int M = 1024; // rows of A / C
    constexpr int N = 1024; // cols of B / C
    constexpr int K = 1024; // contraction dimension
    constexpr int TILE = 32; // MMA-friendly tile width for the outer maps

    auto& arch = gpu::rocm::ROCM_ARCH_GFX1201;

    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto [block, matmul_node] = build_offloaded_mma_structure(builder, M, N, K, TILE, TILE, 16, arch);

    dump_sdfg(builder.subject(), "0.init");

    EXPECT_NO_THROW(builder.subject().validate());

    passes::expansion::expand_single_node(builder, block, matmul_node, gpu::GpuMmaExpander(&arch));

    dump_sdfg(builder.subject(), "1.expanded");

    EXPECT_NO_THROW(builder.subject().validate());

    test::utils::test_codegen(builder.subject(), "result", true);
}

TEST(ROCMMMATest, 1K_1K_1K_64x64_gfx1201) {
    constexpr int M = 1024; // rows of A / C
    constexpr int N = 1024; // cols of B / C
    constexpr int K = 1024; // contraction dimension
    constexpr int TILE = 64; // MMA-friendly tile width for the outer maps

    auto& arch = gpu::rocm::ROCM_ARCH_GFX1201;

    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto [block, matmul_node] = build_offloaded_mma_structure(builder, M, N, K, TILE, TILE, 16, arch);

    dump_sdfg(builder.subject(), "0.init");

    EXPECT_NO_THROW(builder.subject().validate());

    passes::expansion::expand_single_node(builder, block, matmul_node, gpu::GpuMmaExpander(&arch));

    dump_sdfg(builder.subject(), "1.expanded");

    EXPECT_NO_THROW(builder.subject().validate());

    test::utils::test_codegen(builder.subject(), "result", true);
}

TEST(ROCMMMATest, 1K_1K_1K_64x64_gfx90a) {
    constexpr int M = 1024; // rows of A / C
    constexpr int N = 1024; // cols of B / C
    constexpr int K = 1024; // contraction dimension
    constexpr int TILE = 64; // MMA-friendly tile width for the outer maps

    auto& arch = gpu::rocm::ROCM_ARCH_GFX1201;

    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto [block, matmul_node] = build_offloaded_mma_structure(builder, M, N, K, TILE, TILE, 16, arch);

    dump_sdfg(builder.subject(), "0.init");

    EXPECT_NO_THROW(builder.subject().validate());

    passes::expansion::expand_single_node(builder, block, matmul_node, gpu::GpuMmaExpander(&arch));

    dump_sdfg(builder.subject(), "1.expanded");

    EXPECT_NO_THROW(builder.subject().validate());

    test::utils::test_codegen(builder.subject(), "result", true);
}

TEST(ROCMMMATest, ExpansionEmitsWaveMap_gfx90a) {
    constexpr int TILE = 32; // 2x2 waves of 16x16 MMA blocks

    auto& arch = gpu::rocm::ROCM_ARCH_GFX90A;

    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto [block, matmul_node] = build_offloaded_mma_structure(builder, 1024, 1024, 1024, TILE, TILE, 16, arch);
    passes::expansion::expand_single_node(builder, block, matmul_node, gpu::GpuMmaExpander(&arch));
    EXPECT_NO_THROW(builder.subject().validate());

    analysis::AnalysisManager am(builder.subject());
    structured_control_flow::StructuredLoop* x_block = nullptr;
    structured_control_flow::StructuredLoop* y_block = nullptr;
    for (auto* loop : am.get<analysis::LoopAnalysis>().loops()) {
        auto* sl = dynamic_cast<structured_control_flow::StructuredLoop*>(loop);
        if (!sl || sl->schedule_type().category() != structured_control_flow::ScheduleTypeCategory::Offloader) {
            continue;
        }
        auto level = gpu::ScheduleType_GPU_Offload::target_level(sl->schedule_type());
        if (level == gpu::TargetLevel::X_BLOCK) {
            x_block = sl;
        } else if (level == gpu::TargetLevel::Y_BLOCK) {
            y_block = sl;
        }
    }
    ASSERT_NE(x_block, nullptr);
    ASSERT_NE(y_block, nullptr);
    EXPECT_EQ(x_block->indvar()->get_name().rfind("wave_row", 0), 0u);
    EXPECT_EQ(gpu::ScheduleType_GPU_Offload::parallel_size(x_block->schedule_type())->as_int(), 2);
    EXPECT_EQ(gpu::ScheduleType_GPU_Offload::lanes(x_block->schedule_type())->as_int(), 64);
    EXPECT_EQ(gpu::ScheduleType_GPU_Offload::parallel_size(y_block->schedule_type())->as_int(), 2);

    // Fragment addresses use the wave index directly: no thread-index division.
    size_t loads = 0;
    visitor::for_each_block(builder.subject().root(), [&](structured_control_flow::Block& b) {
        for (auto* lib : b.dataflow().library_nodes()) {
            if (auto* load = dynamic_cast<gpu::GpuMmaFragmentLoadNode*>(lib)) {
                ++loads;
                auto s = load->toStr();
                EXPECT_EQ(s.find("idiv"), std::string::npos) << s;
                if (load->fragment_type() == gpu::MmaFragmentType::A) {
                    EXPECT_NE(s.find(x_block->indvar()->get_name()), std::string::npos) << s;
                }
            }
        }
    });
    EXPECT_EQ(loads, 3u); // A, B, C

    auto out = test::utils::test_codegen(builder.subject(), "wave_map", true);
    EXPECT_NE(out.main_function.find("dim3((int)(128), (int)(2), (int)(1))"), std::string::npos) << out.main_function;
}

namespace {

structured_control_flow::StructuredLoop* find_loop(analysis::AnalysisManager& am, const std::string& prefix) {
    for (auto* loop : am.get<analysis::LoopAnalysis>().loops()) {
        auto* sl = dynamic_cast<structured_control_flow::StructuredLoop*>(loop);
        if (sl && sl->indvar()->get_name().rfind(prefix, 0) == 0) {
            return sl;
        }
    }
    return nullptr;
}

const data_flow::AccessNode* find_access(structured_control_flow::StructuredLoop& loop, const std::string& container) {
    const data_flow::AccessNode* found = nullptr;
    visitor::for_each_block(loop.root(), [&](structured_control_flow::Block& b) {
        for (auto* a : b.dataflow().data_nodes()) {
            if (a->data() == container) {
                found = a;
            }
        }
    });
    return found;
}

const tiles::TileCopyNode* find_copy_from(structured_control_flow::ControlFlowNode& root, const std::string& src) {
    const tiles::TileCopyNode* found = nullptr;
    visitor::for_each_block(root, [&](structured_control_flow::Block& b) {
        for (auto* lib : b.dataflow().library_nodes()) {
            auto* copy = dynamic_cast<tiles::TileCopyNode*>(lib);
            if (!copy) {
                continue;
            }
            for (auto& e : b.dataflow().in_edges(*copy)) {
                auto* a = dynamic_cast<const data_flow::AccessNode*>(&e.src());
                if (e.dst_conn() == "_src" && a && a->data() == src) {
                    found = copy;
                }
            }
        }
    });
    return found;
}

} // namespace

// Step 3: per-wave slots, and the lanes of each wave share the copy.
TEST(ROCMMMATest, LocalStorageOnWaveMapSharesCopyAcrossLanes_gfx90a) {
    auto& arch = gpu::rocm::ROCM_ARCH_GFX90A;
    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto [block, matmul_node] = build_offloaded_mma_structure(builder, 1024, 1024, 1024, 32, 32, 16, arch);
    passes::expansion::expand_single_node(builder, block, matmul_node, gpu::GpuMmaExpander(&arch));

    for (const std::string container : {"A", "B"}) {
        analysis::AnalysisManager am(builder.subject());
        auto* dummy = find_loop(am, "dummy");
        ASSERT_NE(dummy, nullptr);
        auto* access = find_access(*dummy, container);
        ASSERT_NE(access, nullptr) << container;
        transformations::LocalStorage ls(*dummy, *access);
        ASSERT_TRUE(ls.can_be_applied(builder, am)) << container;
        ls.apply(builder, am);
    }
    EXPECT_NO_THROW(builder.subject().validate());

    // A: slot per wave_row (x), cooperative over wave_col (y) + the 64 lanes.
    auto* copy_a = find_copy_from(builder.subject().root(), "A");
    ASSERT_NE(copy_a, nullptr);
    EXPECT_EQ(copy_a->coop_axes(), std::vector<int>{1});
    EXPECT_EQ(copy_a->coop_lanes(), 64u);
    EXPECT_TRUE(symbolic::eq(copy_a->coop_threads(), symbolic::integer(128))) << copy_a->coop_threads()->__str__();
    EXPECT_NE(copy_a->plan().dst.offset()->__str__().find("wave_row"), std::string::npos);

    // B: slot per wave_col (y), cooperative over the whole wave_row axis (2 waves x 64 lanes).
    auto* copy_b = find_copy_from(builder.subject().root(), "B");
    ASSERT_NE(copy_b, nullptr);
    EXPECT_EQ(copy_b->coop_axes(), std::vector<int>{0});
    EXPECT_EQ(copy_b->coop_lanes(), 1u);
    EXPECT_TRUE(symbolic::eq(copy_b->coop_threads(), symbolic::integer(128))) << copy_b->coop_threads()->__str__();
    EXPECT_NE(copy_b->plan().dst.offset()->__str__().find("wave_col"), std::string::npos);

    // Two slots per operand (one per wave), not one per thread.
    for (const auto& name : builder.subject().containers()) {
        if (name.rfind("__daisy_local_storage_", 0) == 0) {
            auto& type = static_cast<const types::Array&>(builder.subject().type(name));
            EXPECT_TRUE(symbolic::eq(type.num_elements(), symbolic::integer(2))) << name;
        }
    }

    auto out = test::utils::test_codegen(builder.subject(), "wave_map_ls", true);
    std::string kernels;
    for (auto& [name, snippet] : out.snippets) {
        kernels += snippet.content;
    }
    EXPECT_NE(kernels.find("threadIdx.x % 64 + threadIdx.y * (64)"), std::string::npos) << kernels;
}

} // namespace sdfg::rocm
