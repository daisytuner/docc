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
#include "sdfg/targets/gpu/gpu_mma_fragment.h"
#include "sdfg/targets/gpu/gpu_mma_fragment_load_node.h"
#include "sdfg/targets/gpu/gpu_mma_matmul_node.h"
#include "sdfg/targets/gpu/gpu_offload_schedule_type.h"
#include "sdfg/targets/gpu/gpu_types.h"
#include "sdfg/targets/rocm/rocm.h"
#include "sdfg/targets/rocm/rocm_arch.h"
#include "sdfg/targets/rocm/rocm_mma_bare_dispatcher.h"
#include "sdfg/tiles/library_nodes/tile_copy_node.h"
#include "sdfg/tiles/transformations/local_storage.h"
#include "sdfg/tiles/transformations/software_pipelining.h"
#include "sdfg/tiles/transformations/tile_vectorizer.h"
#include "sdfg/transformations/loop_tiling.h"
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
    const gpu::GpuArch& arch,
    bool b_k_major = false,
    types::PrimitiveType input_type = types::PrimitiveType::Half,
    types::PrimitiveType output_type = types::PrimitiveType::Half
) {
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(input_type);
    types::Pointer dev_pointer_type(base_desc);
    dev_pointer_type.storage_type().value("AMD_Generic");
    types::Scalar out_desc(output_type);
    types::Pointer out_pointer_type(out_desc);
    out_pointer_type.storage_type().value("AMD_Generic");
    types::Scalar index_desc(types::PrimitiveType::Int32);
    index_desc.storage_type().value("AMD_Generic");

    // Matrix pointers (device buffers) and the map induction variables.
    builder.add_container("A", dev_pointer_type, true);
    builder.add_container("B", dev_pointer_type, true);
    builder.add_container("C", out_pointer_type, true);
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
    // K-major B (C = A @ W^T with W[N][K] row-major): column-major [K, TILE] slice at row j of W.
    math::tensor::TensorLayout b_layout = b_k_major ? math::tensor::TensorLayout(
                                                          {symbolic::integer(K), symbolic::integer(tile_n)},
                                                          {symbolic::integer(1), symbolic::integer(K)},
                                                          symbolic::mul(col, symbolic::integer(K))
                                                      )
                                                    : math::tensor::TensorLayout(
                                                          {symbolic::integer(K), symbolic::integer(tile_n)},
                                                          {symbolic::integer(N), symbolic::integer(1)},
                                                          col
                                                      );
    math::tensor::TensorLayout c_layout(
        {symbolic::integer(tile_m), symbolic::integer(tile_n)},
        {symbolic::integer(N), symbolic::integer(1)},
        symbolic::add(symbolic::mul(row, symbolic::integer(N)), col)
    );

    types::Tensor a_tensor(base_desc, a_layout);
    types::Tensor b_tensor(base_desc, b_layout);
    types::Tensor c_tensor(out_desc, c_layout);

    auto matmul_types = input_type == output_type
                            ? math::tensor::ContractionTypes::uniform(input_type)
                            : math::tensor::ContractionTypes{input_type, types::PrimitiveType::Void, output_type};
    auto& matmul_node = static_cast<math::tensor::MatMulNode&>(builder.add_library_node<math::tensor::MatMulNode>(
        block, DebugInfo(), a_layout, b_layout, matmul_types, &c_layout
    ));

    builder.add_computational_memlet(block, a_node, matmul_node, "A", {}, a_tensor, block.debug_info());
    builder.add_computational_memlet(block, b_node, matmul_node, "B", {}, b_tensor, block.debug_info());
    builder.add_computational_memlet(block, c_node, matmul_node, "Y", {}, c_tensor, block.debug_info());

    return {block, matmul_node};
}

/// The single GpuMmaMatmulNode produced by expansion; its block size is the one actually applied.
static const gpu::GpuMmaMatmulNode* find_mma_matmul(structured_control_flow::ControlFlowNode& root) {
    const gpu::GpuMmaMatmulNode* found = nullptr;
    visitor::for_each_block(root, [&](structured_control_flow::Block& b) {
        for (auto* lib : b.dataflow().library_nodes()) {
            if (auto* mma = dynamic_cast<gpu::GpuMmaMatmulNode*>(lib)) {
                found = mma;
            }
        }
    });
    return found;
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

    auto& arch = gpu::rocm::ROCM_ARCH_GFX90A;

    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto [block, matmul_node] = build_offloaded_mma_structure(builder, M, N, K, TILE, TILE, 16, arch);

    dump_sdfg(builder.subject(), "0.init");

    EXPECT_NO_THROW(builder.subject().validate());

    passes::expansion::expand_single_node(builder, block, matmul_node, gpu::GpuMmaExpander(&arch));

    dump_sdfg(builder.subject(), "1.expanded");

    EXPECT_NO_THROW(builder.subject().validate());

    test::utils::test_codegen(builder.subject(), "result", true);
}

TEST(ROCMMMATest, 1K_1K_1K_64x64_gfx90a_with_explicit_32x32x8_blocks) {
    constexpr int M = 1024; // rows of A / C
    constexpr int N = 1024; // cols of B / C
    constexpr int K = 1024; // contraction dimension
    constexpr int TILE = 64; // MMA-friendly tile width for the outer maps

    auto& arch = gpu::rocm::ROCM_ARCH_GFX90A;

    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto [block, matmul_node] = build_offloaded_mma_structure(builder, M, N, K, TILE, TILE, 16, arch);

    dump_sdfg(builder.subject(), "0.init");

    EXPECT_NO_THROW(builder.subject().validate());

    auto override_block = gpu::MmaBlockSize::parse_block_size("32x32x8");
    ASSERT_EQ(override_block.m, 32);
    ASSERT_EQ(override_block.n, 32);
    ASSERT_EQ(override_block.k, 8);
    passes::expansion::expand_single_node(builder, block, matmul_node, gpu::GpuMmaExpander(&arch, &override_block));

    dump_sdfg(builder.subject(), "1.expanded");

    EXPECT_NO_THROW(builder.subject().validate());

    // The explicit 32x32x8 hint is valid on gfx90a (CDNA) and must drive the emitted MMA.
    auto* mma = find_mma_matmul(builder.subject().root());
    ASSERT_NE(mma, nullptr);
    EXPECT_EQ(mma->mma_block_size().m, 32);
    EXPECT_EQ(mma->mma_block_size().n, 32);
    EXPECT_EQ(mma->mma_block_size().k, 8);
    EXPECT_EQ(mma->implementation_type().value(), gpu::rocm::ImplementationType_ROCM_MMA_BARE.value());
    ASSERT_TRUE(builder.subject().exists("mma_a0"));
    auto& regs_type = builder.subject().type("mma_a0");
    EXPECT_EQ(regs_type.storage_type().value(), gpu::rocm::RocmMmaSupport::MMA_REGS_STORAGE_TYPE);

    test::utils::test_codegen(builder.subject(), "result", true);
}

TEST(ROCMMMATest, DefaultBlockSizeIs16x16x16_gfx90a) {
    constexpr int M = 1024; // rows of A / C
    constexpr int N = 1024; // cols of B / C
    constexpr int K = 1024; // contraction dimension
    constexpr int TILE = 64; // MMA-friendly tile width for the outer maps

    auto& arch = gpu::rocm::ROCM_ARCH_GFX90A;

    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto [block, matmul_node] = build_offloaded_mma_structure(builder, M, N, K, TILE, TILE, 16, arch);

    EXPECT_NO_THROW(builder.subject().validate());

    // No block-size hint: expansion must fall back to the 16x16x16 default.
    passes::expansion::expand_single_node(builder, block, matmul_node, gpu::GpuMmaExpander(&arch));

    EXPECT_NO_THROW(builder.subject().validate());

    auto* mma = find_mma_matmul(builder.subject().root());
    ASSERT_NE(mma, nullptr);
    EXPECT_EQ(mma->mma_block_size().m, 16);
    EXPECT_EQ(mma->mma_block_size().n, 16);
    EXPECT_EQ(mma->mma_block_size().k, 16);
}

TEST(ROCMMMATest, ExplicitBlockSizeIgnoredOnRdna_gfx1201) {
    constexpr int M = 1024; // rows of A / C
    constexpr int N = 1024; // cols of B / C
    constexpr int K = 1024; // contraction dimension
    constexpr int TILE = 64; // MMA-friendly tile width for the outer maps

    auto& arch = gpu::rocm::ROCM_ARCH_GFX1201;

    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto [block, matmul_node] = build_offloaded_mma_structure(builder, M, N, K, TILE, TILE, 16, arch);

    EXPECT_NO_THROW(builder.subject().validate());

    // 32x32x8 is CDNA-only; on gfx1201 (RDNA) the hint is invalid and must be ignored,
    // leaving the 16x16x16 default in place.
    auto override_block = gpu::MmaBlockSize::parse_block_size("32x32x8");
    ASSERT_EQ(override_block.m, 32);
    ASSERT_EQ(override_block.n, 32);
    ASSERT_EQ(override_block.k, 8);
    auto result =
        passes::expansion::expand_single_node(builder, block, matmul_node, gpu::GpuMmaExpander(&arch, &override_block));

    EXPECT_NO_THROW(builder.subject().validate());

    EXPECT_FALSE(result.expanded);
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

    // One unit-stride K-block loop (1024/16 blocks) directly holding loads and MMA; no helper loop.
    std::vector<structured_control_flow::StructuredLoop*> sequential;
    for (auto* loop : am.get<analysis::LoopAnalysis>().loops()) {
        auto* sl = dynamic_cast<structured_control_flow::StructuredLoop*>(loop);
        if (sl && sl->schedule_type().category() != structured_control_flow::ScheduleTypeCategory::Offloader) {
            sequential.push_back(sl);
        }
    }
    ASSERT_EQ(sequential.size(), 1u);
    EXPECT_TRUE(symbolic::eq(sequential.front()->stride(), symbolic::one()));
    EXPECT_TRUE(symbolic::eq(sequential.front()->num_iterations(), symbolic::integer(64)));

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
    EXPECT_EQ(loads, 2u); // A, B: a MatMul overwrites C

    auto out = test::utils::test_codegen(builder.subject(), "wave_map", true);
    EXPECT_NE(out.main_function.find("dim3((int)(128), (int)(2), (int)(1))"), std::string::npos) << out.main_function;

    // fp16 inputs accumulate in fp32 (native MFMA accumulator); C/D fragments stay fp16, C zero-filled for the
    // converting add since a MatMul overwrites its output.
    std::string kernels;
    for (auto& [name, snippet] : out.snippets) {
        kernels += snippet.content;
    }
    // The static 128x2 block bounds the kernel, so the compiler may use the full register file.
    EXPECT_NE(kernels.find("__global__ void __launch_bounds__(256) kernel_"), std::string::npos) << kernels;
    EXPECT_NE(kernels.find("rocwmma::accumulator, 16, 16, 16, rocwmma::float32_t> mma_acc"), std::string::npos)
        << kernels;
    EXPECT_NE(kernels.find("rocwmma::accumulator, 16, 16, 16, rocwmma::float16_t> mma_c"), std::string::npos)
        << kernels;
}

namespace {

structured_control_flow::StructuredLoop* find_loop(analysis::AnalysisManager& am, const std::string& indvar) {
    for (auto* loop : am.get<analysis::LoopAnalysis>().loops()) {
        auto* sl = dynamic_cast<structured_control_flow::StructuredLoop*>(loop);
        if (sl && sl->indvar()->get_name() == indvar) {
            return sl;
        }
    }
    return nullptr;
}

/// Strip-mine the MMA K-block loop `tile_k0` into K-panels of @p blocks MMA blocks:
/// `tile_k0_tile0` walks the panels, `tile_k0` (inner) the blocks of one panel.
void strip_mine_k(builder::StructuredSDFGBuilder& builder, size_t blocks) {
    analysis::AnalysisManager am(builder.subject());
    auto* k = find_loop(am, "tile_k0");
    ASSERT_NE(k, nullptr);
    transformations::LoopTiling tiling(*k, blocks, /*simplify_bounds=*/true);
    ASSERT_TRUE(tiling.can_be_applied(builder, am));
    tiling.apply(builder, am);
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

void localize(builder::StructuredSDFGBuilder& builder, const std::string& loop_indvar, bool transpose_b = false) {
    for (const std::string container : {"A", "B"}) {
        analysis::AnalysisManager am(builder.subject());
        auto* loop = find_loop(am, loop_indvar);
        ASSERT_NE(loop, nullptr);
        auto* access = find_access(*loop, container);
        ASSERT_NE(access, nullptr) << container;
        transformations::LocalStorage ls(*loop, *access, false, false, transpose_b && container == "B");
        ASSERT_TRUE(ls.can_be_applied(builder, am)) << container;
        ls.apply(builder, am);
    }
}

/// The fragment load reading the local buffer staged from @p container.
const gpu::GpuMmaFragmentLoadNode*
find_local_load(structured_control_flow::ControlFlowNode& root, const std::string& container) {
    const gpu::GpuMmaFragmentLoadNode* found = nullptr;
    const std::string prefix = "__daisy_local_storage_" + container;
    visitor::for_each_block(root, [&](structured_control_flow::Block& b) {
        for (auto* lib : b.dataflow().library_nodes()) {
            auto* load = dynamic_cast<gpu::GpuMmaFragmentLoadNode*>(lib);
            if (!load) {
                continue;
            }
            for (auto& e : b.dataflow().in_edges(*load)) {
                auto* a = dynamic_cast<const data_flow::AccessNode*>(&e.src());
                if (e.dst_conn() == "ptr" && a && a->data().rfind(prefix, 0) == 0) {
                    found = load;
                }
            }
        }
    });
    return found;
}

} // namespace

TEST(ROCMMMATest, FragmentLayoutFromTensorLayoutKeepsMajorness) {
    auto s16 = symbolic::integer(16);
    math::tensor::TensorLayout row({s16, s16}, {symbolic::integer(40), symbolic::one()}, symbolic::symbol("o"));
    auto r = gpu::GpuMmaFromMemoryLayout::from_tensor_layout(row);
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(r->layout, gpu::MMA_LAYOUT_ROW_MAJOR);
    EXPECT_TRUE(symbolic::eq(r->ldstride, symbolic::integer(40)));
    EXPECT_TRUE(symbolic::eq(r->offset, symbolic::symbol("o")));

    math::tensor::TensorLayout col({s16, s16}, {symbolic::one(), symbolic::integer(24)}, symbolic::zero());
    auto c = gpu::GpuMmaFromMemoryLayout::from_tensor_layout(col);
    ASSERT_TRUE(c.has_value());
    EXPECT_EQ(c->layout, gpu::MMA_LAYOUT_COL_MAJOR);
    EXPECT_TRUE(symbolic::eq(c->ldstride, symbolic::integer(24)));

    // Round trip through the fragment's own tensor layout.
    auto back = c->to_tensor_layout(gpu::MmaBlockSize(16, 16, 16), gpu::MmaFragmentType::B);
    ASSERT_TRUE(back.has_value());
    EXPECT_EQ(back->is_2d_col_or_row_major(), math::tensor::TensorLayout::LAYOUT_COL_MAJOR);
}

TEST(ROCMMMATest, FragmentLoadRelocalizesToBufferView) {
    auto& arch = gpu::rocm::ROCM_ARCH_GFX90A;
    auto* mma = arch.mma_support();
    gpu::GpuMmaTiling tiling{
        .mma_block_size = gpu::MmaBlockSize(16, 16, 16),
        .threads_per_mma_wavefront = 32,
        .wave_tiling = {.blocks_m = 1, .blocks_n = 1},
        .macro_tiling = {.blocks_m = 1, .blocks_n = 1},
        .acc_type = types::PrimitiveType::Half,
    };

    ASSERT_TRUE(mma->is_valid_block_size(tiling.mma_block_size, types::PrimitiveType::Half, tiling.acc_type));

    builder::StructuredSDFGBuilder builder("reloc", FunctionType_CPU);
    auto& block = builder.add_block(builder.subject().root());
    gpu::GpuMmaFromMemoryLayout global{
        .offset = symbolic::symbol("g"), .ldstride = symbolic::integer(1024), .layout = gpu::MMA_LAYOUT_ROW_MAJOR
    };

    auto& load = static_cast<gpu::GpuMmaFragmentLoadNode&>(builder.add_library_node<gpu::GpuMmaFragmentLoadNode>(
        block,
        DebugInfo(),
        tiling.mma_block_size,
        gpu::MmaFragmentType::A,
        global,
        types::PrimitiveType::Half,
        mma->get_mma_impl_type(tiling)
    ));
    auto s16 = symbolic::integer(16);
    auto view_offset =
        symbolic::add(symbolic::mul(symbolic::integer(1024), symbolic::symbol("w")), symbolic::symbol("kk"));

    // A 16x16 sub-window of a [16][64] slot: ld = 64, offset = slot + window origin.
    math::tensor::TensorLayout view({s16, s16}, {symbolic::integer(64), symbolic::one()}, view_offset);
    ASSERT_TRUE(load.can_relocalize_operand(gpu::GpuMmaFragmentLoadNode::PTR_INPUT_IDX, view));
    ASSERT_TRUE(load.relocalize_operand(gpu::GpuMmaFragmentLoadNode::PTR_INPUT_IDX, view));
    EXPECT_TRUE(symbolic::eq(load.layout().offset, view_offset));
    EXPECT_TRUE(symbolic::eq(load.layout().ldstride, symbolic::integer(64)));
    EXPECT_EQ(load.layout().layout, gpu::MMA_LAYOUT_ROW_MAJOR);

    // A view of a different shape than the fragment is rejected.
    math::tensor::TensorLayout
        wide({s16, symbolic::integer(64)}, {symbolic::integer(64), symbolic::one()}, symbolic::zero());
    EXPECT_FALSE(load.can_relocalize_operand(gpu::GpuMmaFragmentLoadNode::PTR_INPUT_IDX, wide));
}

// Per-wave slots fold into one block tile, staged by the whole block (all lanes of all waves).
TEST(ROCMMMATest, LocalStorageOnWaveMapSharesCopyAcrossLanes_gfx90a) {
    auto& arch = gpu::rocm::ROCM_ARCH_GFX90A;
    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto [block, matmul_node] = build_offloaded_mma_structure(builder, 1024, 1024, 1024, 32, 32, 16, arch);
    passes::expansion::expand_single_node(builder, block, matmul_node, gpu::GpuMmaExpander(&arch));

    strip_mine_k(builder, 2);
    localize(builder, "tile_k0");
    EXPECT_NO_THROW(builder.subject().validate());

    // Both operands: whole-block copy (2x2 waves x 64 lanes), no per-wave slot in the target.
    for (const char* operand : {"A", "B"}) {
        auto* copy = find_copy_from(builder.subject().root(), operand);
        ASSERT_NE(copy, nullptr) << operand;
        EXPECT_TRUE(copy->coop_axes().empty()) << operand;
        EXPECT_EQ(copy->coop_lanes(), 1u) << operand;
        EXPECT_TRUE(symbolic::eq(copy->coop_threads(), symbolic::integer(256))) << copy->coop_threads()->__str__();
        EXPECT_EQ(copy->plan().dst.offset()->__str__().find("wave_"), std::string::npos) << operand;
    }

    // One block tile of 32 padded rows per operand. A fragments read 8 B k-runs per lane, so
    // 4 halves spread a phase's rows over all banks; B fragments read k strided, where 4 or 8
    // halves are equally conflict-free and 8 keeps rows 16-byte aligned.
    for (const auto& name : builder.subject().containers()) {
        if (name.rfind("__daisy_local_storage_", 0) == 0) {
            auto& type = static_cast<const types::Array&>(builder.subject().type(name));
            EXPECT_TRUE(symbolic::eq(type.num_elements(), symbolic::integer(32))) << name;
            auto* row = dynamic_cast<const types::Array*>(&type.element_type());
            ASSERT_NE(row, nullptr) << name;
            const int64_t padded = name.rfind("__daisy_local_storage_A", 0) == 0 ? 36 : 40;
            EXPECT_TRUE(symbolic::eq(row->num_elements(), symbolic::integer(padded))) << name;
        }
    }

    auto out = test::utils::test_codegen(builder.subject(), "wave_map_ls", true);
    std::string kernels;
    for (auto& [name, snippet] : out.snippets) {
        kernels += snippet.content;
    }
    // Re-staged every panel: a leading barrier fences the previous panel's reads.
    auto panel = kernels.find("for(tile_k0_tile0");
    ASSERT_NE(panel, std::string::npos) << kernels;
    auto copy = kernels.find("__tc_tid", panel);
    auto barrier = kernels.find("__syncthreads", panel);
    ASSERT_NE(copy, std::string::npos) << kernels;
    EXPECT_LT(barrier, copy) << kernels;
}

// Steps 4+5: K strip-mined into 32-wide panels (2 MMA blocks); each fragment reads its
// wave's 16x16 window of the padded block-tile buffers.
TEST(ROCMMMATest, LocalStorageFragmentsReadOwnSlotWindow_gfx90a) {
    auto& arch = gpu::rocm::ROCM_ARCH_GFX90A;
    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto [block, matmul_node] = build_offloaded_mma_structure(builder, 1024, 1024, 1024, 32, 32, 16, arch);
    passes::expansion::expand_single_node(builder, block, matmul_node, gpu::GpuMmaExpander(&arch));
    strip_mine_k(builder, 2);
    localize(builder, "tile_k0");
    EXPECT_NO_THROW(builder.subject().validate());

    analysis::AnalysisManager am(builder.subject());
    auto wave_row = find_loop(am, "wave_row0")->indvar();
    auto wave_col = find_loop(am, "wave_col0")->indvar();
    // Element column of the fragment within the panel: 16 * (block - panel start).
    auto k_in_panel = symbolic::
        mul(symbolic::integer(16),
            symbolic::sub(find_loop(am, "tile_k0")->indvar(), find_loop(am, "tile_k0_tile0")->indvar()));

    // A tile [32 m][32+4], B tile [32 k][32+8].
    auto* a = find_local_load(builder.subject().root(), "A");
    ASSERT_NE(a, nullptr);
    EXPECT_TRUE(
        symbolic::
            eq(a->layout().offset,
               symbolic::expand(symbolic::add(symbolic::mul(symbolic::integer(576), wave_row), k_in_panel)))
    ) << a->layout().offset->__str__();
    EXPECT_TRUE(symbolic::eq(a->layout().ldstride, symbolic::integer(36)));

    auto* b = find_local_load(builder.subject().root(), "B");
    ASSERT_NE(b, nullptr);
    EXPECT_TRUE(
        symbolic::
            eq(b->layout().offset,
               symbolic::expand(
                   symbolic::
                       add(symbolic::mul(symbolic::integer(16), wave_col),
                           symbolic::mul(symbolic::integer(40), k_in_panel))
               ))
    ) << b->layout().offset->__str__();
    EXPECT_TRUE(symbolic::eq(b->layout().ldstride, symbolic::integer(40)));
}

// Staging around the whole K-block loop: the block tile holds the full K panel.
TEST(ROCMMMATest, LocalStorageFragmentsReadSubWindow_gfx90a) {
    auto& arch = gpu::rocm::ROCM_ARCH_GFX90A;
    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto [block, matmul_node] = build_offloaded_mma_structure(builder, 1024, 1024, 64, 32, 32, 16, arch);
    passes::expansion::expand_single_node(builder, block, matmul_node, gpu::GpuMmaExpander(&arch));
    localize(builder, "tile_k0");
    EXPECT_NO_THROW(builder.subject().validate());

    analysis::AnalysisManager am(builder.subject());
    auto wave_row = find_loop(am, "wave_row0")->indvar();
    auto wave_col = find_loop(am, "wave_col0")->indvar();
    auto tile_k = find_loop(am, "tile_k0")->indvar();

    // A tile = [32][64+4]; the fragment window starts at row 16*wave_row, column 16*tile_k.
    auto* a = find_local_load(builder.subject().root(), "A");
    ASSERT_NE(a, nullptr);
    EXPECT_TRUE(
        symbolic::eq(
            a->layout().offset,
            symbolic::add(symbolic::mul(symbolic::integer(1088), wave_row), symbolic::mul(symbolic::integer(16), tile_k))
        )
    ) << a->layout().offset->__str__();
    EXPECT_TRUE(symbolic::eq(a->layout().ldstride, symbolic::integer(68)));

    // B tile = [64][32+8]; the fragment window starts at row 16*tile_k, column 16*wave_col.
    auto* b = find_local_load(builder.subject().root(), "B");
    ASSERT_NE(b, nullptr);
    EXPECT_TRUE(
        symbolic::eq(
            b->layout().offset,
            symbolic::add(symbolic::mul(symbolic::integer(16), wave_col), symbolic::mul(symbolic::integer(640), tile_k))
        )
    ) << b->layout().offset->__str__();
    EXPECT_TRUE(symbolic::eq(b->layout().ldstride, symbolic::integer(40)));
}

// Double-buffered panels: fragments select the stage by an offset bias (not a memlet
// subscript), and per-thread-slot copies stay synchronous (no CDNA global_load_lds).
TEST(ROCMMMATest, SoftwarePipeliningBiasesFragmentStage_gfx90a) {
    auto& arch = gpu::rocm::ROCM_ARCH_GFX90A;
    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto [block, matmul_node] = build_offloaded_mma_structure(builder, 1024, 1024, 1024, 32, 32, 16, arch);
    passes::expansion::expand_single_node(builder, block, matmul_node, gpu::GpuMmaExpander(&arch));
    strip_mine_k(builder, 2);
    localize(builder, "tile_k0");

    auto offset_before = find_local_load(builder.subject().root(), "A")->layout().offset;
    symbolic::Expression a_stage_stride = symbolic::one();
    for (const auto& name : builder.subject().containers()) {
        if (name.rfind("__daisy_local_storage_A", 0) == 0) {
            const types::IType* t = &builder.subject().type(name);
            while (auto* arr = dynamic_cast<const types::Array*>(t)) {
                a_stage_stride = symbolic::mul(a_stage_stride, arr->num_elements());
                t = &arr->element_type();
            }
        }
    }

    {
        analysis::AnalysisManager am(builder.subject());
        transformations::SoftwarePipelining sp(*find_loop(am, "tile_k0_tile0"), 2);
        ASSERT_TRUE(sp.can_be_applied(builder, am));
        sp.apply(builder, am);
    }
    EXPECT_NO_THROW(builder.subject().validate());

    analysis::AnalysisManager am(builder.subject());
    auto panel = find_loop(am, "tile_k0_tile0");
    auto stage = symbolic::mod(symbolic::div(panel->indvar(), symbolic::integer(2)), symbolic::integer(2));
    auto* a = find_local_load(builder.subject().root(), "A");
    ASSERT_NE(a, nullptr);
    EXPECT_TRUE(
        symbolic::
            eq(symbolic::expand(symbolic::sub(a->layout().offset, offset_before)),
               symbolic::expand(symbolic::mul(stage, a_stage_stride)))
    ) << a->layout().offset->__str__();

    auto out = test::utils::test_codegen(builder.subject(), "pipelined", true);
    std::string kernels;
    for (auto& [name, snippet] : out.snippets) {
        kernels += snippet.content;
    }
    EXPECT_EQ(kernels.find("global_load_lds"), std::string::npos) << kernels;
    EXPECT_NE(kernels.find("load_matrix_sync(mma_a0, (reinterpret_cast"), std::string::npos) << kernels;
}

// A/B fragments declared with the memory layout must use the layout-less overload (the runtime-layout one
// round-trips through a temporary); accumulators have no layout and must keep it.
TEST(ROCMMMATest, FragmentLoadOverload_gfx90a) {
    auto& arch = gpu::rocm::ROCM_ARCH_GFX90A;
    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto [block, matmul_node] = build_offloaded_mma_structure(builder, 1024, 1024, 1024, 32, 32, 16, arch);
    passes::expansion::expand_single_node(builder, block, matmul_node, gpu::GpuMmaExpander(&arch));

    auto out = test::utils::test_codegen(builder.subject(), "overload", true);
    std::string kernels;
    for (auto& [name, snippet] : out.snippets) {
        kernels += snippet.content;
    }
    auto load_line = [&](const std::string& frag) {
        auto pos = kernels.find("load_matrix_sync(" + frag + ",");
        EXPECT_NE(pos, std::string::npos) << frag << "\n" << kernels;
        return pos == std::string::npos ? std::string() : kernels.substr(pos, kernels.find('\n', pos) - pos);
    };
    EXPECT_EQ(load_line("mma_a0").find("rocwmma::mem_"), std::string::npos) << kernels;
    EXPECT_EQ(load_line("mma_b0").find("rocwmma::mem_"), std::string::npos) << kernels;
    // A MatMul overwrites C: it never loads it, and stores the accumulator with its memory layout.
    EXPECT_EQ(kernels.find("load_matrix_sync(mma_c0"), std::string::npos) << kernels;
    auto store = kernels.find("store_matrix_sync(");
    ASSERT_NE(store, std::string::npos) << kernels;
    EXPECT_NE(kernels.substr(store, kernels.find('\n', store) - store).find("rocwmma::mem_row_major"), std::string::npos)
        << kernels;
}

// K-major B: the transposed block tile [32 n][32+4 k] is read column-major by the fragments,
// and TileVectorizer stages it with the 4x4 coalesced block copy.
TEST(ROCMMMATest, LocalStorageTransposedBStagesKMajor_gfx90a) {
    auto& arch = gpu::rocm::ROCM_ARCH_GFX90A;
    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto [block, matmul_node] = build_offloaded_mma_structure(builder, 1024, 1024, 1024, 32, 32, 16, arch);
    passes::expansion::expand_single_node(builder, block, matmul_node, gpu::GpuMmaExpander(&arch));
    strip_mine_k(builder, 2);
    localize(builder, "tile_k0", /*transpose_b=*/true);
    EXPECT_NO_THROW(builder.subject().validate());

    for (const auto& name : builder.subject().containers()) {
        if (name.rfind("__daisy_local_storage_B", 0) == 0) {
            auto& type = static_cast<const types::Array&>(builder.subject().type(name));
            EXPECT_TRUE(symbolic::eq(type.num_elements(), symbolic::integer(32))) << name;
            auto& row = static_cast<const types::Array&>(type.element_type());
            EXPECT_TRUE(symbolic::eq(row.num_elements(), symbolic::integer(36))) << name;
        }
    }

    analysis::AnalysisManager am(builder.subject());
    auto wave_col = find_loop(am, "wave_col0")->indvar();
    auto k_in_panel = symbolic::
        mul(symbolic::integer(16),
            symbolic::sub(find_loop(am, "tile_k0")->indvar(), find_loop(am, "tile_k0_tile0")->indvar()));
    auto* b = find_local_load(builder.subject().root(), "B");
    ASSERT_NE(b, nullptr);
    EXPECT_EQ(b->layout().layout, gpu::MMA_LAYOUT_COL_MAJOR);
    EXPECT_TRUE(symbolic::eq(b->layout().ldstride, symbolic::integer(36)));
    EXPECT_TRUE(
        symbolic::
            eq(b->layout().offset,
               symbolic::expand(symbolic::add(symbolic::mul(symbolic::integer(576), wave_col), k_in_panel)))
    ) << b->layout().offset->__str__();

    {
        transformations::TileVectorizer tv(*find_loop(am, "tile_k0_tile0"));
        ASSERT_TRUE(tv.can_be_applied(builder, am));
        tv.apply(builder, am);
    }
    auto* copy_b = find_copy_from(builder.subject().root(), "B");
    ASSERT_NE(copy_b, nullptr);
    EXPECT_EQ(copy_b->atom(), tiles::CopyAtom::CoalesceSync);
    EXPECT_EQ(copy_b->bytes(), 8u);
    auto* copy_a = find_copy_from(builder.subject().root(), "A");
    ASSERT_NE(copy_a, nullptr);
    EXPECT_EQ(copy_a->atom(), tiles::CopyAtom::VectorSync);

    auto out = test::utils::test_codegen(builder.subject(), "kmajor_b", true);
    std::string kernels;
    for (auto& [name, snippet] : out.snippets) {
        kernels += snippet.content;
    }
    EXPECT_NE(kernels.find("make_uint2"), std::string::npos) << kernels;
    EXPECT_NE(kernels.find("rocwmma::mem_col_major"), std::string::npos) << kernels;
}

// Register-staged pipelining: panel 0 staged in a prologue; in the loop the next panel is
// loaded into per-thread registers before the compute and stored after a barrier.
TEST(ROCMMMATest, SoftwarePipeliningRegisterStaged_gfx90a) {
    auto& arch = gpu::rocm::ROCM_ARCH_GFX90A;
    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto [block, matmul_node] = build_offloaded_mma_structure(builder, 1024, 1024, 1024, 32, 32, 16, arch);
    passes::expansion::expand_single_node(builder, block, matmul_node, gpu::GpuMmaExpander(&arch));
    strip_mine_k(builder, 2);
    localize(builder, "tile_k0", /*transpose_b=*/true);
    {
        analysis::AnalysisManager am(builder.subject());
        transformations::SoftwarePipelining sp(*find_loop(am, "tile_k0_tile0"), 2, false, /*register_staged=*/true);
        ASSERT_TRUE(sp.can_be_applied(builder, am));
        sp.apply(builder, am);
    }
    EXPECT_NO_THROW(builder.subject().validate());

    size_t stages = 0;
    for (const auto& name : builder.subject().containers()) {
        if (name.rfind("__daisy_stage_", 0) == 0) {
            ++stages;
            auto& type = static_cast<const types::Array&>(builder.subject().type(name));
            EXPECT_EQ(type.element_type().primitive_type(), types::PrimitiveType::UInt32) << name;
        }
        if (name.rfind("__daisy_local_storage_", 0) == 0) {
            // Still one shared buffer per operand: no stage axis.
            auto& type = static_cast<const types::Array&>(builder.subject().type(name));
            EXPECT_TRUE(symbolic::eq(type.num_elements(), symbolic::integer(32))) << name;
        }
    }
    EXPECT_EQ(stages, 2u);

    // Loop body: [if load; compute; barrier; if store; barrier], no full copies inside.
    analysis::AnalysisManager am(builder.subject());
    auto* panel = find_loop(am, "tile_k0_tile0");
    size_t loads = 0, stores = 0, full = 0;
    visitor::for_each_block(panel->root(), [&](structured_control_flow::Block& b) {
        for (auto* lib : b.dataflow().library_nodes()) {
            if (auto* tc = dynamic_cast<tiles::TileCopyNode*>(lib)) {
                loads += tc->phase() == tiles::CopyPhase::LoadRegs;
                stores += tc->phase() == tiles::CopyPhase::StoreRegs;
                full += tc->phase() == tiles::CopyPhase::Full;
            }
        }
    });
    EXPECT_EQ(loads, 2u);
    EXPECT_EQ(stores, 2u);
    EXPECT_EQ(full, 0u);
    {
        // The phases survive a JSON round trip (the gym replays SDFGs through JSON).
        serializer::JSONSerializer ser;
        auto j = ser.serialize(builder.subject());
        auto copy = ser.deserialize(j);
        size_t staged = 0;
        visitor::for_each_block(copy->root(), [&](structured_control_flow::Block& b) {
            for (auto* lib : b.dataflow().library_nodes()) {
                if (auto* tc = dynamic_cast<tiles::TileCopyNode*>(lib)) {
                    staged += tc->phase() != tiles::CopyPhase::Full;
                }
            }
        });
        EXPECT_EQ(staged, 4u);
    }
    auto* body = &panel->root();
    while (body->size() == 1 && dynamic_cast<structured_control_flow::Sequence*>(&body->at(0))) {
        body = static_cast<structured_control_flow::Sequence*>(&body->at(0));
    }
    ASSERT_EQ(body->size(), 5u);
    EXPECT_NE(dynamic_cast<structured_control_flow::IfElse*>(&body->at(0)), nullptr);
    EXPECT_NE(dynamic_cast<structured_control_flow::StructuredLoop*>(&body->at(1)), nullptr);
    EXPECT_NE(dynamic_cast<structured_control_flow::IfElse*>(&body->at(3)), nullptr);

    {
        // Vectorize from the wave map so the prologue copies outside the panel loop widen too.
        transformations::TileVectorizer tv(*find_loop(am, "wave_col0"));
        ASSERT_TRUE(tv.can_be_applied(builder, am));
        tv.apply(builder, am);
    }
    visitor::for_each_block(builder.subject().root(), [&](structured_control_flow::Block& b) {
        for (auto* lib : b.dataflow().library_nodes()) {
            if (auto* tc = dynamic_cast<tiles::TileCopyNode*>(lib)) {
                EXPECT_NE(tc->atom(), tiles::CopyAtom::ScalarSync) << tc->toStr();
            }
        }
    });

    auto out = test::utils::test_codegen(builder.subject(), "reg_staged", true);
    std::string kernels;
    for (auto& [name, snippet] : out.snippets) {
        kernels += snippet.content;
    }
    EXPECT_NE(kernels.find("__daisy_stage_"), std::string::npos) << kernels;
    EXPECT_NE(kernels.find("#pragma unroll"), std::string::npos) << kernels;
    EXPECT_NE(kernels.find("__builtin_amdgcn_sched_barrier(0);"), std::string::npos) << kernels;
    // A's padded rows (36 halves) are 8- but not 16-byte aligned: 16-byte global loads,
    // stored to LDS in 8-byte pieces.
    EXPECT_NE(kernels.find("const uint4 __tc_v"), std::string::npos) << kernels;
    EXPECT_NE(kernels.find("reinterpret_cast<int2*>"), std::string::npos) << kernels;
    EXPECT_EQ(kernels.find("*reinterpret_cast<uint4*>"), std::string::npos) << kernels;
}

// Raw 32x32x8 MFMA atom: only CDNA offers it; fragments are register vectors and every operand
// read from LDS (A row-major, K-major B) is one 8-byte access per lane.
TEST(ROCMMMATest, RawMfma32x32x8_gfx90a) {
    auto block_hint = gpu::MmaBlockSize::parse_block_size("32x32x8");
    EXPECT_EQ(block_hint.m, 32);
    EXPECT_EQ(block_hint.n, 32);
    EXPECT_EQ(block_hint.k, 8);
    auto& arch = gpu::rocm::ROCM_ARCH_GFX90A;

    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto [block, matmul_node] = build_offloaded_mma_structure(builder, 1024, 1024, 1024, 64, 64, 16, arch);
    passes::expansion::expand_single_node(builder, block, matmul_node, gpu::GpuMmaExpander(&arch, &block_hint));
    EXPECT_NO_THROW(builder.subject().validate());

    size_t fragments = 0;
    for (const auto& name : builder.subject().containers()) {
        if (name.rfind("mma_", 0) == 0) {
            ++fragments;
            EXPECT_EQ(builder.subject().type(name).storage_type().value(), gpu::rocm::RocmMmaSupport::MMA_REGS_STORAGE_TYPE)
                << name;
        }
    }
    EXPECT_GT(fragments, 0u);

    strip_mine_k(builder, 8);
    localize(builder, "tile_k0", /*transpose_b=*/true);
    EXPECT_NO_THROW(builder.subject().validate());
    // 8 B lane reads over 16-lane phases: 64 + 4 halves per row spread a phase over all banks.
    for (const auto& name : builder.subject().containers()) {
        if (name.rfind("__daisy_local_storage_", 0) == 0) {
            auto& type = static_cast<const types::Array&>(builder.subject().type(name));
            auto& row = static_cast<const types::Array&>(type.element_type());
            EXPECT_TRUE(symbolic::eq(row.num_elements(), symbolic::integer(68))) << name;
        }
    }
    auto* b = find_local_load(builder.subject().root(), "B");
    ASSERT_NE(b, nullptr);
    EXPECT_EQ(b->layout().layout, gpu::MMA_LAYOUT_COL_MAJOR);
    EXPECT_EQ(b->implementation_type(), gpu::rocm::ImplementationType_ROCM_MMA_BARE);

    auto out = test::utils::test_codegen(builder.subject(), "mfma32", true);
    std::string kernels;
    for (auto& [name, snippet] : out.snippets) {
        kernels += snippet.content;
    }
    EXPECT_EQ(kernels.find("rocwmma::"), std::string::npos) << kernels;
    EXPECT_NE(kernels.find("__builtin_amdgcn_mfma_f32_32x32x8f16"), std::string::npos) << kernels;
    EXPECT_NE(kernels.find("_Float16 __attribute__((ext_vector_type(4))) mma_a0"), std::string::npos) << kernels;
    EXPECT_NE(kernels.find("float __attribute__((ext_vector_type(16))) mma_acc0"), std::string::npos) << kernels;
    // A and B fragment reads from LDS: one vector access each.
    size_t vector_reads = 0;
    for (auto pos = kernels.find("__typeof__(mma_"); pos != std::string::npos;
         pos = kernels.find("__typeof__(mma_", pos + 1)) {
        ++vector_reads;
    }
    EXPECT_GE(vector_reads, 2u) << kernels;
}

TEST(ROCMMMATest, RawMfma32x32x8_128x96Tile_LaunchBounds) {
    auto block_hint = gpu::MmaBlockSize::parse_block_size("32x32x8");
    EXPECT_EQ(block_hint.m, 32);
    EXPECT_EQ(block_hint.n, 32);
    EXPECT_EQ(block_hint.k, 8);
    auto& arch = gpu::rocm::ROCM_ARCH_GFX90A;
    auto tiling = arch.mma_support()->get_mma_tiling(
        {symbolic::integer(128), symbolic::integer(96), symbolic::integer(1024)},
        types::PrimitiveType::Half,
        types::PrimitiveType::Float,
        &block_hint,
        nullptr
    );
    ASSERT_TRUE(tiling.has_value());
    EXPECT_EQ(tiling->macro_tiling.blocks_m, 4);
    EXPECT_EQ(tiling->macro_tiling.blocks_n, 3);

    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto [block, matmul_node] = build_offloaded_mma_structure(builder, 1024, 1152, 1024, 128, 96, 16, arch);
    passes::expansion::expand_single_node(builder, block, matmul_node, gpu::GpuMmaExpander(&arch, &block_hint));
    EXPECT_NO_THROW(builder.subject().validate());

    auto out = test::utils::test_codegen(builder.subject(), "mfma32_128x96", true);
    std::string kernels;
    for (auto& [name, snippet] : out.snippets) {
        kernels += snippet.content;
    }
    // 4 x 3 waves of 64 lanes.
    EXPECT_NE(kernels.find("__global__ void __launch_bounds__(768) kernel_"), std::string::npos) << kernels;
}

// K-major B (C = A @ W^T): the B tile is [k][n] with global strides [1, K]. The copy must sweep the
// unit-stride k fastest across lanes and widen like A; sweeping n fastest reads 2 KB apart per lane.
TEST(ROCMMMATest, LocalStorageKMajorBCopyIsCoalesced_gfx90a) {
    auto& arch = gpu::rocm::ROCM_ARCH_GFX90A;
    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto [block, matmul_node] =
        build_offloaded_mma_structure(builder, 1024, 1024, 1024, 32, 32, 16, arch, /*b_k_major=*/true);
    passes::expansion::expand_single_node(builder, block, matmul_node, gpu::GpuMmaExpander(&arch));
    strip_mine_k(builder, 2);
    localize(builder, "tile_k0");
    EXPECT_NO_THROW(builder.subject().validate());

    auto* b = find_local_load(builder.subject().root(), "B");
    ASSERT_NE(b, nullptr);
    EXPECT_EQ(b->layout().layout, gpu::MMA_LAYOUT_COL_MAJOR);

    {
        analysis::AnalysisManager am(builder.subject());
        transformations::TileVectorizer tv(*find_loop(am, "tile_k0_tile0"));
        ASSERT_TRUE(tv.can_be_applied(builder, am));
        tv.apply(builder, am);
    }
    auto* copy_a = find_copy_from(builder.subject().root(), "A");
    ASSERT_NE(copy_a, nullptr);
    EXPECT_EQ(copy_a->atom(), tiles::CopyAtom::VectorSync) << copy_a->toStr();
    EXPECT_EQ(copy_a->bytes(), 16u) << copy_a->toStr();

    auto* copy_b = find_copy_from(builder.subject().root(), "B");
    ASSERT_NE(copy_b, nullptr);
    // emit_cooperative_copy_loop delinearizes the lane index row-major: the last dim is lane-fastest.
    EXPECT_TRUE(symbolic::eq(copy_b->plan().src.strides().back(), symbolic::one())) << copy_b->toStr();
    EXPECT_EQ(copy_b->atom(), tiles::CopyAtom::VectorSync) << copy_b->toStr();
    EXPECT_EQ(copy_b->bytes(), 16u) << copy_b->toStr();

    auto out = test::utils::test_codegen(builder.subject(), "kmajor_b_global", true);
    std::string kernels;
    for (auto& [name, snippet] : out.snippets) {
        kernels += snippet.content;
    }
    EXPECT_EQ(
        kernels.find("(B))[1024*block_col + 16*tile_k0_tile0 + ((__tc_c) / (32)) + 1024*((__tc_c) % (32))]"),
        std::string::npos
    ) << kernels;
}

// fp32 K-major staging of a row-major B: 4x4 blocks load as coalesced 16-byte rows, regrouped in registers.
TEST(ROCMMMATest, LocalStorageCoalescedBStagesKMajorFp32_gfx90a) {
    auto& arch = gpu::rocm::ROCM_ARCH_GFX90A;
    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto [block, matmul_node] = build_offloaded_mma_structure(
        builder,
        1024,
        1024,
        1024,
        32,
        32,
        16,
        arch,
        /*b_k_major=*/false,
        types::PrimitiveType::Float,
        types::PrimitiveType::Float
    );
    ASSERT_TRUE(passes::expansion::expand_single_node(builder, block, matmul_node, gpu::GpuMmaExpander(&arch)).expanded);
    strip_mine_k(builder, 2);
    localize(builder, "tile_k0", /*transpose_b=*/true);
    EXPECT_NO_THROW(builder.subject().validate());

    {
        analysis::AnalysisManager am(builder.subject());
        transformations::TileVectorizer tv(*find_loop(am, "tile_k0_tile0"));
        ASSERT_TRUE(tv.can_be_applied(builder, am));
        tv.apply(builder, am);
    }
    auto* copy_b = find_copy_from(builder.subject().root(), "B");
    ASSERT_NE(copy_b, nullptr);
    EXPECT_EQ(copy_b->atom(), tiles::CopyAtom::CoalesceSync) << copy_b->toStr();
    EXPECT_EQ(copy_b->bytes(), 16u) << copy_b->toStr();

    auto out = test::utils::test_codegen(builder.subject(), "kmajor_b_fp32", true);
    std::string kernels;
    for (auto& [name, snippet] : out.snippets) {
        kernels += snippet.content;
    }
    EXPECT_NE(kernels.find("const uint4 __tc_v = *reinterpret_cast<const uint4*>"), std::string::npos) << kernels;
    EXPECT_NE(kernels.find("*reinterpret_cast<uint4*>"), std::string::npos) << kernels;
    EXPECT_NE(kernels.find("make_uint4(__tc_x[0][0], __tc_x[1][0], __tc_x[2][0], __tc_x[3][0])"), std::string::npos)
        << kernels;
}

// Register-staged pipelining of the fp32 coalesced block copy: each lane holds 16 words per 4x4 block.
TEST(ROCMMMATest, SoftwarePipeliningRegisterStagedCoalesceFp32_gfx90a) {
    auto& arch = gpu::rocm::ROCM_ARCH_GFX90A;
    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto [block, matmul_node] = build_offloaded_mma_structure(
        builder,
        1024,
        1024,
        1024,
        32,
        32,
        16,
        arch,
        /*b_k_major=*/false,
        types::PrimitiveType::Float,
        types::PrimitiveType::Float
    );
    ASSERT_TRUE(passes::expansion::expand_single_node(builder, block, matmul_node, gpu::GpuMmaExpander(&arch)).expanded);
    strip_mine_k(builder, 2);
    localize(builder, "tile_k0", /*transpose_b=*/true);
    {
        analysis::AnalysisManager am(builder.subject());
        transformations::SoftwarePipelining sp(*find_loop(am, "tile_k0_tile0"), 2, false, /*register_staged=*/true);
        ASSERT_TRUE(sp.can_be_applied(builder, am));
        sp.apply(builder, am);
    }
    {
        analysis::AnalysisManager am(builder.subject());
        transformations::TileVectorizer tv(*find_loop(am, "wave_col0"));
        ASSERT_TRUE(tv.can_be_applied(builder, am));
        tv.apply(builder, am);
    }
    size_t coalesced = 0;
    visitor::for_each_block(builder.subject().root(), [&](structured_control_flow::Block& b) {
        for (auto* lib : b.dataflow().library_nodes()) {
            if (auto* tc = dynamic_cast<tiles::TileCopyNode*>(lib)) {
                EXPECT_NE(tc->atom(), tiles::CopyAtom::ScalarSync) << tc->toStr();
                coalesced += tc->atom() == tiles::CopyAtom::CoalesceSync;
            }
        }
    });
    EXPECT_GT(coalesced, 0u);

    auto out = test::utils::test_codegen(builder.subject(), "reg_staged_fp32_coalesce", true);
    std::string kernels;
    for (auto& [name, snippet] : out.snippets) {
        kernels += snippet.content;
    }
    EXPECT_NE(kernels.find("[16 * __tc_i + 15]"), std::string::npos) << kernels;
    EXPECT_NE(kernels.find("make_uint4("), std::string::npos) << kernels;
}

} // namespace sdfg::rocm

namespace sdfg::rocm {

namespace {

std::string int8_kernels(const gpu::rocm::RocmArch& arch, const std::string& group) {
    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg_int8", FunctionType_CPU);
    auto [block, matmul_node] = build_offloaded_mma_structure(
        builder,
        1024,
        1024,
        1024,
        32,
        32,
        16,
        arch,
        /*b_k_major=*/false,
        types::PrimitiveType::Int8,
        types::PrimitiveType::Int32
    );
    EXPECT_NO_THROW(builder.subject().validate());
    auto outcome = passes::expansion::expand_single_node(builder, block, matmul_node, gpu::GpuMmaExpander(&arch));
    EXPECT_TRUE(outcome.expanded);
    EXPECT_NO_THROW(builder.subject().validate());

    std::string kernels;
    for (auto& [name, snippet] : test::utils::test_codegen(builder.subject(), group, true).snippets) {
        kernels += snippet.content;
    }
    return kernels;
}

} // namespace

TEST(ROCMMMATest, Int8_FullRateBlocks_gfx942) {
    // CDNA3's native Int8 MFMA is 16x16x32; rocWMMA would pad 16x16x16 to it at half the rate.
    auto kernels = int8_kernels(gpu::rocm::ROCM_ARCH_GFX942, "int8_gfx942");
    EXPECT_NE(kernels.find("rocwmma::matrix_a, 16, 16, 32, int8_t"), std::string::npos) << kernels;
    EXPECT_NE(kernels.find("rocwmma::matrix_b, 16, 16, 32, int8_t"), std::string::npos) << kernels;
    EXPECT_NE(kernels.find("rocwmma::accumulator, 16, 16, 32, int32_t> mma_acc"), std::string::npos) << kernels;
    EXPECT_NE(kernels.find("rocwmma::mma_sync("), std::string::npos) << kernels;
    // The Int32 accumulator is the output: C is neither loaded nor added, since a MatMul overwrites it.
    EXPECT_EQ(kernels.find("mma_c"), std::string::npos) << kernels;
}

TEST(ROCMMMATest, Int8_NativeBlocks_gfx90a) {
    auto kernels = int8_kernels(gpu::rocm::ROCM_ARCH_GFX90A, "int8_gfx90a");
    EXPECT_NE(kernels.find("rocwmma::matrix_a, 16, 16, 16, int8_t"), std::string::npos) << kernels;
    EXPECT_NE(kernels.find("rocwmma::accumulator, 16, 16, 16, int32_t> mma_acc"), std::string::npos) << kernels;
}

TEST(ROCMMMATest, Int8_Support) {
    auto* mma = gpu::rocm::ROCM_ARCH_GFX942.mma_support();
    ASSERT_NE(mma, nullptr);
    EXPECT_TRUE(mma->supported_types(types::PrimitiveType::Int8, types::PrimitiveType::Int32));
    // An Int8 output would wrap the Int32 sums.
    EXPECT_FALSE(mma->supported_types(types::PrimitiveType::Int8, types::PrimitiveType::Int8));
    EXPECT_EQ(
        mma->get_accumulator_type(types::PrimitiveType::Int8, types::PrimitiveType::Int32, types::PrimitiveType::Void),
        types::PrimitiveType::Int32
    );
    EXPECT_TRUE(mma->is_valid_block_size({16, 16, 32}, types::PrimitiveType::Int8, types::PrimitiveType::Int32));
    EXPECT_TRUE(mma->is_valid_block_size({16, 16, 16}, types::PrimitiveType::Int8, types::PrimitiveType::Int32));
    EXPECT_FALSE(mma->is_valid_block_size({32, 32, 8}, types::PrimitiveType::Int8, types::PrimitiveType::Int32));
    EXPECT_FALSE(
        gpu::rocm::ROCM_ARCH_GFX90A.mma_support()
            ->is_valid_block_size({16, 16, 32}, types::PrimitiveType::Int8, types::PrimitiveType::Int32)
    );

    // K = 48 only divides by 16: the half-rate block still applies.
    auto tiling = mma->get_mma_tiling(
        {symbolic::integer(64), symbolic::integer(64), symbolic::integer(48)},
        types::PrimitiveType::Int8,
        types::PrimitiveType::Int32,
        nullptr,
        nullptr
    );
    ASSERT_TRUE(tiling.has_value());
    EXPECT_EQ(tiling->mma_block_size.k, 16);
}

} // namespace sdfg::rocm

namespace sdfg {

TEST(ROCMMMATest, ClippedFragmentLayoutReportsClippedShape) {
    auto s16 = symbolic::integer(16);
    auto rows = symbolic::sub(symbolic::integer(1000), symbolic::symbol("i0"));
    gpu::GpuMmaFromMemoryLayout layout{
        .offset = symbolic::zero(),
        .ldstride = symbolic::integer(1000),
        .layout = gpu::MMA_LAYOUT_ROW_MAJOR,
        .rows = rows,
    };
    ASSERT_TRUE(layout.clipped());
    auto t = layout.to_tensor_layout(gpu::MmaBlockSize(16, 16, 16), gpu::MmaFragmentType::A);
    ASSERT_TRUE(t.has_value());
    EXPECT_TRUE(symbolic::eq(t->shape()[0], symbolic::min(s16, rows)));
    EXPECT_TRUE(symbolic::eq(t->shape()[1], s16));
    // Both the block shape (a padded local buffer) and the clipped shape relocalize.
    EXPECT_TRUE(layout.matches_access_shape({s16, s16}, gpu::MmaBlockSize(16, 16, 16), gpu::MmaFragmentType::A));
    EXPECT_TRUE(layout.matches_access_shape(t->shape(), gpu::MmaBlockSize(16, 16, 16), gpu::MmaFragmentType::A));

    nlohmann::json j;
    gpu::serialize_mma_from_memory_layout(j, layout);
    auto back = gpu::deserialize_mma_from_memory_layout(j);
    ASSERT_FALSE(back.rows.is_null());
    EXPECT_TRUE(symbolic::eq(back.rows, rows));
    EXPECT_TRUE(back.cols.is_null());
}

TEST(ROCMMMATest, RaggedTilingCoversPaddedExtents) {
    auto* mma = gpu::rocm::ROCM_ARCH_GFX90A.mma_support();
    gpu::MmaBlockSize b16(16, 16, 16);
    auto exact16 = mma->get_mma_tiling(
        {symbolic::integer(64), symbolic::integer(64), symbolic::integer(1000)},
        types::PrimitiveType::Half,
        types::PrimitiveType::Float,
        &b16,
        nullptr
    );
    EXPECT_FALSE(exact16.has_value());
    auto ragged = mma->get_ragged_mma_tiling(
        {symbolic::integer(64), symbolic::integer(64), symbolic::integer(1000)},
        types::PrimitiveType::Half,
        types::PrimitiveType::Half,
        types::PrimitiveType::Float,
        &b16,
        nullptr
    );
    ASSERT_TRUE(ragged.has_value());
    EXPECT_EQ(ragged->macro_tiling.blocks_m, 4);
    EXPECT_EQ(ragged->macro_tiling.blocks_n, 4);
}

} // namespace sdfg
