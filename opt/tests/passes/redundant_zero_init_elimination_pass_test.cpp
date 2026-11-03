#include "sdfg/passes/redundant_zero_init_elimination_pass.h"

#include <gtest/gtest.h>

#include "sdfg/analysis/analysis.h"
#include "sdfg/analysis/loop_analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_nodes/barrier_local_node.h"
#include "sdfg/structured_control_flow/for.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/targets/gpu/gpu_mma_fill_node.h"
#include "sdfg/targets/gpu/gpu_mma_fragment_eltwise_add_node.h"
#include "sdfg/targets/gpu/gpu_mma_fragment_load_node.h"
#include "sdfg/targets/gpu/gpu_mma_fragment_store_node.h"
#include "sdfg/targets/gpu/gpu_mma_matmul_node.h"
#include "sdfg/targets/gpu/gpu_offload_schedule_type.h"
#include "sdfg/targets/rocm/rocm.h"
#include "sdfg/targets/rocm/rocm_arch.h"
#include "sdfg/targets/rocm/rocm_mma_fragment_dispatcher.h"
#include "sdfg/tiles/library_nodes/tile_copy_node.h"
#include "sdfg/visitor/for_each.h"

using namespace sdfg;

namespace {

using structured_control_flow::Sequence;
namespace sym = symbolic;

constexpr int M = 2;
constexpr int N = 17920;
constexpr int K = 1536;
const gpu::MmaBlockSize kBlock{16, 16, 16};
const auto& kMma = gpu::rocm::ImplementationType_ROCM_MMA_FRAGMENT;

const std::string kY = "__daisy_hip_100_mm";

types::Pointer half_ptr() {
    return types::Pointer(types::Scalar(types::PrimitiveType::Half));
}
types::Pointer float_ptr() {
    return types::Pointer(types::Scalar(types::PrimitiveType::Float));
}

types::Pointer fragment_type(types::PrimitiveType element, gpu::MmaFragmentType frag, gpu::MmaFragmentLayout layout) {
    types::Pointer type{types::Scalar(element)};
    type.storage_type().value(gpu::rocm::RocmMmaSupport::MMA_FRAGMENT_STORAGE_TYPE);
    type.storage_type().args(
        {sym::integer(kBlock.m),
         sym::integer(kBlock.n),
         sym::integer(kBlock.k),
         sym::integer(static_cast<int>(frag)),
         sym::integer(static_cast<int>(layout))}
    );
    return type;
}

structured_control_flow::ScheduleType rocm_offload(gpu::TargetLevel level, int parallel_size) {
    return gpu::ScheduleType_GPU_Offload::create<rocm::ScheduleType_ROCM_Offload>(level, sym::integer(parallel_size));
}

structured_control_flow::ScheduleType unrolled() {
    auto schedule = structured_control_flow::ScheduleType_Sequential::create();
    structured_control_flow::ScheduleType_Unroll::set(schedule);
    return schedule;
}

structured_control_flow::Map& add_map(
    builder::StructuredSDFGBuilder& builder,
    Sequence& scope,
    const std::string& indvar,
    int bound,
    int step,
    const structured_control_flow::ScheduleType& schedule
) {
    auto iv = sym::symbol(indvar);
    return builder
        .add_map(scope, iv, sym::Lt(iv, sym::integer(bound)), sym::zero(), sym::add(iv, sym::integer(step)), schedule);
}

void add_fragment_load(
    builder::StructuredSDFGBuilder& builder,
    structured_control_flow::Block& block,
    const std::string& frag,
    const std::string& memory,
    gpu::MmaFragmentType frag_type,
    const gpu::GpuMmaFromMemoryLayout& layout
) {
    auto& load = builder.add_library_node<
        gpu::GpuMmaFragmentLoadNode>(block, DebugInfo(), kBlock, frag_type, layout, types::PrimitiveType::Half, kMma);
    builder.add_computational_memlet(
        block,
        builder.add_access(block, frag),
        load,
        load.input(gpu::GpuMmaFragmentLoadNode::FRAG_INPUT_IDX),
        {},
        half_ptr()
    );
    builder.add_computational_memlet(
        block,
        builder.add_access(block, memory),
        load,
        load.input(gpu::GpuMmaFragmentLoadNode::PTR_INPUT_IDX),
        {},
        half_ptr()
    );
}

void add_tile_copy(
    builder::StructuredSDFGBuilder& builder,
    structured_control_flow::Block& block,
    const std::string& buffer,
    const std::string& global,
    int rows,
    const sym::Expression& src_offset,
    tiles::TileGuard guard
) {
    tiles::TiledCopy plan;
    plan.src = tiles::Layout({sym::integer(rows), sym::integer(128)}, {sym::integer(K), sym::one()}, src_offset);
    plan.dst = tiles::Layout({sym::integer(rows), sym::integer(128)}, {sym::integer(132), sym::one()}, sym::zero());
    plan.atom = tiles::CopyAtom::VectorSync;
    guard.tile_sizes = {sym::integer(rows), sym::integer(128)};
    auto& copy = builder.add_library_node<tiles::TileCopyNode>(
        block,
        DebugInfo(),
        rocm::ImplementationType_ROCM,
        plan,
        tiles::CopyDirection::In,
        size_t(16),
        guard,
        std::vector<int>{},
        sym::Expression(sym::integer(256))
    );
    builder.add_computational_memlet(block, builder.add_access(block, buffer), copy, copy.input(0), {}, half_ptr());
    builder.add_computational_memlet(block, builder.add_access(block, global), copy, copy.input(1), {}, half_ptr());
}

/**
 * Gate/up projection of the Qwen MLP after ROCm scheduling (gfx90a): `Y[2, 17920] = x[2, 1536] @ w_gate_up^T`.
 *
 *   map _i0 (X_GRID) / _j0 (X_BLOCK):                       Y[_i0, _j0] = 0           <- naive MatMul init nest
 *   map _j1_tile0 (X_GRID) / _i1_tile0 (Y_GRID) / wave_row0 (X_BLOCK) / wave_col0 (Y_BLOCK):
 *       GpuMmaFill(acc)
 *       for tile_k0_tile0:  barrier; tile_copy x, w -> LDS; barrier
 *           for tile_k0:    MmaFragmentLoad(a, b); GpuMmaMatmul(acc, a, b)
 *       MmaFragmentLoad(c <- Y); MmaFragmentEltwiseAdd(d = acc + c); MmaFragmentStore(d -> Y)
 *
 * The MMA nest is what GpuMmaEinsumTransform (+ LoopTiling, LocalStorage) emits for the accumulation nest.
 */
structured_control_flow::Map& build_qwen_gate_up_matmul(builder::StructuredSDFGBuilder& builder, bool annotate) {
    auto& root = builder.subject().root();
    auto& arch = gpu::rocm::ROCM_ARCH_GFX90A;

    types::Pointer device_half{types::Scalar(types::PrimitiveType::Half)};
    device_half.storage_type() = types::StorageType("AMD_Generic");
    builder.add_container("x", device_half, true);
    builder.add_container("w_gate_up", device_half, true);
    builder.add_container(kY, device_half);

    for (auto& [name, rows] : std::vector<std::pair<std::string, int>>{{"lds_x", 16}, {"lds_w", 64}}) {
        types::Array row(types::Scalar(types::PrimitiveType::Half), sym::integer(132));
        types::Array buffer(row, sym::integer(rows));
        buffer.storage_type() = types::StorageType("NV_Shared");
        builder.add_container(name, buffer);
    }

    builder.add_container(
        "mma_acc0", fragment_type(types::PrimitiveType::Float, gpu::MmaFragmentType::C, gpu::MMA_LAYOUT_UNSPECIFIED)
    );
    builder.add_container(
        "mma_a0", fragment_type(types::PrimitiveType::Half, gpu::MmaFragmentType::A, gpu::MMA_LAYOUT_ROW_MAJOR)
    );
    builder.add_container(
        "mma_b0", fragment_type(types::PrimitiveType::Half, gpu::MmaFragmentType::B, gpu::MMA_LAYOUT_COL_MAJOR)
    );
    for (auto* name : {"mma_c0", "mma_d0"}) {
        builder.add_container(
            name, fragment_type(types::PrimitiveType::Half, gpu::MmaFragmentType::C, gpu::MMA_LAYOUT_UNSPECIFIED)
        );
    }
    for (auto* name : {"_i0", "_j0", "_j1_tile0", "_i1_tile0", "wave_row0", "wave_col0", "tile_k0_tile0", "tile_k0"}) {
        builder.add_container(name, types::Scalar(types::PrimitiveType::Int32));
    }

    // ---- Zero-init nest (naive MatMul expansion, offloaded) ----
    auto init_schedule = rocm_offload(gpu::TargetLevel::X_GRID, M);
    structured_control_flow::ScheduleType_NoFusion::set(init_schedule);
    if (annotate) {
        structured_control_flow::ScheduleType_ZeroInit::set(init_schedule);
    }
    auto& init = add_map(builder, root, "_i0", M, 1, init_schedule);
    auto& init_j = add_map(builder, init.root(), "_j0", N, 1, rocm_offload(gpu::TargetLevel::X_BLOCK, 128));
    {
        auto& block = builder.add_block(init_j.root());
        auto& zero = builder.add_constant(block, "0.0", types::Scalar(types::PrimitiveType::Half));
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
        builder.add_computational_memlet(block, zero, tasklet, "_in", {}, types::Scalar(types::PrimitiveType::Half));
        builder.add_computational_memlet(
            block,
            tasklet,
            "_out",
            builder.add_access(block, kY),
            {sym::add(sym::mul(sym::symbol("_i0"), sym::integer(N)), sym::symbol("_j0"))},
            half_ptr()
        );
    }

    // ---- MMA nest ----
    auto& grid_x = add_map(builder, root, "_j1_tile0", N, 64, rocm_offload(gpu::TargetLevel::X_GRID, 280));
    auto& grid_y = add_map(builder, grid_x.root(), "_i1_tile0", M, 16, rocm_offload(gpu::TargetLevel::Y_GRID, 1));
    auto wave_row_schedule = gpu::ScheduleType_GPU_Offload::create(arch, gpu::TargetLevel::X_BLOCK, sym::integer(1));
    gpu::ScheduleType_GPU_Offload::lanes(wave_row_schedule, sym::integer(64));
    auto& wave_row = add_map(builder, grid_y.root(), "wave_row0", 1, 1, wave_row_schedule);
    auto& wave_col = add_map(
        builder,
        wave_row.root(),
        "wave_col0",
        4,
        1,
        gpu::ScheduleType_GPU_Offload::create(arch, gpu::TargetLevel::Y_BLOCK, sym::integer(4))
    );
    auto& wave = wave_col.root();

    {
        auto& block = builder.add_block(wave);
        auto& fill =
            builder.add_library_node<gpu::GpuMmaFillNode>(block, DebugInfo(), kBlock, types::PrimitiveType::Float, kMma);
        builder.add_computational_memlet(
            block,
            builder.add_access(block, "mma_acc0"),
            fill,
            fill.input(gpu::GpuMmaFillNode::Y_INPUT_IDX),
            {},
            float_ptr()
        );
    }

    auto tk_tile = sym::symbol("tile_k0_tile0");
    auto& k_panel = builder.add_for(
        wave, tk_tile, sym::Lt(tk_tile, sym::integer(K / 16)), sym::zero(), sym::add(tk_tile, sym::integer(8))
    );
    builder.update_schedule_type(k_panel, unrolled());
    builder.add_library_node<data_flow::BarrierLocalNode>(builder.add_block(k_panel.root()), DebugInfo());
    add_tile_copy(
        builder,
        builder.add_block(k_panel.root()),
        "lds_x",
        "x",
        16,
        sym::parse("1536*_i1_tile0 + 16*tile_k0_tile0"),
        {.dims = {{.axis = 0, .base = sym::symbol("_i1_tile0"), .max = sym::integer(M - 1)}}}
    );
    add_tile_copy(
        builder,
        builder.add_block(k_panel.root()),
        "lds_w",
        "w_gate_up",
        64,
        sym::parse("1536*_j1_tile0 + 16*tile_k0_tile0"),
        {}
    );
    builder.add_library_node<data_flow::BarrierLocalNode>(builder.add_block(k_panel.root()), DebugInfo());

    auto tk = sym::symbol("tile_k0");
    auto& k_loop = builder.add_for(
        k_panel.root(), tk, sym::Lt(tk, sym::add(tk_tile, sym::integer(8))), tk_tile, sym::add(tk, sym::one())
    );
    builder.update_schedule_type(k_loop, unrolled());
    {
        auto& block = builder.add_block(k_loop.root());
        add_fragment_load(
            builder,
            block,
            "mma_a0",
            "lds_x",
            gpu::MmaFragmentType::A,
            {.offset = sym::parse("16*tile_k0 - 16*tile_k0_tile0 + 2112*wave_row0"),
             .ldstride = sym::integer(132),
             .layout = gpu::MMA_LAYOUT_ROW_MAJOR}
        );
        add_fragment_load(
            builder,
            block,
            "mma_b0",
            "lds_w",
            gpu::MmaFragmentType::B,
            {.offset = sym::parse("16*tile_k0 - 16*tile_k0_tile0 + 2112*wave_col0"),
             .ldstride = sym::integer(132),
             .layout = gpu::MMA_LAYOUT_COL_MAJOR}
        );
    }
    {
        auto& block = builder.add_block(k_loop.root());
        auto& mma = builder.add_library_node<gpu::GpuMmaMatmulNode>(
            block, DebugInfo(), kBlock, types::PrimitiveType::Half, types::PrimitiveType::Float, kMma
        );
        builder.add_computational_memlet(
            block,
            builder.add_access(block, "mma_acc0"),
            mma,
            mma.input(gpu::GpuMmaMatmulNode::Y_INPUT_IDX),
            {},
            float_ptr()
        );
        builder.add_computational_memlet(
            block,
            builder.add_access(block, "mma_a0"),
            mma,
            mma.input(gpu::GpuMmaMatmulNode::A_INPUT_IDX),
            {},
            half_ptr()
        );
        builder.add_computational_memlet(
            block,
            builder.add_access(block, "mma_b0"),
            mma,
            mma.input(gpu::GpuMmaMatmulNode::B_INPUT_IDX),
            {},
            half_ptr()
        );
    }

    // Y is accumulated into: load it as C, add the accumulator and store it back (clipped to the M = 2 rows).
    const gpu::GpuMmaFromMemoryLayout y_layout{
        .offset = sym::parse("17920*_i1_tile0 + _j1_tile0 + 16*wave_col0 + 286720*wave_row0"),
        .ldstride = sym::integer(N),
        .layout = gpu::MMA_LAYOUT_ROW_MAJOR,
        .rows = sym::parse("2 - _i1_tile0 - 16*wave_row0"),
    };
    add_fragment_load(builder, builder.add_block(wave), "mma_c0", kY, gpu::MmaFragmentType::C, y_layout);
    {
        auto& block = builder.add_block(wave);
        auto& add = builder.add_library_node<gpu::GpuMmaFragmentEltwiseAddNode>(
            block, DebugInfo(), kBlock, types::PrimitiveType::Half, types::PrimitiveType::Float, kMma
        );
        builder.add_computational_memlet(
            block,
            builder.add_access(block, "mma_d0"),
            add,
            add.input(gpu::GpuMmaFragmentEltwiseAddNode::FRAG_D_INPUT_IDX),
            {},
            half_ptr()
        );
        builder.add_computational_memlet(
            block,
            builder.add_access(block, "mma_acc0"),
            add,
            add.input(gpu::GpuMmaFragmentEltwiseAddNode::FRAG_ACC_INPUT_IDX),
            {},
            float_ptr()
        );
        builder.add_computational_memlet(
            block,
            builder.add_access(block, "mma_c0"),
            add,
            add.input(gpu::GpuMmaFragmentEltwiseAddNode::FRAG_C_INPUT_IDX),
            {},
            half_ptr()
        );
    }
    {
        auto& block = builder.add_block(wave);
        auto& store = builder.add_library_node<gpu::GpuMmaFragmentStoreNode>(
            block, DebugInfo(), kBlock, gpu::MmaFragmentType::C, y_layout, types::PrimitiveType::Half, kMma
        );
        builder.add_computational_memlet(
            block,
            builder.add_access(block, "mma_d0"),
            store,
            store.input(gpu::GpuMmaFragmentStoreNode::FRAG_INPUT_IDX),
            {},
            half_ptr()
        );
        builder.add_computational_memlet(
            block,
            builder.add_access(block, kY),
            store,
            store.input(gpu::GpuMmaFragmentStoreNode::PTR_INPUT_IDX),
            {},
            half_ptr()
        );
    }

    return init;
}

struct Counts {
    size_t fills = 0;
    size_t fills_of_c = 0;
    size_t c_loads_of_y = 0;
    size_t stores_to_y = 0;
    size_t zero_stores_to_y = 0;
};

Counts count(Sequence& root) {
    Counts counts;
    visitor::for_each_block(root, [&](structured_control_flow::Block& block) {
        auto& dataflow = block.dataflow();
        auto src_of = [&](data_flow::CodeNode& node, size_t idx) {
            return static_cast<data_flow::AccessNode&>(dataflow.in_edge_for_connector(node, node.input(idx))->src())
                .data();
        };
        for (auto* lib : dataflow.library_nodes()) {
            if (auto* fill = dynamic_cast<gpu::GpuMmaFillNode*>(lib)) {
                ++counts.fills;
                counts.fills_of_c += src_of(*fill, gpu::GpuMmaFillNode::Y_INPUT_IDX) == "mma_c0";
            } else if (auto* load = dynamic_cast<gpu::GpuMmaFragmentLoadNode*>(lib)) {
                counts.c_loads_of_y += load->fragment_type() == gpu::MmaFragmentType::C &&
                                       src_of(*load, gpu::GpuMmaFragmentLoadNode::PTR_INPUT_IDX) == kY;
            } else if (auto* store = dynamic_cast<gpu::GpuMmaFragmentStoreNode*>(lib)) {
                counts.stores_to_y += src_of(*store, gpu::GpuMmaFragmentStoreNode::PTR_INPUT_IDX) == kY;
            }
        }
        for (auto* tasklet : dataflow.tasklets()) {
            auto& in = *dataflow.in_edges(*tasklet).begin();
            auto& out = *dataflow.out_edges(*tasklet).begin();
            auto* constant = dynamic_cast<data_flow::ConstantNode*>(&in.src());
            counts.zero_stores_to_y += tasklet->is_assign() && constant != nullptr && constant->data() == "0.0" &&
                                       static_cast<data_flow::AccessNode&>(out.dst()).data() == kY;
        }
    });
    return counts;
}

} // namespace

TEST(RedundantZeroInitEliminationPassTest, QwenGateUpMatmul_WithoutAnnotationIsNoOp) {
    builder::StructuredSDFGBuilder builder("qwen_gate_up", FunctionType_CPU);
    build_qwen_gate_up_matmul(builder, /*annotate=*/false);
    EXPECT_NO_THROW(builder.subject().validate());

    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::RedundantZeroInitEliminationPass pass;
    EXPECT_FALSE(pass.run(builder, analysis_manager));
    EXPECT_EQ(builder.subject().root().size(), 2);
}

TEST(RedundantZeroInitEliminationPassTest, QwenGateUpMatmul_EliminatesInitNest) {
    builder::StructuredSDFGBuilder builder("qwen_gate_up", FunctionType_CPU);
    build_qwen_gate_up_matmul(builder, /*annotate=*/true);
    EXPECT_NO_THROW(builder.subject().validate());

    auto& root = builder.subject().root();
    ASSERT_EQ(root.size(), 2);
    auto before = count(root);
    EXPECT_EQ(before.zero_stores_to_y, 1);
    EXPECT_EQ(before.c_loads_of_y, 1);
    EXPECT_EQ(before.stores_to_y, 1);
    EXPECT_EQ(before.fills, 1);
    EXPECT_EQ(before.fills_of_c, 0);

    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::RedundantZeroInitEliminationPass pass;
    EXPECT_TRUE(pass.run(builder, analysis_manager));
    EXPECT_NO_THROW(builder.subject().validate());

    // Only the MMA nest remains; its C fragment is zero-filled instead of loaded from Y.
    ASSERT_EQ(root.size(), 1);
    auto* mma_nest = dynamic_cast<structured_control_flow::Map*>(&root.at(0));
    ASSERT_NE(mma_nest, nullptr);
    EXPECT_EQ(mma_nest->indvar()->get_name(), "_j1_tile0");
    auto after = count(root);
    EXPECT_EQ(after.zero_stores_to_y, 0);
    EXPECT_EQ(after.c_loads_of_y, 0);
    EXPECT_EQ(after.stores_to_y, 1);
    EXPECT_EQ(after.fills, 2);
    EXPECT_EQ(after.fills_of_c, 1);

    // Nothing left to eliminate.
    analysis::AnalysisManager fresh_analysis(builder.subject());
    EXPECT_FALSE(pass.run(builder, fresh_analysis));
}
