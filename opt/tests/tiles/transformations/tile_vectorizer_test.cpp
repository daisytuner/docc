#include "sdfg/tiles/transformations/tile_vectorizer.h"

#include <gtest/gtest.h>

#include <functional>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/if_else.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/targets/cuda/cuda.h"
#include "sdfg/targets/gpu/gpu_schedule_type.h"
#include "sdfg/tiles/library_nodes/tile_copy_node.h"
#include "sdfg/tiles/transformations/local_storage.h"

using namespace sdfg;
using transformations::LocalStorage;
using transformations::TileVectorizer;

namespace {

// A [4][32] fp32 tile of A staged cooperatively under a CUDA block map, as the scalar
// TileCopyNode LocalStorage emits.
std::pair<tiles::TileCopyNode*, structured_control_flow::Map*>
stage_scalar_copy(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& am) {
    auto& seq = builder.subject().root();
    types::Scalar loop_var(types::PrimitiveType::Int32);
    types::Scalar elem(types::PrimitiveType::Float);
    types::Pointer ptr(elem);
    auto b = symbolic::symbol("b");
    auto i = symbolic::symbol("i");
    auto k = symbolic::symbol("k");
    auto N = symbolic::symbol("N");
    builder.add_container("N", loop_var, true);
    builder.add_container("A", ptr, true);
    builder.add_container("C", ptr, true);
    builder.add_container("b", loop_var);
    builder.add_container("i", loop_var);
    builder.add_container("k", loop_var);

    auto sched_b = gpu::ScheduleType_GPU_Offload::create<
        cuda::ScheduleType_CUDA_Offload>(gpu::TargetLevel::X_BLOCK, symbolic::integer(32));
    auto& map_b =
        builder
            .add_map(seq, b, symbolic::Lt(b, N), symbolic::integer(0), symbolic::add(b, symbolic::integer(1)), sched_b);
    auto& loop_i = builder.add_for(
        map_b.root(),
        i,
        symbolic::Lt(i, symbolic::integer(4)),
        symbolic::integer(0),
        symbolic::add(i, symbolic::integer(1))
    );
    auto& loop_k = builder.add_for(
        loop_i.root(),
        k,
        symbolic::Lt(k, symbolic::integer(32)),
        symbolic::integer(0),
        symbolic::add(k, symbolic::integer(1))
    );
    auto& block = builder.add_block(loop_k.root());
    auto& c_in = builder.add_access(block, "C");
    auto& a_in = builder.add_access(block, "A");
    auto& c_out = builder.add_access(block, "C");
    auto& t = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, c_in, t, "_in1", {b}, ptr);
    builder
        .add_computational_memlet(block, a_in, t, "_in2", {symbolic::add(symbolic::mul(i, symbolic::integer(32)), k)}, ptr);
    builder.add_computational_memlet(block, t, "_out", c_out, {b}, ptr);

    LocalStorage ls(loop_i, a_in);
    EXPECT_TRUE(ls.can_be_applied(builder, am));
    ls.apply(builder, am);

    tiles::TileCopyNode* copy = nullptr;
    std::function<void(structured_control_flow::ControlFlowNode&)> find =
        [&](structured_control_flow::ControlFlowNode& n) {
            if (auto* blk = dynamic_cast<structured_control_flow::Block*>(&n)) {
                for (auto& node : blk->dataflow().nodes()) {
                    if (auto* tc = dynamic_cast<tiles::TileCopyNode*>(&node)) {
                        copy = tc;
                    }
                }
            } else if (auto* s = dynamic_cast<structured_control_flow::Sequence*>(&n)) {
                for (size_t x = 0; x < s->size(); ++x) {
                    find(s->at(x));
                }
            } else if (auto* m = dynamic_cast<structured_control_flow::Map*>(&n)) {
                find(m->root());
            } else if (auto* f = dynamic_cast<structured_control_flow::StructuredLoop*>(&n)) {
                find(f->root());
            } else if (auto* ie = dynamic_cast<structured_control_flow::IfElse*>(&n)) {
                for (size_t x = 0; x < ie->size(); ++x) {
                    find(ie->at(x).first);
                }
            }
        };
    find(map_b.root());
    return {copy, &map_b};
}

} // namespace

// A fully-covering 2D fp32 cooperative copy (a scalar TileCopyNode, as LocalStorage
// emits it) is widened by TileVectorizer to a 16-byte VectorSync float4: the inner
// tile coord is unit-stride in source and buffer, and the flat coverage (128) / row
// (32) are multiples of 4.
TEST(TileVectorizerTest, WidensContiguousScalarCopyToFloat4) {
    builder::StructuredSDFGBuilder builder("tv_f4", FunctionType_CPU);
    analysis::AnalysisManager am(builder.subject());
    auto [copy, map_b] = stage_scalar_copy(builder, am);
    ASSERT_NE(copy, nullptr) << "LocalStorage must emit a TileCopyNode for the cooperative tile";
    EXPECT_EQ(copy->atom(), tiles::CopyAtom::ScalarSync) << "LocalStorage must emit a scalar copy";

    TileVectorizer tv(*map_b);
    ASSERT_TRUE(tv.can_be_applied(builder, am));
    tv.apply(builder, am);

    EXPECT_EQ(copy->atom(), tiles::CopyAtom::VectorSync) << "contiguous fp32 cooperative copy must widen";
    EXPECT_EQ(copy->bytes(), 16u) << "fp32 coalesces to a 16-byte float4 transfer";
}

// Buffer rows padded to 34 floats start 8- but not 16-byte aligned: a synchronous copy keeps
// 16-byte global transfers (the buffer side is split at emission), an async copy drops to 8.
TEST(TileVectorizerTest, PaddedBufferRowsLimitOnlyAsyncWidth) {
    for (auto atom : {tiles::CopyAtom::ScalarSync, tiles::CopyAtom::CpAsync}) {
        builder::StructuredSDFGBuilder builder("tv_pad", FunctionType_CPU);
        analysis::AnalysisManager am(builder.subject());
        auto [copy, map_b] = stage_scalar_copy(builder, am);
        ASSERT_NE(copy, nullptr);
        auto plan = copy->plan();
        plan.dst = tiles::Layout(plan.dst.shape(), {symbolic::integer(34), symbolic::one()}, plan.dst.offset());
        copy->set_plan(plan);
        copy->set_atom(atom);

        TileVectorizer tv(*map_b);
        ASSERT_TRUE(tv.can_be_applied(builder, am));
        tv.apply(builder, am);

        EXPECT_EQ(copy->bytes(), atom == tiles::CopyAtom::CpAsync ? 8u : 16u);
    }
}

// A column-major fp16 [8][4] tile with only rows 0..2 valid: after the sweep reorder the ragged
// row axis is the inner (unit-stride) axis, so a 16-byte transfer would read all 8 rows past the
// bound. Only an outer-axis guard, uniform over each transfer, may widen.
TEST(TileVectorizerTest, InnerAxisGuardBlocksWidening) {
    for (bool ragged_rows : {true, false}) {
        builder::StructuredSDFGBuilder builder("tv_guard", FunctionType_CPU);
        types::Scalar loop_var(types::PrimitiveType::Int32);
        types::Scalar elem(types::PrimitiveType::Half);
        types::Pointer ptr(elem);
        builder.add_container("A", ptr, true);
        builder.add_container("buf", types::Array(types::StorageType::NV_Shared(), 16, "", elem, symbolic::integer(48)));
        builder.add_container("b", loop_var);
        auto b = symbolic::symbol("b");
        auto sched_b = gpu::ScheduleType_GPU_Offload::create<
            cuda::ScheduleType_CUDA_Offload>(gpu::TargetLevel::X_BLOCK, symbolic::integer(32));
        auto& map_b = builder.add_map(
            builder.subject().root(),
            b,
            symbolic::Lt(b, symbolic::integer(32)),
            symbolic::integer(0),
            symbolic::add(b, symbolic::integer(1)),
            sched_b
        );
        auto& block = builder.add_block(map_b.root());
        auto& src = builder.add_access(block, "A");
        auto& dst = builder.add_access(block, "buf");

        tiles::TiledCopy plan;
        plan.src = tiles::Layout(
            {symbolic::integer(8), symbolic::integer(4)},
            {symbolic::integer(1), symbolic::integer(16)},
            symbolic::zero()
        );
        plan.dst = tiles::Layout(
            {symbolic::integer(8), symbolic::integer(4)},
            {symbolic::integer(1), symbolic::integer(12)},
            symbolic::zero()
        );
        tiles::TileGuard guard;
        guard.tile_sizes = {symbolic::integer(8), symbolic::integer(4)};
        guard.dims.push_back({ragged_rows ? 0u : 1u, symbolic::zero(), symbolic::integer(2)});
        ASSERT_TRUE(tiles::coalesce_sweep_order(plan, guard));
        ASSERT_EQ(guard.dims.front().axis, ragged_rows ? 1u : 0u);

        auto& copy = static_cast<tiles::TileCopyNode&>(builder.add_library_node<tiles::TileCopyNode>(
            block, DebugInfo(), data_flow::ImplementationType_NONE, plan, tiles::CopyDirection::In, 2, guard
        ));
        builder.add_computational_memlet(block, dst, copy, "_dst", {}, ptr);
        builder.add_computational_memlet(block, src, copy, "_src", {}, ptr);

        analysis::AnalysisManager am(builder.subject());
        TileVectorizer tv(map_b);
        ASSERT_TRUE(tv.can_be_applied(builder, am));
        tv.apply(builder, am);

        if (ragged_rows) {
            EXPECT_EQ(copy.atom(), tiles::CopyAtom::ScalarSync) << copy.toStr();
            EXPECT_EQ(copy.bytes(), 2u);
        } else {
            EXPECT_EQ(copy.atom(), tiles::CopyAtom::VectorSync) << copy.toStr();
            EXPECT_EQ(copy.bytes(), 16u);
        }
    }
}
