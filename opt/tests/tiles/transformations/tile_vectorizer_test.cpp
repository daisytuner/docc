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
#include "sdfg/tiles/library_nodes/async_copy_node.h"
#include "sdfg/tiles/transformations/local_storage.h"

using namespace sdfg;
using transformations::LocalStorage;
using transformations::TileVectorizer;

namespace {

// Count 16-byte VectorCopyNodes and leftover scalar `assign` staging tasklets in a
// subtree.
void count(structured_control_flow::ControlFlowNode& n, size_t& vec16, size_t& assigns) {
    if (auto* blk = dynamic_cast<structured_control_flow::Block*>(&n)) {
        for (auto& node : blk->dataflow().nodes()) {
            if (auto* v = dynamic_cast<tiles::VectorCopyNode*>(&node)) {
                if (v->bytes() == 16u) vec16++;
            }
            if (auto* tk = dynamic_cast<data_flow::Tasklet*>(&node)) {
                if (tk->code() == data_flow::TaskletCode::assign) assigns++;
            }
        }
    } else if (auto* s = dynamic_cast<structured_control_flow::Sequence*>(&n)) {
        for (size_t x = 0; x < s->size(); ++x) count(s->at(x), vec16, assigns);
    } else if (auto* m = dynamic_cast<structured_control_flow::Map*>(&n)) {
        count(m->root(), vec16, assigns);
    } else if (auto* f = dynamic_cast<structured_control_flow::StructuredLoop*>(&n)) {
        count(f->root(), vec16, assigns);
    } else if (auto* ie = dynamic_cast<structured_control_flow::IfElse*>(&n)) {
        for (size_t x = 0; x < ie->size(); ++x) count(ie->at(x).first, vec16, assigns);
    }
}

} // namespace

// A fully-covering 2D fp32 cooperative copy (scalar, as LocalStorage emits it) is
// widened by TileVectorizer to a synchronous 16-byte VectorCopyNode (float4): the
// inner tile coord is unit-stride in source and buffer, and the flat coverage (128)
// / row (32) are multiples of 4. No scalar assign copy should remain.
TEST(TileVectorizerTest, WidensContiguousScalarCopyToFloat4) {
    builder::StructuredSDFGBuilder builder("tv_f4", FunctionType_CPU);
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

    analysis::AnalysisManager am(builder.subject());
    // Stage A cooperatively as a scalar copy (LocalStorage no longer vectorizes).
    LocalStorage ls(loop_i, a_in);
    ASSERT_TRUE(ls.can_be_applied(builder, am));
    ls.apply(builder, am);

    size_t vec16_before = 0, assigns_before = 0;
    count(map_b.root(), vec16_before, assigns_before);
    EXPECT_EQ(vec16_before, 0u) << "LocalStorage must emit only scalar copies";
    EXPECT_GE(assigns_before, 1u);

    TileVectorizer tv(map_b);
    ASSERT_TRUE(tv.can_be_applied(builder, am));
    tv.apply(builder, am);

    size_t vec16 = 0, assigns = 0;
    count(map_b.root(), vec16, assigns);
    EXPECT_EQ(vec16, 1u) << "contiguous fp32 cooperative copy must widen to a float4 VectorCopyNode";
    EXPECT_EQ(assigns, 0u) << "no scalar assign copy should remain for the staged tile";
}
