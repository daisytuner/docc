#include "sdfg/tiles/transformations/software_pipelining.h"
#include "sdfg/tiles/transformations/tile_vectorizer.h"

#include <gtest/gtest.h>

#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/if_else.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/targets/cuda/cuda.h"
#include "sdfg/tiles/library_nodes/pipeline_node.h"
#include "sdfg/tiles/library_nodes/tile_copy_node.h"
#include "sdfg/types/array.h"

using namespace sdfg;

namespace {

constexpr long TILE = 8; // elements staged per panel (one buffer slot)

// Add a cooperative copy-in TileCopyNode block to @p parent: it stages the panel-k
// slice A[k*TILE .. k*TILE + TILE) into buf[0..TILE) (dst stride 1). @p src_stride
// > 1 makes the source non-contiguous (defeats vector widening). The buffer memlet
// is a bare pointer (empty subset); the address lives in the plan.
tiles::TileCopyNode& add_copy(
    builder::StructuredSDFGBuilder& builder,
    structured_control_flow::Sequence& parent,
    const std::string& src,
    const std::string& dst,
    const types::IType& ptr,
    long src_stride = 1
) {
    auto& b = builder.add_block(parent);
    auto& s = builder.add_access(b, src);
    auto& d = builder.add_access(b, dst);
    auto k = symbolic::symbol("k");
    tiles::TiledCopy plan;
    plan.src = tiles::
        Layout({symbolic::integer(TILE)}, {symbolic::integer(src_stride)}, symbolic::mul(k, symbolic::integer(TILE)));
    plan.dst = tiles::Layout({symbolic::integer(TILE)}, {symbolic::integer(1)}, symbolic::integer(0));
    plan.atom = tiles::CopyAtom::ScalarSync;
    auto& node = builder.add_library_node<
        tiles::TileCopyNode>(b, DebugInfo(), data_flow::ImplementationType_NONE, plan, tiles::CopyDirection::In, 4);
    builder.add_computational_memlet(b, d, node, "_dst", {}, ptr);
    builder.add_computational_memlet(b, s, node, "_src", {}, ptr);
    return static_cast<tiles::TileCopyNode&>(node);
}

// A scalar compute block reading buf[0] into C (the consumer the copy feeds).
void add_compute(
    builder::StructuredSDFGBuilder& builder,
    structured_control_flow::Sequence& parent,
    const std::string& buf,
    const types::IType& buf_type
) {
    auto& b = builder.add_block(parent);
    auto& r = builder.add_access(b, buf);
    auto& tk = builder.add_tasklet(b, data_flow::TaskletCode::assign, "_out", {"_in"});
    auto& c = builder.add_access(b, "C");
    builder.add_computational_memlet(b, r, tk, "_in", {symbolic::integer(0)}, buf_type);
    builder.add_computational_memlet(b, tk, "_out", c, {}, types::Scalar(types::PrimitiveType::Float));
}

// GPU map over i { for k in [0,K) { stage A -> buf (TileCopyNode); compute buf -> C } }.
// Returns the panel loop (the `for k`). src_stride/wrap tune the copy shape.
structured_control_flow::For& build(
    builder::StructuredSDFGBuilder& builder,
    int K,
    bool shared = true,
    long src_stride = 1,
    bool wrap = false,
    long pad = 0
) {
    auto& root = builder.subject().root();
    types::Scalar f(types::PrimitiveType::Float);
    types::Pointer aptr(f);
    types::Scalar u64(types::PrimitiveType::UInt64);
    types::Array buf_type(
        shared ? types::StorageType::NV_Shared() : types::StorageType::CPU_Stack(),
        0,
        "",
        f,
        symbolic::integer(TILE + pad)
    );

    builder.add_container("A", aptr, true);
    builder.add_container("buf", buf_type);
    builder.add_container("C", f, true);
    builder.add_container("i", u64);
    builder.add_container("k", u64);

    auto cuda_sched = cuda::ScheduleType_CUDA::create();
    cuda::ScheduleType_CUDA::dimension(cuda_sched, cuda::CUDADimension::X);
    cuda::ScheduleType_CUDA::block_size(cuda_sched, symbolic::integer(32));
    auto i = symbolic::symbol("i");
    auto& gmap = builder.add_map(
        root,
        i,
        symbolic::Lt(i, symbolic::integer(64)),
        symbolic::integer(0),
        symbolic::add(i, symbolic::integer(1)),
        cuda_sched
    );
    auto k = symbolic::symbol("k");
    auto& kloop = builder.add_for(
        gmap.root(),
        k,
        symbolic::Lt(k, symbolic::integer(K)),
        symbolic::integer(0),
        symbolic::add(k, symbolic::integer(1))
    );

    structured_control_flow::Sequence& body = wrap ? builder.add_sequence(kloop.root()) : kloop.root();
    add_copy(builder, body, "A", "buf", aptr, src_stride);
    add_compute(builder, body, "buf", buf_type);
    return static_cast<structured_control_flow::For&>(gmap.root().at(0));
}

// Two cooperative shared operands (buf, buf2), each staged by its own TileCopyNode,
// both consumed by the compute. Returns the panel loop.
structured_control_flow::For& build_two(builder::StructuredSDFGBuilder& builder, int K) {
    auto& root = builder.subject().root();
    types::Scalar f(types::PrimitiveType::Float);
    types::Pointer aptr(f);
    types::Scalar u64(types::PrimitiveType::UInt64);
    types::Array buf_type(types::StorageType::NV_Shared(), 0, "", f, symbolic::integer(TILE));

    builder.add_container("A", aptr, true);
    builder.add_container("B", aptr, true);
    builder.add_container("buf", buf_type);
    builder.add_container("buf2", buf_type);
    builder.add_container("C", f, true);
    builder.add_container("i", u64);
    builder.add_container("k", u64);

    auto cuda_sched = cuda::ScheduleType_CUDA::create();
    cuda::ScheduleType_CUDA::dimension(cuda_sched, cuda::CUDADimension::X);
    cuda::ScheduleType_CUDA::block_size(cuda_sched, symbolic::integer(32));
    auto i = symbolic::symbol("i");
    auto& gmap = builder.add_map(
        root,
        i,
        symbolic::Lt(i, symbolic::integer(64)),
        symbolic::integer(0),
        symbolic::add(i, symbolic::integer(1)),
        cuda_sched
    );
    auto k = symbolic::symbol("k");
    auto& kloop = builder.add_for(
        gmap.root(),
        k,
        symbolic::Lt(k, symbolic::integer(K)),
        symbolic::integer(0),
        symbolic::add(k, symbolic::integer(1))
    );
    add_copy(builder, kloop.root(), "A", "buf", aptr);
    add_copy(builder, kloop.root(), "B", "buf2", aptr);
    {
        auto& b = builder.add_block(kloop.root());
        auto& r1 = builder.add_access(b, "buf");
        auto& r2 = builder.add_access(b, "buf2");
        auto& tk = builder.add_tasklet(b, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
        auto& c = builder.add_access(b, "C");
        builder.add_computational_memlet(b, r1, tk, "_in1", {symbolic::integer(0)}, buf_type);
        builder.add_computational_memlet(b, r2, tk, "_in2", {symbolic::integer(0)}, buf_type);
        builder.add_computational_memlet(b, tk, "_out", c, {}, f);
    }
    return static_cast<structured_control_flow::For&>(gmap.root().at(0));
}

// Like build(), but the cooperative copy-in is nested inside a boundary-guard
// IfElse (as StreamK's ragged panel produces). Returns the panel loop.
structured_control_flow::For& build_guarded(builder::StructuredSDFGBuilder& builder, int K) {
    auto& root = builder.subject().root();
    types::Scalar f(types::PrimitiveType::Float);
    types::Pointer aptr(f);
    types::Scalar u64(types::PrimitiveType::UInt64);
    types::Array buf_type(types::StorageType::NV_Shared(), 0, "", f, symbolic::integer(TILE));

    builder.add_container("A", aptr, true);
    builder.add_container("buf", buf_type);
    builder.add_container("C", f, true);
    builder.add_container("i", u64);
    builder.add_container("k", u64);

    auto cuda_sched = cuda::ScheduleType_CUDA::create();
    cuda::ScheduleType_CUDA::dimension(cuda_sched, cuda::CUDADimension::X);
    cuda::ScheduleType_CUDA::block_size(cuda_sched, symbolic::integer(32));
    auto i = symbolic::symbol("i");
    auto& gmap = builder.add_map(
        root,
        i,
        symbolic::Lt(i, symbolic::integer(64)),
        symbolic::integer(0),
        symbolic::add(i, symbolic::integer(1)),
        cuda_sched
    );
    auto k = symbolic::symbol("k");
    auto& kloop = builder.add_for(
        gmap.root(),
        k,
        symbolic::Lt(k, symbolic::integer(K)),
        symbolic::integer(0),
        symbolic::add(k, symbolic::integer(1))
    );

    auto& if_else = builder.add_if_else(kloop.root());
    auto& guarded = builder.add_case(if_else, symbolic::Lt(k, symbolic::integer(K)));
    add_copy(builder, guarded, "A", "buf", aptr);
    add_compute(builder, kloop.root(), "buf", buf_type);
    return static_cast<structured_control_flow::For&>(gmap.root().at(0));
}

// Count TileCopyNodes with the cp.async atom in a subtree.
size_t count_cp_async(structured_control_flow::ControlFlowNode& scope) {
    size_t n = 0;
    std::function<void(structured_control_flow::ControlFlowNode&)> scan =
        [&](structured_control_flow::ControlFlowNode& node) {
            if (auto* b = dynamic_cast<structured_control_flow::Block*>(&node)) {
                for (auto& dn : b->dataflow().nodes()) {
                    if (auto* tc = dynamic_cast<tiles::TileCopyNode*>(&dn)) {
                        if (tc->atom() == tiles::CopyAtom::CpAsync) n++;
                    }
                }
            } else if (auto* ie = dynamic_cast<structured_control_flow::IfElse*>(&node)) {
                for (size_t i = 0; i < ie->size(); i++) scan(ie->at(i).first);
            } else if (auto* seq = dynamic_cast<structured_control_flow::Sequence*>(&node)) {
                for (size_t i = 0; i < seq->size(); i++) scan(seq->at(i));
            } else if (auto* map = dynamic_cast<structured_control_flow::Map*>(&node)) {
                scan(map->root());
            } else if (auto* loop = dynamic_cast<structured_control_flow::StructuredLoop*>(&node)) {
                scan(loop->root());
            }
        };
    scan(scope);
    return n;
}

// Count TileCopyNodes with a given transfer width in a subtree.
size_t count_bytes(structured_control_flow::ControlFlowNode& scope, size_t bytes) {
    size_t n = 0;
    std::function<void(structured_control_flow::ControlFlowNode&)> scan =
        [&](structured_control_flow::ControlFlowNode& node) {
            if (auto* b = dynamic_cast<structured_control_flow::Block*>(&node)) {
                for (auto& dn : b->dataflow().nodes()) {
                    if (auto* tc = dynamic_cast<tiles::TileCopyNode*>(&dn)) {
                        if (tc->atom() == tiles::CopyAtom::CpAsync && tc->bytes() == bytes) n++;
                    }
                }
            } else if (auto* ie = dynamic_cast<structured_control_flow::IfElse*>(&node)) {
                for (size_t i = 0; i < ie->size(); i++) scan(ie->at(i).first);
            } else if (auto* seq = dynamic_cast<structured_control_flow::Sequence*>(&node)) {
                for (size_t i = 0; i < seq->size(); i++) scan(seq->at(i));
            } else if (auto* map = dynamic_cast<structured_control_flow::Map*>(&node)) {
                scan(map->root());
            } else if (auto* loop = dynamic_cast<structured_control_flow::StructuredLoop*>(&node)) {
                scan(loop->root());
            }
        };
    scan(scope);
    return n;
}

} // namespace

TEST(SoftwarePipeliningTest, SingleOperandStagesOnlyFirstBuffer) {
    builder::StructuredSDFGBuilder builder("sp", FunctionType_CPU);
    auto& kloop = build_two(builder, /*K=*/4);
    auto& sdfg = builder.subject();
    analysis::AnalysisManager am(sdfg);
    transformations::SoftwarePipelining sp(kloop, 2, /*single_operand=*/true);
    ASSERT_TRUE(sp.can_be_applied(builder, am));
    sp.apply(builder, am);

    // Only the first (name-ordered) buffer gains the [stages] axis; the second
    // stays single-buffered.
    auto* buf = dynamic_cast<const types::Array*>(&sdfg.type("buf"));
    ASSERT_NE(buf, nullptr);
    EXPECT_TRUE(symbolic::eq(buf->num_elements(), symbolic::integer(2)));
    auto* buf2 = dynamic_cast<const types::Array*>(&sdfg.type("buf2"));
    ASSERT_NE(buf2, nullptr);
    EXPECT_TRUE(symbolic::eq(buf2->num_elements(), symbolic::integer(TILE))); // unchanged (no stage axis)

    // Exactly one buffer is prefetched via cp.async (prologue + in-loop = 2 copies);
    // buf2 keeps its synchronous copy.
    auto& map_body = static_cast<structured_control_flow::Sequence&>(*kloop.get_parent());
    EXPECT_EQ(count_cp_async(map_body), 2u);
}

TEST(SoftwarePipeliningTest, VectorizeStridesCoopMapAndWidensCpAsync) {
    builder::StructuredSDFGBuilder builder("sp", FunctionType_CPU);
    auto& kloop = build(builder, /*K=*/4);
    auto& sdfg = builder.subject();
    analysis::AnalysisManager am(sdfg);
    transformations::SoftwarePipelining sp(kloop, 2, /*single_operand=*/false);
    ASSERT_TRUE(sp.can_be_applied(builder, am));
    sp.apply(builder, am);
    // Widening now belongs to TileVectorizer, applied over the enclosing offload map
    // (so both the prologue and in-loop cp.async are reached).
    auto* omap = dynamic_cast<structured_control_flow::StructuredLoop*>(kloop.get_parent()->get_parent());
    ASSERT_NE(omap, nullptr);
    transformations::TileVectorizer tv(*omap);
    ASSERT_TRUE(tv.can_be_applied(builder, am));
    tv.apply(builder, am);

    // The contiguous fp32 tile (TILE=8, multiple of 4) widens to a 16-byte float4.
    auto& map_body = static_cast<structured_control_flow::Sequence&>(*kloop.get_parent());
    EXPECT_EQ(count_bytes(map_body, 16), 2u);
    EXPECT_EQ(count_bytes(map_body, 4), 0u);
}

TEST(SoftwarePipeliningTest, VectorizeRejectsNonContiguousSource) {
    builder::StructuredSDFGBuilder builder("sp", FunctionType_CPU);
    auto& kloop = build(builder, /*K=*/4, /*shared=*/true, /*src_stride=*/2); // A[2c] -> not contiguous
    auto& sdfg = builder.subject();
    analysis::AnalysisManager am(sdfg);
    transformations::SoftwarePipelining sp(kloop, 2, /*single_operand=*/false);
    ASSERT_TRUE(sp.can_be_applied(builder, am));
    sp.apply(builder, am);
    auto* omap = dynamic_cast<structured_control_flow::StructuredLoop*>(kloop.get_parent()->get_parent());
    ASSERT_NE(omap, nullptr);
    transformations::TileVectorizer tv(*omap);
    ASSERT_TRUE(tv.can_be_applied(builder, am));
    tv.apply(builder, am);

    // Non-unit source stride: the cp.async must not widen, staying at 4 bytes.
    auto& map_body = static_cast<structured_control_flow::Sequence&>(*kloop.get_parent());
    EXPECT_EQ(count_bytes(map_body, 16), 0u);
    EXPECT_EQ(count_bytes(map_body, 4), 2u);
}

TEST(SoftwarePipeliningTest, CanBeApplied) {
    builder::StructuredSDFGBuilder builder("sp", FunctionType_CPU);
    auto& kloop = build(builder, /*K=*/4);
    analysis::AnalysisManager am(builder.subject());
    transformations::SoftwarePipelining sp(kloop, 2);
    EXPECT_TRUE(sp.can_be_applied(builder, am));
}

// When the panel body is a single wrapper Sequence holding both the copy and the
// compute, the pipeline must peel/shift only the copy — not the compute. A
// regression clones the compute into the prologue (so it reads the just-prefetched
// panel: no overlap). Assert exactly one block writes the compute output C.
TEST(SoftwarePipeliningTest, WrapperSequenceDoesNotShiftCompute) {
    builder::StructuredSDFGBuilder builder("sp", FunctionType_CPU);
    auto& kloop = build(builder, /*K=*/4, /*shared=*/true, /*src_stride=*/1, /*wrap=*/true);
    auto& sdfg = builder.subject();
    analysis::AnalysisManager am(sdfg);
    transformations::SoftwarePipelining sp(kloop, 2, /*single_operand=*/false);
    ASSERT_TRUE(sp.can_be_applied(builder, am));
    sp.apply(builder, am);

    size_t writes_C = 0;
    std::function<void(structured_control_flow::ControlFlowNode&)> scan =
        [&](structured_control_flow::ControlFlowNode& n) {
            if (auto* b = dynamic_cast<structured_control_flow::Block*>(&n)) {
                for (auto* acc : b->dataflow().data_nodes()) {
                    if (acc->data() == "C" && b->dataflow().in_degree(*acc) > 0) writes_C++;
                }
            } else if (auto* ie = dynamic_cast<structured_control_flow::IfElse*>(&n)) {
                for (size_t i = 0; i < ie->size(); i++) scan(ie->at(i).first);
            } else if (auto* seq = dynamic_cast<structured_control_flow::Sequence*>(&n)) {
                for (size_t i = 0; i < seq->size(); i++) scan(seq->at(i));
            } else if (auto* map = dynamic_cast<structured_control_flow::Map*>(&n)) {
                scan(map->root());
            } else if (auto* loop = dynamic_cast<structured_control_flow::StructuredLoop*>(&n)) {
                scan(loop->root());
            }
        };
    scan(static_cast<structured_control_flow::Sequence&>(*kloop.get_parent()));
    EXPECT_EQ(writes_C, 1u) << "the compute must stay in the loop, not be cloned into the prologue";
}

TEST(SoftwarePipeliningTest, RejectsTooFewPanels) {
    builder::StructuredSDFGBuilder builder("sp", FunctionType_CPU);
    auto& kloop = build(builder, /*K=*/1);
    analysis::AnalysisManager am(builder.subject());
    transformations::SoftwarePipelining sp(kloop, 2);
    EXPECT_FALSE(sp.can_be_applied(builder, am));
}

TEST(SoftwarePipeliningTest, RejectsNonShared) {
    builder::StructuredSDFGBuilder builder("sp", FunctionType_CPU);
    auto& kloop = build(builder, /*K=*/4, /*shared=*/false);
    analysis::AnalysisManager am(builder.subject());
    transformations::SoftwarePipelining sp(kloop, 2);
    EXPECT_FALSE(sp.can_be_applied(builder, am));
}

// Regression: the shared-staging gate must descend into IfElse. StreamK's ragged
// panel wraps the cooperative copy-in in a boundary guard; a copy nested in a
// conditional still stages a shared tile, so the panel stays pipelineable and
// apply() prepends the [stages] axis as usual.
TEST(SoftwarePipeliningTest, GuardedCopyInStagesShared) {
    builder::StructuredSDFGBuilder builder("sp", FunctionType_CPU);
    auto& kloop = build_guarded(builder, /*K=*/4);
    auto& sdfg = builder.subject();
    analysis::AnalysisManager am(sdfg);
    transformations::SoftwarePipelining sp(kloop, 2);
    ASSERT_TRUE(sp.can_be_applied(builder, am));
    sp.apply(builder, am);

    // buf gained the leading [2] stage axis; inner is the original [TILE].
    auto* outer = dynamic_cast<const types::Array*>(&sdfg.type("buf"));
    ASSERT_NE(outer, nullptr);
    EXPECT_TRUE(symbolic::eq(outer->num_elements(), symbolic::integer(2)));
    EXPECT_TRUE(outer->storage_type().is_nv_shared());
}

TEST(SoftwarePipeliningTest, StagesBufferAndReindexes) {
    builder::StructuredSDFGBuilder builder("sp", FunctionType_CPU);
    auto& kloop = build(builder, /*K=*/4);
    auto& sdfg = builder.subject();
    // The panel loop's parent (the GPU map body) — becomes [prologue, loop].
    auto& map_body = static_cast<structured_control_flow::Sequence&>(*kloop.get_parent());
    analysis::AnalysisManager am(sdfg);
    transformations::SoftwarePipelining sp(kloop, 2);
    ASSERT_TRUE(sp.can_be_applied(builder, am));
    sp.apply(builder, am);

    // buf gained a leading [2] axis; inner is the original [TILE].
    auto* outer = dynamic_cast<const types::Array*>(&sdfg.type("buf"));
    ASSERT_NE(outer, nullptr);
    EXPECT_TRUE(symbolic::eq(outer->num_elements(), symbolic::integer(2)));
    EXPECT_TRUE(outer->storage_type().is_nv_shared());
    auto* inner = dynamic_cast<const types::Array*>(&outer->element_type());
    ASSERT_NE(inner, nullptr);
    EXPECT_TRUE(symbolic::eq(inner->num_elements(), symbolic::integer(TILE)));

    // A prologue sequence was inserted before the loop.
    ASSERT_EQ(map_body.size(), 2u);
    auto* prologue = dynamic_cast<structured_control_flow::Sequence*>(&map_body.at(0));
    ASSERT_NE(prologue, nullptr);

    // Prologue prefetches panel 0 via cp.async + commits it. The staged copy is a
    // TileCopyNode whose buffer memlet is a bare pointer (address in the plan).
    size_t prologue_async = count_cp_async(*prologue);
    size_t prologue_commit = 0;
    for (size_t i = 0; i < prologue->size(); i++) {
        auto* b = dynamic_cast<structured_control_flow::Block*>(&prologue->at(i));
        if (b == nullptr) {
            continue;
        }
        for (auto& node : b->dataflow().nodes()) {
            if (dynamic_cast<tiles::PipelineCommitNode*>(&node) != nullptr) {
                prologue_commit++;
            }
        }
        for (auto* acc : b->dataflow().data_nodes()) {
            if (acc->data() == "buf") {
                for (auto& m : b->dataflow().out_edges(*acc)) { // bare pointer into the node
                    EXPECT_EQ(m.subset().size(), 0u);
                }
            }
        }
    }
    EXPECT_EQ(prologue_async, 1u);
    EXPECT_EQ(prologue_commit, 1u);

    // The in-loop prefetch is guarded (IfElse) and followed by commit + wait;
    // the compute block still reads buf at the current stage mod(k,2).
    auto& loop_body = kloop.root();
    bool has_guarded_copy = false;
    size_t loop_async = count_cp_async(loop_body);
    size_t loop_commit = 0, loop_wait = 0;
    auto stage = symbolic::mod(symbolic::symbol("k"), symbolic::integer(2));
    std::function<void(structured_control_flow::ControlFlowNode&)> scan =
        [&](structured_control_flow::ControlFlowNode& n) {
            if (auto* b = dynamic_cast<structured_control_flow::Block*>(&n)) {
                for (auto& node : b->dataflow().nodes()) {
                    if (dynamic_cast<tiles::PipelineCommitNode*>(&node) != nullptr) {
                        loop_commit++;
                    }
                    if (dynamic_cast<tiles::PipelineWaitNode*>(&node) != nullptr) {
                        loop_wait++;
                    }
                }
                for (auto* acc : b->dataflow().data_nodes()) {
                    if (acc->data() != "buf") {
                        continue;
                    }
                    for (auto& m : b->dataflow().out_edges(*acc)) { // compute read of buf
                        if (m.subset().size() == 2u && !symbolic::atoms(m.subset().at(0)).empty()) {
                            EXPECT_TRUE(symbolic::eq(m.subset().at(0), stage));
                        }
                    }
                }
            } else if (auto* ie = dynamic_cast<structured_control_flow::IfElse*>(&n)) {
                for (size_t i = 0; i < ie->size(); i++) {
                    scan(ie->at(i).first);
                }
            } else if (auto* seq = dynamic_cast<structured_control_flow::Sequence*>(&n)) {
                for (size_t i = 0; i < seq->size(); i++) {
                    scan(seq->at(i));
                }
            }
        };
    for (size_t i = 0; i < loop_body.size(); i++) {
        if (dynamic_cast<structured_control_flow::IfElse*>(&loop_body.at(i)) != nullptr) {
            has_guarded_copy = true;
        }
        scan(loop_body.at(i));
    }
    EXPECT_TRUE(has_guarded_copy);
    EXPECT_EQ(loop_async, 1u);
    EXPECT_EQ(loop_commit, 1u);
    // Two waits: the guarded prefetch's wait (keep stages-1 in flight) in the
    // `then` branch, and the tail drain wait (keep 0) in the `else` branch.
    EXPECT_EQ(loop_wait, 2u);
}

// Regression: a Padded double-buffer's stage bias must use the buffer's per-stage
// element count (padding included), not the plan's logical tile size — otherwise
// stage 1 addresses the wrong offset (silent wrong results, e.g. the StreamK GEMM).
TEST(SoftwarePipeliningTest, PaddedBufferStageBiasUsesBufferStride) {
    constexpr long PAD = 5; // buffer per-stage stride = TILE + PAD > the logical tile TILE
    builder::StructuredSDFGBuilder builder("sp_pad", FunctionType_CPU);
    auto& kloop = build(builder, /*K=*/4, /*shared=*/true, /*src_stride=*/1, /*wrap=*/false, /*pad=*/PAD);
    auto& sdfg = builder.subject();
    analysis::AnalysisManager am(sdfg);
    transformations::SoftwarePipelining sp(kloop, 2);
    ASSERT_TRUE(sp.can_be_applied(builder, am));
    sp.apply(builder, am);

    // Collect every TileCopyNode's biased buffer offset.
    std::vector<symbolic::Expression> offsets;
    std::function<void(structured_control_flow::ControlFlowNode&)> collect =
        [&](structured_control_flow::ControlFlowNode& node) {
            if (auto* b = dynamic_cast<structured_control_flow::Block*>(&node)) {
                for (auto& dn : b->dataflow().nodes()) {
                    if (auto* tc = dynamic_cast<tiles::TileCopyNode*>(&dn)) {
                        offsets.push_back(tc->plan().dst.offset());
                    }
                }
            } else if (auto* ie = dynamic_cast<structured_control_flow::IfElse*>(&node)) {
                for (size_t i = 0; i < ie->size(); i++) collect(ie->at(i).first);
            } else if (auto* seq = dynamic_cast<structured_control_flow::Sequence*>(&node)) {
                for (size_t i = 0; i < seq->size(); i++) collect(seq->at(i));
            } else if (auto* map = dynamic_cast<structured_control_flow::Map*>(&node)) {
                collect(map->root());
            } else if (auto* loop = dynamic_cast<structured_control_flow::StructuredLoop*>(&node)) {
                collect(loop->root());
            }
        };
    collect(sdfg.root());
    ASSERT_FALSE(offsets.empty());

    // Some biased buffer offset must carry the padded per-stage stride (TILE + PAD)
    // as its stage coefficient — never the logical tile size (TILE), which was the
    // pre-fix bug. (mod() stays opaque symbolically, so match the coefficient.)
    const std::string padded = std::to_string(TILE + PAD);
    const std::string logical = std::to_string(TILE);
    bool found_padded_stride = false;
    for (const auto& off : offsets) {
        const std::string s = off->__str__();
        if (s.find(padded) != std::string::npos) {
            found_padded_stride = true;
        }
        EXPECT_EQ(s.find(logical), std::string::npos)
            << "stage bias used the logical tile size (" << logical << "), not the padded stride: " << s;
    }
    EXPECT_TRUE(found_padded_stride) << "no stage offset used the padded per-stage buffer stride";
}
