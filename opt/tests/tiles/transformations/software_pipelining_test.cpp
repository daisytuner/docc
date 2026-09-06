#include "sdfg/tiles/transformations/software_pipelining.h"

#include <gtest/gtest.h>

#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_nodes/async_copy_node.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/if_else.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/targets/cuda/cuda.h"
#include "sdfg/types/array.h"

using namespace sdfg;

namespace {

// Builds: GPU map over i { for k in [0,K) { copy A[k] -> buf[1][2]; compute buf[1][2] -> C } }
// Returns the panel loop (the `for k`).
structured_control_flow::For& build(builder::StructuredSDFGBuilder& builder, int K, bool shared = true) {
    auto& root = builder.subject().root();
    types::Scalar f(types::PrimitiveType::Float);
    types::Pointer aptr(f);
    types::Scalar u64(types::PrimitiveType::UInt64);

    types::Array buf_inner(f, symbolic::integer(8));
    types::Array buf_type(
        shared ? types::StorageType::NV_Shared() : types::StorageType::CPU_Stack(),
        0,
        "",
        buf_inner,
        symbolic::integer(4)
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

    {
        auto& b = builder.add_block(kloop.root());
        auto& a = builder.add_access(b, "A");
        auto& tk = builder.add_tasklet(b, data_flow::TaskletCode::assign, "_out", {"_in"});
        auto& bufw = builder.add_access(b, "buf");
        builder.add_computational_memlet(b, a, tk, "_in", {k}, aptr);
        builder.add_computational_memlet(b, tk, "_out", bufw, {symbolic::integer(1), symbolic::integer(2)}, buf_type);
    }
    {
        auto& b = builder.add_block(kloop.root());
        auto& bufr = builder.add_access(b, "buf");
        auto& tk = builder.add_tasklet(b, data_flow::TaskletCode::assign, "_out", {"_in"});
        auto& c = builder.add_access(b, "C");
        builder.add_computational_memlet(b, bufr, tk, "_in", {symbolic::integer(1), symbolic::integer(2)}, buf_type);
        builder.add_computational_memlet(b, tk, "_out", c, {}, f);
    }
    return static_cast<structured_control_flow::For&>(gmap.root().at(0));
}

// Two cooperative shared operands (buf, buf2), each staged by its own copy
// block, both consumed by the compute. Returns the panel loop.
structured_control_flow::For& build_two(builder::StructuredSDFGBuilder& builder, int K) {
    auto& root = builder.subject().root();
    types::Scalar f(types::PrimitiveType::Float);
    types::Pointer aptr(f);
    types::Scalar u64(types::PrimitiveType::UInt64);
    types::Array buf_inner(f, symbolic::integer(8));
    types::Array buf_type(types::StorageType::NV_Shared(), 0, "", buf_inner, symbolic::integer(4));

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
    std::vector<std::pair<std::string, std::string>> ops = {{"A", "buf"}, {"B", "buf2"}};
    for (auto& names : ops) {
        auto& b = builder.add_block(kloop.root());
        auto& src = builder.add_access(b, names.first);
        auto& tk = builder.add_tasklet(b, data_flow::TaskletCode::assign, "_out", {"_in"});
        auto& w = builder.add_access(b, names.second);
        builder.add_computational_memlet(b, src, tk, "_in", {k}, aptr);
        builder.add_computational_memlet(b, tk, "_out", w, {symbolic::integer(1), symbolic::integer(2)}, buf_type);
    }
    {
        auto& b = builder.add_block(kloop.root());
        auto& r1 = builder.add_access(b, "buf");
        auto& r2 = builder.add_access(b, "buf2");
        auto& tk = builder.add_tasklet(b, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
        auto& c = builder.add_access(b, "C");
        builder.add_computational_memlet(b, r1, tk, "_in1", {symbolic::integer(1), symbolic::integer(2)}, buf_type);
        builder.add_computational_memlet(b, r2, tk, "_in2", {symbolic::integer(1), symbolic::integer(2)}, buf_type);
        builder.add_computational_memlet(b, tk, "_out", c, {}, f);
    }
    return static_cast<structured_control_flow::For&>(gmap.root().at(0));
}

// A panel loop whose body has a cooperative-copy Map (X_BLOCK over c in [0,COOP))
// writing buf[c] from A[src_coeff*c], then a compute reading buf. Returns the
// panel loop. src_coeff>1 makes the source non-contiguous in c.
structured_control_flow::For& build_coop(builder::StructuredSDFGBuilder& builder, int K, int COOP, int src_coeff = 1) {
    auto& root = builder.subject().root();
    types::Scalar f(types::PrimitiveType::Float);
    types::Pointer aptr(f);
    types::Scalar u64(types::PrimitiveType::UInt64);
    types::Array buf_type(types::StorageType::NV_Shared(), 0, "", f, symbolic::integer(COOP));

    builder.add_container("A", aptr, true);
    builder.add_container("buf", buf_type);
    builder.add_container("C", f, true);
    builder.add_container("i", u64);
    builder.add_container("k", u64);
    builder.add_container("c", u64);

    auto mk_sched = [&]() {
        auto s = cuda::ScheduleType_CUDA::create();
        cuda::ScheduleType_CUDA::dimension(s, cuda::CUDADimension::X);
        cuda::ScheduleType_CUDA::block_size(s, symbolic::integer(32));
        return s;
    };
    auto i = symbolic::symbol("i");
    auto& gmap = builder.add_map(
        root,
        i,
        symbolic::Lt(i, symbolic::integer(64)),
        symbolic::integer(0),
        symbolic::add(i, symbolic::integer(1)),
        mk_sched()
    );
    auto k = symbolic::symbol("k");
    auto& kloop = builder.add_for(
        gmap.root(),
        k,
        symbolic::Lt(k, symbolic::integer(K)),
        symbolic::integer(0),
        symbolic::add(k, symbolic::integer(1))
    );
    auto c = symbolic::symbol("c");
    auto& coop = builder.add_map(
        kloop.root(),
        c,
        symbolic::Lt(c, symbolic::integer(COOP)),
        symbolic::integer(0),
        symbolic::add(c, symbolic::integer(1)),
        mk_sched()
    );
    {
        auto& b = builder.add_block(coop.root());
        auto& a = builder.add_access(b, "A");
        auto& tk = builder.add_tasklet(b, data_flow::TaskletCode::assign, "_out", {"_in"});
        auto& bufw = builder.add_access(b, "buf");
        builder.add_computational_memlet(b, a, tk, "_in", {symbolic::mul(symbolic::integer(src_coeff), c)}, aptr);
        builder.add_computational_memlet(b, tk, "_out", bufw, {c}, buf_type);
    }
    {
        auto& b = builder.add_block(kloop.root());
        auto& bufr = builder.add_access(b, "buf");
        auto& tk = builder.add_tasklet(b, data_flow::TaskletCode::assign, "_out", {"_in"});
        auto& cc = builder.add_access(b, "C");
        builder.add_computational_memlet(b, bufr, tk, "_in", {symbolic::integer(0)}, buf_type);
        builder.add_computational_memlet(b, tk, "_out", cc, {}, f);
    }
    return static_cast<structured_control_flow::For&>(gmap.root().at(0));
}

// Like build(), but the shared copy-in is nested inside a boundary-guard IfElse
// (as StreamK's ragged panel produces). A shared write inside a conditional still
// stages a shared tile, so the staging gate must descend into the IfElse.
structured_control_flow::For& build_guarded(builder::StructuredSDFGBuilder& builder, int K) {
    auto& root = builder.subject().root();
    types::Scalar f(types::PrimitiveType::Float);
    types::Pointer aptr(f);
    types::Scalar u64(types::PrimitiveType::UInt64);
    types::Array buf_inner(f, symbolic::integer(8));
    types::Array buf_type(types::StorageType::NV_Shared(), 0, "", buf_inner, symbolic::integer(4));

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

    // Boundary-guarded cooperative copy-in: buf[1][2] = A[k] under `if (k < K)`.
    {
        auto& if_else = builder.add_if_else(kloop.root());
        auto& guarded = builder.add_case(if_else, symbolic::Lt(k, symbolic::integer(K)));
        auto& b = builder.add_block(guarded);
        auto& a = builder.add_access(b, "A");
        auto& tk = builder.add_tasklet(b, data_flow::TaskletCode::assign, "_out", {"_in"});
        auto& bufw = builder.add_access(b, "buf");
        builder.add_computational_memlet(b, a, tk, "_in", {k}, aptr);
        builder.add_computational_memlet(b, tk, "_out", bufw, {symbolic::integer(1), symbolic::integer(2)}, buf_type);
    }
    {
        auto& b = builder.add_block(kloop.root());
        auto& bufr = builder.add_access(b, "buf");
        auto& tk = builder.add_tasklet(b, data_flow::TaskletCode::assign, "_out", {"_in"});
        auto& c = builder.add_access(b, "C");
        builder.add_computational_memlet(b, bufr, tk, "_in", {symbolic::integer(1), symbolic::integer(2)}, buf_type);
        builder.add_computational_memlet(b, tk, "_out", c, {}, f);
    }
    return static_cast<structured_control_flow::For&>(gmap.root().at(0));
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
    EXPECT_TRUE(symbolic::eq(buf2->num_elements(), symbolic::integer(4))); // unchanged (no stage axis)

    // Exactly one cp.async operand (buf); buf2 keeps its synchronous copy.
    size_t async = 0;
    std::function<void(structured_control_flow::ControlFlowNode&)> scan =
        [&](structured_control_flow::ControlFlowNode& n) {
            if (auto* b = dynamic_cast<structured_control_flow::Block*>(&n)) {
                for (auto& node : b->dataflow().nodes()) {
                    if (dynamic_cast<data_flow::CpAsyncCopyNode*>(&node) != nullptr) {
                        async++;
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
            } else if (auto* loop = dynamic_cast<structured_control_flow::StructuredLoop*>(&n)) {
                scan(loop->root());
            }
        };
    auto& map_body = static_cast<structured_control_flow::Sequence&>(*kloop.get_parent());
    scan(map_body);
    EXPECT_EQ(async, 2u); // one prologue + one in-loop, for buf only
}

TEST(SoftwarePipeliningTest, VectorizeStridesCoopMapAndWidensCpAsync) {
    builder::StructuredSDFGBuilder builder("sp", FunctionType_CPU);
    auto& kloop = build_coop(builder, /*K=*/4, /*COOP=*/8);
    auto& sdfg = builder.subject();
    analysis::AnalysisManager am(sdfg);
    transformations::SoftwarePipelining sp(kloop, 2, /*single_operand=*/false, /*vectorize=*/true);
    ASSERT_TRUE(sp.can_be_applied(builder, am));
    sp.apply(builder, am);

    size_t async = 0, bytes16 = 0, strided4 = 0;
    std::function<void(structured_control_flow::ControlFlowNode&)> scan =
        [&](structured_control_flow::ControlFlowNode& n) {
            if (auto* b = dynamic_cast<structured_control_flow::Block*>(&n)) {
                for (auto& node : b->dataflow().nodes()) {
                    if (auto* cp = dynamic_cast<data_flow::CpAsyncCopyNode*>(&node)) {
                        async++;
                        if (cp->bytes() == 16) {
                            bytes16++;
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
            } else if (auto* map = dynamic_cast<structured_control_flow::Map*>(&n)) {
                if (!map->stride().is_null() && map->stride()->as_int() == 4) {
                    strided4++;
                }
                scan(map->root());
            } else if (auto* loop = dynamic_cast<structured_control_flow::StructuredLoop*>(&n)) {
                scan(loop->root());
            }
        };
    auto& map_body = static_cast<structured_control_flow::Sequence&>(*kloop.get_parent());
    scan(map_body);
    EXPECT_EQ(async, 2u); // prologue + in-loop
    EXPECT_EQ(bytes16, 2u); // both widened to float4
    EXPECT_EQ(strided4, 2u); // both coop maps strided by 4
}

TEST(SoftwarePipeliningTest, VectorizeRejectsNonContiguousSource) {
    builder::StructuredSDFGBuilder builder("sp", FunctionType_CPU);
    auto& kloop = build_coop(builder, /*K=*/4, /*COOP=*/8, /*src_coeff=*/2); // A[2c] -> not contiguous
    auto& sdfg = builder.subject();
    analysis::AnalysisManager am(sdfg);
    transformations::SoftwarePipelining sp(kloop, 2, /*single_operand=*/false, /*vectorize=*/true);
    ASSERT_TRUE(sp.can_be_applied(builder, am));
    sp.apply(builder, am);

    // The non-contiguous source must keep scalar (4-byte) cp.async and unit-stride
    // coop maps — the widening guard must not fire.
    size_t bytes4 = 0, bytes16 = 0, strided4 = 0;
    std::function<void(structured_control_flow::ControlFlowNode&)> scan =
        [&](structured_control_flow::ControlFlowNode& n) {
            if (auto* b = dynamic_cast<structured_control_flow::Block*>(&n)) {
                for (auto& node : b->dataflow().nodes()) {
                    if (auto* cp = dynamic_cast<data_flow::CpAsyncCopyNode*>(&node)) {
                        (cp->bytes() == 16 ? bytes16 : bytes4)++;
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
            } else if (auto* map = dynamic_cast<structured_control_flow::Map*>(&n)) {
                if (!map->stride().is_null() && map->stride()->as_int() == 4) {
                    strided4++;
                }
                scan(map->root());
            } else if (auto* loop = dynamic_cast<structured_control_flow::StructuredLoop*>(&n)) {
                scan(loop->root());
            }
        };
    auto& map_body = static_cast<structured_control_flow::Sequence&>(*kloop.get_parent());
    scan(map_body);
    EXPECT_EQ(bytes16, 0u);
    EXPECT_EQ(bytes4, 2u);
    EXPECT_EQ(strided4, 0u);
}

TEST(SoftwarePipeliningTest, CanBeApplied) {
    builder::StructuredSDFGBuilder builder("sp", FunctionType_CPU);
    auto& kloop = build(builder, /*K=*/4);
    analysis::AnalysisManager am(builder.subject());
    transformations::SoftwarePipelining sp(kloop, 2);
    EXPECT_TRUE(sp.can_be_applied(builder, am));
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

    // buf gained the leading [2] stage axis; inner is the original [4].
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

    // buf gained a leading [2] axis; inner is the original [4][8].
    auto* outer = dynamic_cast<const types::Array*>(&sdfg.type("buf"));
    ASSERT_NE(outer, nullptr);
    EXPECT_TRUE(symbolic::eq(outer->num_elements(), symbolic::integer(2)));
    EXPECT_TRUE(outer->storage_type().is_nv_shared());
    auto* inner = dynamic_cast<const types::Array*>(&outer->element_type());
    ASSERT_NE(inner, nullptr);
    EXPECT_TRUE(symbolic::eq(inner->num_elements(), symbolic::integer(4)));

    // A prologue sequence was inserted before the loop.
    ASSERT_EQ(map_body.size(), 2u);
    auto* prologue = dynamic_cast<structured_control_flow::Sequence*>(&map_body.at(0));
    ASSERT_NE(prologue, nullptr);

    // Prologue prefetches panel 0 via cp.async + commits it. The dst reference
    // memlet addresses a fixed stage slot (leading index independent of k).
    size_t prologue_async = 0, prologue_commit = 0;
    for (size_t i = 0; i < prologue->size(); i++) {
        auto* b = dynamic_cast<structured_control_flow::Block*>(&prologue->at(i));
        if (b == nullptr) {
            continue;
        }
        for (auto& node : b->dataflow().nodes()) {
            if (dynamic_cast<data_flow::CpAsyncCopyNode*>(&node) != nullptr) {
                prologue_async++;
            }
            if (dynamic_cast<data_flow::PipelineCommitNode*>(&node) != nullptr) {
                prologue_commit++;
            }
        }
        for (auto* acc : b->dataflow().data_nodes()) {
            if (acc->data() == "buf") {
                for (auto& m : b->dataflow().out_edges(*acc)) { // address-of buf[stage]
                    ASSERT_EQ(m.subset().size(), 3u);
                    EXPECT_TRUE(symbolic::atoms(m.subset().at(0)).empty());
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
    size_t loop_async = 0, loop_commit = 0, loop_wait = 0;
    auto stage = symbolic::mod(symbolic::symbol("k"), symbolic::integer(2));
    std::function<void(structured_control_flow::ControlFlowNode&)> scan =
        [&](structured_control_flow::ControlFlowNode& n) {
            if (auto* b = dynamic_cast<structured_control_flow::Block*>(&n)) {
                for (auto& node : b->dataflow().nodes()) {
                    if (dynamic_cast<data_flow::CpAsyncCopyNode*>(&node) != nullptr) {
                        loop_async++;
                    }
                    if (dynamic_cast<data_flow::PipelineCommitNode*>(&node) != nullptr) {
                        loop_commit++;
                    }
                    if (dynamic_cast<data_flow::PipelineWaitNode*>(&node) != nullptr) {
                        loop_wait++;
                    }
                }
                for (auto* acc : b->dataflow().data_nodes()) {
                    if (acc->data() != "buf") {
                        continue;
                    }
                    for (auto& m : b->dataflow().out_edges(*acc)) { // compute read of buf
                        if (m.dst_conn() == "ref") {
                            continue; // address-of for the cp.async prefetch
                        }
                        if (m.subset().size() == 3u && !symbolic::atoms(m.subset().at(0)).empty()) {
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
