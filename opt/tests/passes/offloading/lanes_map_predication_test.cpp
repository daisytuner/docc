#include "sdfg/passes/offloading/lanes_map_predication.h"

#include <gtest/gtest.h>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_nodes/barrier_local_node.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/passes/offloading/reduction_shared_memory_delinearization.h"
#include "sdfg/serializer/json_serializer.h"
#include "sdfg/structured_control_flow/if_else.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/structured_control_flow/reduce.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/targets/cuda/cuda.h"
#include "sdfg/targets/gpu/gpu_map_utils.h"
#include "sdfg/targets/gpu/gpu_wave_barrier_node.h"

namespace sdfg {

namespace {

using structured_control_flow::Block;
using structured_control_flow::IfElse;
using structured_control_flow::Sequence;

constexpr int64_t LANES = 32;

/// g X_GRID(1) > i X_BLOCK(2, lanes): the body is built by each test.
struct LanesKernel {
    builder::StructuredSDFGBuilder builder{"lanes_sdfg", FunctionType_CPU};
    types::Scalar f32{types::PrimitiveType::Float};
    types::Pointer dev{types::StorageType::NV_Generic(), 0, "", types::Scalar(types::PrimitiveType::Float)};
    structured_control_flow::Map* map = nullptr;

    explicit LanesKernel(int64_t lanes = LANES) {
        auto& root = builder.subject().root();
        types::Scalar i32(types::PrimitiveType::Int32);
        for (auto name : {"g", "i", "k"}) {
            builder.add_container(name, i32);
        }
        builder.add_container("s", f32);
        builder.add_container("t", f32);
        builder.add_container("dev", dev);

        auto g = symbolic::symbol("g");
        auto i = symbolic::symbol("i");
        auto& grid = builder.add_map(
            root,
            g,
            symbolic::Lt(g, symbolic::one()),
            symbolic::zero(),
            symbolic::add(g, symbolic::one()),
            cuda::ScheduleType_CUDA_Offload::create<
                cuda::ScheduleType_CUDA_Offload>(gpu::TargetLevel::X_GRID, symbolic::one())
        );
        auto schedule = cuda::ScheduleType_CUDA_Offload::create<
            cuda::ScheduleType_CUDA_Offload>(gpu::TargetLevel::X_BLOCK, symbolic::integer(2));
        if (lanes != 1) {
            gpu::ScheduleType_GPU_Offload::lanes(schedule, symbolic::integer(lanes));
        }
        map = &builder.add_map(
            grid.root(),
            i,
            symbolic::Lt(i, symbolic::integer(5)),
            symbolic::zero(),
            symbolic::add(i, symbolic::one()),
            schedule
        );
    }

    /// dev[index] = dev[index] + 1 (a lane-shared read-modify-write).
    Block& increment(Sequence& parent, const symbolic::Expression& index) {
        auto& block = builder.add_block(parent);
        auto& in = builder.add_access(block, "dev");
        auto& out = builder.add_access(block, "dev");
        auto& one = builder.add_constant(block, "1.0f", f32);
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in0", "_in1"});
        builder.add_computational_memlet(block, in, tasklet, "_in0", {index}, dev);
        builder.add_computational_memlet(block, one, tasklet, "_in1", {}, f32);
        builder.add_computational_memlet(block, tasklet, "_out", out, {index}, dev);
        return block;
    }

    /// scalar = dev[i] (a thread-private read of lane-shared memory).
    Block& load(Sequence& parent, const std::string& scalar) {
        auto& block = builder.add_block(parent);
        auto& in = builder.add_access(block, "dev");
        auto& out = builder.add_access(block, scalar);
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
        builder.add_computational_memlet(block, in, tasklet, "_in", {symbolic::symbol("i")}, dev);
        builder.add_computational_memlet(block, tasklet, "_out", out, {}, f32);
        return block;
    }

    bool run() {
        analysis::AnalysisManager analysis_manager(builder.subject());
        passes::LanesMapPredication pass;
        return pass.run_pass(builder, analysis_manager);
    }
};

bool is_leader_guard(const structured_control_flow::ControlFlowNode& node, int64_t lanes = LANES) {
    auto* if_else = dynamic_cast<const IfElse*>(&node);
    return if_else != nullptr && if_else->size() == 1 &&
           symbolic::eq(if_else->at(0).second, gpu::lane_leader_condition(lanes));
}

bool is_wave_barrier(const structured_control_flow::ControlFlowNode& node) {
    auto* block = dynamic_cast<const Block*>(&node);
    if (block == nullptr || block->dataflow().library_nodes().size() != 1) {
        return false;
    }
    return (*block->dataflow().library_nodes().begin())->code() == gpu::LibraryNodeType_GpuWaveBarrier;
}

} // namespace

// The read-modify-write is executed by the leader lane only; the later private read-back by
// every lane needs a wave barrier after it. The thread-private loads stay replicated.
TEST(LanesMapPredicationTest, GuardsSharedWriteAndSynchronizesReadBack) {
    LanesKernel k;
    auto& body = k.map->root();
    auto& before = k.load(body, "s");
    auto kk = symbolic::symbol("k");
    auto& loop = k.builder.add_for(
        body, kk, symbolic::Lt(kk, symbolic::integer(3)), symbolic::zero(), symbolic::add(kk, symbolic::one())
    );
    auto& rmw = k.increment(loop.root(), symbolic::symbol("i"));
    auto& after = k.load(body, "t");

    EXPECT_TRUE(k.run());

    ASSERT_EQ(body.size(), 3u);
    EXPECT_EQ(&body.at(0), &before);
    EXPECT_EQ(&body.at(2), &after);
    ASSERT_EQ(loop.root().size(), 2u);
    ASSERT_TRUE(is_leader_guard(loop.root().at(0)));
    auto& guarded = static_cast<IfElse&>(loop.root().at(0)).at(0).first;
    ASSERT_EQ(guarded.size(), 1u);
    EXPECT_EQ(&guarded.at(0), &rmw);
    EXPECT_TRUE(is_wave_barrier(loop.root().at(1)));
    EXPECT_NO_THROW(k.builder.subject().validate());

    // Idempotent: predicated blocks are recognized and left alone.
    serializer::JSONSerializer serializer;
    auto once = serializer.serialize(k.builder.subject()).dump();
    EXPECT_FALSE(k.run());
    EXPECT_EQ(serializer.serialize(k.builder.subject()).dump(), once);
}

// Consecutive shared writes share one guard; without a lane reading them afterwards, no barrier.
TEST(LanesMapPredicationTest, GroupsRunsWithoutBarrierWhenUnread) {
    LanesKernel k;
    auto& body = k.map->root();
    auto& first = k.increment(body, symbolic::symbol("i"));
    auto& second = k.increment(body, symbolic::add(symbolic::symbol("i"), symbolic::integer(5)));

    EXPECT_TRUE(k.run());

    ASSERT_EQ(body.size(), 1u);
    ASSERT_TRUE(is_leader_guard(body.at(0)));
    auto& guarded = static_cast<IfElse&>(body.at(0)).at(0).first;
    ASSERT_EQ(guarded.size(), 2u);
    EXPECT_EQ(&guarded.at(0), &first);
    EXPECT_EQ(&guarded.at(1), &second);
}

// Thread-private writes, wave-collective nodes and lane-indexed (threadIdx.x) work stay unguarded.
TEST(LanesMapPredicationTest, LeavesPrivateCollectiveAndLaneIndexedBlocks) {
    LanesKernel k;
    auto& body = k.map->root();
    k.load(body, "s");
    auto& barrier = k.builder.add_block(body);
    k.builder.add_library_node<data_flow::BarrierLocalNode>(barrier, DebugInfo());
    k.increment(
        body, symbolic::add(symbolic::mul(symbolic::symbol("i"), symbolic::integer(LANES)), symbolic::threadIdx_x())
    );

    EXPECT_FALSE(k.run());
    for (size_t i = 0; i < body.size(); ++i) {
        EXPECT_TRUE(dynamic_cast<Block*>(&body.at(i)) != nullptr);
    }
}

// An offloaded reduce in a lanes map commits its accumulator from the leader lane, so a later read
// on the other lanes is preceded by a wave barrier. Its body only updates materialized partials.
TEST(LanesMapPredicationTest, SynchronizesReadBackOfOffloadedReduce) {
    LanesKernel k;
    auto& body = k.map->root();
    auto kk = symbolic::symbol("k");
    auto& reduce = k.builder.add_reduce(
        body,
        kk,
        symbolic::Lt(kk, symbolic::integer(4)),
        symbolic::zero(),
        symbolic::add(kk, symbolic::one()),
        {structured_control_flow::ReductionInfo{structured_control_flow::ReductionOperation::Add, "dev"}},
        cuda::ScheduleType_CUDA_Offload::create<
            cuda::ScheduleType_CUDA_Offload>(gpu::TargetLevel::Y_BLOCK, symbolic::integer(4))
    );
    k.increment(reduce.root(), symbolic::symbol("i"));
    auto& after = k.load(body, "s");

    analysis::AnalysisManager analysis_manager(k.builder.subject());
    passes::ReductionSharedMemoryDelinearization materialize;
    ASSERT_TRUE(materialize.run_pass(k.builder, analysis_manager));
    passes::LanesMapPredication pass;
    EXPECT_TRUE(pass.run_pass(k.builder, analysis_manager));

    ASSERT_EQ(body.size(), 3u);
    EXPECT_EQ(&body.at(0), &reduce);
    EXPECT_TRUE(is_wave_barrier(body.at(1)));
    EXPECT_EQ(&body.at(2), &after);
    for (size_t i = 0; i < reduce.root().size(); ++i) {
        EXPECT_FALSE(is_leader_guard(reduce.root().at(i)));
    }
    EXPECT_FALSE(pass.run_pass(k.builder, analysis_manager));
}

// Plain X_BLOCK maps (lanes == 1) run each iteration on one thread: nothing to predicate.
TEST(LanesMapPredicationTest, IgnoresMapsWithoutLanes) {
    LanesKernel k(1);
    k.increment(k.map->root(), symbolic::symbol("i"));
    EXPECT_FALSE(k.run());
    EXPECT_TRUE(dynamic_cast<Block*>(&k.map->root().at(0)) != nullptr);
}

// A predicated block must not produce thread-private values the other lanes consume.
TEST(LanesMapPredicationTest, RejectsPredicatedBlockFeedingOtherLanes) {
    LanesKernel k;
    auto& body = k.map->root();
    auto& block = k.increment(body, symbolic::symbol("i"));
    auto& in = k.builder.add_access(block, "dev");
    auto& t = k.builder.add_access(block, "t");
    auto& copy = k.builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    k.builder.add_computational_memlet(block, in, copy, "_in", {symbolic::symbol("i")}, k.dev);
    k.builder.add_computational_memlet(block, copy, "_out", t, {}, k.f32);

    auto& consumer = k.builder.add_block(body);
    auto& t_in = k.builder.add_access(consumer, "t");
    auto& s = k.builder.add_access(consumer, "s");
    auto& assign = k.builder.add_tasklet(consumer, data_flow::TaskletCode::assign, "_out", {"_in"});
    k.builder.add_computational_memlet(consumer, t_in, assign, "_in", {}, k.f32);
    k.builder.add_computational_memlet(consumer, assign, "_out", s, {}, k.f32);

    EXPECT_THROW(k.run(), InvalidSDFGException);
}

} // namespace sdfg
