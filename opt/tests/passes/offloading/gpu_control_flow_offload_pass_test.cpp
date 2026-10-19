#include "sdfg/passes/offloading/gpu_control_flow_offload_pass.h"

#include <gtest/gtest.h>

#include <set>
#include <string>
#include <vector>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_nodes/stdlib/memset.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/passes/scheduler/cuda_offload_scheduler.h"
#include "sdfg/passes/scheduler/loop_scheduling_pass.h"
#include "sdfg/passes/scheduler/rocm_offload_scheduler.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/for.h"
#include "sdfg/structured_control_flow/if_else.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/structured_control_flow/return.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/structured_control_flow/while.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/targets/cuda/cuda.h"
#include "sdfg/targets/rocm/rocm.h"

using namespace sdfg;
using namespace sdfg::structured_control_flow;

namespace {

struct CUDACase {
    using Pass = passes::CUDAControlFlowOffloadPass;
    using Scheduler = passes::scheduler::CUDAOffloadScheduler;
    using OffloadSchedule = cuda::ScheduleType_CUDA_Offload;

    static const data_flow::ImplementationType& with_transfers() {
        return cuda::ImplementationType_CUDAWithTransfers;
    }
    static const data_flow::ImplementationType& without_transfers() {
        return cuda::ImplementationType_CUDAWithoutTransfers;
    }
};

struct ROCMCase {
    using Pass = passes::ROCMControlFlowOffloadPass;
    using Scheduler = passes::scheduler::ROCMOffloadScheduler;
    using OffloadSchedule = rocm::ScheduleType_ROCM_Offload;

    static const data_flow::ImplementationType& with_transfers() {
        return rocm::ImplementationType_ROCMWithTransfers;
    }
    static const data_flow::ImplementationType& without_transfers() {
        return rocm::ImplementationType_ROCMWithoutTransfers;
    }
};

using GPUCases = ::testing::Types<CUDACase, ROCMCase>;

template<typename Case>
class GPUControlFlowOffloadPassTest : public ::testing::Test {
protected:
    using OffloadSchedule = typename Case::OffloadSchedule;

    builder::StructuredSDFGBuilder builder_{"control_flow_offload_test", FunctionType_CPU};

    types::Scalar float_ = types::Scalar(types::PrimitiveType::Float);
    types::Pointer float_ptr_ = types::Pointer(float_);
    types::Scalar int64_ = types::Scalar(types::PrimitiveType::Int64);

    void SetUp() override {
        builder_.add_container("A", float_ptr_, true);
        builder_.add_container("n", int64_, true);
    }

    Sequence& root() {
        return builder_.subject().root();
    }

    symbolic::Condition n_lt(int64_t bound) {
        return symbolic::Lt(symbolic::symbol("n"), symbolic::integer(bound));
    }

    // A[dst] = A[src]
    Block& add_copy(Sequence& parent, int64_t dst, int64_t src) {
        auto& block = builder_.add_block(parent);
        auto& in = builder_.add_access(block, "A");
        auto& out = builder_.add_access(block, "A");
        auto& tasklet = builder_.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
        builder_.add_computational_memlet(block, in, tasklet, "_in", {symbolic::integer(src)}, float_ptr_);
        builder_.add_computational_memlet(block, tasklet, "_out", out, {symbolic::integer(dst)}, float_ptr_);
        return block;
    }

    // s = A[0] with s being a scalar argument; offloading must reject the scalar output.
    Block& add_scalar_output(Sequence& parent) {
        if (!builder_.subject().exists("s")) {
            builder_.add_container("s", float_, true);
        }
        auto& block = builder_.add_block(parent);
        auto& in = builder_.add_access(block, "A");
        auto& out = builder_.add_access(block, "s");
        auto& tasklet = builder_.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
        builder_.add_computational_memlet(block, in, tasklet, "_in", {symbolic::zero()}, float_ptr_);
        builder_.add_computational_memlet(block, tasklet, "_out", out, {}, float_);
        return block;
    }

    // for i in [1, 16): A[i] = A[i - 1]
    For& add_sequential_for(Sequence& parent) {
        if (!builder_.subject().exists("i")) {
            builder_.add_container("i", int64_);
        }
        auto indvar = symbolic::symbol("i");
        auto& loop = builder_.add_for(
            parent,
            indvar,
            symbolic::Lt(indvar, symbolic::integer(16)),
            symbolic::one(),
            symbolic::add(indvar, symbolic::one())
        );
        auto& block = builder_.add_block(loop.root());
        auto& in = builder_.add_access(block, "A");
        auto& out = builder_.add_access(block, "A");
        auto& tasklet = builder_.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
        builder_
            .add_computational_memlet(block, in, tasklet, "_in", {symbolic::sub(indvar, symbolic::one())}, float_ptr_);
        builder_.add_computational_memlet(block, tasklet, "_out", out, {indvar}, float_ptr_);
        return loop;
    }

    // map j in [0, bound): A[j] = 0.0f
    Map& add_zero_map(Sequence& parent, int64_t bound, const ScheduleType& schedule) {
        if (!builder_.subject().exists("j")) {
            builder_.add_container("j", int64_);
        }
        auto indvar = symbolic::symbol("j");
        auto& map = builder_.add_map(
            parent,
            indvar,
            symbolic::Lt(indvar, symbolic::integer(bound)),
            symbolic::zero(),
            symbolic::add(indvar, symbolic::one()),
            schedule
        );
        auto& block = builder_.add_block(map.root());
        auto& out = builder_.add_access(block, "A");
        auto& constant = builder_.add_constant(block, "0.0f", float_);
        auto& tasklet = builder_.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
        builder_.add_computational_memlet(block, constant, tasklet, "_in", {}, float_);
        builder_.add_computational_memlet(block, tasklet, "_out", out, {indvar}, float_ptr_);
        return map;
    }

    bool run_pass() {
        analysis::AnalysisManager analysis_manager(builder_.subject());
        typename Case::Pass pass;
        return pass.run(builder_, analysis_manager);
    }

    bool run_scheduling() {
        analysis::AnalysisManager analysis_manager(builder_.subject());
        typename Case::Scheduler scheduler;
        passes::scheduler::LoopSchedulingPass pass({&scheduler}, nullptr);
        return pass.run(builder_, analysis_manager);
    }

    bool is_offloaded(const StructuredLoop& loop) {
        return loop.schedule_type().value() == OffloadSchedule::value();
    }

    std::vector<Map*> offloaded_root_maps() {
        std::vector<Map*> maps;
        for (size_t i = 0; i < root().size(); ++i) {
            if (auto* map = dynamic_cast<Map*>(&root().at(i))) {
                if (is_offloaded(*map)) {
                    maps.push_back(map);
                }
            }
        }
        return maps;
    }

    // The offloaded root-level map whose body directly contains `node`, or nullptr.
    Map* enclosing_offloaded_map(ControlFlowNode& node) {
        auto* body = node.get_parent();
        if (body == nullptr) {
            return nullptr;
        }
        auto* map = dynamic_cast<Map*>(body->get_parent());
        if (map == nullptr || !is_offloaded(*map) || map->get_parent() != &root()) {
            return nullptr;
        }
        return map;
    }

    void expect_trivial_offload_map(Map& map) {
        EXPECT_TRUE(symbolic::eq(map.num_iterations(), symbolic::one()));
        EXPECT_EQ(OffloadSchedule::target_level(map.schedule_type()), gpu::TargetLevel::X_GRID);
        EXPECT_EQ(OffloadSchedule::parallel_size(map.schedule_type())->as_int(), 1);
    }

    std::set<std::string> container_names() {
        auto containers = builder_.subject().containers();
        return {containers.begin(), containers.end()};
    }
};

} // namespace

TYPED_TEST_SUITE(GPUControlFlowOffloadPassTest, GPUCases);

// ---- Nothing to do ---------------------------------------------------------

TYPED_TEST(GPUControlFlowOffloadPassTest, EmptyRootReturnsFalse) {
    EXPECT_FALSE(this->run_pass());
    EXPECT_EQ(this->root().size(), 0);
}

TYPED_TEST(GPUControlFlowOffloadPassTest, IgnoresControlFlowNestedInLoops) {
    auto& loop = this->add_sequential_for(this->root());
    auto& if_else = this->builder_.add_if_else(loop.root());
    auto& branch = this->builder_.add_case(if_else, this->n_lt(10));
    auto& block = this->add_copy(branch, 0, 1);

    EXPECT_FALSE(this->run_pass());

    ASSERT_EQ(this->root().size(), 1);
    EXPECT_EQ(&this->root().at(0), &loop);
    EXPECT_EQ(loop.schedule_type().category(), ScheduleTypeCategory::None);
    EXPECT_EQ(if_else.get_parent(), &loop.root());
    EXPECT_EQ(block.get_parent(), &branch);
}

TYPED_TEST(GPUControlFlowOffloadPassTest, OffloadsNodesWithoutComputation) {
    auto& empty_block = this->builder_.add_block(this->root());
    auto& if_else = this->builder_.add_if_else(this->root());
    this->builder_.add_case(if_else, this->n_lt(10));

    EXPECT_TRUE(this->run_pass());

    auto maps = this->offloaded_root_maps();
    ASSERT_EQ(maps.size(), 1);
    auto& map = *maps.front();
    this->expect_trivial_offload_map(map);
    ASSERT_EQ(map.root().size(), 2);
    EXPECT_EQ(&map.root().at(0), &empty_block);
    EXPECT_EQ(&map.root().at(1), &if_else);
}

// ---- Wrapping of eligible node types ---------------------------------------

TYPED_TEST(GPUControlFlowOffloadPassTest, WrapsSingleBlock) {
    auto& block = this->add_copy(this->root(), 0, 1);

    EXPECT_TRUE(this->run_pass());

    auto maps = this->offloaded_root_maps();
    ASSERT_EQ(maps.size(), 1);
    auto& map = *maps.front();
    this->expect_trivial_offload_map(map);
    ASSERT_EQ(map.root().size(), 1);
    EXPECT_EQ(&map.root().at(0), &block);
    // Device allocation and transfers are placed around the offloaded map.
    EXPECT_GT(this->root().size(), 1);
}

TYPED_TEST(GPUControlFlowOffloadPassTest, WrapsIfElse) {
    auto& if_else = this->builder_.add_if_else(this->root());
    auto& then_branch = this->builder_.add_case(if_else, this->n_lt(10));
    this->add_copy(then_branch, 0, 1);
    auto& else_branch = this->builder_.add_case(if_else, symbolic::Not(this->n_lt(10)));
    this->add_copy(else_branch, 1, 0);

    EXPECT_TRUE(this->run_pass());

    auto* map = this->enclosing_offloaded_map(if_else);
    ASSERT_NE(map, nullptr);
    this->expect_trivial_offload_map(*map);
    EXPECT_EQ(map->root().size(), 1);
    EXPECT_EQ(this->offloaded_root_maps().size(), 1);
}

TYPED_TEST(GPUControlFlowOffloadPassTest, WrapsWhileWithBreak) {
    auto& loop = this->builder_.add_while(this->root());
    this->add_copy(loop.root(), 0, 1);
    auto& if_else = this->builder_.add_if_else(loop.root());
    auto& branch = this->builder_.add_case(if_else, this->n_lt(10));
    this->builder_.add_break(branch);

    EXPECT_TRUE(this->run_pass());

    auto* map = this->enclosing_offloaded_map(loop);
    ASSERT_NE(map, nullptr);
    this->expect_trivial_offload_map(*map);
    EXPECT_EQ(map->root().size(), 1);
    EXPECT_EQ(this->offloaded_root_maps().size(), 1);
}

TYPED_TEST(GPUControlFlowOffloadPassTest, WrapsSequence) {
    auto& sequence = this->builder_.add_sequence(this->root());
    this->add_copy(sequence, 0, 1);
    this->add_copy(sequence, 2, 3);

    EXPECT_TRUE(this->run_pass());

    auto* map = this->enclosing_offloaded_map(sequence);
    ASSERT_NE(map, nullptr);
    this->expect_trivial_offload_map(*map);
    EXPECT_EQ(map->root().size(), 1);
    EXPECT_EQ(sequence.size(), 2);
}

// ---- Grouping of consecutive nodes -----------------------------------------

TYPED_TEST(GPUControlFlowOffloadPassTest, MergesConsecutiveNodesInOrder) {
    auto& first = this->add_copy(this->root(), 0, 1);
    auto& if_else = this->builder_.add_if_else(this->root());
    auto& branch = this->builder_.add_case(if_else, this->n_lt(10));
    this->add_copy(branch, 1, 2);
    auto& last = this->add_copy(this->root(), 2, 3);

    EXPECT_TRUE(this->run_pass());

    auto maps = this->offloaded_root_maps();
    ASSERT_EQ(maps.size(), 1);
    auto& map = *maps.front();
    this->expect_trivial_offload_map(map);
    ASSERT_EQ(map.root().size(), 3);
    EXPECT_EQ(&map.root().at(0), &first);
    EXPECT_EQ(&map.root().at(1), &if_else);
    EXPECT_EQ(&map.root().at(2), &last);
}

TYPED_TEST(GPUControlFlowOffloadPassTest, UnscheduledForSeparatesRuns) {
    auto& before = this->add_copy(this->root(), 0, 1);
    auto& loop = this->add_sequential_for(this->root());
    auto& after = this->add_copy(this->root(), 20, 21);

    EXPECT_TRUE(this->run_pass());

    EXPECT_EQ(loop.get_parent(), &this->root());
    EXPECT_EQ(loop.schedule_type().category(), ScheduleTypeCategory::None);

    auto* before_map = this->enclosing_offloaded_map(before);
    auto* after_map = this->enclosing_offloaded_map(after);
    ASSERT_NE(before_map, nullptr);
    ASSERT_NE(after_map, nullptr);
    EXPECT_NE(before_map, after_map);
    EXPECT_EQ(before_map->root().size(), 1);
    EXPECT_EQ(after_map->root().size(), 1);
    EXPECT_EQ(this->offloaded_root_maps().size(), 2);
}

TYPED_TEST(GPUControlFlowOffloadPassTest, AssignmentBlockSeparatesRunsAndStaysOnHost) {
    this->builder_.add_container("k", this->int64_);
    auto& before = this->add_copy(this->root(), 0, 1);
    auto& assignments = this->builder_.add_assignments(this->root(), {{symbolic::symbol("k"), symbolic::integer(4)}});
    auto& after = this->add_copy(this->root(), 2, 3);

    EXPECT_TRUE(this->run_pass());

    EXPECT_EQ(assignments.get_parent(), &this->root());

    auto* before_map = this->enclosing_offloaded_map(before);
    auto* after_map = this->enclosing_offloaded_map(after);
    ASSERT_NE(before_map, nullptr);
    ASSERT_NE(after_map, nullptr);
    EXPECT_NE(before_map, after_map);
    EXPECT_EQ(this->offloaded_root_maps().size(), 2);
}

TYPED_TEST(GPUControlFlowOffloadPassTest, GPULibraryNodeSeparatesRunsAndIsNotWrapped) {
    using Case = TypeParam;
    auto& before = this->add_copy(this->root(), 0, 1);
    auto [memset_block, memset_node] = stdlib::add_memset_block(
        this->builder_, this->root(), "A", symbolic::zero(), symbolic::integer(1024), this->float_ptr_
    );
    memset_node.set_implementation_type(Case::without_transfers());
    auto& after = this->add_copy(this->root(), 2, 3);

    EXPECT_TRUE(this->run_pass());

    EXPECT_EQ(memset_block.get_parent(), &this->root());
    EXPECT_EQ(memset_node.implementation_type().value(), Case::without_transfers().value());

    auto* before_map = this->enclosing_offloaded_map(before);
    auto* after_map = this->enclosing_offloaded_map(after);
    ASSERT_NE(before_map, nullptr);
    ASSERT_NE(after_map, nullptr);
    EXPECT_NE(before_map, after_map);
    EXPECT_EQ(this->offloaded_root_maps().size(), 2);
}

// ---- Ineligible subtrees ---------------------------------------------------

TYPED_TEST(GPUControlFlowOffloadPassTest, RootReturnStaysOnHost) {
    auto& block = this->add_copy(this->root(), 0, 1);
    auto& ret = this->builder_.add_return(this->root(), "");

    EXPECT_TRUE(this->run_pass());

    EXPECT_EQ(ret.get_parent(), &this->root());
    auto* map = this->enclosing_offloaded_map(block);
    ASSERT_NE(map, nullptr);
    EXPECT_EQ(map->root().size(), 1);
}

TYPED_TEST(GPUControlFlowOffloadPassTest, SkipsIfElseContainingReturn) {
    auto& if_else = this->builder_.add_if_else(this->root());
    auto& branch = this->builder_.add_case(if_else, this->n_lt(10));
    this->add_copy(branch, 0, 1);
    this->builder_.add_return(branch, "");

    EXPECT_FALSE(this->run_pass());

    ASSERT_EQ(this->root().size(), 1);
    EXPECT_EQ(&this->root().at(0), &if_else);
    EXPECT_TRUE(this->offloaded_root_maps().empty());
}

TYPED_TEST(GPUControlFlowOffloadPassTest, SkipsWhileContainingOffloadedMap) {
    using OffloadSchedule = typename TestFixture::OffloadSchedule;
    auto& loop = this->builder_.add_while(this->root());
    this->add_copy(loop.root(), 0, 1);
    auto& inner = this->add_zero_map(
        loop.root(),
        128,
        OffloadSchedule::template create<OffloadSchedule>(gpu::TargetLevel::X_GRID, symbolic::integer(128))
    );
    auto& if_else = this->builder_.add_if_else(loop.root());
    auto& branch = this->builder_.add_case(if_else, this->n_lt(10));
    this->builder_.add_break(branch);

    EXPECT_FALSE(this->run_pass());

    ASSERT_EQ(this->root().size(), 1);
    EXPECT_EQ(&this->root().at(0), &loop);
    EXPECT_EQ(inner.get_parent(), &loop.root());
    EXPECT_TRUE(this->is_offloaded(inner));
}

// ---- Rollback and fallback -------------------------------------------------

TYPED_TEST(GPUControlFlowOffloadPassTest, RollsBackWhenOffloadIsNotApplicable) {
    auto& block = this->add_scalar_output(this->root());
    auto containers_before = this->container_names();

    EXPECT_FALSE(this->run_pass());

    ASSERT_EQ(this->root().size(), 1);
    EXPECT_EQ(&this->root().at(0), &block);
    EXPECT_EQ(this->container_names(), containers_before);
    EXPECT_TRUE(this->offloaded_root_maps().empty());
}

TYPED_TEST(GPUControlFlowOffloadPassTest, FallsBackToSingleNodesWhenRunFails) {
    auto& scalar_block = this->add_scalar_output(this->root());
    auto& copy_block = this->add_copy(this->root(), 0, 1);

    EXPECT_TRUE(this->run_pass());

    EXPECT_EQ(scalar_block.get_parent(), &this->root());
    auto* map = this->enclosing_offloaded_map(copy_block);
    ASSERT_NE(map, nullptr);
    this->expect_trivial_offload_map(*map);
    EXPECT_EQ(map->root().size(), 1);
    EXPECT_EQ(this->offloaded_root_maps().size(), 1);
}

TYPED_TEST(GPUControlFlowOffloadPassTest, SecondRunIsNoOp) {
    this->add_copy(this->root(), 0, 1);

    EXPECT_TRUE(this->run_pass());
    size_t root_size = this->root().size();
    auto containers = this->container_names();

    EXPECT_FALSE(this->run_pass());
    EXPECT_EQ(this->root().size(), root_size);
    EXPECT_EQ(this->container_names(), containers);
    EXPECT_EQ(this->offloaded_root_maps().size(), 1);
}

// ---- Integration with the offload schedulers -------------------------------

TYPED_TEST(GPUControlFlowOffloadPassTest, SchedulerOffloadsRootControlFlowWithoutLoops) {
    auto& if_else = this->builder_.add_if_else(this->root());
    auto& branch = this->builder_.add_case(if_else, this->n_lt(10));
    this->add_copy(branch, 0, 1);

    EXPECT_TRUE(this->run_scheduling());

    auto* map = this->enclosing_offloaded_map(if_else);
    ASSERT_NE(map, nullptr);
    this->expect_trivial_offload_map(*map);
    EXPECT_EQ(this->offloaded_root_maps().size(), 1);
}

TYPED_TEST(GPUControlFlowOffloadPassTest, SchedulerOffloadsRootBlockNextToParallelMap) {
    this->add_zero_map(this->root(), 1024, ScheduleType_Sequential::create());
    auto& block = this->add_copy(this->root(), 0, 1);

    EXPECT_TRUE(this->run_scheduling());

    auto* map = this->enclosing_offloaded_map(block);
    ASSERT_NE(map, nullptr);
    this->expect_trivial_offload_map(*map);
    EXPECT_EQ(map->root().size(), 1);
    // The parallel map and the wrapped block; transfer blocks around the parallel map stay unwrapped.
    EXPECT_EQ(this->offloaded_root_maps().size(), 2);
}

TYPED_TEST(GPUControlFlowOffloadPassTest, SchedulerDoesNotWrapExtractedLibraryNodeTransfers) {
    using Case = TypeParam;
    auto [memset_block, memset_node] = stdlib::add_memset_block(
        this->builder_, this->root(), "A", symbolic::zero(), symbolic::integer(1024), this->float_ptr_
    );
    memset_node.set_implementation_type(Case::with_transfers());

    EXPECT_TRUE(this->run_scheduling());

    EXPECT_EQ(memset_node.implementation_type().value(), Case::without_transfers().value());
    EXPECT_EQ(memset_block.get_parent(), &this->root());
    EXPECT_EQ(this->root().size(), 3);
    EXPECT_TRUE(this->offloaded_root_maps().empty());
}
