#include <gtest/gtest.h>

#include "sdfg/analysis/analysis.h"
#include "sdfg/analysis/loop_analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_nodes/math/tensor/reduce_ops/softmax_node.h"
#include "sdfg/data_flow/library_nodes/stdlib/memset.h"
#include "sdfg/passes/scheduler/cuda_offload_scheduler.h"
#include "sdfg/passes/scheduler/cuda_scheduler.h"
#include "sdfg/passes/scheduler/loop_scheduling_pass.h"
#include "sdfg/passes/scheduler/rocm_offload_scheduler.h"
#include "sdfg/passes/scheduler/rocm_scheduler.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/targets/cuda/cuda.h"
#include "sdfg/targets/rocm/rocm.h"

using namespace sdfg;

namespace {

struct CUDATraits {
    static const data_flow::ImplementationType& with_transfers() {
        return cuda::ImplementationType_CUDAWithTransfers;
    }
    static const data_flow::ImplementationType& without_transfers() {
        return cuda::ImplementationType_CUDAWithoutTransfers;
    }
    static auto gpu_schedule() {
        return cuda::ScheduleType_CUDA::create();
    }
};

struct ROCMTraits {
    static const data_flow::ImplementationType& with_transfers() {
        return rocm::ImplementationType_ROCMWithTransfers;
    }
    static const data_flow::ImplementationType& without_transfers() {
        return rocm::ImplementationType_ROCMWithoutTransfers;
    }
    static auto gpu_schedule() {
        return rocm::ScheduleType_ROCM::create();
    }
};

template<typename S, typename T>
struct SchedulerCase {
    using Scheduler = S;
    using Traits = T;
};

using SchedulerCases = ::testing::Types<
    SchedulerCase<passes::scheduler::CUDAScheduler, CUDATraits>,
    SchedulerCase<passes::scheduler::CUDAOffloadScheduler, CUDATraits>,
    SchedulerCase<passes::scheduler::ROCMScheduler, ROCMTraits>,
    SchedulerCase<passes::scheduler::ROCMOffloadScheduler, ROCMTraits>>;

template<typename Case>
class GPUPostScheduleTest : public ::testing::Test {
protected:
    using Scheduler = typename Case::Scheduler;
    using Traits = typename Case::Traits;

    builder::StructuredSDFGBuilder builder_{"post_schedule_test", FunctionType_CPU};
    Scheduler scheduler_;

    types::Scalar float_ = types::Scalar(types::PrimitiveType::Float);
    types::Pointer float_ptr_ = types::Pointer(float_);

    stdlib::MemsetNode& add_memset_with_transfers() {
        builder_.add_container("buf", float_ptr_, true);
        auto [block, memset_node] = stdlib::add_memset_block(
            builder_, builder_.subject().root(), "buf", symbolic::zero(), symbolic::integer(1024), float_ptr_
        );
        memset_node.set_implementation_type(Traits::with_transfers());
        return memset_node;
    }

    structured_control_flow::Map&
    add_copy_map(const std::string& dst, const structured_control_flow::ScheduleType& schedule) {
        auto& sdfg = builder_.subject();
        if (!sdfg.exists("A")) {
            builder_.add_container("A", float_ptr_, true);
        }
        builder_.add_container("i", types::Scalar(types::PrimitiveType::Int64));
        auto indvar = symbolic::symbol("i");
        auto& map = builder_.add_map(
            sdfg.root(),
            indvar,
            symbolic::Lt(indvar, symbolic::integer(128)),
            symbolic::zero(),
            symbolic::add(indvar, symbolic::one()),
            schedule
        );
        auto& block = builder_.add_block(map.root());
        auto& a_in = builder_.add_access(block, "A");
        auto& out = builder_.add_access(block, dst);
        auto& tasklet = builder_.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
        builder_.add_computational_memlet(block, a_in, tasklet, "_in", {indvar}, float_ptr_);
        if (dst == "A") {
            builder_.add_computational_memlet(block, tasklet, "_out", out, {indvar}, float_ptr_);
        } else {
            builder_.add_computational_memlet(block, tasklet, "_out", out, {}, float_);
        }
        return map;
    }

    bool run_scheduling() {
        analysis::AnalysisManager analysis_manager(builder_.subject());
        passes::scheduler::LoopSchedulingPass pass({&scheduler_}, nullptr);
        return pass.run(builder_, analysis_manager);
    }

    void expect_no_gpu_scheduled_loops() {
        analysis::AnalysisManager analysis_manager(builder_.subject());
        auto& loop_analysis = analysis_manager.get<analysis::LoopAnalysis>();
        for (auto* loop : loop_analysis.loops()) {
            if (auto* sloop = dynamic_cast<structured_control_flow::StructuredLoop*>(loop)) {
                EXPECT_EQ(sloop->schedule_type().category(), structured_control_flow::ScheduleTypeCategory::None);
            }
        }
    }
};

} // namespace

TYPED_TEST_SUITE(GPUPostScheduleTest, SchedulerCases);

TYPED_TEST(GPUPostScheduleTest, ExtractsLibraryNodeTransfersWithoutLoops) {
    using Traits = typename TestFixture::Traits;
    auto& memset_node = this->add_memset_with_transfers();
    ASSERT_EQ(this->builder_.subject().root().size(), 1);

    EXPECT_TRUE(this->run_scheduling());

    EXPECT_EQ(memset_node.implementation_type().value(), Traits::without_transfers().value());
    EXPECT_EQ(this->builder_.subject().root().size(), 3);
}

TYPED_TEST(GPUPostScheduleTest, NoChangesWithoutLoopsOrLibraryNodes) {
    auto& sdfg = this->builder_.subject();
    this->builder_.add_container("x", this->float_);
    this->builder_.add_container("y", this->float_);
    auto& block = this->builder_.add_block(sdfg.root());
    auto& x = this->builder_.add_access(block, "x");
    auto& y = this->builder_.add_access(block, "y");
    auto& tasklet = this->builder_.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    this->builder_.add_computational_memlet(block, x, tasklet, "_in", {}, this->float_);
    this->builder_.add_computational_memlet(block, tasklet, "_out", y, {}, this->float_);

    EXPECT_FALSE(this->run_scheduling());
    EXPECT_EQ(sdfg.root().size(), 1);
}

TYPED_TEST(GPUPostScheduleTest, ExtractsLibraryNodeTransfersWithOnlySkippedLoops) {
    using Traits = typename TestFixture::Traits;
    auto& sdfg = this->builder_.subject();
    this->builder_.add_container("A", this->float_ptr_, true);
    this->builder_.add_container("i", types::Scalar(types::PrimitiveType::Int64));
    auto indvar = symbolic::symbol("i");
    auto& loop = this->builder_.add_for(
        sdfg.root(),
        indvar,
        symbolic::Lt(indvar, symbolic::integer(128)),
        symbolic::one(),
        symbolic::add(indvar, symbolic::one())
    );
    auto& block = this->builder_.add_block(loop.root());
    auto& a_in = this->builder_.add_access(block, "A");
    auto& a_out = this->builder_.add_access(block, "A");
    auto& tasklet = this->builder_.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    this->builder_
        .add_computational_memlet(block, a_in, tasklet, "_in", {symbolic::sub(indvar, symbolic::one())}, this->float_ptr_);
    this->builder_.add_computational_memlet(block, tasklet, "_out", a_out, {indvar}, this->float_ptr_);

    auto& memset_node = this->add_memset_with_transfers();
    size_t root_size = sdfg.root().size();

    EXPECT_TRUE(this->run_scheduling());

    EXPECT_EQ(memset_node.implementation_type().value(), Traits::without_transfers().value());
    EXPECT_EQ(sdfg.root().size(), root_size + 2);
    this->expect_no_gpu_scheduled_loops();
}

TYPED_TEST(GPUPostScheduleTest, ExtractsLibraryNodeTransfersWithOnlyUnschedulableLoops) {
    using Traits = typename TestFixture::Traits;
    auto& sdfg = this->builder_.subject();
    this->builder_.add_container("s", this->float_, true);
    auto& map = this->add_copy_map("s", structured_control_flow::ScheduleType_Sequential::create());
    {
        analysis::AnalysisManager analysis_manager(sdfg);
        ASSERT_FALSE(this->scheduler_.can_apply_schedule(this->builder_, analysis_manager, map));
    }

    auto& memset_node = this->add_memset_with_transfers();

    EXPECT_TRUE(this->run_scheduling());

    EXPECT_EQ(memset_node.implementation_type().value(), Traits::without_transfers().value());
    this->expect_no_gpu_scheduled_loops();
}

TYPED_TEST(GPUPostScheduleTest, ExtractsLibraryNodeTransfersNextToAlreadyScheduledMap) {
    using Traits = typename TestFixture::Traits;
    auto& map = this->add_copy_map("A", Traits::gpu_schedule());
    auto& memset_node = this->add_memset_with_transfers();
    auto schedule_before = map.schedule_type().value();

    EXPECT_TRUE(this->run_scheduling());

    EXPECT_EQ(memset_node.implementation_type().value(), Traits::without_transfers().value());
    EXPECT_EQ(map.schedule_type().value(), schedule_before);
}

TEST(CUDAOffloadPostScheduleTest, ExtractsSoftmaxTransfersWithoutLoops) {
    builder::StructuredSDFGBuilder builder("softmax_post_schedule_test", FunctionType_CPU);
    auto& sdfg = builder.subject();

    types::Scalar desc(types::PrimitiveType::Float);
    types::Pointer ptr_type(desc);
    builder.add_container("X", ptr_type, true);
    builder.add_container("Y", ptr_type, true);

    auto& block = builder.add_block(sdfg.root());
    auto& x_node = builder.add_access(block, "X");
    auto& y_node = builder.add_access(block, "Y");

    std::vector<symbolic::Expression> shape = {symbolic::integer(64), symbolic::integer(128)};
    auto& softmax_node = static_cast<math::tensor::SoftmaxNode&>(
        builder.add_library_node<math::tensor::SoftmaxNode>(block, DebugInfo(), shape, std::vector<int64_t>{-1}, false)
    );
    softmax_node.set_implementation_type(cuda::ImplementationType_CUDAWithTransfers);

    types::Tensor tensor_type(desc, shape);
    builder.add_computational_memlet(block, y_node, softmax_node, "Y", {}, tensor_type);
    builder.add_computational_memlet(block, x_node, softmax_node, "X", {}, tensor_type);

    analysis::AnalysisManager analysis_manager(sdfg);
    passes::scheduler::CUDAOffloadScheduler scheduler;
    passes::scheduler::LoopSchedulingPass pass({&scheduler}, nullptr);

    EXPECT_TRUE(pass.run(builder, analysis_manager));
    EXPECT_EQ(softmax_node.implementation_type().value(), cuda::ImplementationType_CUDAWithoutTransfers.value());
}
