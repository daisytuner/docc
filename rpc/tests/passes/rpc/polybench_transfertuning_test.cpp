#include <gtest/gtest.h>
#include <memory>

#include "sdfg/analysis/loop_analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/passes/normalization/normalization.h"
#include "sdfg/passes/pipeline.h"
#include "sdfg/passes/rpc/rpc_scheduler.h"
#include "sdfg/passes/scheduler/loop_scheduling_pass.h"
#include "sdfg/structured_sdfg.h"

#include "fixtures/polybench.h"
#include "sdfg/passes/scheduler/scheduler_registry.h"

using namespace sdfg;

// Runs the RPC scheduling pass on a polybench SDFG after normalizing loops
// (data parallelism + loop normalization), matching the real compiler pipeline.
// This exercises the full LoopSchedulingPass traversal including the CHILDREN
// descent path, catching regressions like use-after-free when find()
// invalidates the analysis cache via cutout().
static bool run_rpc_scheduling(std::unique_ptr<StructuredSDFG> init_sdfg) {
    builder::StructuredSDFGBuilder builder(init_sdfg);
    analysis::AnalysisManager analysis_manager(builder.subject());

    passes::Pipeline data_parallelism = passes::Pipeline::data_parallelism();
    data_parallelism.run(builder, analysis_manager);

    passes::Pipeline lp_pipeline = passes::normalization::loop_normalization();
    lp_pipeline.run(builder, analysis_manager);

    analysis_manager.invalidate_all();

    passes::scheduler::LoopSchedulingPass
        loop_scheduling_pass({passes::scheduler::SchedulerRegistry::instance().get_loop_scheduler("rpc")}, nullptr);
    return loop_scheduling_pass.run(builder, analysis_manager);
}

// Tests with multiple loop nests where the scheduler must descend (CHILDREN)
// into children because the outermost loop can't be directly transformed.

TEST(PolybenchRPCSchedulerTest, Correlation) { EXPECT_TRUE(run_rpc_scheduling(correlation())); }

TEST(PolybenchRPCSchedulerTest, Covariance) { EXPECT_TRUE(run_rpc_scheduling(covariance())); }

TEST(PolybenchRPCSchedulerTest, Gemm) { EXPECT_TRUE(run_rpc_scheduling(gemm())); }

TEST(PolybenchRPCSchedulerTest, Gemver) { EXPECT_TRUE(run_rpc_scheduling(gemver())); }

TEST(PolybenchRPCSchedulerTest, Gesummv) { EXPECT_TRUE(run_rpc_scheduling(gesummv())); }


TEST(PolybenchRPCSchedulerTest, Syr2k) { EXPECT_TRUE(run_rpc_scheduling(syr2k())); }

TEST(PolybenchRPCSchedulerTest, Syrk) { EXPECT_TRUE(run_rpc_scheduling(syrk())); }

TEST(PolybenchRPCSchedulerTest, Atax) { EXPECT_TRUE(run_rpc_scheduling(atax())); }

TEST(PolybenchRPCSchedulerTest, Bicg) { EXPECT_TRUE(run_rpc_scheduling(bicg())); }

TEST(PolybenchRPCSchedulerTest, Mvt) { EXPECT_TRUE(run_rpc_scheduling(mvt())); }

TEST(PolybenchRPCSchedulerTest, Cholesky) { EXPECT_TRUE(run_rpc_scheduling(cholesky())); }

TEST(PolybenchRPCSchedulerTest, Trmm) { EXPECT_TRUE(run_rpc_scheduling(trmm())); }

TEST(PolybenchRPCSchedulerTest, Doitgen) { EXPECT_NO_THROW(run_rpc_scheduling(doitgen())); }

TEST(PolybenchRPCSchedulerTest, Jacobi2d) { EXPECT_NO_THROW(run_rpc_scheduling(jacobi_2d())); }

TEST(PolybenchRPCSchedulerTest, Fdtd2d) { EXPECT_TRUE(run_rpc_scheduling(fdtd_2d())); }
