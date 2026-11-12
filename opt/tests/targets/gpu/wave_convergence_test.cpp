#include <gtest/gtest.h>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/targets/gpu/gpu_map_utils.h"
#include "sdfg/targets/gpu/gpu_offload_schedule_type.h"
#include "sdfg/targets/rocm/rocm.h"
#include "sdfg/targets/rocm/rocm_offload_dispatcher_strategy.h"

namespace sdfg::gpu {

namespace {

namespace hip = ::sdfg::rocm;

constexpr int64_t kWave = 64;

/// Builds a kernel nest top-down and asks whether a point at the innermost scope is
/// wave-converged. N is a kernel argument (uniform, never written in the kernel).
struct Nest {
    builder::StructuredSDFGBuilder builder{"conv_sdfg", FunctionType_CPU};
    structured_control_flow::Sequence* scope = &builder.subject().root();

    Nest() {
        types::Scalar i32(types::PrimitiveType::Int32);
        for (auto name : {"g", "y", "x", "k", "N"}) {
            builder.add_container(name, i32, name == std::string("N"));
        }
    }

    Nest& map(const std::string& var, TargetLevel level, int64_t psize, int64_t trip, int64_t lanes = 1) {
        auto sched =
            hip::ScheduleType_ROCM_Offload::create<hip::ScheduleType_ROCM_Offload>(level, symbolic::integer(psize));
        if (lanes != 1) {
            ScheduleType_GPU_Offload::lanes(sched, symbolic::integer(lanes));
        }
        auto i = symbolic::symbol(var);
        auto& m = builder.add_map(
            *scope,
            i,
            symbolic::Lt(i, symbolic::integer(trip)),
            symbolic::zero(),
            symbolic::add(i, symbolic::one()),
            sched
        );
        scope = &m.root();
        return *this;
    }

    Nest& branch(const symbolic::Condition& cond) {
        auto& if_else = builder.add_if_else(*scope);
        scope = &builder.add_case(if_else, cond);
        return *this;
    }

    Nest& loop(const std::string& var, const symbolic::Expression& bound) {
        auto i = symbolic::symbol(var);
        auto& f =
            builder.add_for(*scope, i, symbolic::Lt(i, bound), symbolic::zero(), symbolic::add(i, symbolic::one()));
        scope = &f.root();
        return *this;
    }

    bool converged() {
        auto& point = builder.add_block(*scope);
        analysis::AnalysisManager analysis_manager(builder.subject());
        return is_wave_converged(point, analysis_manager, kWave);
    }
};

auto sym(const char* name) {
    return symbolic::symbol(name);
}

} // namespace

TEST(WaveConvergenceTest, WholeWaveBlockWithoutGuard) {
    EXPECT_TRUE(Nest().map("g", TargetLevel::X_GRID, 4, 4).map("x", TargetLevel::X_BLOCK, 64, 128).converged());
}

TEST(WaveConvergenceTest, RaggedBlockGuardSplitsWave) {
    EXPECT_FALSE(Nest().map("g", TargetLevel::X_GRID, 4, 4).map("x", TargetLevel::X_BLOCK, 64, 100).converged());
}

TEST(WaveConvergenceTest, PartialWaveBlock) {
    EXPECT_FALSE(Nest().map("g", TargetLevel::X_GRID, 4, 4).map("x", TargetLevel::X_BLOCK, 32, 32).converged());
}

TEST(WaveConvergenceTest, OuterBlockGuardUniformOnlyForWholeWaveRows) {
    // Y advances every bx threads: with bx = 64 a Y guard is per-wave, with bx = 32 it splits waves.
    EXPECT_TRUE(
        Nest()
            .map("g", TargetLevel::X_GRID, 4, 4)
            .map("y", TargetLevel::Y_BLOCK, 4, 6)
            .map("x", TargetLevel::X_BLOCK, 64, 128)
            .converged()
    );
    EXPECT_FALSE(
        Nest()
            .map("g", TargetLevel::X_GRID, 4, 4)
            .map("y", TargetLevel::Y_BLOCK, 4, 6)
            .map("x", TargetLevel::X_BLOCK, 32, 32)
            .converged()
    );
}

TEST(WaveConvergenceTest, WaveGranularMapGuardIsUniform) {
    // lanes == wave: each wave owns one index, so the ragged guard keeps waves whole.
    EXPECT_TRUE(Nest().map("g", TargetLevel::X_GRID, 4, 4).map("x", TargetLevel::X_BLOCK, 2, 3, kWave).converged());
}

TEST(WaveConvergenceTest, BranchesNeedUniformConditions) {
    EXPECT_FALSE(
        Nest()
            .map("g", TargetLevel::X_GRID, 4, 4)
            .map("x", TargetLevel::X_BLOCK, 64, 128)
            .branch(symbolic::Lt(sym("x"), sym("N")))
            .converged()
    );
    EXPECT_TRUE(
        Nest()
            .map("g", TargetLevel::X_GRID, 4, 4)
            .map("x", TargetLevel::X_BLOCK, 64, 128)
            .branch(symbolic::Lt(sym("g"), sym("N")))
            .converged()
    );
}

TEST(WaveConvergenceTest, SequentialLoopsNeedUniformBounds) {
    EXPECT_FALSE(
        Nest().map("g", TargetLevel::X_GRID, 4, 4).map("x", TargetLevel::X_BLOCK, 64, 128).loop("k", sym("x")).converged()
    );
    EXPECT_TRUE(
        Nest()
            .map("g", TargetLevel::X_GRID, 4, 4)
            .map("x", TargetLevel::X_BLOCK, 64, 128)
            .loop("k", sym("N"))
            .branch(symbolic::Lt(sym("k"), sym("g")))
            .converged()
    );
}

TEST(WaveConvergenceTest, RocmShuffleIsSynchronizedUnlessConverged) {
    hip::ROCMOffloadDispatcherStrategy strategy;
    EXPECT_EQ(strategy.warp_shuffle_xor("v", "m", true).rfind("__shfl_xor(v, m, ", 0), 0u);
    EXPECT_EQ(strategy.warp_shuffle_xor("v", "m", false).rfind("__shfl_xor_sync(", 0), 0u);
}

} // namespace sdfg::gpu
