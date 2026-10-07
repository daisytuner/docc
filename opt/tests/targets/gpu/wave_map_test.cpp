#include <gtest/gtest.h>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/codegen/language_extensions/c_language_extension.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/serializer/json_serializer.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/targets/gpu/gpu_map_utils.h"
#include "sdfg/targets/gpu/gpu_offload_map_dispatcher.h"
#include "sdfg/targets/gpu/gpu_offload_schedule_type.h"
#include "sdfg/targets/rocm/rocm.h"
#include "sdfg/targets/rocm/rocm_offload_dispatcher_strategy.h"
#include "sdfg/tiles/tile.h"

namespace sdfg::gpu {

namespace {

namespace hip = ::sdfg::rocm;

structured_control_flow::ScheduleType rocm_sched(TargetLevel level, int64_t ps, int64_t lanes = 1) {
    auto s = hip::ScheduleType_ROCM_Offload::create<hip::ScheduleType_ROCM_Offload>(level, symbolic::integer(ps));
    if (lanes != 1) {
        ScheduleType_GPU_Offload::lanes(s, symbolic::integer(lanes));
    }
    return s;
}

/// g X_GRID(4) > w X_BLOCK(2, lanes) > c Y_BLOCK(2): out[4g + 2w + c] = 1
struct WaveKernel {
    builder::StructuredSDFGBuilder builder{"wave_sdfg", FunctionType_CPU};
    structured_control_flow::Map* grid = nullptr;
    structured_control_flow::Map* wave = nullptr;

    explicit WaveKernel(int64_t lanes, TargetLevel wave_level = TargetLevel::X_BLOCK) {
        auto& root = builder.subject().root();
        types::Scalar i32(types::PrimitiveType::Int32);
        types::Scalar f32(types::PrimitiveType::Float);
        types::Pointer dev_ptr(types::StorageType("AMD_Generic"), 0, "", f32);
        builder.add_container("g", i32);
        builder.add_container("w", i32);
        builder.add_container("c", i32);
        builder.add_container("__daisy_hip_out", dev_ptr);

        auto g = symbolic::symbol("g");
        auto w = symbolic::symbol("w");
        auto c = symbolic::symbol("c");
        auto one = symbolic::integer(1);

        grid = &builder.add_map(
            root,
            g,
            symbolic::Lt(g, symbolic::integer(4)),
            symbolic::zero(),
            symbolic::add(g, one),
            rocm_sched(TargetLevel::X_GRID, 4)
        );
        auto wave_sched =
            hip::ScheduleType_ROCM_Offload::create<hip::ScheduleType_ROCM_Offload>(wave_level, symbolic::integer(2));
        if (lanes != 1) {
            // Bypass the setter's level check so validation can be exercised on deserialized-like input.
            wave_sched.set_property("lanes", std::to_string(lanes));
        }
        wave = &builder.add_map(
            grid->root(), w, symbolic::Lt(w, symbolic::integer(2)), symbolic::zero(), symbolic::add(w, one), wave_sched
        );
        auto& col = builder.add_map(
            wave->root(),
            c,
            symbolic::Lt(c, symbolic::integer(2)),
            symbolic::zero(),
            symbolic::add(c, one),
            rocm_sched(TargetLevel::Y_BLOCK, 2)
        );

        auto& block = builder.add_block(col.root());
        auto& out = builder.add_access(block, "__daisy_hip_out");
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
        auto& constant = builder.add_constant(block, "1.0f", f32);
        builder.add_computational_memlet(block, constant, tasklet, "_in", {}, f32);
        auto idx = symbolic::
            add(symbolic::mul(symbolic::integer(4), g), symbolic::add(symbolic::mul(symbolic::integer(2), w), c));
        builder.add_computational_memlet(block, tasklet, "_out", out, {idx}, dev_ptr);
    }

    std::pair<std::string, std::string> generate() {
        auto& sdfg = builder.subject();
        codegen::CLanguageExtension language_extension(sdfg);
        auto instrumentation = codegen::InstrumentationPlan::none(sdfg);
        auto arg_capture = codegen::ArgCapturePlan::none(sdfg);
        analysis::AnalysisManager analysis_manager(sdfg);
        GPUOffloadMapDispatcher dispatcher(
            language_extension,
            sdfg,
            analysis_manager,
            *grid,
            *instrumentation,
            *arg_capture,
            std::make_unique<hip::ROCMOffloadDispatcherStrategy>(sdfg, *grid)
        );
        codegen::PrettyPrinter main_stream;
        codegen::PrettyPrinter globals_stream;
        codegen::CodeSnippetFactory snippets;
        dispatcher.dispatch_node(main_stream, globals_stream, snippets);
        std::string kernel_name = "kernel_wave_sdfg_" + std::to_string(grid->element_id());
        return {main_stream.str(), snippets.snippets().at(kernel_name).stream().str()};
    }
};

} // namespace

TEST(WaveMapTest, LanesDefaultsToOne) {
    auto s = rocm_sched(TargetLevel::X_BLOCK, 128);
    EXPECT_EQ(ScheduleType_GPU_Offload::lanes(s)->as_int(), 1);
    EXPECT_EQ(ScheduleType_GPU_Offload::threads(s)->as_int(), 128);
    EXPECT_EQ(s.properties().count("lanes"), 0u);
}

TEST(WaveMapTest, LanesMultiplyThreads) {
    auto s = rocm_sched(TargetLevel::X_BLOCK, 2, 64);
    EXPECT_EQ(ScheduleType_GPU_Offload::lanes(s)->as_int(), 64);
    EXPECT_EQ(ScheduleType_GPU_Offload::parallel_size(s)->as_int(), 2);
    EXPECT_EQ(ScheduleType_GPU_Offload::threads(s)->as_int(), 128);
}

TEST(WaveMapTest, LanesRejectedOffXBlock) {
    for (auto level : {TargetLevel::Y_BLOCK, TargetLevel::Z_BLOCK, TargetLevel::X_GRID, TargetLevel::WARP}) {
        auto s = rocm_sched(level, 2);
        EXPECT_THROW(ScheduleType_GPU_Offload::lanes(s, symbolic::integer(64)), InvalidSDFGException);
    }
    auto s = rocm_sched(TargetLevel::X_BLOCK, 2);
    EXPECT_THROW(ScheduleType_GPU_Offload::lanes(s, symbolic::integer(0)), InvalidSDFGException);
}

TEST(WaveMapTest, LanesJsonRoundTrip) {
    auto s = rocm_sched(TargetLevel::X_BLOCK, 2, 64);
    serializer::JSONSerializer serializer;
    nlohmann::json j;
    serializer.schedule_type_to_json(j, s);
    auto back = serializer.json_to_schedule_type(j);
    EXPECT_EQ(back.value(), s.value());
    EXPECT_EQ(ScheduleType_GPU_Offload::lanes(back)->as_int(), 64);
    EXPECT_EQ(ScheduleType_GPU_Offload::threads(back)->as_int(), 128);
}

TEST(WaveMapTest, ValidateAcceptsWarpSizedLanes) {
    WaveKernel k(64);
    analysis::AnalysisManager am(k.builder.subject());
    EXPECT_NO_THROW(validate_wave_maps(*k.grid, am, 64));
}

TEST(WaveMapTest, ValidateRejectsNonWarpLanes) {
    WaveKernel k(32);
    analysis::AnalysisManager am(k.builder.subject());
    EXPECT_THROW(validate_wave_maps(*k.grid, am, 64), InvalidSDFGException);
}

TEST(WaveMapTest, ValidateRejectsLanesOffXBlock) {
    WaveKernel k(64, TargetLevel::Z_BLOCK);
    analysis::AnalysisManager am(k.builder.subject());
    EXPECT_THROW(validate_wave_maps(*k.grid, am, 64), InvalidSDFGException);
}

TEST(WaveMapTest, ValidateRejectsNestedWarp) {
    WaveKernel k(64);
    auto& sdfg_builder = k.builder;
    sdfg_builder.add_container("l", types::Scalar(types::PrimitiveType::Int32));
    auto l = symbolic::symbol("l");
    sdfg_builder.add_map(
        k.wave->root(),
        l,
        symbolic::Lt(l, symbolic::integer(4)),
        symbolic::zero(),
        symbolic::add(l, symbolic::integer(1)),
        rocm_sched(TargetLevel::WARP, 64)
    );
    analysis::AnalysisManager am(sdfg_builder.subject());
    EXPECT_THROW(validate_wave_maps(*k.grid, am, 64), InvalidSDFGException);
}

TEST(WaveMapTest, CodegenWaveIndexAndLaunch) {
    const int64_t L = hip::rocm_wavefront_size();
    WaveKernel k(L);
    auto [host, kernel] = k.generate();
    const std::string threads = std::to_string(2 * L);
    const std::string per = " / " + std::to_string(L) + ")";

    EXPECT_NE(
        host.find("dim3((int)(4), (int)(1), (int)(1)), dim3((int)(" + threads + "), (int)(2), (int)(1))"),
        std::string::npos
    ) << host;
    EXPECT_NE(kernel.find("(threadIdx.x" + per), std::string::npos) << kernel;
    EXPECT_NE(kernel.find("(blockDim.x" + per), std::string::npos) << kernel;
}

TEST(WaveMapTest, CodegenLanesOneUnchanged) {
    WaveKernel k(1);
    auto [host, kernel] = k.generate();
    EXPECT_NE(host.find("dim3((int)(4), (int)(1), (int)(1)), dim3((int)(2), (int)(2), (int)(1))"), std::string::npos)
        << host;
    EXPECT_EQ(kernel.find("threadIdx.x /"), std::string::npos) << kernel;
    EXPECT_EQ(kernel.find("blockDim.x /"), std::string::npos) << kernel;
}

TEST(WaveMapTest, TileClassifyCarriesLanes) {
    auto wave = tiles::AxisSchedule::classify(rocm_sched(TargetLevel::X_BLOCK, 2, 64));
    ASSERT_TRUE(wave.has_value());
    EXPECT_EQ(wave->level(), tiles::Level::Group);
    EXPECT_EQ(wave->space(), tiles::Space::Shared);
    EXPECT_EQ(wave->parallel_size()->as_int(), 2);
    EXPECT_EQ(wave->lanes()->as_int(), 64);

    auto thread = tiles::AxisSchedule::classify(rocm_sched(TargetLevel::X_BLOCK, 128));
    ASSERT_TRUE(thread.has_value());
    EXPECT_EQ(thread->lanes()->as_int(), 1);
}

} // namespace sdfg::gpu
