// Nest-level schedule annotations (no-fusion, zero-init) must stay on the outermost loop of their nest
// across transformations that rebuild or re-schedule loops, and zero-init must be dropped by fusion.

#include <gtest/gtest.h>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/structured_control_flow/for.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/targets/gpu/gpu_schedule_type.h"
#include "sdfg/targets/rocm/rocm.h"
#include "sdfg/transformations/loop_interchange.h"
#include "sdfg/transformations/loop_tiling.h"
#include "sdfg/transformations/map_fusion.h"
#include "sdfg/transformations/offloading/rocm_offload_transform.h"
#include "sdfg/transformations/omp_transform.h"

using namespace sdfg;
using structured_control_flow::ScheduleType_NoFusion;
using structured_control_flow::ScheduleType_ZeroInit;

namespace {

structured_control_flow::ScheduleType annotated_sequential() {
    auto schedule = structured_control_flow::ScheduleType_Sequential::create();
    ScheduleType_ZeroInit::set(schedule);
    ScheduleType_NoFusion::set(schedule);
    return schedule;
}

bool has_nest_annotations(const structured_control_flow::StructuredLoop& loop) {
    return ScheduleType_ZeroInit::is_set(loop.schedule_type()) && ScheduleType_NoFusion::is_set(loop.schedule_type());
}

bool has_any_nest_annotation(const structured_control_flow::StructuredLoop& loop) {
    return loop.schedule_type().properties().contains("init") || loop.schedule_type().properties().contains("fusion");
}

/// Zero-init nest `A[i*M + j] = 0` with an annotated outer map over i and an inner map (or for) over j.
std::pair<structured_control_flow::StructuredLoop*, structured_control_flow::StructuredLoop*>
add_zero_init_nest(builder::StructuredSDFGBuilder& builder, bool inner_as_for = false) {
    auto& root = builder.subject().root();
    types::Scalar f32(types::PrimitiveType::Float);
    types::Pointer f32ptr(f32);
    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("A", f32ptr, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);
    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");
    const auto N = symbolic::integer(64);
    const auto M = symbolic::integer(32);

    auto& outer = builder.add_map(
        root, i, symbolic::Lt(i, N), symbolic::zero(), symbolic::add(i, symbolic::one()), annotated_sequential()
    );
    structured_control_flow::StructuredLoop* inner = nullptr;
    if (inner_as_for) {
        inner =
            &builder.add_for(outer.root(), j, symbolic::Lt(j, M), symbolic::zero(), symbolic::add(j, symbolic::one()));
    } else {
        inner = &builder.add_map(
            outer.root(),
            j,
            symbolic::Lt(j, M),
            symbolic::zero(),
            symbolic::add(j, symbolic::one()),
            structured_control_flow::ScheduleType_Sequential::create()
        );
    }
    auto& block = builder.add_block(inner->root());
    auto& zero = builder.add_constant(block, "0.0", f32);
    auto& out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, zero, tasklet, "_in", {}, f32);
    builder.add_computational_memlet(block, tasklet, "_out", out, {symbolic::add(symbolic::mul(i, M), j)}, f32ptr);
    return {&outer, inner};
}

} // namespace

TEST(NestAnnotationsTest, UpdateScheduleTypeKeepsNestAnnotations) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);
    auto [outer, inner] = add_zero_init_nest(builder);

    builder.update_schedule_type(
        *outer, structured_control_flow::ScheduleType("OTHER", structured_control_flow::ScheduleTypeCategory::None)
    );
    EXPECT_EQ(outer->schedule_type().value(), "OTHER");
    EXPECT_TRUE(has_nest_annotations(*outer));

    // Explicitly set annotations win over the inherited ones.
    auto explicit_schedule = structured_control_flow::ScheduleType_Sequential::create();
    ScheduleType_NoFusion::set(explicit_schedule, false);
    builder.update_schedule_type(*outer, explicit_schedule);
    EXPECT_TRUE(ScheduleType_ZeroInit::is_set(outer->schedule_type()));
    EXPECT_FALSE(ScheduleType_NoFusion::is_set(outer->schedule_type()));

    builder.update_schedule_type(*outer, structured_control_flow::ScheduleType_Sequential::create(), false);
    EXPECT_FALSE(has_any_nest_annotation(*outer));
}

TEST(NestAnnotationsTest, LoopTilingMapKeepsAnnotationOnOuterTileLoop) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);
    auto [outer, inner] = add_zero_init_nest(builder);

    analysis::AnalysisManager analysis_manager(builder.subject());
    transformations::LoopTiling tiling(*outer, 8);
    ASSERT_TRUE(tiling.can_be_applied(builder, analysis_manager));
    tiling.apply(builder, analysis_manager);

    auto& tile_loop = *tiling.outer_loop();
    EXPECT_EQ(&builder.subject().root().at(0), &tile_loop);
    EXPECT_TRUE(has_nest_annotations(tile_loop));
    EXPECT_FALSE(has_any_nest_annotation(*tiling.inner_loop()));
    EXPECT_FALSE(has_any_nest_annotation(*inner));
}

TEST(NestAnnotationsTest, LoopTilingForKeepsAnnotationOnOuterTileLoop) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);
    auto& root = builder.subject().root();
    types::Scalar f32(types::PrimitiveType::Float);
    types::Pointer f32ptr(f32);
    builder.add_container("A", f32ptr, true);
    builder.add_container("i", types::Scalar(types::PrimitiveType::UInt64));
    auto i = symbolic::symbol("i");
    auto& loop =
        builder
            .add_for(root, i, symbolic::Lt(i, symbolic::integer(64)), symbolic::zero(), symbolic::add(i, symbolic::one()));
    builder.update_schedule_type(loop, annotated_sequential());
    auto& block = builder.add_block(loop.root());
    auto& zero = builder.add_constant(block, "0.0", f32);
    auto& out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, zero, tasklet, "_in", {}, f32);
    builder.add_computational_memlet(block, tasklet, "_out", out, {i}, f32ptr);

    analysis::AnalysisManager analysis_manager(builder.subject());
    transformations::LoopTiling tiling(loop, 8);
    ASSERT_TRUE(tiling.can_be_applied(builder, analysis_manager));
    tiling.apply(builder, analysis_manager);

    EXPECT_TRUE(has_nest_annotations(*tiling.outer_loop()));
    EXPECT_FALSE(has_any_nest_annotation(*tiling.inner_loop()));
}

TEST(NestAnnotationsTest, LoopInterchangeKeepsAnnotationOnOutermostLoop) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);
    auto [outer, inner] = add_zero_init_nest(builder);

    analysis::AnalysisManager analysis_manager(builder.subject());
    transformations::LoopInterchange interchange(*outer, *inner);
    ASSERT_TRUE(interchange.can_be_applied(builder, analysis_manager));
    interchange.apply(builder, analysis_manager);

    auto* new_outer = dyn_cast<structured_control_flow::StructuredLoop*>(&builder.subject().root().at(0));
    ASSERT_NE(new_outer, nullptr);
    auto* new_inner = dyn_cast<structured_control_flow::StructuredLoop*>(&new_outer->root().at(0));
    ASSERT_NE(new_inner, nullptr);
    EXPECT_EQ(new_outer->indvar()->get_name(), "j");
    EXPECT_EQ(new_inner->indvar()->get_name(), "i");
    EXPECT_TRUE(has_nest_annotations(*new_outer));
    EXPECT_FALSE(has_any_nest_annotation(*new_inner));
}

TEST(NestAnnotationsTest, LoopInterchangeWithInnerForKeepsAnnotationOnOutermostLoop) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);
    auto [outer, inner] = add_zero_init_nest(builder, /*inner_as_for=*/true);

    analysis::AnalysisManager analysis_manager(builder.subject());
    transformations::LoopInterchange interchange(*outer, *inner);
    ASSERT_TRUE(interchange.can_be_applied(builder, analysis_manager));
    interchange.apply(builder, analysis_manager);

    auto* new_outer = dyn_cast<structured_control_flow::StructuredLoop*>(&builder.subject().root().at(0));
    ASSERT_NE(new_outer, nullptr);
    auto* new_inner = dyn_cast<structured_control_flow::StructuredLoop*>(&new_outer->root().at(0));
    ASSERT_NE(new_inner, nullptr);
    EXPECT_TRUE(has_nest_annotations(*new_outer));
    EXPECT_FALSE(has_any_nest_annotation(*new_inner));
}

TEST(NestAnnotationsTest, RocmOffloadKeepsNestAnnotations) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);
    auto [outer, inner] = add_zero_init_nest(builder);
    auto& outer_map = static_cast<structured_control_flow::Map&>(*outer);

    analysis::AnalysisManager analysis_manager(builder.subject());
    rocm::ROCMOffloadTransform offload(outer_map, symbolic::integer(64), gpu::TargetLevel::X_GRID);
    ASSERT_TRUE(offload.can_be_applied(builder, analysis_manager));
    offload.apply(builder, analysis_manager);

    EXPECT_EQ(outer_map.schedule_type().value(), rocm::ScheduleType_ROCM_Offload::value());
    EXPECT_TRUE(has_nest_annotations(outer_map));
}

TEST(NestAnnotationsTest, OmpTransformKeepsNestAnnotations) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);
    auto [outer, inner] = add_zero_init_nest(builder);
    auto& outer_map = static_cast<structured_control_flow::Map&>(*outer);

    analysis::AnalysisManager analysis_manager(builder.subject());
    transformations::OMPTransform omp(outer_map);
    ASSERT_TRUE(omp.can_be_applied(builder, analysis_manager));
    omp.apply(builder, analysis_manager);

    EXPECT_NE(outer_map.schedule_type().value(), structured_control_flow::ScheduleType_Sequential::value());
    EXPECT_TRUE(has_nest_annotations(outer_map));
}

TEST(NestAnnotationsTest, MapFusionDropsZeroInit) {
    // T[i] = 0; B[j] = T[j] * 2.0 -- the zero init is fusable (no no-fusion mark) and must lose its annotation.
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);
    auto& root = builder.subject().root();
    types::Scalar f32(types::PrimitiveType::Float);
    types::Array array_desc(f32, {symbolic::symbol("N")});
    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("T", array_desc);
    builder.add_container("B", array_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);
    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");

    auto init_schedule = structured_control_flow::ScheduleType_Sequential::create();
    ScheduleType_ZeroInit::set(init_schedule);
    auto& init_map = builder.add_map(
        root,
        i,
        symbolic::Lt(i, symbolic::symbol("N")),
        symbolic::zero(),
        symbolic::add(i, symbolic::one()),
        init_schedule
    );
    {
        auto& block = builder.add_block(init_map.root());
        auto& zero = builder.add_constant(block, "0.0", f32);
        auto& out = builder.add_access(block, "T");
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
        builder.add_computational_memlet(block, zero, tasklet, "_in", {}, f32);
        builder.add_computational_memlet(block, tasklet, "_out", out, {i}, array_desc);
    }
    auto& consumer = builder.add_map(
        root,
        j,
        symbolic::Lt(j, symbolic::symbol("N")),
        symbolic::zero(),
        symbolic::add(j, symbolic::one()),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    {
        auto& block = builder.add_block(consumer.root());
        auto& in = builder.add_access(block, "T");
        auto& two = builder.add_constant(block, "2.0", f32);
        auto& out = builder.add_access(block, "B");
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_mul, "_out", {"_in1", "_in2"});
        builder.add_computational_memlet(block, in, tasklet, "_in1", {j}, array_desc);
        builder.add_computational_memlet(block, two, tasklet, "_in2", {});
        builder.add_computational_memlet(block, tasklet, "_out", out, {j}, array_desc);
    }

    analysis::AnalysisManager analysis_manager(builder.subject());
    transformations::MapFusion fusion(init_map, consumer);
    ASSERT_TRUE(fusion.can_be_applied(builder, analysis_manager));
    fusion.apply(builder, analysis_manager);

    for (size_t idx = 0; idx < root.size(); ++idx) {
        if (auto* loop = dyn_cast<structured_control_flow::StructuredLoop*>(&root.at(idx))) {
            EXPECT_FALSE(ScheduleType_ZeroInit::is_set(loop->schedule_type()));
        }
    }
}
