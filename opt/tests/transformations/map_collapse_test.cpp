#include "sdfg/transformations/map_collapse.h"

#include <gtest/gtest.h>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/structured_control_flow/if_else.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg_debug_dump.h"

using namespace sdfg;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build a 2-level perfectly-nested map: outer i in [0, N), inner j in [0, M).
/// A single empty block sits in the innermost body.
/// Returns references to the outer map and the inner map.
static std::pair<structured_control_flow::Map*, structured_control_flow::Map*>
build_2d_map_nest(builder::StructuredSDFGBuilder& builder) {
    auto& root = builder.subject().root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("M", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);

    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");
    auto N = symbolic::symbol("N");
    auto M = symbolic::symbol("M");

    auto& outer = builder.add_map(
        root,
        i,
        symbolic::Lt(i, N),
        symbolic::integer(0),
        symbolic::add(i, symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );

    auto& inner = builder.add_map(
        outer.root(),
        j,
        symbolic::Lt(j, M),
        symbolic::integer(0),
        symbolic::add(j, symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );

    builder.add_block(inner.root());

    return {&outer, &inner};
}

// ---------------------------------------------------------------------------
// CanBeApplied — positive cases
// ---------------------------------------------------------------------------

TEST(MapCollapseTest, CanBeApplied_2D) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);
    auto [outer, inner] = build_2d_map_nest(builder);

    analysis::AnalysisManager am(builder.subject());
    transformations::MapCollapse t(*outer, 2);
    EXPECT_TRUE(t.can_be_applied(builder, am));
}

TEST(MapCollapseTest, CanBeApplied_3D) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);
    auto& root = builder.subject().root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("M", sym_desc, true);
    builder.add_container("P", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);
    builder.add_container("k", sym_desc);

    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");
    auto k = symbolic::symbol("k");
    auto N = symbolic::symbol("N");
    auto M = symbolic::symbol("M");
    auto P = symbolic::symbol("P");

    auto& outer = builder.add_map(
        root,
        i,
        symbolic::Lt(i, N),
        symbolic::integer(0),
        symbolic::add(i, symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    auto& middle = builder.add_map(
        outer.root(),
        j,
        symbolic::Lt(j, M),
        symbolic::integer(0),
        symbolic::add(j, symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    auto& inner = builder.add_map(
        middle.root(),
        k,
        symbolic::Lt(k, P),
        symbolic::integer(0),
        symbolic::add(k, symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    builder.add_block(inner.root());

    analysis::AnalysisManager am(builder.subject());
    transformations::MapCollapse t(outer, 3);
    EXPECT_TRUE(t.can_be_applied(builder, am));
}

// ---------------------------------------------------------------------------
// CanBeApplied — negative cases
// ---------------------------------------------------------------------------

TEST(MapCollapseTest, CannotApply_CountTooSmall) {
    // count < 2 must be rejected
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);
    auto [outer, inner] = build_2d_map_nest(builder);

    analysis::AnalysisManager am(builder.subject());
    transformations::MapCollapse t(*outer, 1);
    EXPECT_FALSE(t.can_be_applied(builder, am));
}

TEST(MapCollapseTest, CannotApply_NotEnoughNesting) {
    // count=3 but the nest is only 2 deep
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);
    auto [outer, inner] = build_2d_map_nest(builder);

    analysis::AnalysisManager am(builder.subject());
    transformations::MapCollapse t(*outer, 3);
    EXPECT_FALSE(t.can_be_applied(builder, am));
}

TEST(MapCollapseTest, CanBeApplied_Imperfect_BlockBeforeInnerMap) {
    // The outer map body contains two children: an extra block + the inner map.
    // This is no longer perfectly nested, but the imperfect (CUDA-style) collapse
    // can flatten it by guarding the block with `inner_idx == 0`.
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);
    auto& root = builder.subject().root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("M", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);

    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");

    auto& outer = builder.add_map(
        root,
        i,
        symbolic::Lt(i, symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(i, symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );

    // extra block before the inner map → not perfectly nested
    builder.add_block(outer.root());

    auto& inner = builder.add_map(
        outer.root(),
        j,
        symbolic::Lt(j, symbolic::symbol("M")),
        symbolic::integer(0),
        symbolic::add(j, symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    builder.add_block(inner.root());

    analysis::AnalysisManager am(builder.subject());
    transformations::MapCollapse t(outer, 2);
    EXPECT_TRUE(t.can_be_applied(builder, am));
}

TEST(MapCollapseTest, CannotApply_InnerBoundDependsOnOuterIndvar) {
    // inner map goes from 0 to i, so its bound references the outer indvar
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);
    auto& root = builder.subject().root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);

    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");

    auto& outer = builder.add_map(
        root,
        i,
        symbolic::Lt(i, symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(i, symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );

    // inner bound is `i` — depends on outer indvar
    auto& inner = builder.add_map(
        outer.root(),
        j,
        symbolic::Lt(j, i),
        symbolic::integer(0),
        symbolic::add(j, symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    builder.add_block(inner.root());

    analysis::AnalysisManager am(builder.subject());
    transformations::MapCollapse t(outer, 2);
    EXPECT_FALSE(t.can_be_applied(builder, am));
}

TEST(MapCollapseTest, CannotApply_InnerInitDependsOnOuterIndvar) {
    // inner map starts from i instead of 0, so its init references the outer indvar
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);
    auto& root = builder.subject().root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("M", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);

    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");

    auto& outer = builder.add_map(
        root,
        i,
        symbolic::Lt(i, symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(i, symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );

    // inner init is `i` — depends on outer indvar
    auto& inner = builder.add_map(
        outer.root(),
        j,
        symbolic::Lt(j, symbolic::symbol("M")),
        i,
        symbolic::add(j, symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    builder.add_block(inner.root());

    analysis::AnalysisManager am(builder.subject());
    transformations::MapCollapse t(outer, 2);
    EXPECT_FALSE(t.can_be_applied(builder, am));
}

// ---------------------------------------------------------------------------
// Apply — structural checks
// ---------------------------------------------------------------------------

TEST(MapCollapseTest, Apply_2D_Structure) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);
    auto [outer, inner] = build_2d_map_nest(builder);

    analysis::AnalysisManager am(builder.subject());
    transformations::MapCollapse t(*outer, 2);
    ASSERT_TRUE(t.can_be_applied(builder, am));
    t.apply(builder, am);

    auto* collapsed = t.collapsed_loop();
    ASSERT_NE(collapsed, nullptr);

    // The root must contain the collapsed loop as its only child
    EXPECT_EQ(builder.subject().root().size(), 1);
    EXPECT_EQ(&builder.subject().root().at(0), collapsed);

    // The collapsed body must contain an empty recovery block + the original inner block
    auto& body = collapsed->root();
    EXPECT_EQ(body.size(), 2);
    EXPECT_TRUE(dyn_cast<structured_control_flow::AssignmentBlock*>(&body.at(0)) != nullptr);
    EXPECT_TRUE(dyn_cast<structured_control_flow::Block*>(&body.at(1)) != nullptr);
}

TEST(MapCollapseTest, Apply_2D_IndvarTransitions) {
    // Check that the transition to the first body element regenerates i and j
    // from the new collapsed induction variable civ:
    //   i = civ / M
    //   j = civ % M
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);
    auto [outer, inner] = build_2d_map_nest(builder);

    analysis::AnalysisManager am(builder.subject());
    transformations::MapCollapse t(*outer, 2);
    ASSERT_TRUE(t.can_be_applied(builder, am));
    t.apply(builder, am);

    auto* collapsed = t.collapsed_loop();
    ASSERT_NE(collapsed, nullptr);

    auto civ = collapsed->indvar();
    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");
    auto M = symbolic::symbol("M");

    // Transition to the first element inside the collapsed loop body
    const auto transition = dyn_cast<AssignmentBlock*>(&collapsed->root().at(0));
    ASSERT_TRUE(transition);
    const auto& assignments = transition->assignments();

    // Both original indvars must be re-defined
    ASSERT_TRUE(assignments.count(i)) << "'i' must be assigned in the transition to the first body element";
    ASSERT_TRUE(assignments.count(j)) << "'j' must be assigned in the transition to the first body element";

    // i = civ / M
    EXPECT_TRUE(symbolic::eq(assignments.at(i), symbolic::div(civ, M))) << "Expected i = civ / M";
    // j = civ % M
    EXPECT_TRUE(symbolic::eq(assignments.at(j), symbolic::mod(civ, M))) << "Expected j = civ % M";
}

TEST(MapCollapseTest, Apply_2D_CollapsedRange) {
    // Collapsed loop range must be [0, N*M) with step 1.
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);
    auto [outer, inner] = build_2d_map_nest(builder);

    analysis::AnalysisManager am(builder.subject());
    transformations::MapCollapse t(*outer, 2);
    ASSERT_TRUE(t.can_be_applied(builder, am));
    t.apply(builder, am);

    auto* collapsed = t.collapsed_loop();
    ASSERT_NE(collapsed, nullptr);

    auto civ = collapsed->indvar();
    auto N = symbolic::symbol("N");
    auto M = symbolic::symbol("M");

    EXPECT_TRUE(symbolic::eq(collapsed->init(), symbolic::integer(0)));
    EXPECT_TRUE(symbolic::eq(collapsed->condition(), symbolic::Lt(civ, symbolic::mul(N, M))))
        << "Collapsed condition must be civ < N*M";
    EXPECT_TRUE(symbolic::eq(collapsed->update(), symbolic::add(civ, symbolic::integer(1))));
}

TEST(MapCollapseTest, Apply_3D_IndvarTransitions) {
    // For a 3-level nest (i, j, k) over [0,N) x [0,M) x [0,P):
    //   i = civ / (M*P)
    //   j = (civ / P) % M
    //   k = civ % P
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);
    auto& root = builder.subject().root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("M", sym_desc, true);
    builder.add_container("P", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);
    builder.add_container("k", sym_desc);

    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");
    auto k = symbolic::symbol("k");
    auto N = symbolic::symbol("N");
    auto M = symbolic::symbol("M");
    auto P = symbolic::symbol("P");

    auto& outer = builder.add_map(
        root,
        i,
        symbolic::Lt(i, N),
        symbolic::integer(0),
        symbolic::add(i, symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    auto& middle = builder.add_map(
        outer.root(),
        j,
        symbolic::Lt(j, M),
        symbolic::integer(0),
        symbolic::add(j, symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    auto& inner = builder.add_map(
        middle.root(),
        k,
        symbolic::Lt(k, P),
        symbolic::integer(0),
        symbolic::add(k, symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    builder.add_block(inner.root());

    analysis::AnalysisManager am(builder.subject());
    transformations::MapCollapse t(outer, 3);
    ASSERT_TRUE(t.can_be_applied(builder, am));
    t.apply(builder, am);

    auto* collapsed = t.collapsed_loop();
    ASSERT_NE(collapsed, nullptr);

    auto civ = collapsed->indvar();
    const auto transition = dyn_cast<AssignmentBlock*>(&collapsed->root().at(0));
    ASSERT_TRUE(transition);
    const auto& assignments = transition->assignments();

    ASSERT_TRUE(assignments.count(i)) << "'i' not assigned";
    ASSERT_TRUE(assignments.count(j)) << "'j' not assigned";
    ASSERT_TRUE(assignments.count(k)) << "'k' not assigned";

    // i = civ / (M*P)
    EXPECT_TRUE(symbolic::eq(assignments.at(i), symbolic::div(civ, symbolic::mul(M, P)))) << "Expected i = civ / (M*P)";
    // j = (civ / P) % M
    EXPECT_TRUE(symbolic::eq(assignments.at(j), symbolic::mod(symbolic::div(civ, P), M)))
        << "Expected j = (civ / P) % M";
    // k = civ % P
    EXPECT_TRUE(symbolic::eq(assignments.at(k), symbolic::mod(civ, P))) << "Expected k = civ % P";
}

// ---------------------------------------------------------------------------
// 3D nest — partial 2D collapse of outer two loops (i,j) leaving k intact
// ---------------------------------------------------------------------------

/// Build a 3-level perfectly-nested map: i in [0,N), j in [0,M), k in [0,P).
/// Returns {outer, middle, inner}.
static std::tuple<structured_control_flow::Map*, structured_control_flow::Map*, structured_control_flow::Map*>
build_3d_map_nest(builder::StructuredSDFGBuilder& builder) {
    auto& root = builder.subject().root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("M", sym_desc, true);
    builder.add_container("P", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);
    builder.add_container("k", sym_desc);

    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");
    auto k = symbolic::symbol("k");
    auto N = symbolic::symbol("N");
    auto M = symbolic::symbol("M");
    auto P = symbolic::symbol("P");

    auto& outer = builder.add_map(
        root,
        i,
        symbolic::Lt(i, N),
        symbolic::integer(0),
        symbolic::add(i, symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    auto& middle = builder.add_map(
        outer.root(),
        j,
        symbolic::Lt(j, M),
        symbolic::integer(0),
        symbolic::add(j, symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    auto& inner = builder.add_map(
        middle.root(),
        k,
        symbolic::Lt(k, P),
        symbolic::integer(0),
        symbolic::add(k, symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    builder.add_block(inner.root());

    return {&outer, &middle, &inner};
}

TEST(MapCollapseTest, Apply_3D_CollapseOuter2_Structure) {
    // Collapse only the outer two maps (i, j) — k must survive as an inner map.
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);
    auto [outer, middle, inner] = build_3d_map_nest(builder);

    analysis::AnalysisManager am(builder.subject());
    transformations::MapCollapse t(*outer, 2);
    ASSERT_TRUE(t.can_be_applied(builder, am));
    t.apply(builder, am);

    auto* collapsed = t.collapsed_loop();
    ASSERT_NE(collapsed, nullptr);

    // Root has exactly the collapsed map
    EXPECT_EQ(builder.subject().root().size(), 1);
    EXPECT_EQ(&builder.subject().root().at(0), collapsed);

    // collapsed body: empty recovery block + the surviving k map
    auto& body = collapsed->root();
    EXPECT_EQ(body.size(), 2);
    EXPECT_TRUE(dyn_cast<structured_control_flow::AssignmentBlock*>(&body.at(0)) != nullptr);
    auto* k_map = dyn_cast<structured_control_flow::Map*>(&body.at(1));
    ASSERT_NE(k_map, nullptr) << "Inner k map must survive as direct child of collapsed loop";

    // k map body still contains exactly the original block
    EXPECT_EQ(k_map->root().size(), 1);
    EXPECT_TRUE(dyn_cast<structured_control_flow::Block*>(&k_map->root().at(0)) != nullptr);
}

TEST(MapCollapseTest, Apply_3D_CollapseOuter2_IndvarTransitions) {
    // After collapsing (i, j) with civ:
    //   transition into first body element: i = civ / M, j = civ % M
    //   k is NOT touched by this transition (it keeps its own map loop)
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);
    auto [outer, middle, inner] = build_3d_map_nest(builder);

    analysis::AnalysisManager am(builder.subject());
    transformations::MapCollapse t(*outer, 2);
    ASSERT_TRUE(t.can_be_applied(builder, am));
    t.apply(builder, am);

    auto* collapsed = t.collapsed_loop();
    ASSERT_NE(collapsed, nullptr);

    auto civ = collapsed->indvar();
    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");
    auto k = symbolic::symbol("k");
    auto M = symbolic::symbol("M");

    const auto transition = dyn_cast<AssignmentBlock*>(&collapsed->root().at(0));
    ASSERT_TRUE(transition);
    const auto& assignments = transition->assignments();

    ASSERT_TRUE(assignments.count(i)) << "'i' must be assigned";
    ASSERT_TRUE(assignments.count(j)) << "'j' must be assigned";
    EXPECT_FALSE(assignments.count(k)) << "'k' must NOT be assigned here";

    EXPECT_TRUE(symbolic::eq(assignments.at(i), symbolic::div(civ, M))) << "Expected i = civ / M";
    EXPECT_TRUE(symbolic::eq(assignments.at(j), symbolic::mod(civ, M))) << "Expected j = civ % M";
}

TEST(MapCollapseTest, Apply_3D_CollapseOuter2_CollapsedRange) {
    // Collapsed loop runs [0, N*M); k map is unchanged with bound P.
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);
    auto [outer, middle, inner] = build_3d_map_nest(builder);

    analysis::AnalysisManager am(builder.subject());
    transformations::MapCollapse t(*outer, 2);
    ASSERT_TRUE(t.can_be_applied(builder, am));
    t.apply(builder, am);

    auto* collapsed = t.collapsed_loop();
    ASSERT_NE(collapsed, nullptr);

    auto civ = collapsed->indvar();
    auto N = symbolic::symbol("N");
    auto M = symbolic::symbol("M");
    auto P = symbolic::symbol("P");

    EXPECT_TRUE(symbolic::eq(collapsed->init(), symbolic::integer(0)));
    EXPECT_TRUE(symbolic::eq(collapsed->condition(), symbolic::Lt(civ, symbolic::mul(N, M))))
        << "Collapsed condition must be civ < N*M";
    EXPECT_TRUE(symbolic::eq(collapsed->update(), symbolic::add(civ, symbolic::integer(1))));

    // The surviving k map must still have its original bound P
    auto* k_map = dyn_cast<structured_control_flow::Map*>(&collapsed->root().at(1));
    ASSERT_NE(k_map, nullptr);
    EXPECT_TRUE(symbolic::eq(k_map->condition(), symbolic::Lt(symbolic::symbol("k"), P)))
        << "k map bound must remain < P";
}

// ---------------------------------------------------------------------------
// 3D nest — partial 2D collapse of middle two loops (j,k) leaving i intact
// ---------------------------------------------------------------------------

TEST(MapCollapseTest, Apply_3D_CollapseMiddle2_Structure) {
    // Collapse only the middle pair (j, k) starting from the middle map.
    // After: outer i map → collapsed jk map → original block.
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);
    auto [outer, middle, inner] = build_3d_map_nest(builder);

    analysis::AnalysisManager am(builder.subject());
    transformations::MapCollapse t(*middle, 2);
    ASSERT_TRUE(t.can_be_applied(builder, am));
    t.apply(builder, am);

    auto* collapsed = t.collapsed_loop();
    ASSERT_NE(collapsed, nullptr);

    // Root still has exactly one child: the outer i map (unchanged)
    EXPECT_EQ(builder.subject().root().size(), 1);
    auto* i_map = dyn_cast<structured_control_flow::Map*>(&builder.subject().root().at(0));
    ASSERT_NE(i_map, nullptr) << "Outer i map must still be the root child";

    // Outer i map body: exactly the collapsed jk map
    EXPECT_EQ(i_map->root().size(), 1);
    auto* jk_map = dyn_cast<structured_control_flow::Map*>(&i_map->root().at(0));
    ASSERT_NE(jk_map, nullptr) << "Collapsed jk map must be inside the outer i map";
    EXPECT_EQ(jk_map, collapsed);

    // Collapsed body: empty recovery block + the original block
    EXPECT_EQ(collapsed->root().size(), 2);
    EXPECT_TRUE(dyn_cast<structured_control_flow::AssignmentBlock*>(&collapsed->root().at(0)) != nullptr);
    EXPECT_TRUE(dyn_cast<structured_control_flow::Block*>(&collapsed->root().at(1)) != nullptr);
}

TEST(MapCollapseTest, Apply_3D_CollapseMiddle2_IndvarTransitions) {
    // After collapsing (j, k) with civ:
    //   transition into first body element: j = civ / P, k = civ % P
    //   i is NOT touched
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);
    auto [outer, middle, inner] = build_3d_map_nest(builder);

    analysis::AnalysisManager am(builder.subject());
    transformations::MapCollapse t(*middle, 2);
    ASSERT_TRUE(t.can_be_applied(builder, am));
    t.apply(builder, am);

    auto* collapsed = t.collapsed_loop();
    ASSERT_NE(collapsed, nullptr);

    auto civ = collapsed->indvar();
    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");
    auto k = symbolic::symbol("k");
    auto P = symbolic::symbol("P");

    const auto transition = dyn_cast<AssignmentBlock*>(&collapsed->root().at(0));
    ASSERT_TRUE(transition);
    const auto& assignments = transition->assignments();

    EXPECT_FALSE(assignments.count(i)) << "'i' must NOT be assigned";
    ASSERT_TRUE(assignments.count(j)) << "'j' must be assigned";
    ASSERT_TRUE(assignments.count(k)) << "'k' must be assigned";

    EXPECT_TRUE(symbolic::eq(assignments.at(j), symbolic::div(civ, P))) << "Expected j = civ / P";
    EXPECT_TRUE(symbolic::eq(assignments.at(k), symbolic::mod(civ, P))) << "Expected k = civ % P";
}

TEST(MapCollapseTest, Apply_3D_CollapseMiddle2_CollapsedRange) {
    // Collapsed jk loop runs [0, M*P); outer i loop is unchanged with bound N.
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);
    auto [outer, middle, inner] = build_3d_map_nest(builder);

    analysis::AnalysisManager am(builder.subject());
    transformations::MapCollapse t(*middle, 2);
    ASSERT_TRUE(t.can_be_applied(builder, am));
    t.apply(builder, am);

    auto* collapsed = t.collapsed_loop();
    ASSERT_NE(collapsed, nullptr);

    auto civ = collapsed->indvar();
    auto N = symbolic::symbol("N");
    auto M = symbolic::symbol("M");
    auto P = symbolic::symbol("P");

    EXPECT_TRUE(symbolic::eq(collapsed->init(), symbolic::integer(0)));
    EXPECT_TRUE(symbolic::eq(collapsed->condition(), symbolic::Lt(civ, symbolic::mul(M, P))))
        << "Collapsed condition must be civ < M*P";
    EXPECT_TRUE(symbolic::eq(collapsed->update(), symbolic::add(civ, symbolic::integer(1))));

    // Outer i map must still have its original bound N
    auto* i_map = dyn_cast<structured_control_flow::Map*>(&builder.subject().root().at(0));
    ASSERT_NE(i_map, nullptr);
    EXPECT_TRUE(symbolic::eq(i_map->condition(), symbolic::Lt(symbolic::symbol("i"), N)))
        << "Outer i map bound must remain < N";
}

// ---------------------------------------------------------------------------
// 4D nest — two sequential collapse(2): first on i (collapses i,j), then on k (collapses k,l)
// ---------------------------------------------------------------------------

TEST(MapCollapseTest, Apply_4D_CollapsePairs) {
    // Build 4-level nest: i in [0,N), j in [0,M), k in [0,P), l in [0,Q)
    // Step 1: collapse(2) on i → collapsed_ij in [0, N*M), body = k → l → block
    // Step 2: collapse(2) on k → collapsed_kl in [0, P*Q), body = block
    // Result: collapsed_ij → collapsed_kl → block
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);
    auto& root = builder.subject().root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("M", sym_desc, true);
    builder.add_container("P", sym_desc, true);
    builder.add_container("Q", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);
    builder.add_container("k", sym_desc);
    builder.add_container("l", sym_desc);

    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");
    auto k = symbolic::symbol("k");
    auto l = symbolic::symbol("l");
    auto N = symbolic::symbol("N");
    auto M = symbolic::symbol("M");
    auto P = symbolic::symbol("P");
    auto Q = symbolic::symbol("Q");

    auto& map_i = builder.add_map(
        root,
        i,
        symbolic::Lt(i, N),
        symbolic::integer(0),
        symbolic::add(i, symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    auto& map_j = builder.add_map(
        map_i.root(),
        j,
        symbolic::Lt(j, M),
        symbolic::integer(0),
        symbolic::add(j, symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    auto& map_k = builder.add_map(
        map_j.root(),
        k,
        symbolic::Lt(k, P),
        symbolic::integer(0),
        symbolic::add(k, symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    auto& map_l = builder.add_map(
        map_k.root(),
        l,
        symbolic::Lt(l, Q),
        symbolic::integer(0),
        symbolic::add(l, symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    builder.add_block(map_l.root());

    analysis::AnalysisManager am(builder.subject());

    // --- First collapse: (i, j) ---
    transformations::MapCollapse t1(map_i, 2);
    ASSERT_TRUE(t1.can_be_applied(builder, am));
    t1.apply(builder, am);

    auto* collapsed_ij = t1.collapsed_loop();
    ASSERT_NE(collapsed_ij, nullptr);

    // After first collapse: root → collapsed_ij → k → l → block
    EXPECT_EQ(builder.subject().root().size(), 1);
    EXPECT_EQ(&builder.subject().root().at(0), collapsed_ij);

    auto civ_ij = collapsed_ij->indvar();

    // collapsed_ij range: [0, N*M)
    EXPECT_TRUE(symbolic::eq(collapsed_ij->init(), symbolic::integer(0)));
    EXPECT_TRUE(symbolic::eq(collapsed_ij->condition(), symbolic::Lt(civ_ij, symbolic::mul(N, M))));
    EXPECT_TRUE(symbolic::eq(collapsed_ij->update(), symbolic::add(civ_ij, symbolic::integer(1))));

    // collapsed_ij body: empty recovery block + the k map
    EXPECT_EQ(collapsed_ij->root().size(), 2);
    EXPECT_TRUE(dyn_cast<structured_control_flow::AssignmentBlock*>(&collapsed_ij->root().at(0)) != nullptr);
    auto* surviving_k = dyn_cast<structured_control_flow::Map*>(&collapsed_ij->root().at(1));
    ASSERT_NE(surviving_k, nullptr) << "k map must survive as child of collapsed_ij";

    // Indvar recovery for i, j (on the empty recovery block's transition)
    {
        const auto transition = dyn_cast<AssignmentBlock*>(&collapsed_ij->root().at(0));
        ASSERT_TRUE(transition);
        const auto& asgn = transition->assignments();
        ASSERT_TRUE(asgn.count(i)) << "'i' must be assigned";
        ASSERT_TRUE(asgn.count(j)) << "'j' must be assigned";
        EXPECT_FALSE(asgn.count(k)) << "'k' must NOT be assigned here";
        EXPECT_FALSE(asgn.count(l)) << "'l' must NOT be assigned here";

        EXPECT_TRUE(symbolic::eq(asgn.at(i), symbolic::div(civ_ij, M)));
        EXPECT_TRUE(symbolic::eq(asgn.at(j), symbolic::mod(civ_ij, M)));
    }

    // k map body: l map
    EXPECT_EQ(surviving_k->root().size(), 1);
    auto* surviving_l = dyn_cast<structured_control_flow::Map*>(&surviving_k->root().at(0));
    ASSERT_NE(surviving_l, nullptr) << "l map must survive inside k map";

    // l map body: the original block
    EXPECT_EQ(surviving_l->root().size(), 1);
    EXPECT_TRUE(dyn_cast<structured_control_flow::Block*>(&surviving_l->root().at(0)) != nullptr);

    // --- Second collapse: (k, l) ---
    transformations::MapCollapse t2(*surviving_k, 2);
    ASSERT_TRUE(t2.can_be_applied(builder, am));
    t2.apply(builder, am);

    auto* collapsed_kl = t2.collapsed_loop();
    ASSERT_NE(collapsed_kl, nullptr);

    auto civ_kl = collapsed_kl->indvar();

    // Final structure: root → collapsed_ij → collapsed_kl → block
    EXPECT_EQ(builder.subject().root().size(), 1);

    auto* final_outer = dyn_cast<structured_control_flow::Map*>(&builder.subject().root().at(0));
    ASSERT_NE(final_outer, nullptr);
    // collapsed_ij must still be the root child
    EXPECT_EQ(final_outer, collapsed_ij);

    // collapsed_ij body: empty recovery block + collapsed_kl
    EXPECT_EQ(collapsed_ij->root().size(), 2);
    auto* inner_map = dyn_cast<structured_control_flow::Map*>(&collapsed_ij->root().at(1));
    ASSERT_NE(inner_map, nullptr);
    EXPECT_EQ(inner_map, collapsed_kl);

    // collapsed_kl range: [0, P*Q)
    EXPECT_TRUE(symbolic::eq(collapsed_kl->init(), symbolic::integer(0)));
    EXPECT_TRUE(symbolic::eq(collapsed_kl->condition(), symbolic::Lt(civ_kl, symbolic::mul(P, Q))));
    EXPECT_TRUE(symbolic::eq(collapsed_kl->update(), symbolic::add(civ_kl, symbolic::integer(1))));

    // collapsed_kl body: empty recovery block + the original block
    EXPECT_EQ(collapsed_kl->root().size(), 2);
    EXPECT_TRUE(dyn_cast<structured_control_flow::Block*>(&collapsed_kl->root().at(1)) != nullptr);

    // Indvar recovery for k, l (on the empty recovery block's transition)
    {
        const auto transition = dyn_cast<AssignmentBlock*>(&collapsed_kl->root().at(0));
        ASSERT_TRUE(transition);
        const auto& asgn = transition->assignments();
        ASSERT_TRUE(asgn.count(k)) << "'k' must be assigned";
        ASSERT_TRUE(asgn.count(l)) << "'l' must be assigned";
        EXPECT_FALSE(asgn.count(i)) << "'i' must NOT be assigned here";
        EXPECT_FALSE(asgn.count(j)) << "'j' must NOT be assigned here";

        EXPECT_TRUE(symbolic::eq(asgn.at(k), symbolic::div(civ_kl, Q)));
        EXPECT_TRUE(symbolic::eq(asgn.at(l), symbolic::mod(civ_kl, Q)));
    }

    // Verify that collapsed_ij indvar recovery is still correct after second collapse
    {
        const auto transition = dyn_cast<AssignmentBlock*>(&collapsed_ij->root().at(0));
        ASSERT_TRUE(transition);
        const auto& asgn = transition->assignments();
        ASSERT_TRUE(asgn.count(i)) << "'i' must still be assigned after second collapse";
        ASSERT_TRUE(asgn.count(j)) << "'j' must still be assigned after second collapse";

        EXPECT_TRUE(symbolic::eq(asgn.at(i), symbolic::div(civ_ij, M)))
            << "i = civ_ij / M must survive the second collapse";
        EXPECT_TRUE(symbolic::eq(asgn.at(j), symbolic::mod(civ_ij, M)))
            << "j = civ_ij % M must survive the second collapse";
    }
}

// ---------------------------------------------------------------------------
// Computation preservation — single collapse
// ---------------------------------------------------------------------------

TEST(MapCollapseTest, Apply_2D_WithComputation) {
    // Build: A[i*M + j] = A[i*M + j]  (assign tasklet) inside a 2D map nest
    // After collapse: the block with its tasklet and memlets must survive intact,
    // and the transition must define i = civ / M, j = civ % M.
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);
    auto& root = builder.subject().root();

    types::Scalar float_desc(types::PrimitiveType::Float);
    types::Pointer ptr_desc(float_desc);
    types::Pointer opaque_desc;
    types::Scalar sym_desc(types::PrimitiveType::UInt64);

    builder.add_container("A", opaque_desc, true);
    builder.add_container("N", sym_desc, true);
    builder.add_container("M", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);

    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");
    auto N = symbolic::symbol("N");
    auto M = symbolic::symbol("M");

    auto& outer = builder.add_map(
        root,
        i,
        symbolic::Lt(i, N),
        symbolic::integer(0),
        symbolic::add(i, symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    auto& inner = builder.add_map(
        outer.root(),
        j,
        symbolic::Lt(j, M),
        symbolic::integer(0),
        symbolic::add(j, symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );

    // Computation: A[i*M + j] = A[i*M + j]
    auto index_expr = symbolic::add(symbolic::mul(i, M), j);
    auto& block = builder.add_block(inner.root());
    auto& A_in = builder.add_access(block, "A");
    auto& A_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, A_in, tasklet, "_in", {index_expr}, ptr_desc);
    builder.add_computational_memlet(block, tasklet, "_out", A_out, {index_expr}, ptr_desc);

    analysis::AnalysisManager am(builder.subject());
    transformations::MapCollapse t(outer, 2);
    ASSERT_TRUE(t.can_be_applied(builder, am));
    t.apply(builder, am);

    auto* collapsed = t.collapsed_loop();
    ASSERT_NE(collapsed, nullptr);

    // Structure: root → collapsed → [empty recovery block, block]
    EXPECT_EQ(builder.subject().root().size(), 1);
    EXPECT_EQ(collapsed->root().size(), 2);
    auto* body_block = dyn_cast<structured_control_flow::Block*>(&collapsed->root().at(1));
    ASSERT_NE(body_block, nullptr);

    // Transition assignments
    auto civ = collapsed->indvar();
    const auto transition = dyn_cast<AssignmentBlock*>(&collapsed->root().at(0));
    ASSERT_TRUE(transition);
    const auto& asgn = transition->assignments();
    ASSERT_TRUE(asgn.count(i));
    ASSERT_TRUE(asgn.count(j));
    EXPECT_TRUE(symbolic::eq(asgn.at(i), symbolic::div(civ, M)));
    EXPECT_TRUE(symbolic::eq(asgn.at(j), symbolic::mod(civ, M)));

    // The block must still have exactly 1 tasklet
    auto tasklets = body_block->dataflow().tasklets();
    EXPECT_EQ(tasklets.size(), 1);

    // The tasklet must still have in-edge and out-edge with the original index expression
    auto* t_node = *tasklets.begin();
    auto in_edges = body_block->dataflow().in_edges(*t_node);
    auto out_edges = body_block->dataflow().out_edges(*t_node);
    EXPECT_EQ(body_block->dataflow().in_degree(*t_node), 1);
    EXPECT_EQ(body_block->dataflow().out_degree(*t_node), 1);

    // Verify memlet subsets are inlined to the collapsed induction variable.
    // MapCollapse now substitutes the recovered induction variables (i -> civ / M,
    // j -> civ % M) directly into the memlet subsets so downstream analyses see the
    // relation to civ without needing a separate SymbolPropagation pass. The transition
    // assignments verified above are still kept as a fallback.
    symbolic::ExpressionMapping recovery;
    recovery[i] = symbolic::div(civ, M);
    recovery[j] = symbolic::mod(civ, M);
    auto expected_index = symbolic::subs(index_expr, recovery);

    EXPECT_EQ((*in_edges.begin()).subset().size(), 1);
    EXPECT_TRUE(symbolic::eq((*in_edges.begin()).subset()[0], expected_index))
        << "Input memlet subset must be inlined to (civ / M) * M + civ % M";

    EXPECT_EQ((*out_edges.begin()).subset().size(), 1);
    EXPECT_TRUE(symbolic::eq((*out_edges.begin()).subset()[0], expected_index))
        << "Output memlet subset must be inlined to (civ / M) * M + civ % M";
}

// ---------------------------------------------------------------------------
// Computation preservation — double collapse (4D)
// ---------------------------------------------------------------------------

TEST(MapCollapseTest, Apply_4D_WithComputation_CollapsePairs) {
    // Build 4-level nest: A[i*M*P*Q + j*P*Q + k*Q + l] = A[...] (assign)
    // Collapse(2) on i then collapse(2) on k.
    // Verify that after both collapses the block, tasklet, and memlet subsets
    // are preserved, and both transition assignment sets are correct.
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);
    auto& root = builder.subject().root();

    types::Scalar float_desc(types::PrimitiveType::Float);
    types::Pointer ptr_desc(float_desc);
    types::Pointer opaque_desc;
    types::Scalar sym_desc(types::PrimitiveType::UInt64);

    builder.add_container("A", opaque_desc, true);
    builder.add_container("N", sym_desc, true);
    builder.add_container("M", sym_desc, true);
    builder.add_container("P", sym_desc, true);
    builder.add_container("Q", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);
    builder.add_container("k", sym_desc);
    builder.add_container("l", sym_desc);

    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");
    auto k = symbolic::symbol("k");
    auto l = symbolic::symbol("l");
    auto N = symbolic::symbol("N");
    auto M = symbolic::symbol("M");
    auto P = symbolic::symbol("P");
    auto Q = symbolic::symbol("Q");

    auto& map_i = builder.add_map(
        root,
        i,
        symbolic::Lt(i, N),
        symbolic::integer(0),
        symbolic::add(i, symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    auto& map_j = builder.add_map(
        map_i.root(),
        j,
        symbolic::Lt(j, M),
        symbolic::integer(0),
        symbolic::add(j, symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    auto& map_k = builder.add_map(
        map_j.root(),
        k,
        symbolic::Lt(k, P),
        symbolic::integer(0),
        symbolic::add(k, symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    auto& map_l = builder.add_map(
        map_k.root(),
        l,
        symbolic::Lt(l, Q),
        symbolic::integer(0),
        symbolic::add(l, symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );

    // Computation: A[i*M*P*Q + j*P*Q + k*Q + l] = A[...]
    auto index_expr = symbolic::add(
        symbolic::add(symbolic::mul(i, symbolic::mul(M, symbolic::mul(P, Q))), symbolic::mul(j, symbolic::mul(P, Q))),
        symbolic::add(symbolic::mul(k, Q), l)
    );
    auto& block = builder.add_block(map_l.root());
    auto& A_in = builder.add_access(block, "A");
    auto& A_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, A_in, tasklet, "_in", {index_expr}, ptr_desc);
    builder.add_computational_memlet(block, tasklet, "_out", A_out, {index_expr}, ptr_desc);

    analysis::AnalysisManager am(builder.subject());

    // --- First collapse: (i, j) ---
    transformations::MapCollapse t1(map_i, 2);
    ASSERT_TRUE(t1.can_be_applied(builder, am));
    t1.apply(builder, am);

    auto* collapsed_ij = t1.collapsed_loop();
    ASSERT_NE(collapsed_ij, nullptr);
    auto civ_ij = collapsed_ij->indvar();

    // --- Second collapse: (k, l) ---
    auto* surviving_k = dyn_cast<structured_control_flow::Map*>(&collapsed_ij->root().at(1));
    ASSERT_NE(surviving_k, nullptr);
    // collapsed_ij body: [empty recovery block, k map]
    // k map body: l map
    // l map body: block
    // We need the k map to collapse (k, l):

    transformations::MapCollapse t2(*surviving_k, 2);
    ASSERT_TRUE(t2.can_be_applied(builder, am));
    t2.apply(builder, am);

    auto* collapsed_kl = t2.collapsed_loop();
    ASSERT_NE(collapsed_kl, nullptr);
    auto civ_kl = collapsed_kl->indvar();

    // Final structure: root → collapsed_ij → [empty, collapsed_kl → [empty, block]]
    EXPECT_EQ(builder.subject().root().size(), 1);
    EXPECT_EQ(collapsed_ij->root().size(), 2);
    EXPECT_EQ(collapsed_kl->root().size(), 2);

    auto* final_block = dyn_cast<structured_control_flow::Block*>(&collapsed_kl->root().at(1));
    ASSERT_NE(final_block, nullptr) << "Innermost element must be the original block";

    // Verify collapsed_ij transition assignments are still correct
    {
        const auto transition = dyn_cast<AssignmentBlock*>(&collapsed_ij->root().at(0));
        ASSERT_TRUE(transition);
        const auto& asgn = transition->assignments();
        ASSERT_TRUE(asgn.count(i));
        ASSERT_TRUE(asgn.count(j));
        EXPECT_TRUE(symbolic::eq(asgn.at(i), symbolic::div(civ_ij, M)));
        EXPECT_TRUE(symbolic::eq(asgn.at(j), symbolic::mod(civ_ij, M)));
    }

    // Verify collapsed_kl transition assignments
    {
        const auto transition = dyn_cast<AssignmentBlock*>(&collapsed_kl->root().at(0));
        ASSERT_TRUE(transition);
        const auto& asgn = transition->assignments();
        ASSERT_TRUE(asgn.count(k));
        ASSERT_TRUE(asgn.count(l));
        EXPECT_TRUE(symbolic::eq(asgn.at(k), symbolic::div(civ_kl, Q)));
        EXPECT_TRUE(symbolic::eq(asgn.at(l), symbolic::mod(civ_kl, Q)));
    }

    // The block must still have exactly 1 tasklet
    auto tasklets = final_block->dataflow().tasklets();
    EXPECT_EQ(tasklets.size(), 1);

    // The tasklet memlets must still reference the original index expression (i*M*P*Q + j*P*Q + k*Q + l)
    auto* t_node = *tasklets.begin();
    auto in_edges = final_block->dataflow().in_edges(*t_node);
    auto out_edges = final_block->dataflow().out_edges(*t_node);
    EXPECT_EQ(final_block->dataflow().in_degree(*t_node), 1);
    EXPECT_EQ(final_block->dataflow().out_degree(*t_node), 1);

    // Both collapses inline their recovered induction variables into the memlet subsets:
    //   i -> civ_ij / M, j -> civ_ij % M   (first collapse)
    //   k -> civ_kl / Q, l -> civ_kl % Q   (second collapse)
    symbolic::ExpressionMapping recovery;
    recovery[i] = symbolic::div(civ_ij, M);
    recovery[j] = symbolic::mod(civ_ij, M);
    recovery[k] = symbolic::div(civ_kl, Q);
    recovery[l] = symbolic::mod(civ_kl, Q);
    auto expected_index = symbolic::subs(index_expr, recovery);

    EXPECT_EQ((*in_edges.begin()).subset().size(), 1);
    EXPECT_TRUE(symbolic::eq((*in_edges.begin()).subset()[0], expected_index))
        << "Input memlet subset must be inlined in terms of civ_ij and civ_kl";

    EXPECT_EQ((*out_edges.begin()).subset().size(), 1);
    EXPECT_TRUE(symbolic::eq((*out_edges.begin()).subset()[0], expected_index))
        << "Output memlet subset must be inlined in terms of civ_ij and civ_kl";
}

// ---------------------------------------------------------------------------
// CollapsedLoop accessor before apply
// ---------------------------------------------------------------------------

TEST(MapCollapseTest, CollapsedLoopAccessorBeforeApply_outerloop) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);
    auto [outer, inner] = build_2d_map_nest(builder);

    transformations::MapCollapse t(*outer, 2);
    EXPECT_EQ(t.collapsed_loop(), outer);
}

// ---------------------------------------------------------------------------
// Serialization
// ---------------------------------------------------------------------------

TEST(MapCollapseTest, Serialization) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);
    auto [outer, inner] = build_2d_map_nest(builder);

    size_t loop_id = outer->element_id();
    size_t count = 2;

    transformations::MapCollapse t(*outer, count);

    nlohmann::json j;
    EXPECT_NO_THROW(t.to_json(j));

    EXPECT_EQ(j["transformation_type"], "MapCollapse");
    EXPECT_TRUE(j.contains("subgraph"));
    EXPECT_TRUE(j.contains("parameters"));
    EXPECT_EQ(j["subgraph"]["0"]["element_id"], loop_id);
    EXPECT_EQ(j["subgraph"]["0"]["type"], "map");
    EXPECT_EQ(j["parameters"]["count"], count);
}

TEST(MapCollapseTest, Deserialization) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);
    auto [outer, inner] = build_2d_map_nest(builder);

    size_t loop_id = outer->element_id();

    nlohmann::json j;
    j["transformation_type"] = "MapCollapse";
    j["subgraph"] = {{"0", {{"element_id", loop_id}, {"type", "map"}}}};
    j["parameters"] = {{"count", 2}};

    EXPECT_NO_THROW({
        auto deserialized = transformations::MapCollapse::from_json(builder, j);
        EXPECT_EQ(deserialized.name(), "MapCollapse");
    });
}

// ===========================================================================
// Imperfect (CUDA-style) collapse
//
// When the outer map's body is not a single perfectly-nested map, the collapse
// flattens the outer dimension against a virtual inner dimension whose extent is
// the maximum of all sibling-map bounds. Each sibling inner map body is guarded
// by `inner_idx < bound_i`, and every non-collapsed ("skipped") element is
// replicated on every inner thread (it stays a direct child of the collapsed
// body, with no guard).
//
// Expected transformed structure for outer i in [0,N) with sibling maps
// j in [0,M) and k in [0,P):
//
//   map civ in [0, N*max(M,P)):
//     [0] recovery block; transition assigns i = civ / max(M,P), t = civ % max(M,P)
//     [1] if (t < M):  { entry block (j = t); <body of j map> }
//     [2] if (t < P):  { entry block (k = t); <body of k map> }
//
// A skipped element B between the maps would simply appear as a direct child
// (e.g. [2] B, [3] if (t < P) ...), replicated across all inner threads.
// ===========================================================================

// ---------------------------------------------------------------------------
// Imperfect helpers
// ---------------------------------------------------------------------------

/// Build an outer map i in [0, N) whose body holds two sibling inner maps:
/// j in [0, M) (with a block) followed by k in [0, P) (with a block).
/// Returns {outer, map_j, map_k}.
static std::tuple<structured_control_flow::Map*, structured_control_flow::Map*, structured_control_flow::Map*>
build_2d_sibling_maps_nest(builder::StructuredSDFGBuilder& builder) {
    auto& root = builder.subject().root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("M", sym_desc, true);
    builder.add_container("P", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);
    builder.add_container("k", sym_desc);

    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");
    auto k = symbolic::symbol("k");
    auto N = symbolic::symbol("N");
    auto M = symbolic::symbol("M");
    auto P = symbolic::symbol("P");

    auto& outer = builder.add_map(
        root,
        i,
        symbolic::Lt(i, N),
        symbolic::integer(0),
        symbolic::add(i, symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );

    auto& map_j = builder.add_map(
        outer.root(),
        j,
        symbolic::Lt(j, M),
        symbolic::integer(0),
        symbolic::add(j, symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    builder.add_block(map_j.root());

    auto& map_k = builder.add_map(
        outer.root(),
        k,
        symbolic::Lt(k, P),
        symbolic::integer(0),
        symbolic::add(k, symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    builder.add_block(map_k.root());

    return {&outer, &map_j, &map_k};
}

/// Build an outer map i in [0, N) whose body holds a skipped block followed by a
/// single collapsible inner map j in [0, M) (with a block).
/// Returns {outer, map_j, block_a}.
static std::tuple<structured_control_flow::Map*, structured_control_flow::Map*, structured_control_flow::Block*>
build_imperfect_block_then_map(builder::StructuredSDFGBuilder& builder) {
    auto& root = builder.subject().root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("M", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);

    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");
    auto N = symbolic::symbol("N");
    auto M = symbolic::symbol("M");

    auto& outer = builder.add_map(
        root,
        i,
        symbolic::Lt(i, N),
        symbolic::integer(0),
        symbolic::add(i, symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );

    auto& block_a = builder.add_block(outer.root());

    auto& map_j = builder.add_map(
        outer.root(),
        j,
        symbolic::Lt(j, M),
        symbolic::integer(0),
        symbolic::add(j, symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    builder.add_block(map_j.root());

    return {&outer, &map_j, &block_a};
}

/// Locate the recovered virtual inner index `t` inside a recovery transition.
/// It is the assigned symbol whose value equals `civ % bmax`.
template<typename Assignments>
static symbolic::Expression
find_inner_index(const Assignments& assignments, const symbolic::Expression& civ, const symbolic::Expression& bmax) {
    for (const auto& entry : assignments) {
        if (symbolic::eq(entry.second, symbolic::mod(civ, bmax))) {
            return entry.first;
        }
    }
    return SymEngine::null;
}

// ---------------------------------------------------------------------------
// Imperfect — CanBeApplied positive cases
// ---------------------------------------------------------------------------

TEST(MapCollapseTest, CanBeApplied_Imperfect_TwoSiblingMaps) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);
    auto [outer, map_j, map_k] = build_2d_sibling_maps_nest(builder);

    analysis::AnalysisManager am(builder.subject());
    transformations::MapCollapse t(*outer, 2);
    EXPECT_TRUE(t.can_be_applied(builder, am));
}

TEST(MapCollapseTest, CanBeApplied_Imperfect_MapThenTrailingBlock) {
    // Outer body: inner map j in [0, M) followed by a trailing skipped block.
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);
    auto& root = builder.subject().root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("M", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);

    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");

    auto& outer = builder.add_map(
        root,
        i,
        symbolic::Lt(i, symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(i, symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );

    auto& inner = builder.add_map(
        outer.root(),
        j,
        symbolic::Lt(j, symbolic::symbol("M")),
        symbolic::integer(0),
        symbolic::add(j, symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    builder.add_block(inner.root());

    // trailing skipped block after the inner map
    builder.add_block(outer.root());

    analysis::AnalysisManager am(builder.subject());
    transformations::MapCollapse t(outer, 2);
    EXPECT_TRUE(t.can_be_applied(builder, am));
}

TEST(MapCollapseTest, CanBeApplied_Imperfect_NonCollapsibleSiblingMapSkipped) {
    // Outer body: a collapsible map j in [0, M) and a non-contiguous (stride 2)
    // sibling map kk. The non-collapsible map is treated as a skipped element
    // (guarded by inner_idx == 0), so the collapse is still applicable.
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);
    auto& root = builder.subject().root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("M", sym_desc, true);
    builder.add_container("P", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);
    builder.add_container("kk", sym_desc);

    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");
    auto kk = symbolic::symbol("kk");

    auto& outer = builder.add_map(
        root,
        i,
        symbolic::Lt(i, symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(i, symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );

    auto& map_j = builder.add_map(
        outer.root(),
        j,
        symbolic::Lt(j, symbolic::symbol("M")),
        symbolic::integer(0),
        symbolic::add(j, symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    builder.add_block(map_j.root());

    // Non-contiguous sibling map (stride 2) → not collapsible, but skippable.
    auto& map_kk = builder.add_map(
        outer.root(),
        kk,
        symbolic::Lt(kk, symbolic::symbol("P")),
        symbolic::integer(0),
        symbolic::add(kk, symbolic::integer(2)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    builder.add_block(map_kk.root());

    analysis::AnalysisManager am(builder.subject());
    transformations::MapCollapse t(outer, 2);
    EXPECT_TRUE(t.can_be_applied(builder, am));
}

// ---------------------------------------------------------------------------
// Imperfect — CanBeApplied negative cases
// ---------------------------------------------------------------------------

TEST(MapCollapseTest, CannotApply_Imperfect_NoInnerMap) {
    // Outer body contains only blocks — nothing to flatten.
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);
    auto& root = builder.subject().root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);

    auto i = symbolic::symbol("i");

    auto& outer = builder.add_map(
        root,
        i,
        symbolic::Lt(i, symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(i, symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    builder.add_block(outer.root());
    builder.add_block(outer.root());

    analysis::AnalysisManager am(builder.subject());
    transformations::MapCollapse t(outer, 2);
    EXPECT_FALSE(t.can_be_applied(builder, am));
}

TEST(MapCollapseTest, CannotApply_Imperfect_CountGreaterThan2) {
    // Imperfect nests are only collapsed one level at a time (count must be 2).
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);
    auto [outer, map_j, map_k] = build_2d_sibling_maps_nest(builder);

    analysis::AnalysisManager am(builder.subject());
    transformations::MapCollapse t(*outer, 3);
    EXPECT_FALSE(t.can_be_applied(builder, am));
}

TEST(MapCollapseTest, CannotApply_Imperfect_OuterNonContiguous) {
    // Outer map has stride 2 → not collapsible.
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);
    auto& root = builder.subject().root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("M", sym_desc, true);
    builder.add_container("P", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);
    builder.add_container("k", sym_desc);

    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");
    auto k = symbolic::symbol("k");

    auto& outer = builder.add_map(
        root,
        i,
        symbolic::Lt(i, symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(i, symbolic::integer(2)), // stride 2
        structured_control_flow::ScheduleType_Sequential::create()
    );

    auto& map_j = builder.add_map(
        outer.root(),
        j,
        symbolic::Lt(j, symbolic::symbol("M")),
        symbolic::integer(0),
        symbolic::add(j, symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    builder.add_block(map_j.root());

    auto& map_k = builder.add_map(
        outer.root(),
        k,
        symbolic::Lt(k, symbolic::symbol("P")),
        symbolic::integer(0),
        symbolic::add(k, symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    builder.add_block(map_k.root());

    analysis::AnalysisManager am(builder.subject());
    transformations::MapCollapse t(outer, 2);
    EXPECT_FALSE(t.can_be_applied(builder, am));
}

// ---------------------------------------------------------------------------
// Imperfect — Apply structural checks (two sibling maps)
// ---------------------------------------------------------------------------

TEST(MapCollapseTest, Apply_Imperfect_TwoSiblings_Structure) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);
    auto [outer, map_j, map_k] = build_2d_sibling_maps_nest(builder);

    analysis::AnalysisManager am(builder.subject());
    transformations::MapCollapse t(*outer, 2);
    ASSERT_TRUE(t.can_be_applied(builder, am));
    t.apply(builder, am);

    auto* collapsed = t.collapsed_loop();
    ASSERT_NE(collapsed, nullptr);

    // Root holds exactly the collapsed map.
    EXPECT_EQ(builder.subject().root().size(), 1);
    EXPECT_EQ(&builder.subject().root().at(0), collapsed);

    // Body: recovery block + one guard (IfElse) per original sibling map.
    auto& body = collapsed->root();
    ASSERT_EQ(body.size(), 3);
    EXPECT_TRUE(dyn_cast<structured_control_flow::AssignmentBlock*>(&body.at(0)) != nullptr);

    auto* guard_j = dyn_cast<structured_control_flow::IfElse*>(&body.at(1));
    auto* guard_k = dyn_cast<structured_control_flow::IfElse*>(&body.at(2));
    ASSERT_NE(guard_j, nullptr) << "First sibling must be wrapped in a guard";
    ASSERT_NE(guard_k, nullptr) << "Second sibling must be wrapped in a guard";
    EXPECT_EQ(guard_j->size(), 1) << "Guard must have a single case (no else)";
    EXPECT_EQ(guard_k->size(), 1) << "Guard must have a single case (no else)";
}

TEST(MapCollapseTest, Apply_Imperfect_TwoSiblings_CollapsedRange) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);
    auto [outer, map_j, map_k] = build_2d_sibling_maps_nest(builder);

    analysis::AnalysisManager am(builder.subject());
    transformations::MapCollapse t(*outer, 2);
    ASSERT_TRUE(t.can_be_applied(builder, am));
    t.apply(builder, am);

    auto* collapsed = t.collapsed_loop();
    ASSERT_NE(collapsed, nullptr);

    auto civ = collapsed->indvar();
    auto N = symbolic::symbol("N");
    auto M = symbolic::symbol("M");
    auto P = symbolic::symbol("P");
    auto bmax = symbolic::max(M, P);

    // Collapsed range is [0, N * max(M, P)) with unit stride.
    EXPECT_TRUE(symbolic::eq(collapsed->init(), symbolic::integer(0)));
    EXPECT_TRUE(symbolic::eq(collapsed->condition(), symbolic::Lt(civ, symbolic::mul(N, bmax))))
        << "Collapsed condition must be civ < N*max(M,P)";
    EXPECT_TRUE(symbolic::eq(collapsed->update(), symbolic::add(civ, symbolic::integer(1))));
}

TEST(MapCollapseTest, Apply_Imperfect_TwoSiblings_RecoveryTransition) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);
    auto [outer, map_j, map_k] = build_2d_sibling_maps_nest(builder);

    analysis::AnalysisManager am(builder.subject());
    transformations::MapCollapse t(*outer, 2);
    ASSERT_TRUE(t.can_be_applied(builder, am));
    t.apply(builder, am);

    auto* collapsed = t.collapsed_loop();
    ASSERT_NE(collapsed, nullptr);

    auto civ = collapsed->indvar();
    auto i = symbolic::symbol("i");
    auto M = symbolic::symbol("M");
    auto P = symbolic::symbol("P");
    auto bmax = symbolic::max(M, P);

    // The recovery transition (after the recovery block) defines exactly the
    // outer index i and the virtual inner index t.
    const auto transition = dyn_cast<AssignmentBlock*>(&collapsed->root().at(0));
    ASSERT_TRUE(transition);
    const auto& assignments = transition->assignments();
    EXPECT_EQ(assignments.size(), 2u) << "Recovery must assign exactly i and t";

    // i = civ / max(M, P)
    ASSERT_TRUE(assignments.count(i)) << "'i' must be recovered";
    EXPECT_TRUE(symbolic::eq(assignments.at(i), symbolic::div(civ, bmax))) << "Expected i = civ / max(M,P)";

    // t = civ % max(M, P)
    auto t_idx = find_inner_index(assignments, civ, bmax);
    ASSERT_FALSE(t_idx.is_null()) << "Virtual inner index t = civ % max(M,P) must be recovered";
}

TEST(MapCollapseTest, Apply_Imperfect_TwoSiblings_GuardConditions) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);
    auto [outer, map_j, map_k] = build_2d_sibling_maps_nest(builder);

    analysis::AnalysisManager am(builder.subject());
    transformations::MapCollapse t(*outer, 2);
    ASSERT_TRUE(t.can_be_applied(builder, am));
    t.apply(builder, am);

    auto* collapsed = t.collapsed_loop();
    ASSERT_NE(collapsed, nullptr);

    auto civ = collapsed->indvar();
    auto M = symbolic::symbol("M");
    auto P = symbolic::symbol("P");
    auto bmax = symbolic::max(M, P);

    const auto transition = dyn_cast<AssignmentBlock*>(&collapsed->root().at(0));
    ASSERT_TRUE(transition);
    const auto& assignments = transition->assignments();
    auto t_idx = find_inner_index(assignments, civ, bmax);
    ASSERT_FALSE(t_idx.is_null());

    auto& body = collapsed->root();
    auto* guard_j = dyn_cast<structured_control_flow::IfElse*>(&body.at(1));
    auto* guard_k = dyn_cast<structured_control_flow::IfElse*>(&body.at(2));
    ASSERT_NE(guard_j, nullptr);
    ASSERT_NE(guard_k, nullptr);

    // Each sibling map runs only for valid inner indices: t < bound_i.
    EXPECT_TRUE(symbolic::eq(guard_j->at(0).second, symbolic::Lt(t_idx, M))) << "Expected guard t < M";
    EXPECT_TRUE(symbolic::eq(guard_k->at(0).second, symbolic::Lt(t_idx, P))) << "Expected guard t < P";
}

TEST(MapCollapseTest, Apply_Imperfect_TwoSiblings_InnerIndexRecovery) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);
    auto [outer, map_j, map_k] = build_2d_sibling_maps_nest(builder);

    analysis::AnalysisManager am(builder.subject());
    transformations::MapCollapse t(*outer, 2);
    ASSERT_TRUE(t.can_be_applied(builder, am));
    t.apply(builder, am);

    auto* collapsed = t.collapsed_loop();
    ASSERT_NE(collapsed, nullptr);

    auto civ = collapsed->indvar();
    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");
    auto k = symbolic::symbol("k");
    auto M = symbolic::symbol("M");
    auto P = symbolic::symbol("P");
    auto bmax = symbolic::max(M, P);

    const auto transition = dyn_cast<AssignmentBlock*>(&collapsed->root().at(0));
    ASSERT_TRUE(transition);
    const auto& recovery = transition->assignments();
    auto t_idx = find_inner_index(recovery, civ, bmax);
    ASSERT_FALSE(t_idx.is_null());

    auto& body = collapsed->root();
    auto* guard_j = dyn_cast<structured_control_flow::IfElse*>(&body.at(1));
    auto* guard_k = dyn_cast<structured_control_flow::IfElse*>(&body.at(2));
    ASSERT_NE(guard_j, nullptr);
    ASSERT_NE(guard_k, nullptr);

    // Inside each guard, the original inner index is recovered from t.
    const auto& case_j = guard_j->at(0).first;
    const auto& case_k = guard_k->at(0).first;
    ASSERT_GE(case_j.size(), 1u);
    ASSERT_GE(case_k.size(), 1u);

    const auto j_asgn_block = dyn_cast<AssignmentBlock*>(&case_j.at(0));
    ASSERT_TRUE(j_asgn_block);
    const auto& j_asgn = j_asgn_block->assignments();
    const auto k_asgn_block = dyn_cast<AssignmentBlock*>(&case_k.at(0));
    ASSERT_TRUE(k_asgn_block);
    const auto& k_asgn = k_asgn_block->assignments();
    ASSERT_TRUE(j_asgn.count(j)) << "'j' must be recovered inside its guard";
    ASSERT_TRUE(k_asgn.count(k)) << "'k' must be recovered inside its guard";
    EXPECT_TRUE(symbolic::eq(j_asgn.at(j), t_idx)) << "Expected j = t";
    EXPECT_TRUE(symbolic::eq(k_asgn.at(k), t_idx)) << "Expected k = t";
}

TEST(MapCollapseTest, Apply_Imperfect_ScheduleAndOriginalRemoved) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);
    auto [outer, map_j, map_k] = build_2d_sibling_maps_nest(builder);

    analysis::AnalysisManager am(builder.subject());
    transformations::MapCollapse t(*outer, 2);
    ASSERT_TRUE(t.can_be_applied(builder, am));
    t.apply(builder, am);

    auto* collapsed = t.collapsed_loop();
    ASSERT_NE(collapsed, nullptr);

    // Original outer nest replaced by exactly one collapsed map.
    EXPECT_EQ(builder.subject().root().size(), 1);
    EXPECT_EQ(&builder.subject().root().at(0), collapsed);

    // Collapsed map inherits the outer map's schedule type.
    EXPECT_EQ(collapsed->schedule_type().value(), structured_control_flow::ScheduleType_Sequential::value());
}

// ---------------------------------------------------------------------------
// Imperfect — Apply with skipped elements
// ---------------------------------------------------------------------------

TEST(MapCollapseTest, Apply_Imperfect_SkippedBlock_Replicated) {
    // Outer body: skipped block A, then collapsible map j in [0, M).
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);
    auto [outer, map_j, block_a] = build_imperfect_block_then_map(builder);

    dump_sdfg(builder.subject(), "0.init");

    analysis::AnalysisManager am(builder.subject());
    transformations::MapCollapse t(*outer, 2);
    ASSERT_TRUE(t.can_be_applied(builder, am));
    t.apply(builder, am);

    dump_sdfg(builder.subject(), "1.collapsed");

    auto* collapsed = t.collapsed_loop();
    ASSERT_NE(collapsed, nullptr);

    auto civ = collapsed->indvar();
    auto M = symbolic::symbol("M");

    // Collapsed range [0, N*M): only one collapsible map → max-bound is M.
    auto N = symbolic::symbol("N");
    EXPECT_TRUE(symbolic::eq(collapsed->condition(), symbolic::Lt(civ, symbolic::mul(N, M))));

    const auto transition = dyn_cast<AssignmentBlock*>(&collapsed->root().at(0));
    ASSERT_TRUE(transition);
    const auto& recovery = transition->assignments();
    auto t_idx = find_inner_index(recovery, civ, M);
    ASSERT_FALSE(t_idx.is_null());

    // Body: recovery block, replicated block A (unguarded), guard for inner map j.
    auto& body = collapsed->root();
    ASSERT_EQ(body.size(), 3);
    EXPECT_TRUE(dyn_cast<structured_control_flow::AssignmentBlock*>(&body.at(0)) != nullptr);

    // The skipped block is replicated: it stays a plain Block, not wrapped in a
    // guard, so it runs on every inner thread.
    auto* replicated_a = dyn_cast<structured_control_flow::Block*>(&body.at(1));
    EXPECT_NE(replicated_a, nullptr) << "Skipped block must be replicated as a direct child (no guard)";
    EXPECT_TRUE(dyn_cast<structured_control_flow::IfElse*>(&body.at(1)) == nullptr)
        << "Skipped block must NOT be wrapped in an inner==0 guard";

    auto* guard_j = dyn_cast<structured_control_flow::IfElse*>(&body.at(2));
    ASSERT_NE(guard_j, nullptr);
    // Inner map guarded by t < M.
    EXPECT_TRUE(symbolic::eq(guard_j->at(0).second, symbolic::Lt(t_idx, M))) << "Inner map must be guarded by t < M";
}

TEST(MapCollapseTest, Apply_Imperfect_Assignments_Are_LikeAnyOtherNode) {
    // The transition attached to the inner map (inside outer.root()) carries an
    // assignment, violating the "empty transitions in holding sequence" criterion.
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);
    auto& root = builder.subject().root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("M", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);
    builder.add_container("rando", sym_desc);

    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");
    auto rando = symbolic::symbol("rando");

    auto& outer = builder.add_map(
        root,
        i,
        symbolic::Lt(i, symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(i, symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );

    // The inner map has a non-empty transition in outer.root()
    auto& inner = builder.add_map(
        outer.root(),
        j,
        symbolic::Lt(j, symbolic::symbol("M")),
        symbolic::integer(0),
        symbolic::add(j, symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    builder.add_block(inner.root());
    auto& rogue_node = builder.add_assignments(outer.root(), {{rando, symbolic::integer(0)}}); // non-empty transition
                                                                                               // after the inner map

    dump_sdfg(builder.subject(), "0.init");

    analysis::AnalysisManager am(builder.subject());
    transformations::MapCollapse t(outer, 2);
    EXPECT_TRUE(t.can_be_applied(builder, am));
    t.apply(builder, am);

    dump_sdfg(builder.subject(), "1.collapsed");

    auto* collapsed = t.collapsed_loop();
    ASSERT_NE(collapsed, nullptr);

    auto civ = collapsed->indvar();
    auto M = symbolic::symbol("M");

    // Collapsed range [0, N*M): only one collapsible map → max-bound is M.
    auto N = symbolic::symbol("N");
    EXPECT_TRUE(symbolic::eq(collapsed->condition(), symbolic::Lt(civ, symbolic::mul(N, M))));

    const auto transition = dyn_cast<AssignmentBlock*>(&collapsed->root().at(0));
    ASSERT_TRUE(transition);
    const auto& recovery = transition->assignments();
    auto t_idx = find_inner_index(recovery, civ, M);
    ASSERT_FALSE(t_idx.is_null());

    // Body: recovery block, replicated block A (unguarded), guard for inner map j.
    auto& body = collapsed->root();
    ASSERT_EQ(body.size(), 3);
    EXPECT_TRUE(dyn_cast<structured_control_flow::AssignmentBlock*>(&body.at(0)) != nullptr);

    auto* guard_j = dyn_cast<structured_control_flow::IfElse*>(&body.at(1));
    ASSERT_NE(guard_j, nullptr);
    // Inner map guarded by t < M.
    EXPECT_TRUE(symbolic::eq(guard_j->at(0).second, symbolic::Lt(t_idx, M))) << "Inner map must be guarded by t < M";

    // The skipped block is replicated: it stays a plain Block, not wrapped in a
    // guard, so it runs on every inner thread.
    auto* replicated_a = dyn_cast<structured_control_flow::AssignmentBlock*>(&body.at(2));
    EXPECT_NE(replicated_a, nullptr) << "Skipped block must be replicated as a direct child (no guard)";
}

TEST(MapCollapseTest, Apply_Imperfect_PreservesOrder) {
    // Outer body order: map j in [0,M), skipped block A, map k in [0,P).
    // After collapse the elements must appear in the same order, with the skipped
    // block replicated (a plain Block) between the two map guards.
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);
    auto& root = builder.subject().root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("M", sym_desc, true);
    builder.add_container("P", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);
    builder.add_container("k", sym_desc);

    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");
    auto k = symbolic::symbol("k");
    auto M = symbolic::symbol("M");
    auto P = symbolic::symbol("P");

    auto& outer = builder.add_map(
        root,
        i,
        symbolic::Lt(i, symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(i, symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );

    auto& map_j = builder.add_map(
        outer.root(),
        j,
        symbolic::Lt(j, M),
        symbolic::integer(0),
        symbolic::add(j, symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    builder.add_block(map_j.root());

    builder.add_block(outer.root()); // skipped block A between the maps

    auto& map_k = builder.add_map(
        outer.root(),
        k,
        symbolic::Lt(k, P),
        symbolic::integer(0),
        symbolic::add(k, symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    builder.add_block(map_k.root());

    analysis::AnalysisManager am(builder.subject());
    transformations::MapCollapse t(outer, 2);
    ASSERT_TRUE(t.can_be_applied(builder, am));
    t.apply(builder, am);

    auto* collapsed = t.collapsed_loop();
    ASSERT_NE(collapsed, nullptr);

    auto civ = collapsed->indvar();
    auto bmax = symbolic::max(M, P);

    const auto transition = dyn_cast<AssignmentBlock*>(&collapsed->root().at(0));
    ASSERT_TRUE(transition);
    const auto& recovery = transition->assignments();
    auto t_idx = find_inner_index(recovery, civ, bmax);
    ASSERT_FALSE(t_idx.is_null());

    // Body: recovery block + j (t<M) guard, replicated block A, k (t<P) guard.
    auto& body = collapsed->root();
    ASSERT_EQ(body.size(), 4);

    auto* guard_j = dyn_cast<structured_control_flow::IfElse*>(&body.at(1));
    auto* block_a = dyn_cast<structured_control_flow::Block*>(&body.at(2));
    auto* guard_k = dyn_cast<structured_control_flow::IfElse*>(&body.at(3));
    ASSERT_NE(guard_j, nullptr);
    ASSERT_NE(block_a, nullptr) << "Skipped block must be replicated as a plain Block in original order";
    EXPECT_TRUE(dyn_cast<structured_control_flow::IfElse*>(&body.at(2)) == nullptr)
        << "Skipped block must NOT be wrapped in an inner==0 guard";
    ASSERT_NE(guard_k, nullptr);

    EXPECT_TRUE(symbolic::eq(guard_j->at(0).second, symbolic::Lt(t_idx, M))) << "First guard must be t < M";
    EXPECT_TRUE(symbolic::eq(guard_k->at(0).second, symbolic::Lt(t_idx, P))) << "Third element guard must be t < P";
}

// ---------------------------------------------------------------------------
// Imperfect — data-dependency safety (replication model)
// ---------------------------------------------------------------------------

/// Build outer map i in [0,N) with body:
///   block A (skipped):     X[i]        = Y[i]
///   map j in [0,M):        Z[i*M + j]  = X[i]
/// i.e. a skipped producer of X feeding the collapsible inner map (RAW on X).
/// Returns {outer, map_j, block_a}.
static std::tuple<structured_control_flow::Map*, structured_control_flow::Map*, structured_control_flow::Block*>
build_imperfect_producer_then_map(builder::StructuredSDFGBuilder& builder) {
    auto& root = builder.subject().root();

    types::Scalar float_desc(types::PrimitiveType::Float);
    types::Pointer ptr_desc(float_desc);
    types::Pointer opaque_desc;
    types::Scalar sym_desc(types::PrimitiveType::UInt64);

    builder.add_container("X", opaque_desc, true);
    builder.add_container("Y", opaque_desc, true);
    builder.add_container("Z", opaque_desc, true);
    builder.add_container("N", sym_desc, true);
    builder.add_container("M", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);

    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");
    auto N = symbolic::symbol("N");
    auto M = symbolic::symbol("M");

    auto& outer = builder.add_map(
        root,
        i,
        symbolic::Lt(i, N),
        symbolic::integer(0),
        symbolic::add(i, symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );

    // Producer block A: X[i] = Y[i]
    auto& block_a = builder.add_block(outer.root());
    {
        auto& y_in = builder.add_access(block_a, "Y");
        auto& x_out = builder.add_access(block_a, "X");
        auto& tk = builder.add_tasklet(block_a, data_flow::TaskletCode::assign, "_out", {"_in"});
        builder.add_computational_memlet(block_a, y_in, tk, "_in", {i}, ptr_desc);
        builder.add_computational_memlet(block_a, tk, "_out", x_out, {i}, ptr_desc);
    }

    // Collapsible inner map j in [0, M): Z[i*M + j] = X[i]
    auto& map_j = builder.add_map(
        outer.root(),
        j,
        symbolic::Lt(j, M),
        symbolic::integer(0),
        symbolic::add(j, symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    {
        auto& block_b = builder.add_block(map_j.root());
        auto& x_in = builder.add_access(block_b, "X");
        auto& z_out = builder.add_access(block_b, "Z");
        auto& tk = builder.add_tasklet(block_b, data_flow::TaskletCode::assign, "_out", {"_in"});
        builder.add_computational_memlet(block_b, x_in, tk, "_in", {i}, ptr_desc);
        builder.add_computational_memlet(block_b, tk, "_out", z_out, {symbolic::add(symbolic::mul(i, M), j)}, ptr_desc);
    }

    return {&outer, &map_j, &block_a};
}

TEST(MapCollapseTest, CanBeApplied_Imperfect_ProducerConsumedByInnerMap) {
    // A skipped producer block writes X[i]; the collapsible inner map reads X[i].
    // Under replication each inner thread re-runs the producer (it cannot depend on
    // the inner index), so this RAW dependency is safe and the collapse is allowed.
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);
    auto [outer, map_j, block_a] = build_imperfect_producer_then_map(builder);

    analysis::AnalysisManager am(builder.subject());
    transformations::MapCollapse t(*outer, 2);
    EXPECT_TRUE(t.can_be_applied(builder, am));
}

TEST(MapCollapseTest, Apply_Imperfect_Producer_Replicated_NotGuarded) {
    // The producer must be replicated: it stays a plain Block (no inner==0 guard)
    // as a direct child of the collapsed body, ahead of the inner-map guard.
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);
    auto [outer, map_j, block_a] = build_imperfect_producer_then_map(builder);

    analysis::AnalysisManager am(builder.subject());
    transformations::MapCollapse t(*outer, 2);
    ASSERT_TRUE(t.can_be_applied(builder, am));
    t.apply(builder, am);

    auto* collapsed = t.collapsed_loop();
    ASSERT_NE(collapsed, nullptr);

    auto civ = collapsed->indvar();
    auto M = symbolic::symbol("M");

    const auto transition = dyn_cast<AssignmentBlock*>(&collapsed->root().at(0));
    ASSERT_TRUE(transition);
    const auto& recovery = transition->assignments();
    auto t_idx = find_inner_index(recovery, civ, M);
    ASSERT_FALSE(t_idx.is_null());

    // Body: recovery block, replicated producer block (unguarded), inner-map guard.
    auto& body = collapsed->root();
    ASSERT_EQ(body.size(), 3);
    EXPECT_TRUE(dyn_cast<structured_control_flow::AssignmentBlock*>(&body.at(0)) != nullptr);

    auto* producer = dyn_cast<structured_control_flow::Block*>(&body.at(1));
    ASSERT_NE(producer, nullptr) << "Producer must be replicated as a direct child block";
    EXPECT_TRUE(dyn_cast<structured_control_flow::IfElse*>(&body.at(1)) == nullptr)
        << "Producer must NOT be wrapped in an inner==0 guard";

    // The producer keeps its tasklet (X[i] = Y[i]) intact.
    EXPECT_EQ(producer->dataflow().tasklets().size(), 1u);

    auto* guard_j = dyn_cast<structured_control_flow::IfElse*>(&body.at(2));
    ASSERT_NE(guard_j, nullptr);
    EXPECT_TRUE(symbolic::eq(guard_j->at(0).second, symbolic::Lt(t_idx, M)));
}

TEST(MapCollapseTest, CannotApply_Imperfect_CollapsibleMapOutputConsumed) {
    // The collapsible inner map writes X (inner-varying), and a later skipped block
    // reads X. After flattening the consumer would read another thread's data
    // without synchronization, which replication cannot fix → must be rejected.
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);
    auto& root = builder.subject().root();

    types::Scalar float_desc(types::PrimitiveType::Float);
    types::Pointer ptr_desc(float_desc);
    types::Pointer opaque_desc;
    types::Scalar sym_desc(types::PrimitiveType::UInt64);

    builder.add_container("X", opaque_desc, true);
    builder.add_container("Y", opaque_desc, true);
    builder.add_container("Z", opaque_desc, true);
    builder.add_container("N", sym_desc, true);
    builder.add_container("M", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);

    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");
    auto N = symbolic::symbol("N");
    auto M = symbolic::symbol("M");

    auto& outer = builder.add_map(
        root,
        i,
        symbolic::Lt(i, N),
        symbolic::integer(0),
        symbolic::add(i, symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );

    // Collapsible inner map j in [0, M): X[i*M + j] = Y[i*M + j]  (writes X)
    auto& map_j = builder.add_map(
        outer.root(),
        j,
        symbolic::Lt(j, M),
        symbolic::integer(0),
        symbolic::add(j, symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    {
        auto idx = symbolic::add(symbolic::mul(i, M), j);
        auto& block_j = builder.add_block(map_j.root());
        auto& y_in = builder.add_access(block_j, "Y");
        auto& x_out = builder.add_access(block_j, "X");
        auto& tk = builder.add_tasklet(block_j, data_flow::TaskletCode::assign, "_out", {"_in"});
        builder.add_computational_memlet(block_j, y_in, tk, "_in", {idx}, ptr_desc);
        builder.add_computational_memlet(block_j, tk, "_out", x_out, {idx}, ptr_desc);
    }

    // Skipped block B consuming X: Z[i] = X[i]
    auto& block_b = builder.add_block(outer.root());
    {
        auto& x_in = builder.add_access(block_b, "X");
        auto& z_out = builder.add_access(block_b, "Z");
        auto& tk = builder.add_tasklet(block_b, data_flow::TaskletCode::assign, "_out", {"_in"});
        builder.add_computational_memlet(block_b, x_in, tk, "_in", {i}, ptr_desc);
        builder.add_computational_memlet(block_b, tk, "_out", z_out, {i}, ptr_desc);
    }

    analysis::AnalysisManager am(builder.subject());
    transformations::MapCollapse t(outer, 2);
    EXPECT_FALSE(t.can_be_applied(builder, am));
}

TEST(MapCollapseTest, CannotApply_Imperfect_WriteWriteConflict) {
    // Two skipped blocks write the same container X. Even though each is uniform
    // across the inner index, a cross-element write-write conflict has no
    // guaranteed ordering after flattening → conservatively rejected.
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);
    auto& root = builder.subject().root();

    types::Scalar float_desc(types::PrimitiveType::Float);
    types::Pointer ptr_desc(float_desc);
    types::Pointer opaque_desc;
    types::Scalar sym_desc(types::PrimitiveType::UInt64);

    builder.add_container("X", opaque_desc, true);
    builder.add_container("Y", opaque_desc, true);
    builder.add_container("N", sym_desc, true);
    builder.add_container("M", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);

    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");
    auto N = symbolic::symbol("N");
    auto M = symbolic::symbol("M");

    auto& outer = builder.add_map(
        root,
        i,
        symbolic::Lt(i, N),
        symbolic::integer(0),
        symbolic::add(i, symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );

    // Skipped block A: X[i] = Y[i]
    auto& block_a = builder.add_block(outer.root());
    {
        auto& y_in = builder.add_access(block_a, "Y");
        auto& x_out = builder.add_access(block_a, "X");
        auto& tk = builder.add_tasklet(block_a, data_flow::TaskletCode::assign, "_out", {"_in"});
        builder.add_computational_memlet(block_a, y_in, tk, "_in", {i}, ptr_desc);
        builder.add_computational_memlet(block_a, tk, "_out", x_out, {i}, ptr_desc);
    }

    // Collapsible inner map j in [0, M): empty body (no data conflict)
    auto& map_j = builder.add_map(
        outer.root(),
        j,
        symbolic::Lt(j, M),
        symbolic::integer(0),
        symbolic::add(j, symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    builder.add_block(map_j.root());

    // Skipped block C: X[i] = Y[i]  (second writer of X → WAW with block A)
    auto& block_c = builder.add_block(outer.root());
    {
        auto& y_in = builder.add_access(block_c, "Y");
        auto& x_out = builder.add_access(block_c, "X");
        auto& tk = builder.add_tasklet(block_c, data_flow::TaskletCode::assign, "_out", {"_in"});
        builder.add_computational_memlet(block_c, y_in, tk, "_in", {i}, ptr_desc);
        builder.add_computational_memlet(block_c, tk, "_out", x_out, {i}, ptr_desc);
    }

    analysis::AnalysisManager am(builder.subject());
    transformations::MapCollapse t(outer, 2);
    EXPECT_FALSE(t.can_be_applied(builder, am));
}

// ---------------------------------------------------------------------------
// Inlining of recovered induction variables (no SymbolPropagation required)
// ---------------------------------------------------------------------------

TEST(MapCollapseTest, Apply_2D_InlinesRecoveredIndvars) {
    // A[i][j] read/write with subset {i, j}. After collapse the subsets must be
    // inlined directly to {civ / M, civ % M}, while the recovery transition is kept
    // as a fallback.
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);
    auto& root = builder.subject().root();

    types::Scalar float_desc(types::PrimitiveType::Float);
    types::Pointer ptr_desc(float_desc);
    types::Pointer opaque_desc;
    types::Scalar sym_desc(types::PrimitiveType::UInt64);

    builder.add_container("A", opaque_desc, true);
    builder.add_container("N", sym_desc, true);
    builder.add_container("M", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);

    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");
    auto N = symbolic::symbol("N");
    auto M = symbolic::symbol("M");

    auto& outer = builder.add_map(
        root,
        i,
        symbolic::Lt(i, N),
        symbolic::integer(0),
        symbolic::add(i, symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    auto& inner = builder.add_map(
        outer.root(),
        j,
        symbolic::Lt(j, M),
        symbolic::integer(0),
        symbolic::add(j, symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );

    // A[i][j] = A[i][j]  (2D subset)
    auto& block = builder.add_block(inner.root());
    auto& A_in = builder.add_access(block, "A");
    auto& A_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, A_in, tasklet, "_in", {i, j}, ptr_desc);
    builder.add_computational_memlet(block, tasklet, "_out", A_out, {i, j}, ptr_desc);

    analysis::AnalysisManager am(builder.subject());
    transformations::MapCollapse t(outer, 2);
    ASSERT_TRUE(t.can_be_applied(builder, am));
    t.apply(builder, am);

    auto* collapsed = t.collapsed_loop();
    ASSERT_NE(collapsed, nullptr);
    auto civ = collapsed->indvar();

    // Fallback: the recovery transition still defines i and j.
    const auto transition = dyn_cast<AssignmentBlock*>(&collapsed->root().at(0));
    ASSERT_TRUE(transition);
    const auto& asgn = transition->assignments();
    ASSERT_TRUE(asgn.count(i));
    ASSERT_TRUE(asgn.count(j));
    EXPECT_TRUE(symbolic::eq(asgn.at(i), symbolic::div(civ, M)));
    EXPECT_TRUE(symbolic::eq(asgn.at(j), symbolic::mod(civ, M)));

    // Subsets are inlined: {i, j} -> {civ / M, civ % M}, no reference to i or j remains.
    auto* body_block = dyn_cast<structured_control_flow::Block*>(&collapsed->root().at(1));
    ASSERT_NE(body_block, nullptr);
    auto* t_node = *body_block->dataflow().tasklets().begin();
    const auto& in_subset = (*body_block->dataflow().in_edges(*t_node).begin()).subset();
    const auto& out_subset = (*body_block->dataflow().out_edges(*t_node).begin()).subset();

    ASSERT_EQ(in_subset.size(), 2u);
    EXPECT_TRUE(symbolic::eq(in_subset[0], symbolic::div(civ, M)));
    EXPECT_TRUE(symbolic::eq(in_subset[1], symbolic::mod(civ, M)));
    EXPECT_FALSE(symbolic::uses(in_subset[0], i));
    EXPECT_FALSE(symbolic::uses(in_subset[1], j));

    ASSERT_EQ(out_subset.size(), 2u);
    EXPECT_TRUE(symbolic::eq(out_subset[0], symbolic::div(civ, M)));
    EXPECT_TRUE(symbolic::eq(out_subset[1], symbolic::mod(civ, M)));
}

TEST(MapCollapseTest, Apply_2D_KeepsTransitionAsFallbackForAccessNodeContainer) {
    // When an induction variable is used as an access-node *container* (data) name it
    // cannot be replaced by a complex expression: AccessNode::replace only renames
    // symbol->symbol. The container reference must therefore be left untouched and the
    // recovery transition kept so the value is still defined. Memlet subsets in the same
    // block are still inlined normally.
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);
    auto& root = builder.subject().root();

    types::Scalar float_desc(types::PrimitiveType::Float);
    types::Pointer ptr_desc(float_desc);
    types::Pointer opaque_desc;
    types::Scalar sym_desc(types::PrimitiveType::UInt64);

    builder.add_container("A", opaque_desc, true);
    builder.add_container("N", sym_desc, true);
    builder.add_container("M", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);

    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");
    auto N = symbolic::symbol("N");
    auto M = symbolic::symbol("M");

    auto& outer = builder.add_map(
        root,
        i,
        symbolic::Lt(i, N),
        symbolic::integer(0),
        symbolic::add(i, symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    auto& inner = builder.add_map(
        outer.root(),
        j,
        symbolic::Lt(j, M),
        symbolic::integer(0),
        symbolic::add(j, symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );

    // A[j] = i  — reads the induction variable 'i' as a *container* (data), writes A[j].
    auto& block = builder.add_block(inner.root());
    auto& i_in = builder.add_access(block, "i");
    auto& A_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, i_in, tasklet, "_in", {}, sym_desc);
    builder.add_computational_memlet(block, tasklet, "_out", A_out, {j}, ptr_desc);

    analysis::AnalysisManager am(builder.subject());
    transformations::MapCollapse t(outer, 2);
    ASSERT_TRUE(t.can_be_applied(builder, am));
    t.apply(builder, am);

    auto* collapsed = t.collapsed_loop();
    ASSERT_NE(collapsed, nullptr);
    auto civ = collapsed->indvar();

    // Fallback retained: recovery transition still defines i (needed by the access node).
    const auto transition = dyn_cast<AssignmentBlock*>(&collapsed->root().at(0));
    ASSERT_TRUE(transition);
    const auto& asgn = transition->assignments();
    ASSERT_TRUE(asgn.count(i));
    EXPECT_TRUE(symbolic::eq(asgn.at(i), symbolic::div(civ, M)));

    auto* body_block = dyn_cast<structured_control_flow::Block*>(&collapsed->root().at(1));
    ASSERT_NE(body_block, nullptr);
    auto* t_node = *body_block->dataflow().tasklets().begin();

    // The access-node container is NOT rewritten (still "i") — complex expressions cannot
    // be a container name.
    auto& in_src = (*body_block->dataflow().in_edges(*t_node).begin()).src();
    auto* in_access = dyn_cast<data_flow::AccessNode>(&in_src);
    ASSERT_NE(in_access, nullptr);
    EXPECT_EQ(in_access->data(), "i") << "Induction variable used as a container must not be inlined";

    // The ordinary subset {j} is still inlined to civ % M.
    const auto& out_subset = (*body_block->dataflow().out_edges(*t_node).begin()).subset();
    ASSERT_EQ(out_subset.size(), 1u);
    EXPECT_TRUE(symbolic::eq(out_subset[0], symbolic::mod(civ, M)));
}

TEST(MapCollapseTest, Apply_Imperfect_InlinesRecoveredIndvars) {
    // Imperfect (CUDA-style) collapse: outer map i with a skipped block referencing i and a
    // collapsible inner map j whose body references i and j. After collapse the subsets must
    // be inlined to the collapsed index (i -> civ / M, j -> civ % M) without SymbolPropagation.
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);
    auto& root = builder.subject().root();

    types::Scalar float_desc(types::PrimitiveType::Float);
    types::Pointer ptr_desc(float_desc);
    types::Pointer opaque_desc;
    types::Scalar sym_desc(types::PrimitiveType::UInt64);

    builder.add_container("A", opaque_desc, true);
    builder.add_container("B", opaque_desc, true);
    builder.add_container("N", sym_desc, true);
    builder.add_container("M", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);

    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");
    auto N = symbolic::symbol("N");
    auto M = symbolic::symbol("M");

    auto& outer = builder.add_map(
        root,
        i,
        symbolic::Lt(i, N),
        symbolic::integer(0),
        symbolic::add(i, symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );

    // Skipped block: B[i] = B[i]  (replicated on every inner thread; references i only).
    auto& skipped = builder.add_block(outer.root());
    {
        auto& b_in = builder.add_access(skipped, "B");
        auto& b_out = builder.add_access(skipped, "B");
        auto& tk = builder.add_tasklet(skipped, data_flow::TaskletCode::assign, "_out", {"_in"});
        builder.add_computational_memlet(skipped, b_in, tk, "_in", {i}, ptr_desc);
        builder.add_computational_memlet(skipped, tk, "_out", b_out, {i}, ptr_desc);
    }

    // Collapsible inner map j: A[i*M + j] = A[i*M + j].
    auto& map_j = builder.add_map(
        outer.root(),
        j,
        symbolic::Lt(j, M),
        symbolic::integer(0),
        symbolic::add(j, symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    auto index_expr = symbolic::add(symbolic::mul(i, M), j);
    auto& inner_block = builder.add_block(map_j.root());
    auto& a_in = builder.add_access(inner_block, "A");
    auto& a_out = builder.add_access(inner_block, "A");
    auto& inner_tk = builder.add_tasklet(inner_block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(inner_block, a_in, inner_tk, "_in", {index_expr}, ptr_desc);
    builder.add_computational_memlet(inner_block, inner_tk, "_out", a_out, {index_expr}, ptr_desc);

    analysis::AnalysisManager am(builder.subject());
    transformations::MapCollapse t(outer, 2);
    ASSERT_TRUE(t.can_be_applied(builder, am));
    t.apply(builder, am);

    auto* collapsed = t.collapsed_loop();
    ASSERT_NE(collapsed, nullptr);
    auto civ = collapsed->indvar();
    // For a single collapsible inner map the inner extent is its bound M.
    auto expected_i = symbolic::div(civ, M);
    auto expected_j = symbolic::mod(civ, M);

    // Fallback: outer recovery transition still defines i.
    const auto transition = dyn_cast<AssignmentBlock*>(&collapsed->root().at(0));
    ASSERT_TRUE(transition);
    const auto& outer_asgn = transition->assignments();
    ASSERT_TRUE(outer_asgn.count(i));
    EXPECT_TRUE(symbolic::eq(outer_asgn.at(i), expected_i));

    // Skipped block (index 1): B[i] -> B[civ / M].
    auto* skipped_block = dyn_cast<structured_control_flow::Block*>(&collapsed->root().at(1));
    ASSERT_NE(skipped_block, nullptr);
    auto* skipped_tk = *skipped_block->dataflow().tasklets().begin();
    const auto& skipped_subset = (*skipped_block->dataflow().in_edges(*skipped_tk).begin()).subset();
    ASSERT_EQ(skipped_subset.size(), 1u);
    EXPECT_TRUE(symbolic::eq(skipped_subset[0], expected_i));
    EXPECT_FALSE(symbolic::uses(skipped_subset[0], i));

    // Guarded inner map (index 2): A[i*M + j] -> A[(civ/M)*M + civ%M].
    auto* guard = dyn_cast<structured_control_flow::IfElse*>(&collapsed->root().at(2));
    ASSERT_NE(guard, nullptr);
    auto& branch = guard->at(0).first;
    // Branch: [recovery block (j = t), inner block].
    auto* guarded_block = dyn_cast<structured_control_flow::Block*>(&branch.at(1));
    ASSERT_NE(guarded_block, nullptr);
    auto* guarded_tk = *guarded_block->dataflow().tasklets().begin();
    const auto& guarded_subset = (*guarded_block->dataflow().in_edges(*guarded_tk).begin()).subset();

    symbolic::ExpressionMapping recovery;
    recovery[i] = expected_i;
    recovery[j] = expected_j;
    auto expected_index = symbolic::subs(index_expr, recovery);

    ASSERT_EQ(guarded_subset.size(), 1u);
    EXPECT_TRUE(symbolic::eq(guarded_subset[0], expected_index));
    EXPECT_FALSE(symbolic::uses(guarded_subset[0], i));
    EXPECT_FALSE(symbolic::uses(guarded_subset[0], j));
}
