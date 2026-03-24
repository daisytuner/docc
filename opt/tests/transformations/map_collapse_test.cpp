#include "sdfg/transformations/map_collapse.h"

#include <gtest/gtest.h>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/structured_control_flow/map.h"

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

TEST(MapCollapseTest, CannotApply_NotPerfectlyNested) {
    // The outer map body contains two children: an extra block + the inner map.
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
    EXPECT_FALSE(t.can_be_applied(builder, am));
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

TEST(MapCollapseTest, CannotApply_NonEmptyTransitionToInnerMap) {
    // The transition attached to the inner map (inside outer.root()) carries an
    // assignment, violating the "empty transitions in holding sequence" criterion.
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

    // The inner map has a non-empty transition in outer.root()
    auto& inner = builder.add_map(
        outer.root(),
        j,
        symbolic::Lt(j, symbolic::symbol("M")),
        symbolic::integer(0),
        symbolic::add(j, symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create(),
        {{i, symbolic::integer(0)}} // non-empty transition on the inner map
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
    EXPECT_EQ(&builder.subject().root().at(0).first, collapsed);

    // The collapsed body must contain an empty recovery block + the original inner block
    auto& body = collapsed->root();
    EXPECT_EQ(body.size(), 2);
    EXPECT_TRUE(dynamic_cast<structured_control_flow::Block*>(&body.at(0).first) != nullptr);
    EXPECT_TRUE(dynamic_cast<structured_control_flow::Block*>(&body.at(1).first) != nullptr);
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
    const auto& transition = collapsed->root().at(0).second;
    const auto& assignments = transition.assignments();

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
    const auto& assignments = collapsed->root().at(0).second.assignments();

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
    EXPECT_EQ(&builder.subject().root().at(0).first, collapsed);

    // collapsed body: empty recovery block + the surviving k map
    auto& body = collapsed->root();
    EXPECT_EQ(body.size(), 2);
    EXPECT_TRUE(dynamic_cast<structured_control_flow::Block*>(&body.at(0).first) != nullptr);
    auto* k_map = dynamic_cast<structured_control_flow::Map*>(&body.at(1).first);
    ASSERT_NE(k_map, nullptr) << "Inner k map must survive as direct child of collapsed loop";

    // k map body still contains exactly the original block
    EXPECT_EQ(k_map->root().size(), 1);
    EXPECT_TRUE(dynamic_cast<structured_control_flow::Block*>(&k_map->root().at(0).first) != nullptr);
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

    const auto& assignments = collapsed->root().at(0).second.assignments();

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
    auto* k_map = dynamic_cast<structured_control_flow::Map*>(&collapsed->root().at(1).first);
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
    auto* i_map = dynamic_cast<structured_control_flow::Map*>(&builder.subject().root().at(0).first);
    ASSERT_NE(i_map, nullptr) << "Outer i map must still be the root child";

    // Outer i map body: exactly the collapsed jk map
    EXPECT_EQ(i_map->root().size(), 1);
    auto* jk_map = dynamic_cast<structured_control_flow::Map*>(&i_map->root().at(0).first);
    ASSERT_NE(jk_map, nullptr) << "Collapsed jk map must be inside the outer i map";
    EXPECT_EQ(jk_map, collapsed);

    // Collapsed body: empty recovery block + the original block
    EXPECT_EQ(collapsed->root().size(), 2);
    EXPECT_TRUE(dynamic_cast<structured_control_flow::Block*>(&collapsed->root().at(0).first) != nullptr);
    EXPECT_TRUE(dynamic_cast<structured_control_flow::Block*>(&collapsed->root().at(1).first) != nullptr);
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

    const auto& assignments = collapsed->root().at(0).second.assignments();

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
    auto* i_map = dynamic_cast<structured_control_flow::Map*>(&builder.subject().root().at(0).first);
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
    EXPECT_EQ(&builder.subject().root().at(0).first, collapsed_ij);

    auto civ_ij = collapsed_ij->indvar();

    // collapsed_ij range: [0, N*M)
    EXPECT_TRUE(symbolic::eq(collapsed_ij->init(), symbolic::integer(0)));
    EXPECT_TRUE(symbolic::eq(collapsed_ij->condition(), symbolic::Lt(civ_ij, symbolic::mul(N, M))));
    EXPECT_TRUE(symbolic::eq(collapsed_ij->update(), symbolic::add(civ_ij, symbolic::integer(1))));

    // collapsed_ij body: empty recovery block + the k map
    EXPECT_EQ(collapsed_ij->root().size(), 2);
    EXPECT_TRUE(dynamic_cast<structured_control_flow::Block*>(&collapsed_ij->root().at(0).first) != nullptr);
    auto* surviving_k = dynamic_cast<structured_control_flow::Map*>(&collapsed_ij->root().at(1).first);
    ASSERT_NE(surviving_k, nullptr) << "k map must survive as child of collapsed_ij";

    // Indvar recovery for i, j (on the empty recovery block's transition)
    {
        const auto& asgn = collapsed_ij->root().at(0).second.assignments();
        ASSERT_TRUE(asgn.count(i)) << "'i' must be assigned";
        ASSERT_TRUE(asgn.count(j)) << "'j' must be assigned";
        EXPECT_FALSE(asgn.count(k)) << "'k' must NOT be assigned here";
        EXPECT_FALSE(asgn.count(l)) << "'l' must NOT be assigned here";

        EXPECT_TRUE(symbolic::eq(asgn.at(i), symbolic::div(civ_ij, M)));
        EXPECT_TRUE(symbolic::eq(asgn.at(j), symbolic::mod(civ_ij, M)));
    }

    // k map body: l map
    EXPECT_EQ(surviving_k->root().size(), 1);
    auto* surviving_l = dynamic_cast<structured_control_flow::Map*>(&surviving_k->root().at(0).first);
    ASSERT_NE(surviving_l, nullptr) << "l map must survive inside k map";

    // l map body: the original block
    EXPECT_EQ(surviving_l->root().size(), 1);
    EXPECT_TRUE(dynamic_cast<structured_control_flow::Block*>(&surviving_l->root().at(0).first) != nullptr);

    // --- Second collapse: (k, l) ---
    transformations::MapCollapse t2(*surviving_k, 2);
    ASSERT_TRUE(t2.can_be_applied(builder, am));
    t2.apply(builder, am);

    auto* collapsed_kl = t2.collapsed_loop();
    ASSERT_NE(collapsed_kl, nullptr);

    auto civ_kl = collapsed_kl->indvar();

    // Final structure: root → collapsed_ij → collapsed_kl → block
    EXPECT_EQ(builder.subject().root().size(), 1);

    auto* final_outer = dynamic_cast<structured_control_flow::Map*>(&builder.subject().root().at(0).first);
    ASSERT_NE(final_outer, nullptr);
    // collapsed_ij must still be the root child
    EXPECT_EQ(final_outer, collapsed_ij);

    // collapsed_ij body: empty recovery block + collapsed_kl
    EXPECT_EQ(collapsed_ij->root().size(), 2);
    auto* inner_map = dynamic_cast<structured_control_flow::Map*>(&collapsed_ij->root().at(1).first);
    ASSERT_NE(inner_map, nullptr);
    EXPECT_EQ(inner_map, collapsed_kl);

    // collapsed_kl range: [0, P*Q)
    EXPECT_TRUE(symbolic::eq(collapsed_kl->init(), symbolic::integer(0)));
    EXPECT_TRUE(symbolic::eq(collapsed_kl->condition(), symbolic::Lt(civ_kl, symbolic::mul(P, Q))));
    EXPECT_TRUE(symbolic::eq(collapsed_kl->update(), symbolic::add(civ_kl, symbolic::integer(1))));

    // collapsed_kl body: empty recovery block + the original block
    EXPECT_EQ(collapsed_kl->root().size(), 2);
    EXPECT_TRUE(dynamic_cast<structured_control_flow::Block*>(&collapsed_kl->root().at(1).first) != nullptr);

    // Indvar recovery for k, l (on the empty recovery block's transition)
    {
        const auto& asgn = collapsed_kl->root().at(0).second.assignments();
        ASSERT_TRUE(asgn.count(k)) << "'k' must be assigned";
        ASSERT_TRUE(asgn.count(l)) << "'l' must be assigned";
        EXPECT_FALSE(asgn.count(i)) << "'i' must NOT be assigned here";
        EXPECT_FALSE(asgn.count(j)) << "'j' must NOT be assigned here";

        EXPECT_TRUE(symbolic::eq(asgn.at(k), symbolic::div(civ_kl, Q)));
        EXPECT_TRUE(symbolic::eq(asgn.at(l), symbolic::mod(civ_kl, Q)));
    }

    // Verify that collapsed_ij indvar recovery is still correct after second collapse
    {
        const auto& asgn = collapsed_ij->root().at(0).second.assignments();
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
    auto* body_block = dynamic_cast<structured_control_flow::Block*>(&collapsed->root().at(1).first);
    ASSERT_NE(body_block, nullptr);

    // Transition assignments
    auto civ = collapsed->indvar();
    const auto& asgn = collapsed->root().at(0).second.assignments();
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

    // Verify memlet subsets still reference i*M + j
    EXPECT_EQ((*in_edges.begin()).subset().size(), 1);
    EXPECT_TRUE(symbolic::eq((*in_edges.begin()).subset()[0], index_expr))
        << "Input memlet subset must still be i*M + j";

    EXPECT_EQ((*out_edges.begin()).subset().size(), 1);
    EXPECT_TRUE(symbolic::eq((*out_edges.begin()).subset()[0], index_expr))
        << "Output memlet subset must still be i*M + j";
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
    auto* surviving_k = dynamic_cast<structured_control_flow::Map*>(&collapsed_ij->root().at(1).first);
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

    auto* final_block = dynamic_cast<structured_control_flow::Block*>(&collapsed_kl->root().at(1).first);
    ASSERT_NE(final_block, nullptr) << "Innermost element must be the original block";

    // Verify collapsed_ij transition assignments are still correct
    {
        const auto& asgn = collapsed_ij->root().at(0).second.assignments();
        ASSERT_TRUE(asgn.count(i));
        ASSERT_TRUE(asgn.count(j));
        EXPECT_TRUE(symbolic::eq(asgn.at(i), symbolic::div(civ_ij, M)));
        EXPECT_TRUE(symbolic::eq(asgn.at(j), symbolic::mod(civ_ij, M)));
    }

    // Verify collapsed_kl transition assignments
    {
        const auto& asgn = collapsed_kl->root().at(0).second.assignments();
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

    EXPECT_EQ((*in_edges.begin()).subset().size(), 1);
    EXPECT_TRUE(symbolic::eq((*in_edges.begin()).subset()[0], index_expr))
        << "Input memlet subset must still be i*M*P*Q + j*P*Q + k*Q + l";

    EXPECT_EQ((*out_edges.begin()).subset().size(), 1);
    EXPECT_TRUE(symbolic::eq((*out_edges.begin()).subset()[0], index_expr))
        << "Output memlet subset must still be i*M*P*Q + j*P*Q + k*Q + l";
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
