#include "sdfg/transformations/collapse_to_depth.h"

#include <gtest/gtest.h>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/structured_control_flow/map.h"

using namespace sdfg;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static structured_control_flow::Map& add_map(
    builder::StructuredSDFGBuilder& builder,
    structured_control_flow::Sequence& parent,
    symbolic::Symbol iv,
    symbolic::Expression bound
) {
    return builder.add_map(
        parent,
        iv,
        symbolic::Lt(iv, bound),
        symbolic::integer(0),
        symbolic::add(iv, symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
}

/// Build a 2-level perfectly-nested map: i in [0,N), j in [0,M) with an empty block.
static structured_control_flow::Map& build_2d_nest(builder::StructuredSDFGBuilder& builder) {
    auto& root = builder.subject().root();
    types::Scalar sym(types::PrimitiveType::UInt64);
    builder.add_container("N", sym, true);
    builder.add_container("M", sym, true);
    builder.add_container("i", sym);
    builder.add_container("j", sym);

    auto& outer = add_map(builder, root, symbolic::symbol("i"), symbolic::symbol("N"));
    auto& inner = add_map(builder, outer.root(), symbolic::symbol("j"), symbolic::symbol("M"));
    builder.add_block(inner.root());
    return outer;
}

/// Build a 3-level nest: i in [0,N), j in [0,M), k in [0,P).
static structured_control_flow::Map& build_3d_nest(builder::StructuredSDFGBuilder& builder) {
    auto& root = builder.subject().root();
    types::Scalar sym(types::PrimitiveType::UInt64);
    builder.add_container("N", sym, true);
    builder.add_container("M", sym, true);
    builder.add_container("P", sym, true);
    builder.add_container("i", sym);
    builder.add_container("j", sym);
    builder.add_container("k", sym);

    auto& outer = add_map(builder, root, symbolic::symbol("i"), symbolic::symbol("N"));
    auto& mid = add_map(builder, outer.root(), symbolic::symbol("j"), symbolic::symbol("M"));
    auto& inner = add_map(builder, mid.root(), symbolic::symbol("k"), symbolic::symbol("P"));
    builder.add_block(inner.root());
    return outer;
}

/// Build a 4-level nest: i in [0,N), j in [0,M), k in [0,P), l in [0,Q).
static structured_control_flow::Map& build_4d_nest(builder::StructuredSDFGBuilder& builder) {
    auto& root = builder.subject().root();
    types::Scalar sym(types::PrimitiveType::UInt64);
    builder.add_container("N", sym, true);
    builder.add_container("M", sym, true);
    builder.add_container("P", sym, true);
    builder.add_container("Q", sym, true);
    builder.add_container("i", sym);
    builder.add_container("j", sym);
    builder.add_container("k", sym);
    builder.add_container("l", sym);

    auto& m_i = add_map(builder, root, symbolic::symbol("i"), symbolic::symbol("N"));
    auto& m_j = add_map(builder, m_i.root(), symbolic::symbol("j"), symbolic::symbol("M"));
    auto& m_k = add_map(builder, m_j.root(), symbolic::symbol("k"), symbolic::symbol("P"));
    auto& m_l = add_map(builder, m_k.root(), symbolic::symbol("l"), symbolic::symbol("Q"));
    builder.add_block(m_l.root());
    return m_i;
}

/// Build a 5-level nest: i,j,k,l,m over N,M,P,Q,R.
static structured_control_flow::Map& build_5d_nest(builder::StructuredSDFGBuilder& builder) {
    auto& root = builder.subject().root();
    types::Scalar sym(types::PrimitiveType::UInt64);
    for (auto* s : {"N", "M", "P", "Q", "R"}) builder.add_container(s, sym, true);
    for (auto* s : {"i", "j", "k", "l", "m"}) builder.add_container(s, sym);

    auto& m_i = add_map(builder, root, symbolic::symbol("i"), symbolic::symbol("N"));
    auto& m_j = add_map(builder, m_i.root(), symbolic::symbol("j"), symbolic::symbol("M"));
    auto& m_k = add_map(builder, m_j.root(), symbolic::symbol("k"), symbolic::symbol("P"));
    auto& m_l = add_map(builder, m_k.root(), symbolic::symbol("l"), symbolic::symbol("Q"));
    auto& m_m = add_map(builder, m_l.root(), symbolic::symbol("m"), symbolic::symbol("R"));
    builder.add_block(m_m.root());
    return m_i;
}

// =====================================================================
// can_be_applied — positive cases
// =====================================================================

TEST(CollapseToDepthTest, CanBeApplied_2D_Target1) {
    builder::StructuredSDFGBuilder builder("test", FunctionType_CPU);
    auto& outer = build_2d_nest(builder);

    analysis::AnalysisManager am(builder.subject());
    transformations::CollapseToDepth t(outer, 1);
    EXPECT_TRUE(t.can_be_applied(builder, am));
}

TEST(CollapseToDepthTest, CanBeApplied_3D_Target1) {
    builder::StructuredSDFGBuilder builder("test", FunctionType_CPU);
    auto& outer = build_3d_nest(builder);

    analysis::AnalysisManager am(builder.subject());
    transformations::CollapseToDepth t(outer, 1);
    EXPECT_TRUE(t.can_be_applied(builder, am));
}

TEST(CollapseToDepthTest, CanBeApplied_3D_Target2) {
    builder::StructuredSDFGBuilder builder("test", FunctionType_CPU);
    auto& outer = build_3d_nest(builder);

    analysis::AnalysisManager am(builder.subject());
    transformations::CollapseToDepth t(outer, 2);
    EXPECT_TRUE(t.can_be_applied(builder, am));
}

TEST(CollapseToDepthTest, CanBeApplied_4D_Target2) {
    builder::StructuredSDFGBuilder builder("test", FunctionType_CPU);
    auto& outer = build_4d_nest(builder);

    analysis::AnalysisManager am(builder.subject());
    transformations::CollapseToDepth t(outer, 2);
    EXPECT_TRUE(t.can_be_applied(builder, am));
}

TEST(CollapseToDepthTest, CanBeApplied_4D_Target1) {
    builder::StructuredSDFGBuilder builder("test", FunctionType_CPU);
    auto& outer = build_4d_nest(builder);

    analysis::AnalysisManager am(builder.subject());
    transformations::CollapseToDepth t(outer, 1);
    EXPECT_TRUE(t.can_be_applied(builder, am));
}

// =====================================================================
// can_be_applied — negative cases
// =====================================================================

TEST(CollapseToDepthTest, CannotApply_TargetZero) {
    builder::StructuredSDFGBuilder builder("test", FunctionType_CPU);
    auto& outer = build_2d_nest(builder);

    analysis::AnalysisManager am(builder.subject());
    transformations::CollapseToDepth t(outer, 0);
    EXPECT_FALSE(t.can_be_applied(builder, am));
}

TEST(CollapseToDepthTest, CannotApply_TargetThree) {
    builder::StructuredSDFGBuilder builder("test", FunctionType_CPU);
    auto& outer = build_3d_nest(builder);

    analysis::AnalysisManager am(builder.subject());
    transformations::CollapseToDepth t(outer, 3);
    EXPECT_FALSE(t.can_be_applied(builder, am));
}

TEST(CollapseToDepthTest, CannotApply_DepthAlreadyAtTarget) {
    // 2-deep nest with target_loops == 2: depth <= target, nothing to collapse
    builder::StructuredSDFGBuilder builder("test", FunctionType_CPU);
    auto& outer = build_2d_nest(builder);

    analysis::AnalysisManager am(builder.subject());
    transformations::CollapseToDepth t(outer, 2);
    EXPECT_FALSE(t.can_be_applied(builder, am));
}

TEST(CollapseToDepthTest, CannotApply_SingleMap_Target1) {
    // Depth == 1, target == 1: nothing to do
    builder::StructuredSDFGBuilder builder("test", FunctionType_CPU);
    auto& root = builder.subject().root();
    types::Scalar sym(types::PrimitiveType::UInt64);
    builder.add_container("N", sym, true);
    builder.add_container("i", sym);
    auto& map = add_map(builder, root, symbolic::symbol("i"), symbolic::symbol("N"));
    builder.add_block(map.root());

    analysis::AnalysisManager am(builder.subject());
    transformations::CollapseToDepth t(map, 1);
    EXPECT_FALSE(t.can_be_applied(builder, am));
}

TEST(CollapseToDepthTest, CannotApply_InnerBoundDependsOnOuterIndvar) {
    // Inner bound depends on outer indvar → MapCollapse rejects it
    builder::StructuredSDFGBuilder builder("test", FunctionType_CPU);
    auto& root = builder.subject().root();
    types::Scalar sym(types::PrimitiveType::UInt64);
    builder.add_container("N", sym, true);
    builder.add_container("i", sym);
    builder.add_container("j", sym);

    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");

    auto& outer = add_map(builder, root, i, symbolic::symbol("N"));
    // inner bound is `i`
    auto& inner = add_map(builder, outer.root(), j, i);
    builder.add_block(inner.root());

    analysis::AnalysisManager am(builder.subject());
    transformations::CollapseToDepth t(outer, 1);
    EXPECT_FALSE(t.can_be_applied(builder, am));
}

// =====================================================================
// apply — target_loops == 1
// =====================================================================

TEST(CollapseToDepthTest, Apply_2D_Target1_Structure) {
    // 2D → 1 loop
    builder::StructuredSDFGBuilder builder("test", FunctionType_CPU);
    auto& outer = build_2d_nest(builder);

    analysis::AnalysisManager am(builder.subject());
    transformations::CollapseToDepth t(outer, 1);
    ASSERT_TRUE(t.can_be_applied(builder, am));
    t.apply(builder, am);

    auto* result = t.outer_loop();
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(t.inner_loop(), nullptr);

    // Root has one child: the collapsed map
    EXPECT_EQ(builder.subject().root().size(), 1);
    EXPECT_EQ(&builder.subject().root().at(0).first, result);

    // Collapsed range: [0, N*M)
    auto civ = result->indvar();
    auto N = symbolic::symbol("N");
    auto M = symbolic::symbol("M");
    EXPECT_TRUE(symbolic::eq(result->init(), symbolic::integer(0)));
    EXPECT_TRUE(symbolic::eq(result->condition(), symbolic::Lt(civ, symbolic::mul(N, M))));
    EXPECT_TRUE(symbolic::eq(result->update(), symbolic::add(civ, symbolic::integer(1))));

    // Body: empty recovery block + original block
    EXPECT_EQ(result->root().size(), 2);

    // Indvar recovery
    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");
    const auto& asgn = result->root().at(0).second.assignments();
    ASSERT_TRUE(asgn.count(i));
    ASSERT_TRUE(asgn.count(j));
    EXPECT_TRUE(symbolic::eq(asgn.at(i), symbolic::div(civ, M)));
    EXPECT_TRUE(symbolic::eq(asgn.at(j), symbolic::mod(civ, M)));
}

TEST(CollapseToDepthTest, Apply_3D_Target1_Structure) {
    // 3D → 1 loop
    builder::StructuredSDFGBuilder builder("test", FunctionType_CPU);
    auto& outer = build_3d_nest(builder);

    analysis::AnalysisManager am(builder.subject());
    transformations::CollapseToDepth t(outer, 1);
    ASSERT_TRUE(t.can_be_applied(builder, am));
    t.apply(builder, am);

    auto* result = t.outer_loop();
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(t.inner_loop(), nullptr);

    // Collapsed range: [0, N*M*P)
    auto civ = result->indvar();
    auto N = symbolic::symbol("N");
    auto M = symbolic::symbol("M");
    auto P = symbolic::symbol("P");
    EXPECT_TRUE(symbolic::eq(result->condition(), symbolic::Lt(civ, symbolic::mul(N, symbolic::mul(M, P)))));

    // Body: empty recovery block + original block
    EXPECT_EQ(result->root().size(), 2);

    // Indvar recovery: i = civ/(M*P), j = (civ/P)%M, k = civ%P
    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");
    auto k = symbolic::symbol("k");
    const auto& asgn = result->root().at(0).second.assignments();
    ASSERT_TRUE(asgn.count(i));
    ASSERT_TRUE(asgn.count(j));
    ASSERT_TRUE(asgn.count(k));
    EXPECT_TRUE(symbolic::eq(asgn.at(i), symbolic::div(civ, symbolic::mul(M, P))));
    EXPECT_TRUE(symbolic::eq(asgn.at(j), symbolic::mod(symbolic::div(civ, P), M)));
    EXPECT_TRUE(symbolic::eq(asgn.at(k), symbolic::mod(civ, P)));
}

TEST(CollapseToDepthTest, Apply_4D_Target1) {
    // 4D → 1 loop
    builder::StructuredSDFGBuilder builder("test", FunctionType_CPU);
    auto& outer = build_4d_nest(builder);

    analysis::AnalysisManager am(builder.subject());
    transformations::CollapseToDepth t(outer, 1);
    ASSERT_TRUE(t.can_be_applied(builder, am));
    t.apply(builder, am);

    auto* result = t.outer_loop();
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(t.inner_loop(), nullptr);

    // Collapsed range: [0, N*M*P*Q)
    auto civ = result->indvar();
    auto N = symbolic::symbol("N");
    auto M = symbolic::symbol("M");
    auto P = symbolic::symbol("P");
    auto Q = symbolic::symbol("Q");
    EXPECT_TRUE(symbolic::
                    eq(result->condition(), symbolic::Lt(civ, symbolic::mul(N, symbolic::mul(M, symbolic::mul(P, Q)))))
    );

    // Root: single collapsed map
    EXPECT_EQ(builder.subject().root().size(), 1);

    // Body: empty recovery block + original block
    EXPECT_EQ(result->root().size(), 2);
}

// =====================================================================
// apply — target_loops == 2
// =====================================================================

TEST(CollapseToDepthTest, Apply_3D_Target2_Structure) {
    // 3D → 2 loops: outer_count = (3+1)/2 = 2, inner_count = 1
    // Outer (i,j) collapsed, inner k untouched.
    builder::StructuredSDFGBuilder builder("test", FunctionType_CPU);
    auto& outer = build_3d_nest(builder);

    analysis::AnalysisManager am(builder.subject());
    transformations::CollapseToDepth t(outer, 2);
    ASSERT_TRUE(t.can_be_applied(builder, am));
    t.apply(builder, am);

    auto* outer_result = t.outer_loop();
    auto* inner_result = t.inner_loop();
    ASSERT_NE(outer_result, nullptr);
    ASSERT_NE(inner_result, nullptr);

    // Root: single outer collapsed map
    EXPECT_EQ(builder.subject().root().size(), 1);
    EXPECT_EQ(&builder.subject().root().at(0).first, outer_result);

    // Outer range: [0, N*M)
    auto civ_outer = outer_result->indvar();
    auto N = symbolic::symbol("N");
    auto M = symbolic::symbol("M");
    EXPECT_TRUE(symbolic::eq(outer_result->condition(), symbolic::Lt(civ_outer, symbolic::mul(N, M))));

    // Outer body: empty recovery block + inner k map
    EXPECT_EQ(outer_result->root().size(), 2);
    auto* k_map = dynamic_cast<structured_control_flow::Map*>(&outer_result->root().at(1).first);
    ASSERT_NE(k_map, nullptr);
    EXPECT_EQ(k_map, inner_result);

    // Inner k was not collapsed — verify its indvar is still k with bound P
    auto k = symbolic::symbol("k");
    auto P = symbolic::symbol("P");
    EXPECT_TRUE(symbolic::eq(k_map->indvar(), k));
    EXPECT_TRUE(symbolic::eq(k_map->condition(), symbolic::Lt(k, P)));

    // Outer indvar recovery: i = civ / M, j = civ % M
    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");
    const auto& asgn = outer_result->root().at(0).second.assignments();
    ASSERT_TRUE(asgn.count(i));
    ASSERT_TRUE(asgn.count(j));
    EXPECT_TRUE(symbolic::eq(asgn.at(i), symbolic::div(civ_outer, M)));
    EXPECT_TRUE(symbolic::eq(asgn.at(j), symbolic::mod(civ_outer, M)));
}

TEST(CollapseToDepthTest, Apply_4D_Target2_Structure) {
    // 4D → 2 loops: outer_count = (4+1)/2 = 2, inner_count = 2 (even split)
    // Outer (i,j) collapsed, inner (k,l) collapsed.
    builder::StructuredSDFGBuilder builder("test", FunctionType_CPU);
    auto& outer = build_4d_nest(builder);

    analysis::AnalysisManager am(builder.subject());
    transformations::CollapseToDepth t(outer, 2);
    ASSERT_TRUE(t.can_be_applied(builder, am));
    t.apply(builder, am);

    auto* outer_result = t.outer_loop();
    auto* inner_result = t.inner_loop();
    ASSERT_NE(outer_result, nullptr);
    ASSERT_NE(inner_result, nullptr);

    auto N = symbolic::symbol("N");
    auto M = symbolic::symbol("M");
    auto P = symbolic::symbol("P");
    auto Q = symbolic::symbol("Q");

    // Root: collapsed outer map
    EXPECT_EQ(builder.subject().root().size(), 1);
    EXPECT_EQ(&builder.subject().root().at(0).first, outer_result);

    // Outer range: [0, N*M)
    auto civ_outer = outer_result->indvar();
    EXPECT_TRUE(symbolic::eq(outer_result->condition(), symbolic::Lt(civ_outer, symbolic::mul(N, M))));

    // Inner range: [0, P*Q)
    auto civ_inner = inner_result->indvar();
    EXPECT_TRUE(symbolic::eq(inner_result->condition(), symbolic::Lt(civ_inner, symbolic::mul(P, Q))));

    // Outer body: empty recovery block + inner collapsed map
    EXPECT_EQ(outer_result->root().size(), 2);
    auto* inner_map = dynamic_cast<structured_control_flow::Map*>(&outer_result->root().at(1).first);
    ASSERT_NE(inner_map, nullptr);
    EXPECT_EQ(inner_map, inner_result);

    // Inner body: empty recovery block + original block
    EXPECT_EQ(inner_result->root().size(), 2);
    EXPECT_TRUE(dynamic_cast<structured_control_flow::Block*>(&inner_result->root().at(1).first) != nullptr);

    // Outer indvar recovery: i = civ_outer / M, j = civ_outer % M
    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");
    {
        const auto& asgn = outer_result->root().at(0).second.assignments();
        ASSERT_TRUE(asgn.count(i));
        ASSERT_TRUE(asgn.count(j));
        EXPECT_TRUE(symbolic::eq(asgn.at(i), symbolic::div(civ_outer, M)));
        EXPECT_TRUE(symbolic::eq(asgn.at(j), symbolic::mod(civ_outer, M)));
    }

    // Inner indvar recovery: k = civ_inner / Q, l = civ_inner % Q
    auto k = symbolic::symbol("k");
    auto l = symbolic::symbol("l");
    {
        const auto& asgn = inner_result->root().at(0).second.assignments();
        ASSERT_TRUE(asgn.count(k));
        ASSERT_TRUE(asgn.count(l));
        EXPECT_TRUE(symbolic::eq(asgn.at(k), symbolic::div(civ_inner, Q)));
        EXPECT_TRUE(symbolic::eq(asgn.at(l), symbolic::mod(civ_inner, Q)));
    }
}

TEST(CollapseToDepthTest, Apply_5D_Target2_OddSplit) {
    // 5D → 2 loops: outer_count = (5+1)/2 = 3, inner_count = 2
    // Outer (i,j,k) collapsed, inner (l,m) collapsed.
    builder::StructuredSDFGBuilder builder("test", FunctionType_CPU);
    auto& outer = build_5d_nest(builder);

    analysis::AnalysisManager am(builder.subject());
    transformations::CollapseToDepth t(outer, 2);
    ASSERT_TRUE(t.can_be_applied(builder, am));
    t.apply(builder, am);

    auto* outer_result = t.outer_loop();
    auto* inner_result = t.inner_loop();
    ASSERT_NE(outer_result, nullptr);
    ASSERT_NE(inner_result, nullptr);

    auto N = symbolic::symbol("N");
    auto M = symbolic::symbol("M");
    auto P = symbolic::symbol("P");
    auto Q = symbolic::symbol("Q");
    auto R = symbolic::symbol("R");

    // Outer range: [0, N*M*P)
    auto civ_outer = outer_result->indvar();
    EXPECT_TRUE(symbolic::eq(outer_result->condition(), symbolic::Lt(civ_outer, symbolic::mul(N, symbolic::mul(M, P))))
    );

    // Inner range: [0, Q*R)
    auto civ_inner = inner_result->indvar();
    EXPECT_TRUE(symbolic::eq(inner_result->condition(), symbolic::Lt(civ_inner, symbolic::mul(Q, R))));

    // Outer body: empty recovery block + inner collapsed map
    EXPECT_EQ(outer_result->root().size(), 2);
    auto* inner_map = dynamic_cast<structured_control_flow::Map*>(&outer_result->root().at(1).first);
    ASSERT_NE(inner_map, nullptr);
    EXPECT_EQ(inner_map, inner_result);

    // Inner body: empty recovery block + original block
    EXPECT_EQ(inner_result->root().size(), 2);

    // Outer indvar recovery: i = civ/(M*P), j = (civ/P)%M, k = civ%P
    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");
    auto k = symbolic::symbol("k");
    {
        const auto& asgn = outer_result->root().at(0).second.assignments();
        ASSERT_TRUE(asgn.count(i));
        ASSERT_TRUE(asgn.count(j));
        ASSERT_TRUE(asgn.count(k));
        EXPECT_TRUE(symbolic::eq(asgn.at(i), symbolic::div(civ_outer, symbolic::mul(M, P))));
        EXPECT_TRUE(symbolic::eq(asgn.at(j), symbolic::mod(symbolic::div(civ_outer, P), M)));
        EXPECT_TRUE(symbolic::eq(asgn.at(k), symbolic::mod(civ_outer, P)));
    }

    // Inner indvar recovery: l = civ_inner / R, m = civ_inner % R
    auto l = symbolic::symbol("l");
    auto m = symbolic::symbol("m");
    {
        const auto& asgn = inner_result->root().at(0).second.assignments();
        ASSERT_TRUE(asgn.count(l));
        ASSERT_TRUE(asgn.count(m));
        EXPECT_TRUE(symbolic::eq(asgn.at(l), symbolic::div(civ_inner, R)));
        EXPECT_TRUE(symbolic::eq(asgn.at(m), symbolic::mod(civ_inner, R)));
    }
}

// =====================================================================
// Accessor guards
// =====================================================================

TEST(CollapseToDepthTest, AccessorBeforeApply_Throws) {
    builder::StructuredSDFGBuilder builder("test", FunctionType_CPU);
    auto& outer = build_2d_nest(builder);

    transformations::CollapseToDepth t(outer, 1);
    EXPECT_EQ(t.outer_loop(), &outer);
    EXPECT_THROW(t.inner_loop(), InvalidSDFGException);
}

// =====================================================================
// Serialization round-trip
// =====================================================================

TEST(CollapseToDepthTest, Serialization) {
    builder::StructuredSDFGBuilder builder("test", FunctionType_CPU);
    auto& outer = build_3d_nest(builder);

    transformations::CollapseToDepth t(outer, 2);
    nlohmann::json j;
    t.to_json(j);

    EXPECT_EQ(j["transformation_type"], "CollapseToDepth");
    EXPECT_EQ(j["parameters"]["target_loops"], 2);
    EXPECT_EQ(j["subgraph"]["0"]["element_id"], outer.element_id());

    auto t2 = transformations::CollapseToDepth::from_json(builder, j);

    analysis::AnalysisManager am(builder.subject());
    EXPECT_TRUE(t2.can_be_applied(builder, am));
}
