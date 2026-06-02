#include "sdfg/transformations/loop_tile_rescaling.h"

#include <gtest/gtest.h>

#include "sdfg/analysis/analysis.h"
#include "sdfg/analysis/loop_analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/transformations/loop_tiling.h"

using namespace sdfg;

namespace {

class LoopTileRescalingTest : public ::testing::Test {
protected:
    std::unique_ptr<StructuredSDFG> sdfg;
    builder::StructuredSDFGBuilder builder_;

    // Kept after tiling so tests can reference them
    structured_control_flow::StructuredLoop* outer_loop_ = nullptr;

    void SetUp() override {
        builder_ = builder::StructuredSDFGBuilder("tile_rescaling_test", FunctionType_CPU);
        types::Scalar sym_desc(types::PrimitiveType::UInt64);
        builder_.add_container("N", sym_desc, true);
        builder_.add_container("i", sym_desc);

        auto& root = builder_.subject().root();
        auto indvar = symbolic::symbol("i");
        auto bound  = symbolic::symbol("N");
        auto& orig  = builder_.add_for(
            root, indvar, symbolic::Lt(indvar, bound), symbolic::integer(0),
            symbolic::add(indvar, symbolic::integer(1))
        );

        // Add a body block so the loop is non-trivial
        builder_.add_block(orig.root());

        analysis::AnalysisManager analysis_manager(builder_.subject());
        transformations::LoopTiling tiling(orig, 4);
        EXPECT_TRUE(tiling.can_be_applied(builder_, analysis_manager));
        tiling.apply(builder_, analysis_manager);

        // Locate outer loop after tiling (stride > 1)
        analysis::AnalysisManager analysis_manager2(builder_.subject());
        auto& loop_analysis = analysis_manager2.get<analysis::LoopAnalysis>();
        for (auto* loop : loop_analysis.loops()) {
            auto* structured_loop = dynamic_cast<structured_control_flow::StructuredLoop*>(loop);
            if (!structured_loop) continue;
            auto step = symbolic::sub(structured_loop->update(), structured_loop->indvar());
            if (SymEngine::is_a<SymEngine::Integer>(*step) &&
                SymEngine::down_cast<const SymEngine::Integer&>(*step).as_int() > 1) {
                outer_loop_ = structured_loop;
            }
        }
        EXPECT_NE(outer_loop_, nullptr);
    }
    void TearDown() override {};
};

} // namespace

// --- can_be_applied / apply ---

TEST_F(LoopTileRescalingTest, CanBeApplied_AfterTiling) {
    analysis::AnalysisManager analysis_manager(builder_.subject());

    transformations::LoopTileRescaling transformation(*outer_loop_, 8);
    EXPECT_TRUE(transformation.can_be_applied(builder_, analysis_manager));
}

TEST_F(LoopTileRescalingTest, CannotBeApplied_OnUntiled) {
    auto& root = builder_.subject().root();
    auto indvar = symbolic::symbol("j");
    auto& loop = builder_.add_for(
        root, indvar, symbolic::Lt(indvar, symbolic::symbol("N")), symbolic::integer(0),
        symbolic::add(indvar, symbolic::integer(1))
    );
    builder_.add_block(loop.root());

    analysis::AnalysisManager analysis_manager(builder_.subject());
    transformations::LoopTileRescaling t(loop, 8); // stride 1 → cannot apply
    EXPECT_FALSE(t.can_be_applied(builder_, analysis_manager));
}

TEST_F(LoopTileRescalingTest, Apply_ChangesOuterStride) {
    analysis::AnalysisManager analysis_manager(builder_.subject());

    transformations::LoopTileRescaling t(*outer_loop_, 16);
    ASSERT_TRUE(t.can_be_applied(builder_, analysis_manager));
    t.apply(builder_, analysis_manager);

    // Outer update should now be outer_indvar + 16
    auto outer_sym = outer_loop_->indvar();
    EXPECT_TRUE(symbolic::eq(
        outer_loop_->update(),
        symbolic::add(outer_sym, symbolic::integer(16))
    ));
}

TEST_F(LoopTileRescalingTest, Apply_ChangesInnerCondition) {
    analysis::AnalysisManager analysis_manager(builder_.subject());

    transformations::LoopTileRescaling t(*outer_loop_, 16);
    t.apply(builder_, analysis_manager);

    // Inner condition should reference outer_indvar + 16, not + 4
    auto outer_sym  = outer_loop_->indvar();
    auto new_bound  = symbolic::add(outer_sym, symbolic::integer(16));
    auto old_bound  = symbolic::add(outer_sym, symbolic::integer(4));

    // Find the inner loop from the outer body
    auto* inner = dynamic_cast<structured_control_flow::StructuredLoop*>(&outer_loop_->root().at(0).first);
    ASSERT_NE(inner, nullptr);

    auto inner_cond_str = inner->condition()->__str__();
    (void)new_bound; (void)old_bound;
    EXPECT_NE(inner_cond_str.find("16"), std::string::npos);
    EXPECT_EQ(inner_cond_str.find("+ 4"), std::string::npos); // old size gone
}

TEST_F(LoopTileRescalingTest, Apply_Noop_WhenSameSize) {
    analysis::AnalysisManager analysis_manager(builder_.subject());

    // Rescaling to same size must not change anything
    auto before_outer_update = outer_loop_->update();

    transformations::LoopTileRescaling t(*outer_loop_, 4);
    ASSERT_TRUE(t.can_be_applied(builder_, analysis_manager));
    t.apply(builder_, analysis_manager);

    EXPECT_TRUE(symbolic::eq(outer_loop_->update(), before_outer_update));
}

// --- Serialization round-trip ---

TEST_F(LoopTileRescalingTest, Serialization_RoundTrip) {

    transformations::LoopTileRescaling transformation(*outer_loop_, 8);
    nlohmann::json j;
    transformation.to_json(j);

    // Shape
    ASSERT_TRUE(j.contains("transformation_type"));
    EXPECT_EQ(j["transformation_type"], "LoopTileRescaling");
    ASSERT_TRUE(j.contains("subgraph"));
    EXPECT_EQ(j["subgraph"].size(), 1u);
    ASSERT_TRUE(j["subgraph"].contains("0"));
    EXPECT_EQ(j["parameters"]["new_tile_size"], 8u);

    // Deserialize and apply
    auto t2 = transformations::LoopTileRescaling::from_json(builder_, j);
    analysis::AnalysisManager analysis_manager(builder_.subject());
    EXPECT_EQ(t2.name(), "LoopTileRescaling");
    ASSERT_TRUE(t2.can_be_applied(builder_, analysis_manager));
    t2.apply(builder_, analysis_manager);

    EXPECT_TRUE(symbolic::eq(
        outer_loop_->update(),
        symbolic::add(outer_loop_->indvar(), symbolic::integer(8))
    ));
}
