#include "sdfg/passes/tiling_pass.h"

#include <gtest/gtest.h>

#include <vector>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/symbolic/symbolic.h"

using namespace sdfg;

namespace {

// Adds a container `A` (opaque pointer) plus the loop induction variable container.
void add_containers(builder::StructuredSDFGBuilder& builder, const std::string& indvar) {
    if (!builder.subject().exists("A")) {
        types::Pointer opaque_desc;
        builder.add_container("A", opaque_desc, true);
    }
    types::Scalar sym_desc(types::PrimitiveType::Int64);
    builder.add_container(indvar, sym_desc);
}

// Adds a `for (indvar = 0; indvar < bound; indvar += stride)` loop with a single
// A[indvar] = A[indvar] copy body, and returns it.
structured_control_flow::For& add_copy_loop(
    builder::StructuredSDFGBuilder& builder,
    structured_control_flow::Sequence& parent,
    const std::string& indvar,
    int64_t bound,
    int64_t stride
) {
    add_containers(builder, indvar);

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);

    auto sym = symbolic::symbol(indvar);
    auto& loop = builder.add_for(
        parent,
        sym,
        symbolic::Lt(sym, symbolic::integer(bound)),
        symbolic::integer(0),
        symbolic::add(sym, symbolic::integer(stride))
    );

    auto& block = builder.add_block(loop.root());
    auto& A_in = builder.add_access(block, "A");
    auto& A_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, A_in, tasklet, "_in", {sym}, desc);
    builder.add_computational_memlet(block, tasklet, "_out", A_out, {sym}, desc);
    return loop;
}

} // namespace

TEST(TilingPassTest, EmptyLoops_ReturnsFalse) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);
    analysis::AnalysisManager analysis_manager(builder.subject());

    std::vector<structured_control_flow::StructuredLoop*> loops;
    passes::TilingPass pass(loops, 4);
    EXPECT_FALSE(pass.run(builder, analysis_manager));
}

TEST(TilingPassTest, TileSizeZero_ReturnsFalse) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);
    auto& loop = add_copy_loop(builder, builder.subject().root(), "i", 64, 1);

    analysis::AnalysisManager analysis_manager(builder.subject());
    std::vector<structured_control_flow::StructuredLoop*> loops{&loop};
    passes::TilingPass pass(loops, 0);
    EXPECT_FALSE(pass.run(builder, analysis_manager));
}

TEST(TilingPassTest, TileSizeOne_ReturnsFalse) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);
    auto& loop = add_copy_loop(builder, builder.subject().root(), "i", 64, 1);

    analysis::AnalysisManager analysis_manager(builder.subject());
    std::vector<structured_control_flow::StructuredLoop*> loops{&loop};
    passes::TilingPass pass(loops, 1);
    EXPECT_FALSE(pass.run(builder, analysis_manager));
}

TEST(TilingPassTest, SingleContiguousFor_Tiled) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& orig_loop = add_copy_loop(builder, sdfg.root(), "i", 64, 1);

    analysis::AnalysisManager analysis_manager(sdfg);
    std::vector<structured_control_flow::StructuredLoop*> loops{&orig_loop};
    passes::TilingPass pass(loops, 4);
    EXPECT_TRUE(pass.run(builder, analysis_manager));

    // Root now holds the new outer loop, and the vector points to it.
    ASSERT_EQ(sdfg.root().size(), 1);
    auto* outer = dyn_cast<structured_control_flow::For*>(&sdfg.root().at(0));
    ASSERT_NE(outer, nullptr);
    EXPECT_EQ(loops[0], outer);
    EXPECT_NE(loops[0], &orig_loop);

    // Outer stride equals the tile size.
    EXPECT_TRUE(symbolic::eq(outer->update(), symbolic::add(outer->indvar(), symbolic::integer(4))));

    // The original loop is nested inside the outer loop.
    ASSERT_EQ(outer->root().size(), 1);
    EXPECT_EQ(&outer->root().at(0), &orig_loop);
}

TEST(TilingPassTest, MultipleContiguousLoops_AllTiled) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& loop_a = add_copy_loop(builder, sdfg.root(), "i", 64, 1);
    auto& loop_b = add_copy_loop(builder, sdfg.root(), "j", 128, 1);

    analysis::AnalysisManager analysis_manager(sdfg);
    std::vector<structured_control_flow::StructuredLoop*> loops{&loop_a, &loop_b};
    passes::TilingPass pass(loops, 8);
    EXPECT_TRUE(pass.run(builder, analysis_manager));

    // Both vector entries were replaced with the new outer loops.
    EXPECT_NE(loops[0], &loop_a);
    EXPECT_NE(loops[1], &loop_b);
    for (auto* outer : loops) {
        EXPECT_TRUE(symbolic::eq(outer->update(), symbolic::add(outer->indvar(), symbolic::integer(8))));
    }
}

TEST(TilingPassTest, MixedApplicability_OnlyContiguousTiled) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& contiguous = add_copy_loop(builder, sdfg.root(), "i", 64, 1);
    auto& strided = add_copy_loop(builder, sdfg.root(), "j", 64, 2);

    analysis::AnalysisManager analysis_manager(sdfg);
    std::vector<structured_control_flow::StructuredLoop*> loops{&contiguous, &strided};
    passes::TilingPass pass(loops, 4);
    EXPECT_TRUE(pass.run(builder, analysis_manager));

    // Contiguous loop was tiled; strided (non-contiguous) loop pointer is untouched.
    EXPECT_NE(loops[0], &contiguous);
    EXPECT_EQ(loops[1], &strided);
}

TEST(TilingPassTest, AllNonContiguous_ReturnsFalse) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& strided = add_copy_loop(builder, sdfg.root(), "i", 64, 2);

    analysis::AnalysisManager analysis_manager(sdfg);
    std::vector<structured_control_flow::StructuredLoop*> loops{&strided};
    passes::TilingPass pass(loops, 4);
    EXPECT_FALSE(pass.run(builder, analysis_manager));
    EXPECT_EQ(loops[0], &strided);
}
