#include <gtest/gtest.h>

#include <memory>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/parallelization/passes/for_classification.h"
#include "sdfg/structured_control_flow/for.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/transformations/loop_interchange.h"
#include "sdfg/transformations/loop_skewing.h"

namespace parallelogram_tiling_test {

using namespace sdfg;

// Gauss-Seidel-like nest with dependences (0, 1) and (1, -1):
//   for i in [1, N): for j in [1, M-1): A[i][j] = A[i-1][j+1] + A[i][j-1]
std::unique_ptr<builder::StructuredSDFGBuilder> build_gauss_seidel() {
    auto builder = std::make_unique<builder::StructuredSDFGBuilder>("gauss_seidel", FunctionType_CPU);
    types::Scalar idx(types::PrimitiveType::Int64);
    builder->add_container("N", idx, true);
    builder->add_container("M", idx, true);
    builder->add_container("i", idx);
    builder->add_container("j", idx);
    types::Scalar elem(types::PrimitiveType::Double);
    types::Array row(elem, symbolic::symbol("M"));
    types::Pointer desc(row);
    builder->add_container("A", desc, true);

    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");
    auto& loop_i = builder->add_for(
        builder->subject().root(),
        i,
        symbolic::Lt(i, symbolic::symbol("N")),
        symbolic::one(),
        symbolic::add(i, symbolic::one())
    );
    auto& loop_j = builder->add_for(
        loop_i.root(),
        j,
        symbolic::Lt(j, symbolic::sub(symbolic::symbol("M"), symbolic::one())),
        symbolic::one(),
        symbolic::add(j, symbolic::one())
    );
    auto& block = builder->add_block(loop_j.root());
    auto& a_in = builder->add_access(block, "A");
    auto& a_out = builder->add_access(block, "A");
    auto& tasklet = builder->add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder->add_computational_memlet(
        block, a_in, tasklet, "_in1", {symbolic::sub(i, symbolic::one()), symbolic::add(j, symbolic::one())}, desc
    );
    builder->add_computational_memlet(block, a_in, tasklet, "_in2", {i, symbolic::sub(j, symbolic::one())}, desc);
    builder->add_computational_memlet(block, tasklet, "_out", a_out, {i, j}, desc);
    return builder;
}

structured_control_flow::StructuredLoop& loop_at(structured_control_flow::Sequence& root, std::vector<size_t> path) {
    structured_control_flow::ControlFlowNode* node = &root.at(path[0]);
    for (size_t k = 1; k < path.size(); k++) {
        node = &dynamic_cast<structured_control_flow::StructuredLoop*>(node)->root().at(path[k]);
    }
    return *dynamic_cast<structured_control_flow::StructuredLoop*>(node);
}

template<typename T>
void apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& am, T&& transformation) {
    ASSERT_TRUE(transformation.can_be_applied(builder, am));
    transformation.apply(builder, am);
    am.invalidate_all();
}

void expect_loop(structured_control_flow::StructuredLoop& loop, const std::string& indvar, int64_t stride, bool is_map) {
    EXPECT_EQ(loop.indvar()->get_name(), indvar);
    EXPECT_TRUE(symbolic::eq(symbolic::sub(loop.update(), loop.indvar()), symbolic::integer(stride)));
    EXPECT_EQ(dynamic_cast<structured_control_flow::Map*>(&loop) != nullptr, is_map);
}

// Skewing j by s*i gives distances (0, 1), (1, s-1) in (i, j'). Interchange needs s >= 1;
// the new inner loop i is parallel only if the outer j' carries every dependence, i.e. s >= 2.
TEST(ParallelogramTilingTest, Wavefront_Unskewed_Illegal) {
    auto builder = build_gauss_seidel();
    analysis::AnalysisManager am(builder->subject());
    auto& root = builder->subject().root();

    transformations::LoopInterchange interchange(loop_at(root, {0}), loop_at(root, {0, 0}));
    EXPECT_FALSE(interchange.can_be_applied(*builder, am));
}

TEST(ParallelogramTilingTest, Wavefront_Skew1_InnerSequential) {
    auto builder = build_gauss_seidel();
    analysis::AnalysisManager am(builder->subject());
    auto& root = builder->subject().root();

    ASSERT_NO_FATAL_FAILURE(
        apply(*builder, am, transformations::LoopSkewing(loop_at(root, {0}), loop_at(root, {0, 0}), 1))
    );
    ASSERT_NO_FATAL_FAILURE(
        apply(*builder, am, transformations::LoopInterchange(loop_at(root, {0}), loop_at(root, {0, 0})))
    );

    // (1, -1) skewed by 1 is (1, 0) in (i, j'), i.e. (0, 1) in (j', i): still carried by the inner i.
    parallelization::ForClassificationPass classification;
    classification.run(*builder, am);
    expect_loop(loop_at(root, {0}), "j", 1, false);
    expect_loop(loop_at(root, {0, 0}), "i", 1, false);
}

TEST(ParallelogramTilingTest, Wavefront_Skew2_InnerParallel) {
    auto builder = build_gauss_seidel();
    analysis::AnalysisManager am(builder->subject());
    auto& root = builder->subject().root();

    ASSERT_NO_FATAL_FAILURE(
        apply(*builder, am, transformations::LoopSkewing(loop_at(root, {0}), loop_at(root, {0, 0}), 2))
    );
    ASSERT_NO_FATAL_FAILURE(
        apply(*builder, am, transformations::LoopInterchange(loop_at(root, {0}), loop_at(root, {0, 0})))
    );

    parallelization::ForClassificationPass classification;
    classification.run(*builder, am);
    expect_loop(loop_at(root, {0}), "j", 1, false);
    expect_loop(loop_at(root, {0, 0}), "i", 1, true);
}

} // namespace parallelogram_tiling_test
