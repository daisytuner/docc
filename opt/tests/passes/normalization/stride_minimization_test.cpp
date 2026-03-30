#include "sdfg/passes/normalization/stride_minimization.h"
#include "sdfg/passes/normalization/perfect_loop_distribution.h"

#include <gtest/gtest.h>
#include <sdfg/passes/pipeline.h>

#include "fixtures/polybench.h"

using namespace sdfg;

TEST(StrideMinimizationTest, EmptyLoop) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar sym_desc(types::PrimitiveType::Int32);
    builder.add_container("N", sym_desc, true);
    builder.add_container("M", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);

    // Define loop 1
    auto bound = symbolic::symbol("N");
    auto indvar = symbolic::symbol("i");

    auto& loop = builder.add_map(
        root,
        indvar,
        symbolic::Lt(symbolic::symbol("i"), symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    auto& body = loop.root();

    // Define loop 2
    auto bound_2 = symbolic::symbol("M");
    auto indvar_2 = symbolic::symbol("j");

    auto& loop_2 = builder.add_map(
        body,
        indvar_2,
        symbolic::Lt(symbolic::symbol("j"), symbolic::symbol("M")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("j"), symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    auto& body_2 = loop_2.root();

    // Add computation
    auto& block = builder.add_block(body_2);

    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::normalization::StrideMinimization stride_minimization;
    EXPECT_FALSE(stride_minimization.run_pass(builder, analysis_manager));
}

TEST(StrideMinimizationTest, is_admissible) {
    std::vector<std::string> current_1 = {"i", "j", "k"};
    std::vector<std::string> target_1 = {"j", "i", "k"};
    std::unordered_set<std::string> allowed_swaps_1 = {"i_j"};
    EXPECT_TRUE(passes::normalization::StrideMinimization::is_admissible(current_1, target_1, allowed_swaps_1));

    std::unordered_set<std::string> allowed_swaps_2 = {"j_i"};
    EXPECT_FALSE(passes::normalization::StrideMinimization::is_admissible(current_1, target_1, allowed_swaps_2));

    std::unordered_set<std::string> allowed_swaps_3 = {"j_k"};
    EXPECT_FALSE(passes::normalization::StrideMinimization::is_admissible(current_1, target_1, allowed_swaps_3));

    std::vector<std::string> current_2 = {"i", "k", "j"};
    std::vector<std::string> target_2 = {"k", "i", "j"};
    std::unordered_set<std::string> allowed_swaps_4 = {"i_k", "k_j"};
    EXPECT_TRUE(passes::normalization::StrideMinimization::is_admissible(current_2, target_2, allowed_swaps_4));

    std::vector<std::string> target_3 = {"k", "j", "i"};
    EXPECT_FALSE(passes::normalization::StrideMinimization::is_admissible(current_2, target_3, allowed_swaps_4));
}

TEST(StrideMinimizationTest, SingleLoop_NotApplicable) {
    // A single loop (no nesting) with a memory access - size < 2
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);
    auto& root = builder.subject().root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);

    types::Scalar elem_desc(types::PrimitiveType::Double);
    types::Array arr_1d(elem_desc, symbolic::symbol("N"));
    builder.add_container("A", arr_1d, true);

    auto& loop = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1))
    );

    auto& block = builder.add_block(loop.root());
    auto& a_out = builder.add_access(block, "A");
    auto& const_node = builder.add_constant(block, "0.0", elem_desc);
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, const_node, tasklet, "_in", {});
    builder.add_computational_memlet(block, tasklet, "_out", a_out, {symbolic::symbol("i")});

    analysis::AnalysisManager am(builder.subject());
    passes::normalization::StrideMinimization sm;
    EXPECT_FALSE(sm.run_pass(builder, am));
}

TEST(StrideMinimizationTest, AlreadyOptimal_NotApplicable) {
    // 2D loop nest with A[i][j] access - already optimal stride order
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);
    auto& root = builder.subject().root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("M", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);

    types::Scalar elem_desc(types::PrimitiveType::Double);
    types::Array arr_1d(elem_desc, symbolic::symbol("M"));
    types::Pointer arr_2d(arr_1d);
    builder.add_container("A", arr_2d, true);

    auto& loop_i = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1))
    );
    auto& loop_j = builder.add_for(
        loop_i.root(),
        symbolic::symbol("j"),
        symbolic::Lt(symbolic::symbol("j"), symbolic::symbol("M")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("j"), symbolic::integer(1))
    );

    auto& block = builder.add_block(loop_j.root());
    auto& a_out = builder.add_access(block, "A");
    auto& const_node = builder.add_constant(block, "0.0", elem_desc);
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, const_node, tasklet, "_in", {});
    builder.add_computational_memlet(block, tasklet, "_out", a_out, {symbolic::symbol("i"), symbolic::symbol("j")});

    analysis::AnalysisManager am(builder.subject());
    passes::normalization::StrideMinimization sm;
    EXPECT_FALSE(sm.run_pass(builder, am));
}

TEST(StrideMinimizationTest, AllowedSwaps_PerfectlyNested) {
    // Two perfectly nested loops - swap should be allowed
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);
    auto& root = builder.subject().root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("M", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);

    auto& loop_i = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1))
    );
    auto& loop_j = builder.add_for(
        loop_i.root(),
        symbolic::symbol("j"),
        symbolic::Lt(symbolic::symbol("j"), symbolic::symbol("M")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("j"), symbolic::integer(1))
    );
    builder.add_block(loop_j.root());

    analysis::AnalysisManager am(builder.subject());
    std::vector<structured_control_flow::ControlFlowNode*> nested = {&loop_i, &loop_j};
    auto swaps = passes::normalization::StrideMinimization::allowed_swaps(builder, am, nested);
    EXPECT_TRUE(swaps.find("i_j") != swaps.end());
}

TEST(StrideMinimizationTest, AllowedSwaps_NotPerfectlyNested) {
    // Outer loop has two children - i<->j and i<->k swaps should not be allowed
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);
    auto& root = builder.subject().root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("M", sym_desc, true);
    builder.add_container("P", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);
    builder.add_container("k", sym_desc);

    auto& loop_i = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1))
    );
    auto& loop_j = builder.add_for(
        loop_i.root(),
        symbolic::symbol("j"),
        symbolic::Lt(symbolic::symbol("j"), symbolic::symbol("M")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("j"), symbolic::integer(1))
    );
    builder.add_block(loop_j.root());
    auto& loop_k = builder.add_for(
        loop_i.root(),
        symbolic::symbol("k"),
        symbolic::Lt(symbolic::symbol("k"), symbolic::symbol("P")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("k"), symbolic::integer(1))
    );
    builder.add_block(loop_k.root());

    analysis::AnalysisManager am(builder.subject());
    // Path [i, j]: i has multiple children -> i<->j not swappable
    std::vector<structured_control_flow::ControlFlowNode*> path1 = {&loop_i, &loop_j};
    auto swaps1 = passes::normalization::StrideMinimization::allowed_swaps(builder, am, path1);
    EXPECT_TRUE(swaps1.empty());

    // Path [i, k]: same reason
    std::vector<structured_control_flow::ControlFlowNode*> path2 = {&loop_i, &loop_k};
    auto swaps2 = passes::normalization::StrideMinimization::allowed_swaps(builder, am, path2);
    EXPECT_TRUE(swaps2.empty());
}

TEST(StrideMinimizationTest, Apply_SimpleInterchange) {
    // 2D loop nest with A[j][i] access - should interchange to [j, i]
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);
    auto& root = builder.subject().root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("M", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);

    types::Scalar elem_desc(types::PrimitiveType::Double);
    types::Array arr_1d(elem_desc, symbolic::symbol("N"));
    types::Pointer arr_2d(arr_1d);
    builder.add_container("A", arr_2d, true);

    auto& loop_i = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1))
    );
    auto& loop_j = builder.add_for(
        loop_i.root(),
        symbolic::symbol("j"),
        symbolic::Lt(symbolic::symbol("j"), symbolic::symbol("M")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("j"), symbolic::integer(1))
    );

    auto& block = builder.add_block(loop_j.root());
    auto& a_out = builder.add_access(block, "A");
    auto& const_node = builder.add_constant(block, "0.0", elem_desc);
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, const_node, tasklet, "_in", {});
    builder.add_computational_memlet(block, tasklet, "_out", a_out, {symbolic::symbol("j"), symbolic::symbol("i")});

    analysis::AnalysisManager am(builder.subject());
    passes::normalization::StrideMinimization sm;
    EXPECT_TRUE(sm.run_pass(builder, am));

    // Verify the loops were interchanged: outer should be j, inner should be i
    auto& new_root = builder.subject().root();
    auto* outer = dynamic_cast<structured_control_flow::StructuredLoop*>(&new_root.at(0).first);
    ASSERT_NE(outer, nullptr);
    EXPECT_EQ(outer->indvar()->get_name(), "j");

    auto* inner = dynamic_cast<structured_control_flow::StructuredLoop*>(&outer->root().at(0).first);
    ASSERT_NE(inner, nullptr);
    EXPECT_EQ(inner->indvar()->get_name(), "i");

    // Second run should not apply (already optimal)
    EXPECT_FALSE(sm.run_pass(builder, am));
}

TEST(StrideMinimizationTest, OverlappingPaths_RequiresTwoIterations) {
    // Outer loop i with two branches, each needing a j<->k or l<->m interchange:
    //   for i [0, N):
    //     for j [0, M):
    //       for k [0, P):
    //         A[k][j] = ...   // path [i,j,k] wants j<->k swap
    //     for l [0, Q):
    //       for m [0, R):
    //         B[m][l] = ...   // path [i,l,m] wants l<->m swap
    //
    // Both paths share outer loop i and only one gets optimized per pass call.
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);
    auto& root = builder.subject().root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("M", sym_desc, true);
    builder.add_container("P", sym_desc, true);
    builder.add_container("Q", sym_desc, true);
    builder.add_container("R", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);
    builder.add_container("k", sym_desc);
    builder.add_container("l", sym_desc);
    builder.add_container("m", sym_desc);

    types::Scalar elem_desc(types::PrimitiveType::Double);
    types::Array arr_A_1d(elem_desc, symbolic::symbol("P"));
    types::Pointer arr_A_2d(arr_A_1d);
    builder.add_container("A", arr_A_2d, true);

    types::Array arr_B_1d(elem_desc, symbolic::symbol("R"));
    types::Pointer arr_B_2d(arr_B_1d);
    builder.add_container("B", arr_B_2d, true);

    // Outer loop: for i [0, N)
    auto& loop_i = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1))
    );

    // Branch 1: for j [0, M), for k [0, P), A[k][j] = ...
    auto& loop_j = builder.add_for(
        loop_i.root(),
        symbolic::symbol("j"),
        symbolic::Lt(symbolic::symbol("j"), symbolic::symbol("M")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("j"), symbolic::integer(1))
    );
    auto& loop_k = builder.add_for(
        loop_j.root(),
        symbolic::symbol("k"),
        symbolic::Lt(symbolic::symbol("k"), symbolic::symbol("P")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("k"), symbolic::integer(1))
    );
    {
        auto& block = builder.add_block(loop_k.root());
        auto& a_out = builder.add_access(block, "A");
        auto& const_node = builder.add_constant(block, "0.0", elem_desc);
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
        builder.add_computational_memlet(block, const_node, tasklet, "_in", {});
        builder.add_computational_memlet(block, tasklet, "_out", a_out, {symbolic::symbol("k"), symbolic::symbol("j")});
    }

    // Branch 2: for l [0, Q), for m [0, R), B[m][l] = ...
    auto& loop_l = builder.add_for(
        loop_i.root(),
        symbolic::symbol("l"),
        symbolic::Lt(symbolic::symbol("l"), symbolic::symbol("Q")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("l"), symbolic::integer(1))
    );
    auto& loop_m = builder.add_for(
        loop_l.root(),
        symbolic::symbol("m"),
        symbolic::Lt(symbolic::symbol("m"), symbolic::symbol("R")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("m"), symbolic::integer(1))
    );
    {
        auto& block = builder.add_block(loop_m.root());
        auto& b_out = builder.add_access(block, "B");
        auto& const_node = builder.add_constant(block, "0.0", elem_desc);
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
        builder.add_computational_memlet(block, const_node, tasklet, "_in", {});
        builder.add_computational_memlet(block, tasklet, "_out", b_out, {symbolic::symbol("m"), symbolic::symbol("l")});
    }

    analysis::AnalysisManager am(builder.subject());
    passes::normalization::StrideMinimization sm;

    // First iteration: optimizes one of the two paths
    EXPECT_TRUE(sm.run_pass(builder, am));

    // Second iteration: optimizes the remaining path
    EXPECT_TRUE(sm.run_pass(builder, am));

    // Third iteration: nothing left to optimize
    EXPECT_FALSE(sm.run_pass(builder, am));
}

// Check
/***
for (i = 0; i < _PB_M-1; i++)
    C[i][i] = SCALAR_VAL(1.0);

for (i = 0; i < _PB_M-1; i++)
    for (j = i+1; j < _PB_M; j++)
        C[i][j] = SCALAR_VAL(0.0);

for (i = 0; i < _PB_M-1; i++)
    for (k = 0; k < _PB_N; k++)
        for (j = i+1; j < _PB_M; j++)
            C[i][j] += (D[k][i] * D[k][j]);

for (i = 0; i < _PB_M-1; i++)
    for (j = i+1; j < _PB_M; j++)
        C[j][i] = C[i][j];
***/

TEST(StrideMinimizationTest, Polybench_correlation) {
    // Build SDFG
    auto init_sdfg = correlation();
    auto builder = std::make_unique<builder::StructuredSDFGBuilder>(init_sdfg);

    auto analysis_manager = std::make_unique<analysis::AnalysisManager>(builder->subject());

    // todo: get outermost loop

    // Pass
    passes::Pipeline pipeline("Perfect Loop Distribution");

    // Register passes for loop normalization

    pipeline.register_pass<passes::normalization::PerfectLoopDistributionPass>();
    pipeline.register_pass<passes::normalization::StrideMinimization>();

    pipeline.run(*builder, *analysis_manager);

    auto& sdfg = builder->subject();

    {
        auto& root = sdfg.root();
        EXPECT_EQ(root.size(), 4);

        auto loop_i_1 = dynamic_cast<structured_control_flow::StructuredLoop*>(&root.at(0).first);
        EXPECT_TRUE(loop_i_1 != nullptr);
        EXPECT_TRUE(SymEngine::eq(*loop_i_1->init(), *symbolic::integer(0)));
        EXPECT_TRUE(SymEngine::
                        eq(*loop_i_1->condition(),
                           *symbolic::Lt(loop_i_1->indvar(), symbolic::sub(symbolic::symbol("M"), symbolic::integer(1)))
                        ));
        EXPECT_TRUE(SymEngine::eq(*loop_i_1->update(), *symbolic::add(loop_i_1->indvar(), symbolic::integer(1))));
        EXPECT_EQ(loop_i_1->root().size(), 1);

        auto loop_i_2 = dynamic_cast<structured_control_flow::StructuredLoop*>(&root.at(1).first);
        EXPECT_TRUE(loop_i_2 != nullptr);
        EXPECT_TRUE(SymEngine::eq(*loop_i_2->init(), *symbolic::integer(0)));
        EXPECT_TRUE(SymEngine::
                        eq(*loop_i_2->condition(),
                           *symbolic::Lt(loop_i_2->indvar(), symbolic::sub(symbolic::symbol("M"), symbolic::integer(1)))
                        ));
        EXPECT_TRUE(SymEngine::eq(*loop_i_2->update(), *symbolic::add(loop_i_2->indvar(), symbolic::integer(1))));
        EXPECT_EQ(loop_i_2->root().size(), 1);
        auto loop_j_2 = dynamic_cast<structured_control_flow::StructuredLoop*>(&loop_i_2->root().at(0).first);
        EXPECT_TRUE(loop_j_2 != nullptr);
        EXPECT_TRUE(SymEngine::eq(*loop_j_2->init(), *symbolic::add(loop_i_2->indvar(), symbolic::integer(1))));
        EXPECT_TRUE(SymEngine::eq(*loop_j_2->condition(), *symbolic::Lt(loop_j_2->indvar(), symbolic::symbol("M"))));
        EXPECT_TRUE(SymEngine::eq(*loop_j_2->update(), *symbolic::add(loop_j_2->indvar(), symbolic::integer(1))));

        auto loop_i_3 = dynamic_cast<structured_control_flow::StructuredLoop*>(&root.at(2).first);
        EXPECT_TRUE(loop_i_3 != nullptr);
        EXPECT_TRUE(SymEngine::eq(*loop_i_3->init(), *symbolic::integer(0)));
        EXPECT_TRUE(SymEngine::
                        eq(*loop_i_3->condition(),
                           *symbolic::Lt(loop_i_3->indvar(), symbolic::sub(symbolic::symbol("M"), symbolic::integer(1)))
                        ));
        EXPECT_TRUE(SymEngine::eq(*loop_i_3->update(), *symbolic::add(loop_i_3->indvar(), symbolic::integer(1))));
        EXPECT_EQ(loop_i_3->root().size(), 1);
        auto loop_k_3 = dynamic_cast<structured_control_flow::StructuredLoop*>(&loop_i_3->root().at(0).first);
        EXPECT_TRUE(loop_k_3 != nullptr);
        EXPECT_TRUE(SymEngine::eq(*loop_k_3->init(), *symbolic::integer(0)));
        EXPECT_TRUE(SymEngine::eq(*loop_k_3->condition(), *symbolic::Lt(loop_k_3->indvar(), symbolic::symbol("N"))));
        EXPECT_TRUE(SymEngine::eq(*loop_k_3->update(), *symbolic::add(loop_k_3->indvar(), symbolic::integer(1))));
        auto loop_j_3 = dynamic_cast<structured_control_flow::StructuredLoop*>(&loop_k_3->root().at(0).first);
        EXPECT_TRUE(loop_j_3 != nullptr);
        EXPECT_TRUE(SymEngine::eq(*loop_j_3->init(), *symbolic::add(loop_i_3->indvar(), symbolic::integer(1))));
        EXPECT_TRUE(SymEngine::eq(*loop_j_3->condition(), *symbolic::Lt(loop_j_3->indvar(), symbolic::symbol("M"))));
        EXPECT_TRUE(SymEngine::eq(*loop_j_3->update(), *symbolic::add(loop_j_3->indvar(), symbolic::integer(1))));
        EXPECT_EQ(loop_j_3->root().size(), 1);

        auto loop_i_4 = dynamic_cast<structured_control_flow::StructuredLoop*>(&root.at(3).first);
        EXPECT_TRUE(loop_i_4 != nullptr);
        EXPECT_TRUE(SymEngine::eq(*loop_i_4->init(), *symbolic::integer(0)));
        EXPECT_TRUE(SymEngine::
                        eq(*loop_i_4->condition(),
                           *symbolic::Lt(loop_i_4->indvar(), symbolic::sub(symbolic::symbol("M"), symbolic::integer(1)))
                        ));
        EXPECT_TRUE(SymEngine::eq(*loop_i_4->update(), *symbolic::add(loop_i_4->indvar(), symbolic::integer(1))));
        EXPECT_EQ(loop_i_4->root().size(), 1);
        auto loop_j_4 = dynamic_cast<structured_control_flow::StructuredLoop*>(&loop_i_4->root().at(0).first);
        EXPECT_TRUE(loop_j_4 != nullptr);
        EXPECT_TRUE(SymEngine::eq(*loop_j_4->init(), *symbolic::add(loop_i_4->indvar(), symbolic::integer(1))));
        EXPECT_TRUE(SymEngine::eq(*loop_j_4->condition(), *symbolic::Lt(loop_j_4->indvar(), symbolic::symbol("M"))));
        EXPECT_TRUE(SymEngine::eq(*loop_j_4->update(), *symbolic::add(loop_j_4->indvar(), symbolic::integer(1))));
    }
}

/***
for (i = 0; i < _PB_M; i++)
for (j = i; j < _PB_M; j++)
    C[i][j] = SCALAR_VAL(0.0);

for (i = 0; i < _PB_M; i++)
for (k = 0; k < _PB_N; k++)
    for (j = i; j < _PB_M; j++)
        C[i][j] += (D[k][i] * D[k][j]);

for (i = 0; i < _PB_M; i++)
for (j = i; j < _PB_M; j++)
    C[j][i] = C[i][j];
***/

TEST(StrideMinimizationTest, Polybench_covariance) {
    // Build SDFG
    auto init_sdfg = covariance();
    auto builder = std::make_unique<builder::StructuredSDFGBuilder>(init_sdfg);

    auto analysis_manager = std::make_unique<analysis::AnalysisManager>(builder->subject());

    // todo: get outermost loop

    // Pass
    passes::Pipeline pipeline("Perfect Loop Distribution");

    // Register passes for loop normalization

    pipeline.register_pass<passes::normalization::PerfectLoopDistributionPass>();
    pipeline.register_pass<passes::normalization::StrideMinimization>();

    pipeline.run(*builder, *analysis_manager);

    auto& sdfg = builder->subject();

    // Check

    {
        auto& root = sdfg.root();
        EXPECT_EQ(root.size(), 3);

        auto loop_i_2 = dynamic_cast<structured_control_flow::StructuredLoop*>(&root.at(0).first);
        EXPECT_TRUE(loop_i_2 != nullptr);
        EXPECT_TRUE(SymEngine::eq(*loop_i_2->init(), *symbolic::integer(0)));
        EXPECT_TRUE(SymEngine::eq(*loop_i_2->condition(), *symbolic::Lt(loop_i_2->indvar(), symbolic::symbol("M"))));
        EXPECT_TRUE(SymEngine::eq(*loop_i_2->update(), *symbolic::add(loop_i_2->indvar(), symbolic::integer(1))));
        EXPECT_EQ(loop_i_2->root().size(), 1);
        auto loop_j_2 = dynamic_cast<structured_control_flow::StructuredLoop*>(&loop_i_2->root().at(0).first);
        EXPECT_TRUE(loop_j_2 != nullptr);
        EXPECT_TRUE(SymEngine::eq(*loop_j_2->init(), *loop_i_2->indvar()));
        EXPECT_TRUE(SymEngine::eq(*loop_j_2->condition(), *symbolic::Lt(loop_j_2->indvar(), symbolic::symbol("M"))));
        EXPECT_TRUE(SymEngine::eq(*loop_j_2->update(), *symbolic::add(loop_j_2->indvar(), symbolic::integer(1))));

        auto loop_i_3 = dynamic_cast<structured_control_flow::StructuredLoop*>(&root.at(1).first);
        EXPECT_TRUE(loop_i_3 != nullptr);
        EXPECT_TRUE(SymEngine::eq(*loop_i_3->init(), *symbolic::integer(0)));
        EXPECT_TRUE(SymEngine::eq(*loop_i_3->condition(), *symbolic::Lt(loop_i_3->indvar(), symbolic::symbol("M"))));
        EXPECT_TRUE(SymEngine::eq(*loop_i_3->update(), *symbolic::add(loop_i_3->indvar(), symbolic::integer(1))));
        EXPECT_EQ(loop_i_3->root().size(), 1);
        auto loop_k_3 = dynamic_cast<structured_control_flow::StructuredLoop*>(&loop_i_3->root().at(0).first);
        EXPECT_TRUE(loop_k_3 != nullptr);
        EXPECT_TRUE(SymEngine::eq(*loop_k_3->init(), *symbolic::integer(0)));
        EXPECT_TRUE(SymEngine::eq(*loop_k_3->condition(), *symbolic::Lt(loop_k_3->indvar(), symbolic::symbol("N"))));
        EXPECT_TRUE(SymEngine::eq(*loop_k_3->update(), *symbolic::add(loop_k_3->indvar(), symbolic::integer(1))));
        auto loop_j_3 = dynamic_cast<structured_control_flow::StructuredLoop*>(&loop_k_3->root().at(0).first);
        EXPECT_TRUE(loop_j_3 != nullptr);
        EXPECT_TRUE(SymEngine::eq(*loop_j_3->init(), *loop_i_3->indvar()));
        EXPECT_TRUE(SymEngine::eq(*loop_j_3->condition(), *symbolic::Lt(loop_j_3->indvar(), symbolic::symbol("M"))));
        EXPECT_TRUE(SymEngine::eq(*loop_j_3->update(), *symbolic::add(loop_j_3->indvar(), symbolic::integer(1))));
        EXPECT_EQ(loop_j_3->root().size(), 1);

        auto loop_i_4 = dynamic_cast<structured_control_flow::StructuredLoop*>(&root.at(2).first);
        EXPECT_TRUE(loop_i_4 != nullptr);
        EXPECT_TRUE(SymEngine::eq(*loop_i_4->init(), *symbolic::integer(0)));
        EXPECT_TRUE(SymEngine::eq(*loop_i_4->condition(), *symbolic::Lt(loop_i_4->indvar(), symbolic::symbol("M"))));
        EXPECT_TRUE(SymEngine::eq(*loop_i_4->update(), *symbolic::add(loop_i_4->indvar(), symbolic::integer(1))));
        EXPECT_EQ(loop_i_4->root().size(), 1);
        auto loop_j_4 = dynamic_cast<structured_control_flow::StructuredLoop*>(&loop_i_4->root().at(0).first);
        EXPECT_TRUE(loop_j_4 != nullptr);
        EXPECT_TRUE(SymEngine::eq(*loop_j_4->init(), *loop_i_4->indvar()));
        EXPECT_TRUE(SymEngine::eq(*loop_j_4->condition(), *symbolic::Lt(loop_j_4->indvar(), symbolic::symbol("M"))));
        EXPECT_TRUE(SymEngine::eq(*loop_j_4->update(), *symbolic::add(loop_j_4->indvar(), symbolic::integer(1))));
    }
}

TEST(StrideMinimizationTest, Polybench_gemm) {
    // Build SDFG
    auto init_sdfg = gemm();

    auto builder = std::make_unique<builder::StructuredSDFGBuilder>(init_sdfg);

    auto analysis_manager = std::make_unique<analysis::AnalysisManager>(builder->subject());

    // todo: get outermost loop
    passes::Pipeline data_parallism = passes::Pipeline::data_parallelism();
    data_parallism.run(*builder, *analysis_manager);

    // Pass
    passes::Pipeline pipeline("Perfect Loop Distribution");

    // Register passes for loop normalization

    pipeline.register_pass<passes::normalization::PerfectLoopDistributionPass>();
    pipeline.register_pass<passes::normalization::StrideMinimization>();

    pipeline.run(*builder, *analysis_manager);

    auto& sdfg = builder->subject();

    // Check
    /***
     for (i = 0; i < _PB_N; i++) {
        for (j = 0; j < _PB_N; j++)
            C[i][j] *= beta;
    }
    for (i = 0; i < _PB_N; i++) {
        for (k = 0; k < _PB_N; k++) {
            for (j = 0; j < _PB_N; j++)
                C[i][j] += alpha * A[i][k] * B[k][j];
        }
    }
    ***/

    {
        auto& root = sdfg.root();
        EXPECT_EQ(root.size(), 2);

        auto loop_i_1 = dynamic_cast<structured_control_flow::StructuredLoop*>(&root.at(0).first);
        EXPECT_TRUE(loop_i_1 != nullptr);
        EXPECT_TRUE(SymEngine::eq(*loop_i_1->init(), *symbolic::integer(0)));
        EXPECT_TRUE(SymEngine::eq(*loop_i_1->condition(), *symbolic::Lt(loop_i_1->indvar(), symbolic::symbol("N"))));
        EXPECT_TRUE(SymEngine::eq(*loop_i_1->update(), *symbolic::add(loop_i_1->indvar(), symbolic::integer(1))));
        EXPECT_EQ(loop_i_1->root().size(), 1);
        auto loop_j_1 = dynamic_cast<structured_control_flow::StructuredLoop*>(&loop_i_1->root().at(0).first);
        EXPECT_TRUE(loop_j_1 != nullptr);
        EXPECT_TRUE(SymEngine::eq(*loop_j_1->init(), *symbolic::integer(0)));
        EXPECT_TRUE(SymEngine::eq(*loop_j_1->condition(), *symbolic::Lt(loop_j_1->indvar(), symbolic::symbol("M"))));
        EXPECT_TRUE(SymEngine::eq(*loop_j_1->update(), *symbolic::add(loop_j_1->indvar(), symbolic::integer(1))));

        auto loop_i_2 = dynamic_cast<structured_control_flow::StructuredLoop*>(&root.at(1).first);
        EXPECT_TRUE(loop_i_2 != nullptr);
        EXPECT_TRUE(SymEngine::eq(*loop_i_2->init(), *symbolic::integer(0)));
        EXPECT_TRUE(SymEngine::eq(*loop_i_2->condition(), *symbolic::Lt(loop_i_2->indvar(), symbolic::symbol("N"))));
        EXPECT_TRUE(SymEngine::eq(*loop_i_2->update(), *symbolic::add(loop_i_2->indvar(), symbolic::integer(1))));
        EXPECT_EQ(loop_i_2->root().size(), 1);
        auto loop_k_2 = dynamic_cast<structured_control_flow::StructuredLoop*>(&loop_i_2->root().at(0).first);
        EXPECT_TRUE(loop_k_2 != nullptr);
        EXPECT_TRUE(SymEngine::eq(*loop_k_2->init(), *symbolic::integer(0)));
        EXPECT_TRUE(SymEngine::eq(*loop_k_2->condition(), *symbolic::Lt(loop_k_2->indvar(), symbolic::symbol("K"))));
        EXPECT_TRUE(SymEngine::eq(*loop_k_2->update(), *symbolic::add(loop_k_2->indvar(), symbolic::integer(1))));
        EXPECT_EQ(loop_k_2->root().size(), 1);
        auto loop_j_2 = dynamic_cast<structured_control_flow::StructuredLoop*>(&loop_k_2->root().at(0).first);
        EXPECT_TRUE(loop_j_2 != nullptr);
        EXPECT_TRUE(SymEngine::eq(*loop_j_2->init(), *symbolic::integer(0)));
        EXPECT_TRUE(SymEngine::eq(*loop_j_2->condition(), *symbolic::Lt(loop_j_2->indvar(), symbolic::symbol("M"))));
        EXPECT_TRUE(SymEngine::eq(*loop_j_2->update(), *symbolic::add(loop_j_2->indvar(), symbolic::integer(1))));
    }
}

TEST(StrideMinimizationTest, Polybench_symm) {
    // Build SDFG
    auto init_sdfg = symm();

    // Pass
    auto builder = std::make_unique<builder::StructuredSDFGBuilder>(init_sdfg);

    auto analysis_manager = std::make_unique<analysis::AnalysisManager>(builder->subject());

    // todo: get outermost loop
    passes::Pipeline data_parallism = passes::Pipeline::data_parallelism();
    data_parallism.run(*builder, *analysis_manager);

    // Pass
    passes::Pipeline pipeline("Perfect Loop Distribution");

    // Register passes for loop normalization

    pipeline.register_pass<passes::normalization::PerfectLoopDistributionPass>();
    pipeline.register_pass<passes::normalization::StrideMinimization>();

    pipeline.run(*builder, *analysis_manager);

    auto& sdfg = builder->subject();


    // Check (outdated)
    /***
    for (i = 0; i < _PB_M; i++) {
        for (j = 0; j < _PB_N; j++ ) {
            temp2[i][j] = 0;
        }
    }
    for (i = 0; i < _PB_M; i++) {
        for (k = 0; k < i; k++) {
            for (j = 0; j < _PB_N; j++ ) {
                C[k][j] += B[i][j] * A[i][k];
                temp2[i][j] += B[k][j] * A[i][k];
            }
        }
        for (j = 0; j < _PB_N; j++ ) {
            C[i][j] = C[i][j] + temp2[i][j];
        }
    }
    ***/

    {
        auto& root = sdfg.root();
        EXPECT_EQ(root.size(), 1);

        auto loop_i = dynamic_cast<structured_control_flow::StructuredLoop*>(&root.at(0).first);
        ASSERT_NE(loop_i, nullptr);
        EXPECT_TRUE(SymEngine::eq(*loop_i->init(), *symbolic::integer(0)));
        EXPECT_TRUE(SymEngine::eq(*loop_i->condition(), *symbolic::Lt(loop_i->indvar(), symbolic::integer(16))));
        EXPECT_TRUE(SymEngine::eq(*loop_i->update(), *symbolic::add(loop_i->indvar(), symbolic::integer(1))));

        auto& body_i = loop_i->root();
        ASSERT_EQ(body_i.size(), 1);

        auto loop_j = dynamic_cast<structured_control_flow::StructuredLoop*>(&body_i.at(0).first);
        ASSERT_NE(loop_j, nullptr);
        EXPECT_TRUE(SymEngine::eq(*loop_j->init(), *symbolic::integer(0)));
        EXPECT_TRUE(SymEngine::eq(*loop_j->condition(), *symbolic::Lt(loop_j->indvar(), symbolic::integer(12))));
        EXPECT_TRUE(SymEngine::eq(*loop_j->update(), *symbolic::add(loop_j->indvar(), symbolic::integer(1))));

        auto& body_j = loop_j->root();
        ASSERT_EQ(body_j.size(), 3);

        // body_j[0]: block (tmp = 0)
        auto* block_init = dynamic_cast<structured_control_flow::Block*>(&body_j.at(0).first);
        EXPECT_NE(block_init, nullptr);

        // body_j[1]: for k [0, i)
        auto loop_k = dynamic_cast<structured_control_flow::StructuredLoop*>(&body_j.at(1).first);
        ASSERT_NE(loop_k, nullptr);
        EXPECT_TRUE(SymEngine::eq(*loop_k->init(), *symbolic::integer(0)));
        EXPECT_TRUE(SymEngine::eq(*loop_k->condition(), *symbolic::Lt(loop_k->indvar(), loop_i->indvar())));
        EXPECT_TRUE(SymEngine::eq(*loop_k->update(), *symbolic::add(loop_k->indvar(), symbolic::integer(1))));
        EXPECT_EQ(loop_k->root().size(), 2);

        // body_j[2]: block (C[i][j] = C[i][j] + tmp)
        auto* block_final = dynamic_cast<structured_control_flow::Block*>(&body_j.at(2).first);
        EXPECT_NE(block_final, nullptr);
    }
}

TEST(StrideMinimizationTest, Polybench_gemver) {
    // Build SDFG
    auto init_sdfg = gemver();

    auto builder = std::make_unique<builder::StructuredSDFGBuilder>(init_sdfg);

    auto analysis_manager = std::make_unique<analysis::AnalysisManager>(builder->subject());

    // todo: get outermost loop
    passes::Pipeline data_parallism = passes::Pipeline::data_parallelism();
    data_parallism.run(*builder, *analysis_manager);

    // Pass
    passes::Pipeline pipeline("Perfect Loop Distribution");

    // Register passes for loop normalization

    pipeline.register_pass<passes::normalization::PerfectLoopDistributionPass>();
    pipeline.register_pass<passes::normalization::StrideMinimization>();

    pipeline.run(*builder, *analysis_manager);

    auto& sdfg = builder->subject();

    // Check
    /***
    for (i = 0; i < _PB_N; i++)
        for (j = 0; j < _PB_N; j++)
            A[i][j] = A[i][j] + u1[i] * v1[j] + u2[i] * v2[j];

    for (j = 0; j < _PB_N; j++)
        for (i = 0; i < _PB_N; i++)
            x[i] = x[i] + A[j][i] * y[j];

    for (i = 0; i < _PB_N; i++)
        x[i] = x[i] + z[i];

    for (i = 0; i < _PB_N; i++)
        for (j = 0; j < _PB_N; j++)
            w[i] = w[i] + A[i][j] * x[j];
    */

    {
        auto& root = sdfg.root();
        EXPECT_EQ(root.size(), 4);

        auto loop_i_1 = dynamic_cast<structured_control_flow::StructuredLoop*>(&root.at(0).first);
        EXPECT_TRUE(loop_i_1 != nullptr);
        EXPECT_EQ(loop_i_1->indvar()->get_name(), "i_1");
        auto& body_i_1 = loop_i_1->root();

        auto loop_j_1 = dynamic_cast<structured_control_flow::StructuredLoop*>(&body_i_1.at(0).first);
        EXPECT_TRUE(loop_j_1 != nullptr);
        EXPECT_EQ(loop_j_1->indvar()->get_name(), "j_1");

        auto loop_j_2 = dynamic_cast<structured_control_flow::StructuredLoop*>(&root.at(1).first);
        EXPECT_TRUE(loop_j_2 != nullptr);
        EXPECT_EQ(loop_j_2->indvar()->get_name(), "j_2");
        auto& body_j_2 = loop_j_2->root();

        auto loop_i_2 = dynamic_cast<structured_control_flow::StructuredLoop*>(&body_j_2.at(0).first);
        EXPECT_TRUE(loop_i_2 != nullptr);
        EXPECT_EQ(loop_i_2->indvar()->get_name(), "i_2");

        auto loop_i_3 = dynamic_cast<structured_control_flow::StructuredLoop*>(&root.at(2).first);
        EXPECT_TRUE(loop_i_3 != nullptr);
        EXPECT_EQ(loop_i_3->indvar()->get_name(), "i_3");

        auto loop_i_4 = dynamic_cast<structured_control_flow::StructuredLoop*>(&root.at(3).first);
        EXPECT_TRUE(loop_i_4 != nullptr);
        EXPECT_EQ(loop_i_4->indvar()->get_name(), "i_4");
        auto& body_i_4 = loop_i_4->root();

        auto loop_j_4 = dynamic_cast<structured_control_flow::StructuredLoop*>(&body_i_4.at(0).first);
        EXPECT_TRUE(loop_j_4 != nullptr);
        EXPECT_EQ(loop_j_4->indvar()->get_name(), "j_4");
    }
}

TEST(StrideMinimizationTest, Polybench_gesummv) {
    // Build SDFG
    auto init_sdfg = gesummv();
    auto builder = std::make_unique<builder::StructuredSDFGBuilder>(init_sdfg);

    auto analysis_manager = std::make_unique<analysis::AnalysisManager>(builder->subject());

    // todo: get outermost loop
    passes::Pipeline data_parallism = passes::Pipeline::data_parallelism();
    data_parallism.run(*builder, *analysis_manager);

    // Pass
    passes::Pipeline pipeline("Perfect Loop Distribution");

    // Register passes for loop normalization

    pipeline.register_pass<passes::normalization::PerfectLoopDistributionPass>();
    pipeline.register_pass<passes::normalization::StrideMinimization>();

    pipeline.run(*builder, *analysis_manager);

    auto& sdfg = builder->subject();

    // Check
    /***
    for (i = 0; i < _PB_N; i++) {
        tmp[i] = SCALAR_VAL(0.0);
    }
    for (i = 0; i < _PB_N; i++) {
        y[i] = SCALAR_VAL(0.0);
    }
    for (i = 0; i < _PB_N; i++) {
        for (j = 0; j < _PB_N; j++) {
            tmp[i] = A[i][j] * x[j] + tmp[i];
            y[i] = B[i][j] * x[j] + y[i];
            }
    }
    for (i = 0; i < _PB_N; i++) {
        y[i] = tmp[i] + y[i];
    }
    */
}

TEST(StrideMinimizationTest, Polybench_syr2k) {
    // Build SDFG
    auto init_sdfg = syr2k();

    auto builder = std::make_unique<builder::StructuredSDFGBuilder>(init_sdfg);

    auto analysis_manager = std::make_unique<analysis::AnalysisManager>(builder->subject());

    // todo: get outermost loop
    passes::Pipeline data_parallism = passes::Pipeline::data_parallelism();
    data_parallism.run(*builder, *analysis_manager);

    // Pass
    passes::Pipeline pipeline("Perfect Loop Distribution");

    // Register passes for loop normalization

    pipeline.register_pass<passes::normalization::PerfectLoopDistributionPass>();
    pipeline.register_pass<passes::normalization::StrideMinimization>();

    pipeline.run(*builder, *analysis_manager);

    auto& sdfg = builder->subject();

    // Check
    /***
    for (i = 0; i < _PB_N; i++) {
        for (j = 0; j <= i; j++) {
            C[i][j] *= beta;
        }
    }

    for (i = 0; i < _PB_N; i++) {
        for (j = 0; j <= i; j++) {
            for (k = 0; k < _PB_M; k++) {
                C[i][j] += A[j][k]*B[i][k] + B[j][k]*A[i][k];
            }
        }
    }
    ***/

    {
        auto& root = sdfg.root();
        EXPECT_EQ(root.size(), 2);

        auto loop_i_1 = dynamic_cast<structured_control_flow::StructuredLoop*>(&root.at(0).first);
        EXPECT_TRUE(loop_i_1 != nullptr);
        EXPECT_TRUE(SymEngine::eq(*loop_i_1->init(), *symbolic::integer(0)));
        EXPECT_TRUE(SymEngine::eq(*loop_i_1->condition(), *symbolic::Lt(loop_i_1->indvar(), symbolic::symbol("N"))));
        EXPECT_TRUE(SymEngine::eq(*loop_i_1->update(), *symbolic::add(loop_i_1->indvar(), symbolic::integer(1))));
        EXPECT_EQ(loop_i_1->root().size(), 1);
        {
            auto loop_j_1 = dynamic_cast<structured_control_flow::StructuredLoop*>(&loop_i_1->root().at(0).first);
            EXPECT_TRUE(loop_j_1 != nullptr);
            EXPECT_TRUE(SymEngine::eq(*loop_j_1->init(), *symbolic::integer(0)));
            EXPECT_TRUE(SymEngine::eq(*loop_j_1->condition(), *symbolic::Le(loop_j_1->indvar(), loop_i_1->indvar())));
            EXPECT_TRUE(SymEngine::eq(*loop_j_1->update(), *symbolic::add(loop_j_1->indvar(), symbolic::integer(1))));
        }

        auto loop_i_2 = dynamic_cast<structured_control_flow::StructuredLoop*>(&root.at(1).first);
        EXPECT_TRUE(loop_i_2 != nullptr);
        EXPECT_TRUE(SymEngine::eq(*loop_i_2->init(), *symbolic::integer(0)));
        EXPECT_TRUE(SymEngine::eq(*loop_i_2->condition(), *symbolic::Lt(loop_i_2->indvar(), symbolic::symbol("N"))));
        EXPECT_TRUE(SymEngine::eq(*loop_i_2->update(), *symbolic::add(loop_i_2->indvar(), symbolic::integer(1))));
        EXPECT_EQ(loop_i_2->root().size(), 1);
        {
            auto loop_j_2 = dynamic_cast<structured_control_flow::StructuredLoop*>(&loop_i_2->root().at(0).first);
            EXPECT_TRUE(loop_j_2 != nullptr);
            EXPECT_TRUE(SymEngine::eq(*loop_j_2->init(), *symbolic::integer(0)));
            EXPECT_TRUE(SymEngine::eq(*loop_j_2->condition(), *symbolic::Le(loop_j_2->indvar(), loop_i_2->indvar())));
            EXPECT_TRUE(SymEngine::eq(*loop_j_2->update(), *symbolic::add(loop_j_2->indvar(), symbolic::integer(1))));
            {
                auto loop_k = dynamic_cast<structured_control_flow::StructuredLoop*>(&loop_j_2->root().at(0).first);
                EXPECT_TRUE(loop_k != nullptr);
                EXPECT_TRUE(SymEngine::eq(*loop_k->init(), *symbolic::integer(0)));
                EXPECT_TRUE(SymEngine::eq(*loop_k->condition(), *symbolic::Lt(loop_k->indvar(), symbolic::symbol("M")))
                );
                EXPECT_TRUE(SymEngine::eq(*loop_k->update(), *symbolic::add(loop_k->indvar(), symbolic::integer(1))));
            }
        }
    }
}

TEST(StrideMinimizationTest, Polybench_syrk) {
    auto init_sdfg = syrk();

    auto builder = std::make_unique<builder::StructuredSDFGBuilder>(init_sdfg);

    auto analysis_manager = std::make_unique<analysis::AnalysisManager>(builder->subject());

    // todo: get outermost loop
    passes::Pipeline data_parallism = passes::Pipeline::data_parallelism();
    data_parallism.run(*builder, *analysis_manager);

    // Pass
    passes::Pipeline pipeline("Perfect Loop Distribution");

    // Register passes for loop normalization

    pipeline.register_pass<passes::normalization::PerfectLoopDistributionPass>();
    pipeline.register_pass<passes::normalization::StrideMinimization>();

    pipeline.run(*builder, *analysis_manager);

    auto& sdfg = builder->subject();

    // Check
    /***
     for (i = 0; i < _PB_N; i++) {
        for (j = 0; j <= i; j++)
            C[i][j] *= beta;
    }

    for (i = 0; i < _PB_N; i++) {
        for (j = 0; j <= i; j++)
            for (k = 0; k < _PB_M; k++) {
                C[i][j] += A[i][k] * A[j][k];
        }
    }
    ***/

    {
        auto& root = sdfg.root();
        EXPECT_EQ(root.size(), 2);

        auto loop_i = dynamic_cast<structured_control_flow::StructuredLoop*>(&root.at(0).first);
        EXPECT_TRUE(loop_i != nullptr);
        EXPECT_TRUE(SymEngine::eq(*loop_i->init(), *symbolic::integer(0)));
        EXPECT_TRUE(SymEngine::eq(*loop_i->condition(), *symbolic::Lt(loop_i->indvar(), symbolic::symbol("N"))));
        EXPECT_TRUE(SymEngine::eq(*loop_i->update(), *symbolic::add(loop_i->indvar(), symbolic::integer(1))));
        EXPECT_EQ(loop_i->root().size(), 1);
        {
            auto loop_j_1 = dynamic_cast<structured_control_flow::StructuredLoop*>(&loop_i->root().at(0).first);
            EXPECT_TRUE(loop_j_1 != nullptr);
            EXPECT_TRUE(SymEngine::eq(*loop_j_1->init(), *symbolic::integer(0)));
            EXPECT_TRUE(SymEngine::eq(*loop_j_1->condition(), *symbolic::Le(loop_j_1->indvar(), loop_i->indvar())));
            EXPECT_TRUE(SymEngine::eq(*loop_j_1->update(), *symbolic::add(loop_j_1->indvar(), symbolic::integer(1))));
        }

        auto loop_i_2 = dynamic_cast<structured_control_flow::StructuredLoop*>(&root.at(1).first);
        EXPECT_TRUE(loop_i_2 != nullptr);
        EXPECT_TRUE(SymEngine::eq(*loop_i_2->init(), *symbolic::integer(0)));
        EXPECT_TRUE(SymEngine::eq(*loop_i_2->condition(), *symbolic::Lt(loop_i_2->indvar(), symbolic::symbol("N"))));
        EXPECT_TRUE(SymEngine::eq(*loop_i_2->update(), *symbolic::add(loop_i_2->indvar(), symbolic::integer(1))));
        EXPECT_EQ(loop_i_2->root().size(), 1);
        {
            auto loop_j_2 = dynamic_cast<structured_control_flow::StructuredLoop*>(&loop_i_2->root().at(0).first);
            EXPECT_TRUE(loop_j_2 != nullptr);
            EXPECT_TRUE(SymEngine::eq(*loop_j_2->init(), *symbolic::integer(0)));
            EXPECT_TRUE(SymEngine::eq(*loop_j_2->condition(), *symbolic::Le(loop_j_2->indvar(), loop_i_2->indvar())));
            EXPECT_TRUE(SymEngine::eq(*loop_j_2->update(), *symbolic::add(loop_j_2->indvar(), symbolic::integer(1))));
            {
                auto loop_k = dynamic_cast<structured_control_flow::StructuredLoop*>(&loop_j_2->root().at(0).first);
                EXPECT_TRUE(loop_k != nullptr);
                EXPECT_TRUE(SymEngine::eq(*loop_k->init(), *symbolic::integer(0)));
                EXPECT_TRUE(SymEngine::eq(*loop_k->condition(), *symbolic::Lt(loop_k->indvar(), symbolic::symbol("M")))
                );
                EXPECT_TRUE(SymEngine::eq(*loop_k->update(), *symbolic::add(loop_k->indvar(), symbolic::integer(1))));
            }
        }
    }
}

TEST(StrideMinimizationTest, Polybench_trmm) {
    // Build SDFG
    auto init_sdfg = trmm();

    auto builder = std::make_unique<builder::StructuredSDFGBuilder>(init_sdfg);

    auto analysis_manager = std::make_unique<analysis::AnalysisManager>(builder->subject());

    // todo: get outermost loop
    passes::Pipeline data_parallism = passes::Pipeline::data_parallelism();
    data_parallism.run(*builder, *analysis_manager);

    // Pass
    passes::Pipeline pipeline("Perfect Loop Distribution");

    // Register passes for loop normalization

    pipeline.register_pass<passes::normalization::PerfectLoopDistributionPass>();
    pipeline.register_pass<passes::normalization::StrideMinimization>();

    pipeline.run(*builder, *analysis_manager);

    auto& sdfg = builder->subject();

    // Check
    /***
     for (i = 0; i < _PB_M; i++) {
        for (k = i+1; k < _PB_M; k++) {
            for (j = 0; j < _PB_N; j++) {
                B[i][j] += A[k][i] * B[k][j];
            }
        }
        for (j = 0; j < _PB_N; j++) {
            B[i][j] = alpha * B[i][j];
        }
    }
    */

    {
        auto& root = sdfg.root();
        EXPECT_EQ(root.size(), 1);

        auto loop_i_1 = dynamic_cast<structured_control_flow::StructuredLoop*>(&root.at(0).first);
        EXPECT_TRUE(loop_i_1 != nullptr);
        EXPECT_TRUE(SymEngine::eq(*loop_i_1->init(), *symbolic::integer(0)));
        EXPECT_TRUE(SymEngine::eq(*loop_i_1->condition(), *symbolic::Lt(loop_i_1->indvar(), symbolic::symbol("M"))));
        EXPECT_TRUE(SymEngine::eq(*loop_i_1->update(), *symbolic::add(loop_i_1->indvar(), symbolic::integer(1))));
        EXPECT_EQ(loop_i_1->root().size(), 2);
        {
            auto loop_k_1 = dynamic_cast<structured_control_flow::StructuredLoop*>(&loop_i_1->root().at(0).first);
            EXPECT_TRUE(loop_k_1 != nullptr);
            EXPECT_TRUE(SymEngine::eq(*loop_k_1->init(), *symbolic::add(loop_i_1->indvar(), symbolic::integer(1))));
            EXPECT_TRUE(SymEngine::eq(*loop_k_1->condition(), *symbolic::Lt(loop_k_1->indvar(), symbolic::symbol("M")))
            );
            EXPECT_TRUE(SymEngine::eq(*loop_k_1->update(), *symbolic::add(loop_k_1->indvar(), symbolic::integer(1))));
            {
                auto loop_j_1 = dynamic_cast<structured_control_flow::StructuredLoop*>(&loop_k_1->root().at(0).first);
                EXPECT_TRUE(loop_j_1 != nullptr);
                EXPECT_TRUE(SymEngine::eq(*loop_j_1->init(), *symbolic::integer(0)));
                EXPECT_TRUE(SymEngine::eq(*loop_j_1->condition(), *symbolic::Lt(loop_j_1->indvar(), symbolic::symbol("N")))
                );
                EXPECT_TRUE(SymEngine::eq(*loop_j_1->update(), *symbolic::add(loop_j_1->indvar(), symbolic::integer(1)))
                );
            }

            auto loop_j_2 = dynamic_cast<structured_control_flow::StructuredLoop*>(&loop_i_1->root().at(1).first);
            EXPECT_TRUE(loop_j_2 != nullptr);
            EXPECT_TRUE(SymEngine::eq(*loop_j_2->init(), *symbolic::integer(0)));
            EXPECT_TRUE(SymEngine::eq(*loop_j_2->condition(), *symbolic::Lt(loop_j_2->indvar(), symbolic::symbol("N")))
            );
            EXPECT_TRUE(SymEngine::eq(*loop_j_2->update(), *symbolic::add(loop_j_2->indvar(), symbolic::integer(1))));
        }
    }
}

TEST(StrideMinimizationTest, Polybench_atax) {
    // Build SDFG
    auto init_sdfg = atax();

    // Pass
    auto builder = std::make_unique<builder::StructuredSDFGBuilder>(init_sdfg);

    auto analysis_manager = std::make_unique<analysis::AnalysisManager>(builder->subject());

    // todo: get outermost loop
    passes::Pipeline data_parallism = passes::Pipeline::data_parallelism();
    data_parallism.run(*builder, *analysis_manager);

    // Pass
    passes::Pipeline pipeline("Perfect Loop Distribution");

    // Register passes for loop normalization

    pipeline.register_pass<passes::normalization::PerfectLoopDistributionPass>();
    pipeline.register_pass<passes::normalization::StrideMinimization>();

    pipeline.run(*builder, *analysis_manager);

    auto& sdfg = builder->subject();

    // Check
    /***
    for (i = 0; i < _PB_N; i++)
        y[i] = 0;

    for (i = 0; i < _PB_M; i++) {
        tmp[i] = SCALAR_VAL(0.0);
    }

    for (i = 0; i < _PB_M; i++) {
        for (j = 0; j < _PB_N; j++)
            tmp[i] = tmp[i] + A[i][j] * x[j];
    }

    for (i = 0; i < _PB_M; i++) {
        for (j = 0; j < _PB_N; j++)
            y[j] = y[j] + A[i][j] * tmp[i];
    }
    */
}

TEST(StrideMinimizationTest, Polybench_bicg) {
    // Build SDFG
    auto init_sdfg = bicg();

    // Pass
    auto builder = std::make_unique<builder::StructuredSDFGBuilder>(init_sdfg);

    auto analysis_manager = std::make_unique<analysis::AnalysisManager>(builder->subject());

    // todo: get outermost loop
    passes::Pipeline data_parallism = passes::Pipeline::data_parallelism();
    data_parallism.run(*builder, *analysis_manager);

    // Pass
    passes::Pipeline pipeline("Perfect Loop Distribution");

    // Register passes for loop normalization

    pipeline.register_pass<passes::normalization::PerfectLoopDistributionPass>();
    pipeline.register_pass<passes::normalization::StrideMinimization>();

    pipeline.run(*builder, *analysis_manager);

    auto& sdfg = builder->subject();

    // Check
    /***
        for (i = 0; i < _PB_M; i++)
            s[i] = 0;
        for (i = 0; i < _PB_N; i++)
            q[i] = SCALAR_VAL(0.0);

        for (i = 0; i < _PB_N; i++) {
            for (j = 0; j < _PB_M; j++) {
                s[j] = s[j] + r[i] * A[i][j];
                q[i] = q[i] + A[i][j] * p[j];
            }
        }
    */
}

TEST(StrideMinimizationTest, Polybench_doitgen) {
    // Build SDFG
    auto init_sdfg = doitgen();

    // Pass
    auto builder = std::make_unique<builder::StructuredSDFGBuilder>(init_sdfg);

    auto analysis_manager = std::make_unique<analysis::AnalysisManager>(builder->subject());

    // todo: get outermost loop
    passes::Pipeline data_parallism = passes::Pipeline::data_parallelism();
    data_parallism.run(*builder, *analysis_manager);

    // Pass
    passes::Pipeline pipeline("Perfect Loop Distribution");

    // Register passes for loop normalization

    pipeline.register_pass<passes::normalization::PerfectLoopDistributionPass>();
    pipeline.register_pass<passes::normalization::StrideMinimization>();

    pipeline.run(*builder, *analysis_manager);

    auto& sdfg = builder->subject();

    // Check
    /***
    for (r = 0; r < _PB_NR; r++) {}
        for (q = 0; q < _PB_NQ; q++)  {
            for (p = 0; p < _PB_NP; p++)  {
                    sum[p] = SCALAR_VAL(0.0);
            }
                for (s = 0; s < _PB_NP; s++) {
                for (p = 0; p < _PB_NP; p++)
                        sum[p] += A[r][q][s] * C4[s][p];
            }
            for (p = 0; p < _PB_NP; p++)
                    A[r][q][p] = sum[p];
        }
    }
    */

    {
        auto& root = sdfg.root();
        EXPECT_EQ(root.size(), 1);

        auto loop_r = dynamic_cast<structured_control_flow::StructuredLoop*>(&root.at(0).first);
        EXPECT_TRUE(loop_r != nullptr);
        EXPECT_EQ(loop_r->indvar()->get_name(), "r");
        auto& body_r = loop_r->root();

        auto loop_q = dynamic_cast<structured_control_flow::StructuredLoop*>(&body_r.at(0).first);
        EXPECT_TRUE(loop_q != nullptr);
        EXPECT_EQ(loop_q->indvar()->get_name(), "q");
        auto& body_q = loop_q->root();

        auto loop_p_1 = dynamic_cast<structured_control_flow::StructuredLoop*>(&body_q.at(0).first);
        EXPECT_TRUE(loop_p_1 != nullptr);
        EXPECT_EQ(loop_p_1->indvar()->get_name(), "p_10");
        auto& body_p_1 = loop_p_1->root();

        auto loop_s = dynamic_cast<structured_control_flow::StructuredLoop*>(&body_q.at(1).first);
        EXPECT_TRUE(loop_s != nullptr);
        EXPECT_EQ(loop_s->indvar()->get_name(), "s");
        auto& body_s = loop_s->root();

        auto loop_p_2 = dynamic_cast<structured_control_flow::StructuredLoop*>(&body_s.at(0).first);
        EXPECT_TRUE(loop_p_2 != nullptr);
        EXPECT_EQ(loop_p_2->indvar()->get_name(), "p_1");

        auto loop_p_3 = dynamic_cast<structured_control_flow::StructuredLoop*>(&body_q.at(2).first);
        EXPECT_TRUE(loop_p_3 != nullptr);
        EXPECT_EQ(loop_p_3->indvar()->get_name(), "p_2");
    }
}

TEST(StrideMinimizationTest, Polybench_mvt) {
    // Build SDFG
    auto init_sdfg = mvt();

    // Pass
    auto builder = std::make_unique<builder::StructuredSDFGBuilder>(init_sdfg);

    auto analysis_manager = std::make_unique<analysis::AnalysisManager>(builder->subject());

    // todo: get outermost loop
    passes::Pipeline data_parallism = passes::Pipeline::data_parallelism();
    data_parallism.run(*builder, *analysis_manager);

    // Pass
    passes::Pipeline pipeline("Perfect Loop Distribution");

    // Register passes for loop normalization

    pipeline.register_pass<passes::normalization::PerfectLoopDistributionPass>();
    pipeline.register_pass<passes::normalization::StrideMinimization>();

    pipeline.run(*builder, *analysis_manager);

    auto& sdfg = builder->subject();

    // Check
    /***
    for (i = 0; i < _PB_N; i++)
        for (j = 0; j < _PB_N; j++)
            x1[i] = x1[i] + A[i][j] * y_1[j];
    for (j = 0; j < _PB_N; j++)
        for (i = 0; i < _PB_N; i++)
            x2[i] = x2[i] + A[j][i] * y_2[j];
    */

    {
        auto& root = sdfg.root();
        EXPECT_EQ(root.size(), 2);

        auto loop_i_1 = dynamic_cast<structured_control_flow::StructuredLoop*>(&root.at(0).first);
        EXPECT_TRUE(loop_i_1 != nullptr);
        EXPECT_TRUE(loop_i_1->indvar()->get_name() == "i_1");
        auto& body_loop_i_1 = loop_i_1->root();

        auto loop_j_1 = dynamic_cast<structured_control_flow::StructuredLoop*>(&body_loop_i_1.at(0).first);
        EXPECT_TRUE(loop_j_1 != nullptr);
        EXPECT_TRUE(loop_j_1->indvar()->get_name() == "j_1");

        auto loop_i_2 = dynamic_cast<structured_control_flow::StructuredLoop*>(&root.at(1).first);
        EXPECT_TRUE(loop_i_2 != nullptr);
        EXPECT_TRUE(loop_i_2->indvar()->get_name() == "j_2");
        auto& body_loop_i_2 = loop_i_2->root();

        auto loop_j_2 = dynamic_cast<structured_control_flow::StructuredLoop*>(&body_loop_i_2.at(0).first);
        EXPECT_TRUE(loop_j_2 != nullptr);
        EXPECT_TRUE(loop_j_2->indvar()->get_name() == "i_2");
    }
}

TEST(StrideMinimizationTest, Polybench_cholesky) {
    // Build SDFG
    auto init_sdfg = cholesky();

    ;

    // Pass
    auto builder = std::make_unique<builder::StructuredSDFGBuilder>(init_sdfg);

    auto analysis_manager = std::make_unique<analysis::AnalysisManager>(builder->subject());

    // todo: get outermost loop
    passes::Pipeline data_parallism = passes::Pipeline::data_parallelism();
    data_parallism.run(*builder, *analysis_manager);

    // Pass
    passes::Pipeline pipeline("Perfect Loop Distribution");

    // Register passes for loop normalization

    pipeline.register_pass<passes::normalization::PerfectLoopDistributionPass>();
    pipeline.register_pass<passes::normalization::StrideMinimization>();

    pipeline.run(*builder, *analysis_manager);

    auto& sdfg = builder->subject();

    // Check
    /***
     for (i = 0; i < _PB_N; i++) {
        //j<i
        for (j = 0; j < i; j++) {
            for (k = 0; k < j; k++) {
                A[i][j] -= A[i][k] * A[j][k];
            }

            A[i][j] /= A[j][j];
        }
        // i==j case
        for (k = 0; k < i; k++) {
            A[i][i] -= A[i][k] * A[i][k];
        }
        A[i][i] = sqrt(A[i][i]);
    }
    */

    {
        auto& root = sdfg.root();
        EXPECT_EQ(root.size(), 1);

        auto loop_i = dynamic_cast<structured_control_flow::StructuredLoop*>(&root.at(0).first);
        EXPECT_TRUE(loop_i != nullptr);
        EXPECT_TRUE(SymEngine::eq(*loop_i->init(), *symbolic::integer(0)));
        EXPECT_TRUE(SymEngine::eq(*loop_i->condition(), *symbolic::Lt(loop_i->indvar(), symbolic::symbol("N"))));
        EXPECT_TRUE(SymEngine::eq(*loop_i->update(), *symbolic::add(loop_i->indvar(), symbolic::integer(1))));
        auto& body_i = loop_i->root();
        EXPECT_EQ(body_i.size(), 3);

        auto loop_j_1 = dynamic_cast<structured_control_flow::StructuredLoop*>(&body_i.at(0).first);
        EXPECT_TRUE(loop_j_1 != nullptr);
        EXPECT_TRUE(SymEngine::eq(*loop_j_1->init(), *symbolic::integer(0)));
        EXPECT_TRUE(SymEngine::eq(*loop_j_1->condition(), *symbolic::Lt(loop_j_1->indvar(), loop_i->indvar())));
        EXPECT_TRUE(SymEngine::eq(*loop_j_1->update(), *symbolic::add(loop_j_1->indvar(), symbolic::integer(1))));
        auto& body_j_1 = loop_j_1->root();
        EXPECT_EQ(body_j_1.size(), 2);

        auto loop_k_1 = dynamic_cast<structured_control_flow::StructuredLoop*>(&body_j_1.at(0).first);
        EXPECT_TRUE(loop_k_1 != nullptr);
        EXPECT_TRUE(SymEngine::eq(*loop_k_1->init(), *symbolic::integer(0)));
        EXPECT_TRUE(SymEngine::eq(*loop_k_1->condition(), *symbolic::Lt(loop_k_1->indvar(), loop_j_1->indvar())));
        EXPECT_TRUE(SymEngine::eq(*loop_k_1->update(), *symbolic::add(loop_k_1->indvar(), symbolic::integer(1))));
        auto& body_k_1 = loop_k_1->root();

        auto loop_k_2 = dynamic_cast<structured_control_flow::StructuredLoop*>(&body_i.at(1).first);
        EXPECT_TRUE(loop_k_2 != nullptr);
        EXPECT_TRUE(SymEngine::eq(*loop_k_2->init(), *symbolic::integer(0)));
        EXPECT_TRUE(SymEngine::eq(*loop_k_2->condition(), *symbolic::Lt(loop_k_2->indvar(), loop_i->indvar())));
        EXPECT_TRUE(SymEngine::eq(*loop_k_2->update(), *symbolic::add(loop_k_2->indvar(), symbolic::integer(1))));
        auto& body_j_2 = loop_k_2->root();
    }
}

TEST(StrideMinimizationTest, Polybench_fdtd_2d) {
    // Build SDFG
    auto init_sdfg = fdtd_2d();

    auto builder = std::make_unique<builder::StructuredSDFGBuilder>(init_sdfg);

    auto analysis_manager = std::make_unique<analysis::AnalysisManager>(builder->subject());

    passes::Pipeline data_parallism = passes::Pipeline::data_parallelism();
    data_parallism.run(*builder, *analysis_manager);

    // Pass
    passes::normalization::PerfectLoopDistributionPass pass;
    passes::normalization::StrideMinimization stride_minimization_pass;

    auto root = &builder->subject().root();

    // todo: get outermost loop
    auto outermost_loop = dynamic_cast<sdfg::structured_control_flow::For*>(&root->at(0).first);

    bool applies;
    do {
        applies = false;
        applies = pass.run_pass(*builder, *analysis_manager);
    } while (applies);

    do {
        applies = false;
        applies = stride_minimization_pass.run_pass(*builder, *analysis_manager);
        EXPECT_FALSE(applies);
    } while (applies);

    auto& sdfg = builder->subject();
}
