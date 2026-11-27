#include "sdfg/reordering/transformations/loop_distribute.h"

#include <gtest/gtest.h>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_nodes/math/blas/gemm_node.h"
#include "sdfg/structured_control_flow/map.h"

using namespace sdfg;

TEST(LoopDistributeTest, BasicDistribution) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);

    types::Pointer opaque_desc;
    builder.add_container("A", opaque_desc, true);
    builder.add_container("B", opaque_desc, true);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);

    // Define loop
    auto bound = symbolic::symbol("N");
    auto indvar = symbolic::symbol("i");
    auto init = symbolic::integer(0);
    auto condition = symbolic::Lt(indvar, bound);
    auto update = symbolic::add(indvar, symbolic::integer(1));

    auto& loop = builder.add_for(root, indvar, condition, init, update);
    auto& body = loop.root();

    // Add first computation (will be distributed)
    auto& block1 = builder.add_block(body);
    auto& A_in = builder.add_access(block1, "A");
    auto& A_out = builder.add_access(block1, "A");
    auto& tasklet1 = builder.add_tasklet(block1, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block1, A_in, tasklet1, "_in", {symbolic::symbol("i")}, desc);
    builder.add_computational_memlet(block1, tasklet1, "_out", A_out, {symbolic::symbol("i")}, desc);

    // Add second computation
    auto& block2 = builder.add_block(body);
    auto& B_in = builder.add_access(block2, "B");
    auto& B_out = builder.add_access(block2, "B");
    auto& tasklet2 = builder.add_tasklet(block2, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block2, B_in, tasklet2, "_in", {symbolic::symbol("i")}, desc);
    builder.add_computational_memlet(block2, tasklet2, "_out", B_out, {symbolic::symbol("i")}, desc);

    // Verify initial state
    EXPECT_EQ(body.size(), 2);

    analysis::AnalysisManager analysis_manager(builder.subject());
    reordering::transformations::LoopDistribute transformation(loop);
    EXPECT_TRUE(transformation.can_be_applied(builder, analysis_manager));

    // Store element counts before transformation
    size_t initial_loop_count = 1;

    transformation.apply(builder, analysis_manager);

    // Verify transformation created a new loop. New semantics: child_ stays
    // in the original loop in place; siblings before/after move into new
    // loops added before/after. Default constructor uses child_ = at(0), so
    // the only sibling (block2) is moved into a suffix loop added AFTER.
    auto& new_sdfg = builder.subject();
    EXPECT_EQ(new_sdfg.root().size(), 2); // Now we should have 2 loops

    // First loop is the original For (kept) holding block1
    auto first_loop = dyn_cast<structured_control_flow::For*>(&new_sdfg.root().at(0));
    EXPECT_TRUE(first_loop != nullptr);
    EXPECT_EQ(first_loop->root().size(), 1);
    EXPECT_EQ(&first_loop->root().at(0), &block1);

    // Second loop is the new suffix holding block2, sequential like the original
    auto second_loop = dyn_cast<structured_control_flow::For*>(&new_sdfg.root().at(1));
    EXPECT_TRUE(second_loop != nullptr);
    EXPECT_EQ(second_loop->root().size(), 1);
    EXPECT_EQ(&second_loop->root().at(0), &block2);
}

TEST(LoopDistributeTest, MapPiecesKeepSchedule) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);
    auto& root = builder.subject().root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    types::Pointer opaque_desc;
    builder.add_container("A", opaque_desc, true);
    builder.add_container("B", opaque_desc, true);
    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);

    auto indvar = symbolic::symbol("i");
    auto schedule = structured_control_flow::ScheduleType_Sequential::create();
    structured_control_flow::ScheduleType_Vectorize::set(schedule);
    auto& loop = builder.add_map(
        root,
        indvar,
        symbolic::Lt(indvar, symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(indvar, symbolic::integer(1)),
        schedule
    );
    for (auto* container : {"A", "B"}) {
        auto& block = builder.add_block(loop.root());
        auto& in = builder.add_access(block, container);
        auto& out = builder.add_access(block, container);
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
        builder.add_computational_memlet(block, in, tasklet, "_in", {indvar}, desc);
        builder.add_computational_memlet(block, tasklet, "_out", out, {indvar}, desc);
    }

    analysis::AnalysisManager analysis_manager(builder.subject());
    reordering::transformations::LoopDistribute transformation(loop);
    ASSERT_TRUE(transformation.can_be_applied(builder, analysis_manager));
    transformation.apply(builder, analysis_manager);

    ASSERT_EQ(root.size(), 2);
    auto* suffix = dyn_cast<structured_control_flow::Map*>(&root.at(1));
    ASSERT_NE(suffix, nullptr);
    EXPECT_TRUE(structured_control_flow::ScheduleType_Vectorize::is_set(suffix->schedule_type()));
}

TEST(LoopDistributeTest, Serialization) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);

    types::Pointer opaque_desc;
    builder.add_container("A", opaque_desc, true);
    builder.add_container("B", opaque_desc, true);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);

    // Define loop
    auto bound = symbolic::symbol("N");
    auto indvar = symbolic::symbol("i");
    auto init = symbolic::integer(0);
    auto condition = symbolic::Lt(indvar, bound);
    auto update = symbolic::add(indvar, symbolic::integer(1));

    auto& loop = builder.add_for(root, indvar, condition, init, update);
    auto& body = loop.root();

    // Add two blocks
    auto& block1 = builder.add_block(body);
    auto& A_in = builder.add_access(block1, "A");
    auto& A_out = builder.add_access(block1, "A");
    auto& tasklet1 = builder.add_tasklet(block1, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block1, A_in, tasklet1, "_in", {symbolic::symbol("i")}, desc);
    builder.add_computational_memlet(block1, tasklet1, "_out", A_out, {symbolic::symbol("i")}, desc);

    auto& block2 = builder.add_block(body);
    auto& B_in = builder.add_access(block2, "B");
    auto& B_out = builder.add_access(block2, "B");
    auto& tasklet2 = builder.add_tasklet(block2, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block2, B_in, tasklet2, "_in", {symbolic::symbol("i")}, desc);
    builder.add_computational_memlet(block2, tasklet2, "_out", B_out, {symbolic::symbol("i")}, desc);

    size_t loop_id = loop.element_id();

    reordering::transformations::LoopDistribute transformation(loop);

    // Test to_json
    nlohmann::json j;
    EXPECT_NO_THROW(transformation.to_json(j));

    // Verify JSON structure
    EXPECT_EQ(j["transformation_type"], "LoopDistribute");
    EXPECT_TRUE(j.contains("subgraph"));
    EXPECT_TRUE(j["subgraph"].contains("0"));
    EXPECT_EQ(j["subgraph"]["0"]["element_id"], loop_id);
    EXPECT_TRUE(j["subgraph"]["0"]["type"] == "for" || j["subgraph"]["0"]["type"] == "map");
}

TEST(LoopDistributeTest, Deserialization) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);

    types::Pointer opaque_desc;
    builder.add_container("A", opaque_desc, true);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);

    // Define loop
    auto bound = symbolic::symbol("N");
    auto indvar = symbolic::symbol("i");
    auto init = symbolic::integer(0);
    auto condition = symbolic::Lt(indvar, bound);
    auto update = symbolic::add(indvar, symbolic::integer(1));

    auto& loop = builder.add_for(root, indvar, condition, init, update);
    auto& body = loop.root();

    auto& block = builder.add_block(body);
    auto& A_in = builder.add_access(block, "A");
    auto& A_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, A_in, tasklet, "_in", {symbolic::symbol("i")}, desc);
    builder.add_computational_memlet(block, tasklet, "_out", A_out, {symbolic::symbol("i")}, desc);

    size_t loop_id = loop.element_id();

    // Create JSON description
    nlohmann::json j;
    j["transformation_type"] = "LoopDistribute";
    j["subgraph"] = {{"0", {{"element_id", loop_id}, {"type", "for"}}}};

    // Test from_json
    EXPECT_NO_THROW({
        auto deserialized = reordering::transformations::LoopDistribute::from_json(builder, j);
        EXPECT_EQ(deserialized.name(), "LoopDistribute");
    });
}

TEST(LoopDistributeTest, CannotApplySingleChild) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);

    types::Pointer opaque_desc;
    builder.add_container("A", opaque_desc, true);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);

    // Define loop with only one child
    auto bound = symbolic::symbol("N");
    auto indvar = symbolic::symbol("i");
    auto init = symbolic::integer(0);
    auto condition = symbolic::Lt(indvar, bound);
    auto update = symbolic::add(indvar, symbolic::integer(1));

    auto& loop = builder.add_for(root, indvar, condition, init, update);
    auto& body = loop.root();

    // Add only one computation
    auto& block = builder.add_block(body);
    auto& A_in = builder.add_access(block, "A");
    auto& A_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, A_in, tasklet, "_in", {symbolic::symbol("i")}, desc);
    builder.add_computational_memlet(block, tasklet, "_out", A_out, {symbolic::symbol("i")}, desc);

    analysis::AnalysisManager analysis_manager(builder.subject());
    reordering::transformations::LoopDistribute transformation(loop);

    // Should not be applicable with only one child
    EXPECT_FALSE(transformation.can_be_applied(builder, analysis_manager));
}

namespace {

// Adds `C = alpha * ref * B + beta * C` as a GEMM library node in `parent`,
// where `ref` is a pointer container holding a per-iteration sub-matrix.
void add_gemm_on_reference(builder::StructuredSDFGBuilder& builder, structured_control_flow::Sequence& parent) {
    types::Scalar elem(types::PrimitiveType::Float);
    types::Pointer ptr(elem);

    auto& block = builder.add_block(parent);
    auto& ref_in = builder.add_access(block, "ref");
    auto& b_in = builder.add_access(block, "B");
    auto& c_io = builder.add_access(block, "C");
    auto& alpha = builder.add_constant(block, "1.0", elem);
    auto& beta = builder.add_constant(block, "0.0", elem);
    auto& gemm = static_cast<math::blas::GEMMNode&>(builder.add_library_node<math::blas::GEMMNode>(
        block,
        DebugInfo(),
        data_flow::ImplementationType_NONE,
        math::blas::BLAS_Precision::s,
        math::blas::BLAS_Layout::RowMajor,
        math::blas::BLAS_Transpose::No,
        math::blas::BLAS_Transpose::No,
        symbolic::integer(4),
        symbolic::integer(6),
        symbolic::integer(8),
        symbolic::integer(8),
        symbolic::integer(6),
        symbolic::integer(6)
    ));
    builder.add_computational_memlet(block, ref_in, gemm, "__A", {}, ptr);
    builder.add_computational_memlet(block, b_in, gemm, "__B", {}, ptr);
    builder.add_computational_memlet(block, c_io, gemm, "__C", {}, ptr);
    builder.add_computational_memlet(block, alpha, gemm, "__alpha", {}, elem);
    builder.add_computational_memlet(block, beta, gemm, "__beta", {}, elem);
}

// Adds `ref = &A[i * 32]` to `parent`, updated every iteration of the enclosing loop.
void add_reference_update(builder::StructuredSDFGBuilder& builder, structured_control_flow::Sequence& parent) {
    types::Scalar elem(types::PrimitiveType::Float);
    types::Pointer ptr(elem);

    auto& block = builder.add_block(parent);
    auto& a_in = builder.add_access(block, "A");
    auto& ref_out = builder.add_access(block, "ref");
    builder
        .add_reference_memlet(block, a_in, ref_out, {symbolic::mul(symbolic::symbol("i"), symbolic::integer(32))}, ptr);
}

} // namespace

TEST(LoopDistributeTest, LoopDependentReference_GemmCrossGroup_NotApplicable) {
    // for (i = 0; i < N; i++) {
    //     ref = &A[i * 32];                // prefix: reference overwritten every iteration
    //     for (j = 0; j < M; j++)          // center: GEMM reads ref
    //         gemm(ref, B, C);
    // }
    // Distributing would run all reference updates before any GEMM, so every
    // GEMM would read the last iteration's sub-matrix.
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);
    auto& root = builder.subject().root();

    types::Scalar elem(types::PrimitiveType::Float);
    types::Pointer ptr(elem);
    types::Pointer opaque_desc;
    builder.add_container("A", opaque_desc, true);
    builder.add_container("B", opaque_desc, true);
    builder.add_container("C", opaque_desc, true);
    builder.add_container("ref", ptr);

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
    add_reference_update(builder, loop_i.root());

    auto& loop_j = builder.add_for(
        loop_i.root(),
        symbolic::symbol("j"),
        symbolic::Lt(symbolic::symbol("j"), symbolic::symbol("M")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("j"), symbolic::integer(1))
    );
    add_gemm_on_reference(builder, loop_j.root());

    analysis::AnalysisManager analysis_manager(builder.subject());
    reordering::transformations::LoopDistribute transformation(loop_i, loop_j);
    EXPECT_FALSE(transformation.can_be_applied(builder, analysis_manager));
}

TEST(LoopDistributeTest, LoopDependentReference_GemmSameGroup_Applicable) {
    // for (i = 0; i < N; i++) {
    //     D[i] = 0.0;                      // prefix: independent work
    //     for (j = 0; j < M; j++) {        // center: reference and its GEMM together
    //         ref = &A[i * 32];
    //         gemm(ref, B, C);
    //     }
    // }
    // The reference is created and consumed in the same distributed piece, so
    // distribution keeps the dependency intact.
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);
    auto& root = builder.subject().root();

    types::Scalar elem(types::PrimitiveType::Float);
    types::Pointer ptr(elem);
    types::Pointer opaque_desc;
    builder.add_container("A", opaque_desc, true);
    builder.add_container("B", opaque_desc, true);
    builder.add_container("C", opaque_desc, true);
    builder.add_container("D", opaque_desc, true);
    builder.add_container("ref", ptr);

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

    auto& indep_block = builder.add_block(loop_i.root());
    auto& d_out = builder.add_access(indep_block, "D");
    auto& const_node = builder.add_constant(indep_block, "0.0", elem);
    auto& indep_tasklet = builder.add_tasklet(indep_block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(indep_block, const_node, indep_tasklet, "_in", {}, elem);
    builder.add_computational_memlet(indep_block, indep_tasklet, "_out", d_out, {symbolic::symbol("i")}, elem);

    auto& loop_j = builder.add_for(
        loop_i.root(),
        symbolic::symbol("j"),
        symbolic::Lt(symbolic::symbol("j"), symbolic::symbol("M")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("j"), symbolic::integer(1))
    );
    add_reference_update(builder, loop_j.root());
    add_gemm_on_reference(builder, loop_j.root());

    analysis::AnalysisManager analysis_manager(builder.subject());
    reordering::transformations::LoopDistribute transformation(loop_i, loop_j);
    EXPECT_TRUE(transformation.can_be_applied(builder, analysis_manager));
}
