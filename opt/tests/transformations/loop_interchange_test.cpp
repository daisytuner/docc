#include "sdfg/transformations/loop_interchange.h"

#include <gtest/gtest.h>

using namespace sdfg;

TEST(LoopInterchangeTest, Map_2D) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Array desc_1(base_desc, symbolic::symbol("M"));
    types::Pointer desc_2(desc_1);

    types::Pointer opaque_desc;
    builder.add_container("A", opaque_desc, true);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
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
    auto& a_in = builder.add_access(block, "A");
    auto& one_node = builder.add_constant(block, "1.0", base_desc);
    auto& a_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder
        .add_computational_memlet(block, a_in, tasklet, "_in1", {symbolic::symbol("i"), symbolic::symbol("j")}, desc_2);
    builder.add_computational_memlet(block, one_node, tasklet, "_in2", {});
    builder
        .add_computational_memlet(block, tasklet, "_out", a_out, {symbolic::symbol("i"), symbolic::symbol("j")}, desc_2);

    analysis::AnalysisManager analysis_manager(builder.subject());
    transformations::LoopInterchange transformation(loop, loop_2);
    EXPECT_TRUE(transformation.can_be_applied(builder, analysis_manager));
    transformation.apply(builder, analysis_manager);

    auto& new_sdfg = builder.subject();
    EXPECT_EQ(new_sdfg.root().size(), 1);
    auto outer_loop = dynamic_cast<structured_control_flow::Map*>(&new_sdfg.root().at(0).first);
    EXPECT_TRUE(outer_loop != nullptr);
    auto inner_loop = dynamic_cast<structured_control_flow::Map*>(&outer_loop->root().at(0).first);
    EXPECT_TRUE(inner_loop != nullptr);

    EXPECT_EQ(outer_loop->indvar()->get_name(), "j");
    EXPECT_EQ(inner_loop->indvar()->get_name(), "i");

    EXPECT_EQ(outer_loop->root().size(), 1);
    EXPECT_EQ(inner_loop->root().size(), 1);
    EXPECT_EQ(&inner_loop->root().at(0).first, &block);
}

TEST(LoopInterchangeTest, Map_2D_Transition) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Array desc_1(base_desc, symbolic::symbol("M"));
    types::Pointer desc_2(desc_1);

    types::Pointer opaque_desc;
    builder.add_container("A", opaque_desc, true);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
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
        structured_control_flow::ScheduleType_Sequential::create(),
        {{symbolic::symbol("i"), symbolic::zero()}}
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
    auto& a_in = builder.add_access(block, "A");
    auto& one_node = builder.add_constant(block, "1.0", base_desc);
    auto& a_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder
        .add_computational_memlet(block, a_in, tasklet, "_in1", {symbolic::symbol("i"), symbolic::symbol("j")}, desc_2);
    builder.add_computational_memlet(block, one_node, tasklet, "_in2", {});
    builder
        .add_computational_memlet(block, tasklet, "_out", a_out, {symbolic::symbol("i"), symbolic::symbol("j")}, desc_2);

    analysis::AnalysisManager analysis_manager(builder.subject());
    transformations::LoopInterchange transformation(loop, loop_2);
    EXPECT_TRUE(transformation.can_be_applied(builder, analysis_manager));
    transformation.apply(builder, analysis_manager);

    auto& new_sdfg = builder.subject();
    EXPECT_EQ(new_sdfg.root().size(), 1);
    auto outer_loop = dynamic_cast<structured_control_flow::Map*>(&new_sdfg.root().at(0).first);
    EXPECT_TRUE(outer_loop != nullptr);
    EXPECT_EQ(new_sdfg.root().at(0).second.assignments().size(), 1);
    EXPECT_TRUE(symbolic::eq(new_sdfg.root().at(0).second.assignments().at(indvar), symbolic::zero()));
    auto inner_loop = dynamic_cast<structured_control_flow::Map*>(&outer_loop->root().at(0).first);
    EXPECT_TRUE(inner_loop != nullptr);
    EXPECT_EQ(outer_loop->root().at(0).second.assignments().size(), 0);

    EXPECT_EQ(outer_loop->indvar()->get_name(), "j");
    EXPECT_EQ(inner_loop->indvar()->get_name(), "i");

    EXPECT_EQ(outer_loop->root().size(), 1);
    EXPECT_EQ(inner_loop->root().size(), 1);
    EXPECT_EQ(&inner_loop->root().at(0).first, &block);
}

TEST(LoopInterchangeTest, DependentLoops) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("M", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Array desc_1(base_desc, symbolic::symbol("M"));
    types::Pointer desc_2(desc_1);

    types::Pointer opaque_desc;
    builder.add_container("A", opaque_desc, true);

    // Define loop
    auto indvar1 = symbolic::symbol("i");
    auto& loop1 = builder.add_map(
        root,
        indvar1,
        symbolic::Lt(symbolic::symbol("i"), symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    auto& body1 = loop1.root();

    auto indvar2 = symbolic::symbol("j");
    auto offset2 = symbolic::add(indvar1, symbolic::integer(1));
    auto& loop2 = builder.add_map(
        body1,
        indvar2,
        symbolic::Lt(symbolic::symbol("j"), symbolic::sub(symbolic::symbol("M"), offset2)),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("j"), symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    auto& body2 = loop2.root();

    // Add computation
    auto& block = builder.add_block(body2);
    auto& A_in = builder.add_access(block, "A");
    auto& A_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(
        block, A_in, tasklet, "_in", {symbolic::symbol("i"), symbolic::add(symbolic::symbol("j"), offset2)}, desc_2
    );
    builder.add_computational_memlet(
        block, tasklet, "_out", A_out, {symbolic::add(symbolic::symbol("j"), offset2), symbolic::symbol("i")}, desc_2
    );

    // Analysis
    analysis::AnalysisManager analysis_manager(builder.subject());
    transformations::LoopInterchange transformation(loop1, loop2);
    EXPECT_FALSE(transformation.can_be_applied(builder, analysis_manager));
}

TEST(LoopInterchangeTest, OuterLoopHasOuterBlocks) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Array desc_1(base_desc, symbolic::symbol("M"));
    types::Pointer desc_2(desc_1);

    types::Pointer opaque_desc;
    builder.add_container("A", opaque_desc, true);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
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
    auto& blocker = builder.add_block(body);

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
    auto& a_in = builder.add_access(block, "A");
    auto& i = builder.add_access(block, "i");
    auto& a_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder
        .add_computational_memlet(block, a_in, tasklet, "_in1", {symbolic::symbol("i"), symbolic::symbol("j")}, desc_2);
    builder.add_computational_memlet(block, i, tasklet, "_in2", {});
    builder
        .add_computational_memlet(block, tasklet, "_out", a_out, {symbolic::symbol("i"), symbolic::symbol("j")}, desc_2);

    analysis::AnalysisManager analysis_manager(builder.subject());
    transformations::LoopInterchange transformation(loop, loop_2);
    EXPECT_FALSE(transformation.can_be_applied(builder, analysis_manager));
}

TEST(LoopInterchangeTest, Serialization) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Array desc_1(base_desc, symbolic::symbol("M"));
    types::Pointer desc_2(desc_1);

    types::Pointer opaque_desc;
    builder.add_container("A", opaque_desc, true);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("M", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);

    // Define loop 1
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
    auto indvar_2 = symbolic::symbol("j");
    auto& loop_2 = builder.add_map(
        body,
        indvar_2,
        symbolic::Lt(symbolic::symbol("j"), symbolic::symbol("M")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("j"), symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );

    size_t outer_loop_id = loop.element_id();
    size_t inner_loop_id = loop_2.element_id();

    transformations::LoopInterchange transformation(loop, loop_2);

    // Test to_json
    nlohmann::json j;
    EXPECT_NO_THROW(transformation.to_json(j));

    // Verify JSON structure
    EXPECT_EQ(j["transformation_type"], "LoopInterchange");
    EXPECT_TRUE(j.contains("subgraph"));
    EXPECT_TRUE(j["subgraph"].contains("0"));
    EXPECT_TRUE(j["subgraph"].contains("1"));
    EXPECT_EQ(j["subgraph"]["0"]["element_id"], outer_loop_id);
    EXPECT_EQ(j["subgraph"]["1"]["element_id"], inner_loop_id);
    EXPECT_EQ(j["subgraph"]["0"]["type"], "map");
    EXPECT_EQ(j["subgraph"]["1"]["type"], "map");
}

TEST(LoopInterchangeTest, Deserialization) {
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
    builder.add_container("M", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);

    // Define nested loops
    auto indvar = symbolic::symbol("i");
    auto& loop = builder.add_for(
        root,
        indvar,
        symbolic::Lt(symbolic::symbol("i"), symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1))
    );
    auto& body = loop.root();

    auto indvar_2 = symbolic::symbol("j");
    auto& loop_2 = builder.add_for(
        body,
        indvar_2,
        symbolic::Lt(symbolic::symbol("j"), symbolic::symbol("M")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("j"), symbolic::integer(1))
    );

    size_t outer_loop_id = loop.element_id();
    size_t inner_loop_id = loop_2.element_id();

    // Create JSON description
    nlohmann::json j;
    j["transformation_type"] = "LoopInterchange";
    j["subgraph"] = {
        {"0", {{"element_id", outer_loop_id}, {"type", "for"}}}, {"1", {{"element_id", inner_loop_id}, {"type", "for"}}}
    };

    // Test from_json
    EXPECT_NO_THROW({
        auto deserialized = transformations::LoopInterchange::from_json(builder, j);
        EXPECT_EQ(deserialized.name(), "LoopInterchange");
    });
}

TEST(LoopInterchangeTest, CreatedAndDeletedElements) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Array desc_1(base_desc, symbolic::symbol("M"));
    types::Pointer desc_2(desc_1);

    types::Pointer opaque_desc;
    builder.add_container("A", opaque_desc, true);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("M", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);

    // Define loop 1
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

    // Store original loop IDs
    size_t original_outer_id = loop.element_id();
    size_t original_inner_id = loop_2.element_id();

    analysis::AnalysisManager analysis_manager(builder.subject());
    transformations::LoopInterchange transformation(loop, loop_2);
    EXPECT_TRUE(transformation.can_be_applied(builder, analysis_manager));
    transformation.apply(builder, analysis_manager);

    auto& new_sdfg = builder.subject();
    EXPECT_EQ(new_sdfg.root().size(), 1);

    // Verify new outer loop exists and has correct indvar
    auto outer_loop = dynamic_cast<structured_control_flow::Map*>(&new_sdfg.root().at(0).first);
    EXPECT_TRUE(outer_loop != nullptr);
    EXPECT_EQ(outer_loop->indvar()->get_name(), "j");

    // Verify new inner loop exists and has correct indvar
    auto inner_loop = dynamic_cast<structured_control_flow::Map*>(&outer_loop->root().at(0).first);
    EXPECT_TRUE(inner_loop != nullptr);
    EXPECT_EQ(inner_loop->indvar()->get_name(), "i");

    // Verify the original loops were replaced (element IDs should be different)
    EXPECT_NE(outer_loop->element_id(), original_outer_id);
    EXPECT_NE(inner_loop->element_id(), original_inner_id);

    // Verify loop structure is preserved: 2 loops with 1 block
    EXPECT_EQ(outer_loop->root().size(), 1);
    EXPECT_EQ(inner_loop->root().size(), 1);
    EXPECT_EQ(&inner_loop->root().at(0).first, &block);
}

// For-For interchange: A[i][j] = A[i][j] + 1.0 — no cross-iteration deps, should be legal
TEST(LoopInterchangeTest, ForFor_Independent) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Array desc_1(base_desc, symbolic::symbol("M"));
    types::Pointer desc_2(desc_1);

    types::Pointer opaque_desc;
    builder.add_container("A", opaque_desc, true);

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
    auto& body_i = loop_i.root();

    auto& loop_j = builder.add_for(
        body_i,
        symbolic::symbol("j"),
        symbolic::Lt(symbolic::symbol("j"), symbolic::symbol("M")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("j"), symbolic::integer(1))
    );
    auto& body_j = loop_j.root();

    auto& block = builder.add_block(body_j);
    auto& a_in = builder.add_access(block, "A");
    auto& one_node = builder.add_constant(block, "1.0", base_desc);
    auto& a_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder
        .add_computational_memlet(block, a_in, tasklet, "_in1", {symbolic::symbol("i"), symbolic::symbol("j")}, desc_2);
    builder.add_computational_memlet(block, one_node, tasklet, "_in2", {});
    builder
        .add_computational_memlet(block, tasklet, "_out", a_out, {symbolic::symbol("i"), symbolic::symbol("j")}, desc_2);

    analysis::AnalysisManager analysis_manager(sdfg);
    transformations::LoopInterchange transformation(loop_i, loop_j);
    EXPECT_TRUE(transformation.can_be_applied(builder, analysis_manager));
    transformation.apply(builder, analysis_manager);

    auto outer = dynamic_cast<structured_control_flow::For*>(&sdfg.root().at(0).first);
    ASSERT_NE(outer, nullptr);
    auto inner = dynamic_cast<structured_control_flow::For*>(&outer->root().at(0).first);
    ASSERT_NE(inner, nullptr);

    EXPECT_EQ(outer->indvar()->get_name(), "j");
    EXPECT_EQ(inner->indvar()->get_name(), "i");
    EXPECT_EQ(&inner->root().at(0).first, &block);
}

// For-For interchange: A[i+1][j] = A[i][j] — forward dep in i, no dep in j. Legal.
TEST(LoopInterchangeTest, ForFor_ForwardDep) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Array desc_1(base_desc, symbolic::symbol("M"));
    types::Pointer desc_2(desc_1);

    types::Pointer opaque_desc;
    builder.add_container("A", opaque_desc, true);

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
    auto& body_i = loop_i.root();

    auto& loop_j = builder.add_for(
        body_i,
        symbolic::symbol("j"),
        symbolic::Lt(symbolic::symbol("j"), symbolic::symbol("M")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("j"), symbolic::integer(1))
    );
    auto& body_j = loop_j.root();

    // A[i+1][j] = A[i][j]: read from (i,j), write to (i+1,j)
    auto& block = builder.add_block(body_j);
    auto& a_in = builder.add_access(block, "A");
    auto& a_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder
        .add_computational_memlet(block, a_in, tasklet, "_in", {symbolic::symbol("i"), symbolic::symbol("j")}, desc_2);
    builder.add_computational_memlet(
        block,
        tasklet,
        "_out",
        a_out,
        {symbolic::add(symbolic::symbol("i"), symbolic::integer(1)), symbolic::symbol("j")},
        desc_2
    );

    analysis::AnalysisManager analysis_manager(sdfg);
    transformations::LoopInterchange transformation(loop_i, loop_j);
    // Forward dep (d_i=1, d_j=0). After swap: (d_j=0, d_i=1) — lex-non-negative. Legal.
    EXPECT_TRUE(transformation.can_be_applied(builder, analysis_manager));
}

// For-For interchange: Forward propagation pattern — LEGAL
// A[i][j-1] = A[i][j]: read from (i,j), write to (i,j-1)
// This is a WAR (write-after-read) dependency: iteration j writes to j-1, which
// iteration j-1 would read. But j-1 executes BEFORE j, so the read happens first.
// The dependency delta is (0, +1) in the forward direction.
// After swap: (d_j=1, d_i=0) which is lex-positive. LEGAL.
TEST(LoopInterchangeTest, ForFor_ForwardPropagation_Legal) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Array desc_1(base_desc, symbolic::symbol("M"));
    types::Pointer desc_2(desc_1);

    types::Pointer opaque_desc;
    builder.add_container("A", opaque_desc, true);

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
    auto& body_i = loop_i.root();

    auto& loop_j = builder.add_for(
        body_i,
        symbolic::symbol("j"),
        symbolic::Lt(symbolic::symbol("j"), symbolic::symbol("M")),
        symbolic::integer(1),
        symbolic::add(symbolic::symbol("j"), symbolic::integer(1))
    );
    auto& body_j = loop_j.root();

    // A[i][j-1] = A[i][j]: forward propagation (WAR dependency)
    auto& block = builder.add_block(body_j);
    auto& a_in = builder.add_access(block, "A");
    auto& a_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder
        .add_computational_memlet(block, a_in, tasklet, "_in", {symbolic::symbol("i"), symbolic::symbol("j")}, desc_2);
    builder.add_computational_memlet(
        block,
        tasklet,
        "_out",
        a_out,
        {symbolic::symbol("i"), symbolic::sub(symbolic::symbol("j"), symbolic::integer(1))},
        desc_2
    );

    analysis::AnalysisManager analysis_manager(sdfg);
    transformations::LoopInterchange transformation(loop_i, loop_j);
    EXPECT_TRUE(transformation.can_be_applied(builder, analysis_manager));
}

// For-For interchange: Backward stencil pattern — LEGAL
// A[i][j] = A[i][j+1]: read from (i,j+1), write to (i,j)
// This creates a WAR dependency: iteration j reads from location j+1,
// then iteration j+1 writes to that location.
// The dependency delta is (0, +1) — forward in j.
// After swap: (d_j=+1, d_i=0) which is still lex-positive. LEGAL.
TEST(LoopInterchangeTest, ForFor_BackwardStencil_Legal) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Array desc_1(base_desc, symbolic::symbol("M"));
    types::Pointer desc_2(desc_1);

    types::Pointer opaque_desc;
    builder.add_container("A", opaque_desc, true);

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
    auto& body_i = loop_i.root();

    auto& loop_j = builder.add_for(
        body_i,
        symbolic::symbol("j"),
        symbolic::Lt(symbolic::symbol("j"), symbolic::symbol("M")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("j"), symbolic::integer(1))
    );
    auto& body_j = loop_j.root();

    // A[i][j] = A[i][j+1]: backward stencil (WAR dependency)
    auto& block = builder.add_block(body_j);
    auto& a_in = builder.add_access(block, "A");
    auto& a_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(
        block,
        a_in,
        tasklet,
        "_in",
        {symbolic::symbol("i"), symbolic::add(symbolic::symbol("j"), symbolic::integer(1))},
        desc_2
    );
    builder
        .add_computational_memlet(block, tasklet, "_out", a_out, {symbolic::symbol("i"), symbolic::symbol("j")}, desc_2);

    analysis::AnalysisManager analysis_manager(sdfg);
    transformations::LoopInterchange transformation(loop_i, loop_j);
    EXPECT_TRUE(transformation.can_be_applied(builder, analysis_manager));
}

// For-For interchange: Jacobi-like pattern — dep vector (1, -1). After swap: (-1, 1) — illegal!
TEST(LoopInterchangeTest, ForFor_Jacobi_Illegal) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Array desc_1(base_desc, symbolic::symbol("M"));
    types::Pointer desc_2(desc_1);

    types::Pointer opaque_desc;
    builder.add_container("A", opaque_desc, true);

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
    auto& body_i = loop_i.root();

    auto& loop_j = builder.add_for(
        body_i,
        symbolic::symbol("j"),
        symbolic::Lt(symbolic::symbol("j"), symbolic::symbol("M")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("j"), symbolic::integer(1))
    );
    auto& body_j = loop_j.root();

    // A[i+1][j] = A[i][j+1]
    auto& block = builder.add_block(body_j);
    auto& a_in = builder.add_access(block, "A");
    auto& a_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(
        block,
        a_in,
        tasklet,
        "_in",
        {symbolic::symbol("i"), symbolic::add(symbolic::symbol("j"), symbolic::integer(1))},
        desc_2
    );
    builder.add_computational_memlet(
        block,
        tasklet,
        "_out",
        a_out,
        {symbolic::add(symbolic::symbol("i"), symbolic::integer(1)), symbolic::symbol("j")},
        desc_2
    );

    analysis::AnalysisManager analysis_manager(sdfg);
    transformations::LoopInterchange transformation(loop_i, loop_j);
    EXPECT_FALSE(transformation.can_be_applied(builder, analysis_manager));
}

/// FM interchange with non-unit inner stride.
/// Models a tile loop that depends on the time loop after skewing:
///   for t = 0; t < T; t++
///     for tile = 32*t; tile < 32*t + N; tile += 32
///       A[tile] = A[tile] + 1.0
///
/// After interchange (FM, coefficient α = 32):
///   for tile = 0; tile < 32*(T-1) + N; tile += 32
///     for t = max(0, idiv(tile - N, 32) + 1) .. min(T, idiv(tile, 32) + 1)
TEST(LoopInterchangeTest, ForFor_FM_NonUnitStride) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("T", sym_desc, true);
    builder.add_container("N", sym_desc, true);
    builder.add_container("t", sym_desc);
    builder.add_container("tile", sym_desc);

    types::Scalar elem_desc(types::PrimitiveType::Double);
    types::Array arr_desc(elem_desc, symbolic::symbol("N"));
    builder.add_container("A", arr_desc, true);

    // for t = 0; t < T; t++
    auto& loop_t = builder.add_for(
        root,
        symbolic::symbol("t"),
        symbolic::Lt(symbolic::symbol("t"), symbolic::symbol("T")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("t"), symbolic::integer(1))
    );

    // for tile = 32*t; tile < 32*t + N; tile += 32
    auto& loop_tile = builder.add_for(
        loop_t.root(),
        symbolic::symbol("tile"),
        symbolic::
            Lt(symbolic::symbol("tile"),
               symbolic::add(symbolic::mul(symbolic::integer(32), symbolic::symbol("t")), symbolic::symbol("N"))),
        symbolic::mul(symbolic::integer(32), symbolic::symbol("t")),
        symbolic::add(symbolic::symbol("tile"), symbolic::integer(32))
    );

    // A[tile] = A[tile] + 1.0  (self-dep, safe)
    auto& block = builder.add_block(loop_tile.root());
    auto& a_in = builder.add_access(block, "A");
    auto& one_node = builder.add_constant(block, "1.0", elem_desc);
    auto& a_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, a_in, tasklet, "_in1", {symbolic::symbol("tile")}, arr_desc);
    builder.add_computational_memlet(block, one_node, tasklet, "_in2", {});
    builder.add_computational_memlet(block, tasklet, "_out", a_out, {symbolic::symbol("tile")}, arr_desc);

    analysis::AnalysisManager analysis_manager(sdfg);
    transformations::LoopInterchange transformation(loop_t, loop_tile);
    ASSERT_TRUE(transformation.can_be_applied(builder, analysis_manager));
    transformation.apply(builder, analysis_manager);

    // After interchange: outer = tile (stride 32), inner = t (stride 1)
    auto* outer = dynamic_cast<structured_control_flow::For*>(&sdfg.root().at(0).first);
    ASSERT_NE(outer, nullptr);
    EXPECT_EQ(outer->indvar()->get_name(), "tile");

    auto* inner = dynamic_cast<structured_control_flow::For*>(&outer->root().at(0).first);
    ASSERT_NE(inner, nullptr);
    EXPECT_EQ(inner->indvar()->get_name(), "t");

    // New outer stride = old inner stride = 32
    EXPECT_TRUE(symbolic::eq(symbolic::sub(outer->update(), outer->indvar()), symbolic::integer(32)));

    // New inner stride = old outer stride = 1
    EXPECT_TRUE(symbolic::eq(symbolic::sub(inner->update(), inner->indvar()), symbolic::integer(1)));

    // New outer init = inner_init(t=0) = 32*0 = 0
    EXPECT_TRUE(symbolic::eq(outer->init(), symbolic::integer(0)));

    // New inner init uses max(..., idiv(...) + 1) — verify it references tile
    EXPECT_TRUE(symbolic::uses(inner->init(), outer->indvar()->get_name()));
}
