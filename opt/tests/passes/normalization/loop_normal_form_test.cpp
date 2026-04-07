#include <gtest/gtest.h>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/passes/normalization/loop_normal_form.h"
#include "sdfg/passes/pipeline.h"
#include "sdfg/structured_control_flow/for.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/array.h"
#include "sdfg/types/scalar.h"

using namespace sdfg;

// Helper to create a simple loop with body
static void add_simple_body(
    builder::StructuredSDFGBuilder& builder, structured_control_flow::Sequence& loop_root, const std::string& array_name
) {
    types::Scalar elem_desc(types::PrimitiveType::Double);
    auto& block = builder.add_block(loop_root);
    auto& a_node = builder.add_access(block, array_name);
    auto& const_node = builder.add_constant(block, "1.0", elem_desc);
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in1"});
    builder.add_computational_memlet(block, const_node, tasklet, "_in1", {});
    builder.add_computational_memlet(block, tasklet, "_out", a_node, {symbolic::symbol("i")});
}

// Test 1: Loop with non-zero init gets shifted to init=0
TEST(LoopNormalFormTest, ShiftsNonZeroInit) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);

    types::Scalar elem_desc(types::PrimitiveType::Double);
    types::Array arr_desc(elem_desc, symbolic::integer(100));
    builder.add_container("A", arr_desc, true);

    auto& root = builder.subject().root();

    // for i = 5; i < 15; i++
    auto& for_loop = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::integer(15)),
        symbolic::integer(5),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1))
    );
    add_simple_body(builder, for_loop.root(), "A");

    auto sdfg = builder.move();
    builder::StructuredSDFGBuilder builder2(sdfg);
    analysis::AnalysisManager am(builder2.subject());

    // Apply pass once
    passes::normalization::LoopNormalFormPass pass;
    bool applied = pass.run(builder2, am);
    EXPECT_TRUE(applied);

    auto* loop = dynamic_cast<structured_control_flow::For*>(&builder2.subject().root().at(0).first);
    ASSERT_NE(loop, nullptr);

    // After LoopShift: init should be 0
    EXPECT_TRUE(symbolic::eq(loop->init(), symbolic::integer(0)));
    // Condition: i < 10 (shifted and normalized: i + 5 < 15 → i < 15 - 5)
    auto expected_cond = symbolic::Lt(symbolic::symbol("i"), symbolic::integer(10));
    EXPECT_TRUE(symbolic::eq(loop->condition(), expected_cond));
    // Stride still +1
    auto stride = loop->stride();
    ASSERT_FALSE(stride.is_null());
    EXPECT_EQ(stride->as_int(), 1);
    // Update: i + 1
    auto expected_update = symbolic::add(symbolic::symbol("i"), symbolic::integer(1));
    EXPECT_TRUE(symbolic::eq(loop->update(), expected_update));
}

// Test 2: Loop with non-unit stride gets normalized (after init=0)
TEST(LoopNormalFormTest, NormalizesNonUnitStride) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);

    types::Scalar elem_desc(types::PrimitiveType::Double);
    types::Array arr_desc(elem_desc, symbolic::integer(100));
    builder.add_container("A", arr_desc, true);

    auto& root = builder.subject().root();

    // for i = 0; i < 20; i += 4 (already init=0, needs unit stride)
    auto& for_loop = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::integer(20)),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(4))
    );
    add_simple_body(builder, for_loop.root(), "A");

    auto sdfg = builder.move();
    builder::StructuredSDFGBuilder builder2(sdfg);
    analysis::AnalysisManager am(builder2.subject());

    // Apply pass
    passes::normalization::LoopNormalFormPass pass;
    bool applied = pass.run(builder2, am);
    EXPECT_TRUE(applied);

    auto* loop = dynamic_cast<structured_control_flow::For*>(&builder2.subject().root().at(0).first);
    ASSERT_NE(loop, nullptr);

    // After LoopUnitStride: stride should be +1
    auto stride = loop->stride();
    ASSERT_FALSE(stride.is_null());
    EXPECT_EQ(stride->as_int(), 1);
    // Init should still be 0
    EXPECT_TRUE(symbolic::eq(loop->init(), symbolic::integer(0)));
    // Condition: 4*i < 20
    auto expected_cond =
        symbolic::Lt(symbolic::mul(symbolic::integer(4), symbolic::symbol("i")), symbolic::integer(20));
    EXPECT_TRUE(symbolic::eq(loop->condition(), expected_cond));
    // Update: i + 1
    auto expected_update = symbolic::add(symbolic::symbol("i"), symbolic::integer(1));
    EXPECT_TRUE(symbolic::eq(loop->update(), expected_update));
}

// Test 3: Loop with negative stride gets rotated (after init=0)
TEST(LoopNormalFormTest, RotatesNegativeStride) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);

    types::Scalar elem_desc(types::PrimitiveType::Double);
    types::Array arr_desc(elem_desc, symbolic::integer(100));
    builder.add_container("A", arr_desc, true);

    auto& root = builder.subject().root();

    // for i = 10; 0 < i; i-- (init=10, needs shift first, then rotate)
    auto& for_loop = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::integer(0), symbolic::symbol("i")),
        symbolic::integer(10),
        symbolic::sub(symbolic::symbol("i"), symbolic::integer(1))
    );
    add_simple_body(builder, for_loop.root(), "A");

    auto sdfg = builder.move();
    builder::StructuredSDFGBuilder builder2(sdfg);
    analysis::AnalysisManager am(builder2.subject());

    // Apply pass multiple times (shift first, then rotate)
    passes::normalization::LoopNormalFormPass pipeline;
    pipeline.run(builder2, am);

    auto* loop = dynamic_cast<structured_control_flow::For*>(&builder2.subject().root().at(0).first);
    ASSERT_NE(loop, nullptr);

    // After full normalization: stride should be +1
    auto stride = loop->stride();
    ASSERT_FALSE(stride.is_null());
    EXPECT_EQ(stride->as_int(), 1) << "Stride should be +1 after LoopRotate";
    // Init: 0
    EXPECT_TRUE(symbolic::eq(loop->init(), symbolic::integer(0)));
    // Update: i + 1
    auto expected_update = symbolic::add(symbolic::symbol("i"), symbolic::integer(1));
    EXPECT_TRUE(symbolic::eq(loop->update(), expected_update));
}

// Test 4: Already normalized loop - no change
TEST(LoopNormalFormTest, AlreadyNormalized) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);

    types::Scalar elem_desc(types::PrimitiveType::Double);
    types::Array arr_desc(elem_desc, symbolic::integer(100));
    builder.add_container("A", arr_desc, true);

    auto& root = builder.subject().root();

    // for i = 0; i < 10; i++ (already normalized)
    auto& for_loop = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::integer(10)),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1))
    );
    add_simple_body(builder, for_loop.root(), "A");

    auto sdfg = builder.move();
    builder::StructuredSDFGBuilder builder2(sdfg);
    analysis::AnalysisManager am(builder2.subject());

    // Apply pass
    passes::normalization::LoopNormalFormPass pass;
    bool applied = pass.run(builder2, am);
    ASSERT_TRUE(applied) << "Already normalized loop should add final indvar value";

    auto* loop = dynamic_cast<structured_control_flow::For*>(&builder2.subject().root().at(0).first);
    ASSERT_NE(loop, nullptr);

    // Everything should remain unchanged
    EXPECT_TRUE(symbolic::eq(loop->init(), symbolic::integer(0)));
    EXPECT_TRUE(symbolic::eq(loop->condition(), symbolic::Lt(symbolic::symbol("i"), symbolic::integer(10))));
    auto stride = loop->stride();
    ASSERT_FALSE(stride.is_null());
    EXPECT_EQ(stride->as_int(), 1);
    // Update: i + 1
    auto expected_update = symbolic::add(symbolic::symbol("i"), symbolic::integer(1));
    EXPECT_TRUE(symbolic::eq(loop->update(), expected_update));

    auto& finalize_blocks = builder2.subject().root().at(1).second; // LoopIndvarFinalize adds a block
    EXPECT_EQ(finalize_blocks.assignments().size(), 1);
    auto indvar = symbolic::symbol("i");
    ASSERT_TRUE(finalize_blocks.assignments().count(indvar) > 0);
    auto closed_form = finalize_blocks.assignments().at(indvar);
    auto expected_closed_form = symbolic::integer(10); // num_iterations = 10 - 0 = 10
    EXPECT_TRUE(symbolic::eq(closed_form, expected_closed_form));
}

// Test 5: Full pipeline - complex loop with non-zero init + non-unit stride
TEST(LoopNormalFormTest, FullPipelinePositiveStride) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);

    types::Scalar elem_desc(types::PrimitiveType::Double);
    types::Array arr_desc(elem_desc, symbolic::integer(100));
    builder.add_container("A", arr_desc, true);

    auto& root = builder.subject().root();

    // for i = 5; i < 25; i += 4 (iterations: 5, 9, 13, 17, 21)
    auto& for_loop = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::integer(25)),
        symbolic::integer(5),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(4))
    );
    add_simple_body(builder, for_loop.root(), "A");

    auto sdfg = builder.move();
    builder::StructuredSDFGBuilder builder2(sdfg);
    analysis::AnalysisManager am(builder2.subject());

    // Apply full pipeline
    passes::normalization::LoopNormalFormPass pipeline;
    pipeline.run(builder2, am);

    auto* loop = dynamic_cast<structured_control_flow::For*>(&builder2.subject().root().at(0).first);
    ASSERT_NE(loop, nullptr);

    // After full normalization:
    // init = 0
    EXPECT_TRUE(symbolic::eq(loop->init(), symbolic::integer(0)));
    // stride = +1
    auto stride = loop->stride();
    ASSERT_FALSE(stride.is_null());
    EXPECT_EQ(stride->as_int(), 1);
    // Condition: After LoopShift (i→i+5) then LoopUnitStride (i→4*i): 5 + 4*i < 25
    auto expected_cond = symbolic::
        Lt(symbolic::add(symbolic::integer(5), symbolic::mul(symbolic::integer(4), symbolic::symbol("i"))),
           symbolic::integer(25));
    EXPECT_TRUE(symbolic::eq(loop->condition(), expected_cond));
    // Update: i + 1
    auto expected_update = symbolic::add(symbolic::symbol("i"), symbolic::integer(1));
    EXPECT_TRUE(symbolic::eq(loop->update(), expected_update));
}

// Test 6: Full pipeline with negative non-unit stride
TEST(LoopNormalFormTest, FullPipelineNegativeNonUnitStride) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::Int64);
    builder.add_container("i", sym_desc);

    types::Scalar elem_desc(types::PrimitiveType::Double);
    types::Array arr_desc(elem_desc, symbolic::integer(100));
    builder.add_container("A", arr_desc, true);

    auto& root = builder.subject().root();

    // for i = 0; -20 < i; i -= 3 (iterations: 0, -3, -6, -9, -12, -15, -18)
    auto& for_loop = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::integer(-20), symbolic::symbol("i")),
        symbolic::integer(0),
        symbolic::sub(symbolic::symbol("i"), symbolic::integer(3))
    );

    types::Scalar elem_desc2(types::PrimitiveType::Double);
    auto& block = builder.add_block(for_loop.root());
    auto& a_node = builder.add_access(block, "A");
    auto& const_node = builder.add_constant(block, "1.0", elem_desc2);
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in1"});
    builder.add_computational_memlet(block, const_node, tasklet, "_in1", {});
    // Index with 50 + i to stay positive
    builder.add_computational_memlet(
        block, tasklet, "_out", a_node, {symbolic::add(symbolic::integer(50), symbolic::symbol("i"))}
    );

    auto sdfg = builder.move();
    builder::StructuredSDFGBuilder builder2(sdfg);
    analysis::AnalysisManager am(builder2.subject());

    // Apply full pipeline
    passes::normalization::LoopNormalFormPass pipeline;
    pipeline.run(builder2, am);

    auto* loop = dynamic_cast<structured_control_flow::For*>(&builder2.subject().root().at(0).first);
    ASSERT_NE(loop, nullptr);

    // After full normalization (LoopUnitStride then LoopRotate):
    // stride should be +1
    auto stride = loop->stride();
    ASSERT_FALSE(stride.is_null());
    EXPECT_EQ(stride->as_int(), 1)
        << "Final stride should be +1 after LoopUnitStride(-3 -> -1) then LoopRotate(-1 -> +1)";
    // Init: 0
    EXPECT_TRUE(symbolic::eq(loop->init(), symbolic::integer(0)));
    // Update: i + 1
    auto expected_update = symbolic::add(symbolic::symbol("i"), symbolic::integer(1));
    EXPECT_TRUE(symbolic::eq(loop->update(), expected_update));
}

// Test 7: Map loop normalization
TEST(LoopNormalFormTest, MapLoopNormalization) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);

    types::Scalar elem_desc(types::PrimitiveType::Double);
    types::Array arr_desc(elem_desc, symbolic::integer(100));
    builder.add_container("A", arr_desc, true);

    auto& root = builder.subject().root();

    // map i = 2; i < 18; i += 2
    auto& map_loop = builder.add_map(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::integer(18)),
        symbolic::integer(2),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(2)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    add_simple_body(builder, map_loop.root(), "A");

    auto sdfg = builder.move();
    builder::StructuredSDFGBuilder builder2(sdfg);
    analysis::AnalysisManager am(builder2.subject());

    // Apply full pipeline
    passes::normalization::LoopNormalFormPass pipeline;
    pipeline.run(builder2, am);

    auto* loop = dynamic_cast<structured_control_flow::Map*>(&builder2.subject().root().at(0).first);
    ASSERT_NE(loop, nullptr);

    // After full normalization: init=0, stride=+1
    EXPECT_TRUE(symbolic::eq(loop->init(), symbolic::integer(0)));
    auto stride = loop->stride();
    ASSERT_FALSE(stride.is_null());
    EXPECT_EQ(stride->as_int(), 1);
    // Condition: After LoopShift (i→i+2) then LoopUnitStride (i→2*i): 2 + 2*i < 18
    auto expected_cond = symbolic::
        Lt(symbolic::add(symbolic::integer(2), symbolic::mul(symbolic::integer(2), symbolic::symbol("i"))),
           symbolic::integer(18));
    EXPECT_TRUE(symbolic::eq(loop->condition(), expected_cond));
    // Update: i + 1
    auto expected_update = symbolic::add(symbolic::symbol("i"), symbolic::integer(1));
    EXPECT_TRUE(symbolic::eq(loop->update(), expected_update));
}

// Test 8: Nested loops both get normalized
TEST(LoopNormalFormTest, NestedLoops) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);

    types::Scalar elem_desc(types::PrimitiveType::Double);
    types::Array arr_desc(elem_desc, symbolic::integer(100));
    builder.add_container("A", arr_desc, true);

    auto& root = builder.subject().root();

    // Outer: for i = 1; i < 10; i++
    auto& outer_loop = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::integer(10)),
        symbolic::integer(1),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1))
    );

    // Inner: for j = 0; j < 20; j += 2
    auto& inner_loop = builder.add_for(
        outer_loop.root(),
        symbolic::symbol("j"),
        symbolic::Lt(symbolic::symbol("j"), symbolic::integer(20)),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("j"), symbolic::integer(2))
    );
    add_simple_body(builder, inner_loop.root(), "A");

    auto sdfg = builder.move();
    builder::StructuredSDFGBuilder builder2(sdfg);
    analysis::AnalysisManager am(builder2.subject());

    // Apply full pipeline
    passes::normalization::LoopNormalFormPass pipeline;
    pipeline.run(builder2, am);

    auto* outer = dynamic_cast<structured_control_flow::For*>(&builder2.subject().root().at(0).first);
    ASSERT_NE(outer, nullptr);

    // Outer loop: shifted to init=0
    EXPECT_TRUE(symbolic::eq(outer->init(), symbolic::integer(0)));
    auto outer_stride = outer->stride();
    ASSERT_FALSE(outer_stride.is_null());
    EXPECT_EQ(outer_stride->as_int(), 1);
    // Outer condition: i < 9 (shifted and normalized: i + 1 < 10 → i < 9)
    auto expected_outer_cond = symbolic::Lt(symbolic::symbol("i"), symbolic::integer(9));
    EXPECT_TRUE(symbolic::eq(outer->condition(), expected_outer_cond));
    // Outer update: i + 1
    auto expected_outer_update = symbolic::add(symbolic::symbol("i"), symbolic::integer(1));
    EXPECT_TRUE(symbolic::eq(outer->update(), expected_outer_update));

    // Find inner loop (may be preceded by assignment block)
    structured_control_flow::For* inner = nullptr;
    for (size_t i = 0; i < outer->root().size(); ++i) {
        inner = dynamic_cast<structured_control_flow::For*>(&outer->root().at(i).first);
        if (inner) break;
    }
    ASSERT_NE(inner, nullptr);

    // Inner loop: already init=0, should have unit stride now
    EXPECT_TRUE(symbolic::eq(inner->init(), symbolic::integer(0)));
    auto inner_stride = inner->stride();
    ASSERT_FALSE(inner_stride.is_null());
    EXPECT_EQ(inner_stride->as_int(), 1);
    // Inner condition: 2*j < 20
    auto expected_inner_cond =
        symbolic::Lt(symbolic::mul(symbolic::integer(2), symbolic::symbol("j")), symbolic::integer(20));
    EXPECT_TRUE(symbolic::eq(inner->condition(), expected_inner_cond));
    // Inner update: j + 1
    auto expected_inner_update = symbolic::add(symbolic::symbol("j"), symbolic::integer(1));
    EXPECT_TRUE(symbolic::eq(inner->update(), expected_inner_update));
}

// Test 9: Symbolic bounds
TEST(LoopNormalFormTest, SymbolicBounds) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);

    types::Scalar elem_desc(types::PrimitiveType::Double);
    types::Array arr_desc(elem_desc, symbolic::symbol("N"));
    builder.add_container("A", arr_desc, true);

    auto& root = builder.subject().root();

    // for i = 0; i < N; i += 8
    auto& for_loop = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(8))
    );
    add_simple_body(builder, for_loop.root(), "A");

    auto sdfg = builder.move();
    builder::StructuredSDFGBuilder builder2(sdfg);
    analysis::AnalysisManager am(builder2.subject());

    // Apply full pipeline
    passes::normalization::LoopNormalFormPass pipeline;
    pipeline.run(builder2, am);

    auto* loop = dynamic_cast<structured_control_flow::For*>(&builder2.subject().root().at(0).first);
    ASSERT_NE(loop, nullptr);

    // After normalization: stride = +1
    auto stride = loop->stride();
    ASSERT_FALSE(stride.is_null());
    EXPECT_EQ(stride->as_int(), 1);

    // Init: 0
    EXPECT_TRUE(symbolic::eq(loop->init(), symbolic::integer(0)));

    // Condition should be 8*i < N
    auto expected_cond =
        symbolic::Lt(symbolic::mul(symbolic::integer(8), symbolic::symbol("i")), symbolic::symbol("N"));
    EXPECT_TRUE(symbolic::eq(loop->condition(), expected_cond));

    // Update: i + 1
    auto expected_update = symbolic::add(symbolic::symbol("i"), symbolic::integer(1));
    EXPECT_TRUE(symbolic::eq(loop->update(), expected_update));
}
