#include "sdfg/transformations/unroll_transform.h"

#include <gtest/gtest.h>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/structured_control_flow/structured_loop.h"

using namespace sdfg;

/// Build for(i = 0; i < BOUND; i++) { A[i] = A[i] } (BOUND may be constant or symbolic).
static builder::StructuredSDFGBuilder make_loop(structured_control_flow::For*& out, symbolic::Expression bound) {
    builder::StructuredSDFGBuilder builder("unroll_test", FunctionType_CPU);
    auto& root = builder.subject().root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    types::Pointer opaque_desc;
    builder.add_container("A", opaque_desc, true);
    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);

    auto i = symbolic::symbol("i");
    auto& loop =
        builder.add_for(root, i, symbolic::Lt(i, bound), symbolic::integer(0), symbolic::add(i, symbolic::integer(1)));
    auto& block = builder.add_block(loop.root());
    auto& a_in = builder.add_access(block, "A");
    auto& a_out = builder.add_access(block, "A");
    auto& t = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, a_in, t, "_in", {i}, desc);
    builder.add_computational_memlet(block, t, "_out", a_out, {i}, desc);

    out = &loop;
    return builder;
}

TEST(UnrollTransformTest, MarksConstantTripLoopForUnroll) {
    structured_control_flow::For* orig = nullptr;
    auto builder = make_loop(orig, symbolic::integer(8));

    auto sdfg = builder.move();
    builder::StructuredSDFGBuilder b(sdfg);
    analysis::AnalysisManager am(b.subject());

    transformations::UnrollTransform t(*orig);
    EXPECT_TRUE(t.can_be_applied(b, am));
    EXPECT_FALSE(structured_control_flow::ScheduleType_Unroll::is_set(orig->schedule_type()));

    t.apply(b, am);

    // The unroll annotation is set while the schedule kind is preserved.
    EXPECT_TRUE(structured_control_flow::ScheduleType_Unroll::is_set(orig->schedule_type()));
    EXPECT_EQ(orig->schedule_type().value(), structured_control_flow::ScheduleType_Sequential::value());
}

TEST(UnrollTransformTest, NotApplicableToVariableTripLoop) {
    structured_control_flow::For* orig = nullptr;
    auto builder = make_loop(orig, symbolic::symbol("N"));

    auto sdfg = builder.move();
    builder::StructuredSDFGBuilder b(sdfg);
    analysis::AnalysisManager am(b.subject());

    transformations::UnrollTransform t(*orig);
    EXPECT_FALSE(t.can_be_applied(b, am));
}
