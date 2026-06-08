#include "sdfg/passes/normalization/map_fusion.h"

#include <gtest/gtest.h>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/array.h"
#include "sdfg/types/scalar.h"

using namespace sdfg;

TEST(MapFusionPassTest, DoesNotCrashOnEmptyNestedProducerBody) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);
    auto& root = builder.subject().root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("M", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);
    builder.add_container("k", sym_desc);

    types::Scalar float_desc(types::PrimitiveType::Float);
    types::Array array_1d(float_desc, {symbolic::symbol("N")});
    builder.add_container("A", array_1d, true);

    // Producer: Map(i) { Map(j) { } }  — inner body is intentionally empty.
    auto& producer = builder.add_map(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    builder.add_map(
        producer.root(),
        symbolic::symbol("j"),
        symbolic::Lt(symbolic::symbol("j"), symbolic::symbol("M")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("j"), symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );

    // Consumer: Map(k) { Block { } }
    auto& consumer = builder.add_map(
        root,
        symbolic::symbol("k"),
        symbolic::Lt(symbolic::symbol("k"), symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("k"), symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    builder.add_block(consumer.root());

    analysis::AnalysisManager am(builder.subject());
    passes::normalization::MapFusionPass pass;
    EXPECT_FALSE(pass.run(builder, am));
}
TEST(MapFusionPassTest, DoesNotCrashOnEmptyNestedConsumerBody) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);
    auto& root = builder.subject().root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("M", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);
    builder.add_container("k", sym_desc);

    types::Scalar float_desc(types::PrimitiveType::Float);
    types::Array array_1d(float_desc, {symbolic::symbol("N")});
    builder.add_container("A", array_1d, true);

    // Producer: Map(i) { Block { } }
    auto& producer = builder.add_map(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    builder.add_block(producer.root());

    // Consumer: Map(j) { Map(k) { } }  — inner body is intentionally empty.
    auto& consumer = builder.add_map(
        root,
        symbolic::symbol("j"),
        symbolic::Lt(symbolic::symbol("j"), symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("j"), symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    builder.add_map(
        consumer.root(),
        symbolic::symbol("k"),
        symbolic::Lt(symbolic::symbol("k"), symbolic::symbol("M")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("k"), symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );

    analysis::AnalysisManager am(builder.subject());
    passes::normalization::MapFusionPass pass;
    EXPECT_FALSE(pass.run(builder, am));
}
