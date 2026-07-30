#include "sdfg/passes/normalization/map_fusion.h"

#include <gtest/gtest.h>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/array.h"
#include "sdfg/types/pointer.h"
#include "sdfg/types/scalar.h"
#include "sdfg_debug_dump.h"

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

// Reproduces the producer/consumer map pattern:
//   Map #302 [_i4 in 0.._s0] {
//     Block #103 {
//       _tmp_7[_i4] = _tmp_5[_i4] + 1.0;   // fp_add tasklet
//       pf[3 + _s1*_i4] = _tmp_5[_i4];     // assign tasklet
//     }
//   }
//   Map #383 [_slice_iter_0_4 in 0.._s0] {
//     Block #151 {
//       pf[3 + _s1*_slice_iter_0_4] = _tmp_7[_slice_iter_0_4];  // assign tasklet
//     }
//   }
// The consumer map reads _tmp_7 produced by the producer map. Running the
// MapFusionPass twice must be stable: after the first run modifies the SDFG,
// the second run must not modify it again to not stal
TEST(MapFusionPassTest, PipelineStableAfterFusion) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);
    auto& root = builder.subject().root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("_s0", sym_desc, true);
    builder.add_container("_s1", sym_desc, true);
    builder.add_container("_i4", sym_desc);
    builder.add_container("_slice_iter_0_4", sym_desc);

    types::Scalar float_desc(types::PrimitiveType::Float);
    types::Pointer float_ptr(float_desc);
    builder.add_container("_tmp_5", float_ptr, true);
    builder.add_container("_tmp_7", float_ptr, false);
    builder.add_container("pf", float_ptr, true);

    // Producer: Map(_i4) { Block #103 }
    auto& producer = builder.add_map(
        root,
        symbolic::symbol("_i4"),
        symbolic::Lt(symbolic::symbol("_i4"), symbolic::symbol("_s0")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("_i4"), symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    auto& producer_block = builder.add_block(producer.root());
    {
        // _tmp_7[_i4] = _tmp_5[_i4] + 1.0
        auto& acc_tmp5 = builder.add_access(producer_block, "_tmp_5");
        auto& acc_one = builder.add_constant(producer_block, "1.0", float_desc);
        auto& acc_tmp7 = builder.add_access(producer_block, "_tmp_7");
        auto& add_tasklet =
            builder.add_tasklet(producer_block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
        builder
            .add_computational_memlet(producer_block, acc_tmp5, add_tasklet, "_in1", {symbolic::symbol("_i4")}, float_ptr);
        builder.add_computational_memlet(producer_block, acc_one, add_tasklet, "_in2", {}, float_desc);
        builder
            .add_computational_memlet(producer_block, add_tasklet, "_out", acc_tmp7, {symbolic::symbol("_i4")}, float_ptr);

        // pf[3 + _s1*_i4] = _tmp_5[_i4]
        auto& acc_pf = builder.add_access(producer_block, "pf");
        auto& assign_tasklet = builder.add_tasklet(producer_block, data_flow::TaskletCode::assign, "_out", {"_in"});
        builder.add_computational_memlet(
            producer_block, acc_tmp5, assign_tasklet, "_in", {symbolic::symbol("_i4")}, float_ptr
        );
        builder.add_computational_memlet(
            producer_block,
            assign_tasklet,
            "_out",
            acc_pf,
            {symbolic::add(symbolic::integer(3), symbolic::mul(symbolic::symbol("_s1"), symbolic::symbol("_i4")))},
            float_ptr
        );
    }

    // Consumer: Map(_slice_iter_0_4) { Block #151 }
    auto& consumer = builder.add_map(
        root,
        symbolic::symbol("_slice_iter_0_4"),
        symbolic::Lt(symbolic::symbol("_slice_iter_0_4"), symbolic::symbol("_s0")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("_slice_iter_0_4"), symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    auto& consumer_block = builder.add_block(consumer.root());
    {
        // pf[3 + _s1*_slice_iter_0_4] = _tmp_7[_slice_iter_0_4]
        auto& acc_tmp7 = builder.add_access(consumer_block, "_tmp_7");
        auto& acc_pf = builder.add_access(consumer_block, "pf");
        auto& assign_tasklet = builder.add_tasklet(consumer_block, data_flow::TaskletCode::assign, "_out", {"_in"});
        builder.add_computational_memlet(
            consumer_block, acc_tmp7, assign_tasklet, "_in", {symbolic::symbol("_slice_iter_0_4")}, float_ptr
        );
        builder.add_computational_memlet(
            consumer_block,
            assign_tasklet,
            "_out",
            acc_pf,
            {symbolic::
                 add(symbolic::integer(3), symbolic::mul(symbolic::symbol("_s1"), symbolic::symbol("_slice_iter_0_4")))
            },
            float_ptr
        );
    }

    analysis::AnalysisManager am(builder.subject());
    passes::normalization::MapFusionPass pass(true, true);

    // First run: MapFusion may modify the SDFG (fuse producer/consumer).
    dump_sdfg(builder.subject(), "0.before-first");
    bool changed_first = pass.run(builder, am);
    dump_sdfg(builder.subject(), "1.after-first");

    // Second run: MapFusion must be stable and leave the SDFG unchanged.
    bool changed_second = pass.run(builder, am);
    dump_sdfg(builder.subject(), "3.after-second");

    EXPECT_TRUE(changed_first);
    EXPECT_FALSE(changed_second);
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
