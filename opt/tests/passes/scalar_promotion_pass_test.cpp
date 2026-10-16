#include "sdfg/passes/scalar_promotion_pass.h"

#include <gtest/gtest.h>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/structured_control_flow/reduce.h"
#include "sdfg/types/pointer.h"

using namespace sdfg;

namespace {

/// `for i < 128: for k < 256: c[i] += a[256 * i + k]`, the inner loop a Reduce into c, as frontends emit row sums.
std::unique_ptr<StructuredSDFG> row_sum() {
    builder::StructuredSDFGBuilder builder("row_sum", FunctionType_CPU);
    types::Scalar f32(types::PrimitiveType::Float);
    types::Pointer ptr(f32);
    builder.add_container("a", ptr, true);
    builder.add_container("c", ptr, true);
    builder.add_container("i", types::Scalar(types::PrimitiveType::Int64));
    builder.add_container("k", types::Scalar(types::PrimitiveType::Int64));
    auto i = symbolic::symbol("i");
    auto k = symbolic::symbol("k");
    auto& rows = builder.add_map(
        builder.subject().root(),
        i,
        symbolic::Lt(i, symbolic::integer(128)),
        symbolic::zero(),
        symbolic::add(i, symbolic::one()),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    auto& reduce = builder.add_reduce(
        rows.root(),
        k,
        symbolic::Lt(k, symbolic::integer(256)),
        symbolic::zero(),
        symbolic::add(k, symbolic::one()),
        {{structured_control_flow::ReductionOperation::Add, "c"}},
        structured_control_flow::ScheduleType_Sequential::create()
    );
    auto& block = builder.add_block(reduce.root());
    auto& add = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    auto element = symbolic::add(symbolic::mul(i, symbolic::integer(256)), k);
    builder.add_computational_memlet(block, builder.add_access(block, "a"), add, "_in1", {element}, ptr);
    builder.add_computational_memlet(block, builder.add_access(block, "c"), add, "_in2", {i}, ptr);
    builder.add_computational_memlet(block, add, "_out", builder.add_access(block, "c"), {i}, ptr);
    return builder.move();
}

/// The container the single memlet between @p block's access nodes and its tasklet reads (@p read) or writes, with
/// that memlet's subset.
std::pair<std::string, data_flow::Subset> assigned(structured_control_flow::Block& block, bool read) {
    for (auto& memlet : block.dataflow().edges()) {
        auto access = dynamic_cast<const data_flow::AccessNode*>(read ? &memlet.src() : &memlet.dst());
        if (access) {
            return {access->data(), memlet.subset()};
        }
    }
    return {};
}

} // namespace

TEST(ScalarPromotionPassTest, PromotesTheElementAReduceAccumulatesInto) {
    auto sdfg = row_sum();
    builder::StructuredSDFGBuilder builder(sdfg);
    analysis::AnalysisManager analysis_manager(builder.subject());
    EXPECT_TRUE(passes::ScalarPromotionPass().run_pass(builder, analysis_manager));
    builder.subject().validate();

    // acc = c[i]; Reduce k into acc; c[i] = acc
    auto& rows = static_cast<structured_control_flow::Map&>(builder.subject().root().at(0));
    ASSERT_EQ(rows.root().size(), 3);
    auto& reduce = dynamic_cast<structured_control_flow::Reduce&>(rows.root().at(1));
    auto accumulator = reduce.reductions().front().container;
    EXPECT_NE(dynamic_cast<const types::Scalar*>(&builder.subject().type(accumulator)), nullptr);

    auto [loaded, element] = assigned(static_cast<structured_control_flow::Block&>(rows.root().at(0)), true);
    EXPECT_EQ(loaded, "c");
    ASSERT_EQ(element.size(), 1);
    EXPECT_TRUE(symbolic::eq(element.at(0), symbolic::symbol("i")));
    auto [initialized, local] = assigned(static_cast<structured_control_flow::Block&>(rows.root().at(0)), false);
    EXPECT_EQ(initialized, accumulator);
    EXPECT_TRUE(local.empty());
    auto [stored, stored_element] = assigned(static_cast<structured_control_flow::Block&>(rows.root().at(2)), false);
    EXPECT_EQ(stored, "c");
    ASSERT_EQ(stored_element.size(), 1);
    EXPECT_TRUE(symbolic::eq(stored_element.at(0), symbolic::symbol("i")));

    // The body adds into the scalar.
    auto& body = static_cast<structured_control_flow::Block&>(reduce.root().at(0));
    for (auto& memlet : body.dataflow().edges()) {
        for (auto* node : {&memlet.src(), &memlet.dst()}) {
            auto access = dynamic_cast<const data_flow::AccessNode*>(node);
            EXPECT_FALSE(access && access->data() == "c");
            if (access && access->data() == accumulator) {
                EXPECT_TRUE(memlet.subset().empty());
            }
        }
    }
}

TEST(ScalarPromotionPassTest, LeavesReductionsIntoScalars) {
    builder::StructuredSDFGBuilder builder("sum", FunctionType_CPU);
    types::Scalar f32(types::PrimitiveType::Float);
    types::Pointer ptr(f32);
    builder.add_container("a", ptr, true);
    builder.add_container("acc", f32);
    builder.add_container("k", types::Scalar(types::PrimitiveType::Int64));
    auto k = symbolic::symbol("k");
    auto& reduce = builder.add_reduce(
        builder.subject().root(),
        k,
        symbolic::Lt(k, symbolic::integer(256)),
        symbolic::zero(),
        symbolic::add(k, symbolic::one()),
        {{structured_control_flow::ReductionOperation::Add, "acc"}},
        structured_control_flow::ScheduleType_Sequential::create()
    );
    auto& block = builder.add_block(reduce.root());
    auto& add = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, builder.add_access(block, "a"), add, "_in1", {k}, ptr);
    builder.add_computational_memlet(block, builder.add_access(block, "acc"), add, "_in2", {}, f32);
    builder.add_computational_memlet(block, add, "_out", builder.add_access(block, "acc"), {}, f32);

    analysis::AnalysisManager analysis_manager(builder.subject());
    EXPECT_FALSE(passes::ScalarPromotionPass().run_pass(builder, analysis_manager));
}
