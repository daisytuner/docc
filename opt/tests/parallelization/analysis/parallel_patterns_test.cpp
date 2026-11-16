#include "sdfg/parallelization/analysis/parallel_patterns.h"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/parallelization/passes/auto_parallelization.h"
#include "sdfg/serializer/json_serializer.h"
#include "sdfg/types/array.h"

using namespace sdfg;

namespace {

structured_control_flow::Map& add_map(
    builder::StructuredSDFGBuilder& builder,
    structured_control_flow::Sequence& parent,
    const std::string& name,
    symbolic::Expression bound = symbolic::integer(63)
) {
    builder.add_container(name, types::Scalar(types::PrimitiveType::Int64));
    auto i = symbolic::symbol(name);
    return builder.add_map(
        parent,
        i,
        symbolic::Lt(i, bound),
        symbolic::one(),
        symbolic::add(i, symbolic::one()),
        structured_control_flow::ScheduleType_Sequential::create()
    );
}

void add_body(
    builder::StructuredSDFGBuilder& builder,
    structured_control_flow::Sequence& parent,
    const std::vector<data_flow::Subset>& reads,
    const data_flow::Subset& write,
    const std::string& output = "B",
    data_flow::TaskletCode operation = data_flow::TaskletCode::fp_add
) {
    types::Scalar scalar(types::PrimitiveType::Float);
    types::Pointer pointer(scalar);
    auto& block = builder.add_block(parent);
    auto& source = builder.add_access(block, "A");
    auto& destination = builder.add_access(block, output);
    data_flow::Tasklet* previous = nullptr;
    for (size_t i = 1; i < reads.size(); ++i) {
        auto& tasklet = builder.add_tasklet(block, operation, "_out", {"_in1", "_in2"});
        if (previous) {
            auto name = "tmp_" + std::to_string(i);
            builder.add_container(name, scalar);
            auto& temporary = builder.add_access(block, name);
            builder.add_computational_memlet(block, *previous, "_out", temporary, {}, scalar);
            builder.add_computational_memlet(block, temporary, tasklet, "_in1", {}, scalar);
        } else {
            builder.add_computational_memlet(block, source, tasklet, "_in1", reads[0], pointer);
        }
        builder.add_computational_memlet(block, source, tasklet, "_in2", reads[i], pointer);
        previous = &tasklet;
    }
    builder.add_computational_memlet(block, *previous, "_out", destination, write, pointer);
}

void add_arrays(builder::StructuredSDFGBuilder& builder, bool nested = false) {
    types::Scalar scalar(types::PrimitiveType::Float);
    if (nested) {
        types::Array row(scalar, symbolic::integer(64));
        types::Pointer matrix(row);
        builder.add_container("A", matrix, true);
        builder.add_container("B", matrix, true);
    } else {
        builder.add_container("A", types::Pointer(scalar), true);
        builder.add_container("B", types::Pointer(scalar), true);
    }
}

TEST(ParallelPatternsTest, ThreePointNeighborhoodDeduplicatesReadsAndCachesWithoutChangingMap) {
    builder::StructuredSDFGBuilder builder("three_point", FunctionType_CPU);
    add_arrays(builder);
    auto& map = add_map(builder, builder.subject().root(), "i");
    auto i = map.indvar();
    add_body(
        builder,
        map.root(),
        {{symbolic::sub(i, symbolic::one())}, {i}, {symbolic::add(i, symbolic::integer(2))}, {i}},
        {i}
    );
    auto& literal_block = builder.add_block(map.root());
    types::Scalar scalar(types::PrimitiveType::Float);
    builder.add_container("coefficient", scalar);
    auto& literal = builder.add_constant(literal_block, "0.2", scalar);
    auto& coefficient = builder.add_access(literal_block, "coefficient");
    auto& assign = builder.add_tasklet(literal_block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(literal_block, literal, assign, "_in", {}, scalar);
    builder.add_computational_memlet(literal_block, assign, "_out", coefficient, {}, scalar);
    analysis::AnalysisManager manager(builder.subject());
    serializer::JSONSerializer serializer;
    auto before = serializer.serialize(builder.subject());
    auto& patterns = manager.get<parallelization::ParallelPatterns>();
    const auto& result = patterns.stencil(map);
    ASSERT_TRUE(result);
    EXPECT_EQ(result->axes, (std::vector<const structured_control_flow::Map*>{&map}));
    EXPECT_EQ(result->outputs, (std::set<std::string>{"B"}));
    EXPECT_EQ(result->inputs.at("A").offsets, (std::set<parallelization::NeighborhoodOffset>{{-1}, {0}, {2}}));
    EXPECT_EQ(result->inputs.at("A").lower_halo, (parallelization::NeighborhoodOffset{1}));
    EXPECT_EQ(result->inputs.at("A").upper_halo, (parallelization::NeighborhoodOffset{2}));
    EXPECT_EQ(&result, &patterns.stencil(map));
    EXPECT_EQ(before, serializer.serialize(builder.subject()));

    // Changing the map body must not retain its cached neighborhood.
    auto& block = static_cast<structured_control_flow::Block&>(map.root().at(0));
    auto& a = builder.add_access(block, "A");
    auto& b = builder.add_access(block, "B");
    auto& copy = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder
        .add_computational_memlet(block, a, copy, "_in", {i}, types::Pointer(types::Scalar(types::PrimitiveType::Float)));
    builder.add_computational_memlet(
        block,
        copy,
        "_out",
        b,
        {symbolic::add(i, symbolic::one())},
        types::Pointer(types::Scalar(types::PrimitiveType::Float))
    );
    manager.invalidate_all(map);
    EXPECT_FALSE(manager.get<parallelization::ParallelPatterns>().stencil(map));
}

TEST(ParallelPatternsTest, NonlinearFivePointNeighborhoodUsesLogicalCoordinates) {
    builder::StructuredSDFGBuilder builder("five_point", FunctionType_CPU);
    add_arrays(builder, true);
    auto& rows = add_map(builder, builder.subject().root(), "i");
    auto& columns = add_map(builder, rows.root(), "j");
    auto i = rows.indvar(), j = columns.indvar();
    add_body(
        builder,
        columns.root(),
        {{symbolic::sub(i, symbolic::one()), j},
         {i, j},
         {symbolic::add(i, symbolic::one()), j},
         {i, symbolic::sub(j, symbolic::one())},
         {i, symbolic::add(j, symbolic::one())}},
        {i, j},
        "B",
        data_flow::TaskletCode::fp_mul
    );
    analysis::AnalysisManager manager(builder.subject());
    auto& result = manager.get<parallelization::ParallelPatterns>().stencil(rows);
    ASSERT_TRUE(result);
    EXPECT_EQ(
        result->inputs.at("A").offsets,
        (std::set<parallelization::NeighborhoodOffset>{{-1, 0}, {0, -1}, {0, 0}, {0, 1}, {1, 0}})
    );
    EXPECT_EQ(result->inputs.at("A").lower_halo, (parallelization::NeighborhoodOffset{1, 1}));
    EXPECT_EQ(result->inputs.at("A").upper_halo, (parallelization::NeighborhoodOffset{1, 1}));
}

TEST(ParallelPatternsTest, LinearizedMatrixAccessesAreDelinearized) {
    builder::StructuredSDFGBuilder builder("flat_matrix", FunctionType_CPU);
    add_arrays(builder);
    auto& rows = add_map(builder, builder.subject().root(), "i");
    auto& columns = add_map(builder, rows.root(), "j");
    auto index = symbolic::add(symbolic::mul(rows.indvar(), symbolic::integer(64)), columns.indvar());
    add_body(
        builder,
        columns.root(),
        {{symbolic::sub(index, symbolic::integer(64))}, {index}, {symbolic::add(index, symbolic::one())}},
        {index}
    );
    analysis::AnalysisManager manager(builder.subject());
    auto& result = manager.get<parallelization::ParallelPatterns>().stencil(rows);
    ASSERT_TRUE(result);
    EXPECT_EQ(result->inputs.at("A").offsets, (std::set<parallelization::NeighborhoodOffset>{{-1, 0}, {0, 0}, {0, 1}}));
}

TEST(ParallelPatternsTest, RejectsPointwiseIndirectStridedAndInPlaceAccesses) {
    for (const auto& kind : {"pointwise", "indirect", "strided", "in_place", "symbolic_offset"}) {
        SCOPED_TRACE(kind);
        builder::StructuredSDFGBuilder builder("rejected", FunctionType_CPU);
        add_arrays(builder);
        builder.add_container("k", types::Scalar(types::PrimitiveType::Int64), true);
        auto& map = add_map(builder, builder.subject().root(), "i");
        auto i = map.indvar();
        auto second = symbolic::add(i, symbolic::one());
        std::string output = "B";
        if (std::string(kind) == "pointwise") {
            second = i;
        } else if (std::string(kind) == "indirect") {
            second = symbolic::symbol("k");
        } else if (std::string(kind) == "strided") {
            second = symbolic::mul(i, symbolic::integer(2));
        } else if (std::string(kind) == "in_place") {
            output = "A";
        } else {
            second = symbolic::add(i, symbolic::symbol("k"));
        }
        add_body(builder, map.root(), {{i}, {second}}, {i}, output);
        analysis::AnalysisManager manager(builder.subject());
        EXPECT_FALSE(manager.get<parallelization::ParallelPatterns>().stencil(map));
    }
}

TEST(ParallelPatternsTest, RejectsNonRectangularAndSequentialBodies) {
    for (bool sequential : {false, true}) {
        builder::StructuredSDFGBuilder builder("unsupported_nest", FunctionType_CPU);
        add_arrays(builder, true);
        auto& rows = add_map(builder, builder.subject().root(), "i");
        if (sequential) {
            auto j = symbolic::symbol("j");
            builder.add_container("j", types::Scalar(types::PrimitiveType::Int64));
            builder.add_for(
                rows.root(),
                j,
                symbolic::Lt(j, symbolic::integer(63)),
                symbolic::one(),
                symbolic::add(j, symbolic::one())
            );
        } else {
            auto& columns = add_map(builder, rows.root(), "j", rows.indvar());
            add_body(
                builder,
                columns.root(),
                {{rows.indvar(), columns.indvar()}, {rows.indvar(), symbolic::add(columns.indvar(), symbolic::one())}},
                {rows.indvar(), columns.indvar()}
            );
        }
        analysis::AnalysisManager manager(builder.subject());
        EXPECT_FALSE(manager.get<parallelization::ParallelPatterns>().stencil(rows));
    }
}

TEST(ParallelPatternsTest, OutputBaseMustBeLoopInvariant) {
    for (bool varying : {false, true}) {
        SCOPED_TRACE(varying);
        builder::StructuredSDFGBuilder builder("offset_base", FunctionType_CPU);
        add_arrays(builder);
        types::Scalar integer(types::PrimitiveType::Int64);
        builder.add_container("base", integer, !varying);
        auto& map = add_map(builder, builder.subject().root(), "i");
        if (varying) {
            auto& block = builder.add_block(map.root());
            auto& i = builder.add_access(block, "i");
            auto& base = builder.add_access(block, "base");
            auto& copy = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
            builder.add_computational_memlet(block, i, copy, "_in", {}, integer);
            builder.add_computational_memlet(block, copy, "_out", base, {}, integer);
        }
        auto index = symbolic::add(map.indvar(), symbolic::symbol("base"));
        add_body(builder, map.root(), {{index}, {symbolic::add(index, symbolic::one())}}, {index});
        analysis::AnalysisManager manager(builder.subject());
        const auto& result = manager.get<parallelization::ParallelPatterns>().stencil(map);
        if (varying) {
            EXPECT_FALSE(result);
        } else {
            ASSERT_TRUE(result);
            EXPECT_TRUE(symbolic::eq(result->output_index[0], index));
            EXPECT_EQ(result->inputs.at("A").offsets, (std::set<parallelization::NeighborhoodOffset>{{0}, {1}}));
        }
    }
}

} // namespace
