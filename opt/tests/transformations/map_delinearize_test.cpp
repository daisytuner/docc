#include "sdfg/transformations/map_delinearize.h"

#include <gtest/gtest.h>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/structured_control_flow/map.h"

using namespace sdfg;

namespace {

/// A map over i in [0, extent) copying A[read] to B[write].
structured_control_flow::Map& flat_copy(
    builder::StructuredSDFGBuilder& builder,
    int64_t extent,
    const symbolic::Expression& read,
    const symbolic::Expression& write
) {
    types::Scalar element(types::PrimitiveType::Float);
    types::Pointer pointer(element);
    builder.add_container("A", pointer, true);
    builder.add_container("B", pointer, true);
    builder.add_container("i", types::Scalar(types::PrimitiveType::Int64));
    auto i = symbolic::symbol("i");
    auto& map = builder.add_map(
        builder.subject().root(),
        i,
        symbolic::Lt(i, symbolic::integer(extent)),
        symbolic::zero(),
        symbolic::add(i, symbolic::one()),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    auto& block = builder.add_block(map.root());
    auto& a = builder.add_access(block, "A");
    auto& b = builder.add_access(block, "B");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, a, tasklet, "_in", {read}, pointer);
    builder.add_computational_memlet(block, tasklet, "_out", b, {write}, pointer);
    return map;
}

symbolic::Expression subset_of(structured_control_flow::Block& block, const std::string& container) {
    for (auto& memlet : block.dataflow().edges()) {
        for (auto* node : {&memlet.src(), &memlet.dst()}) {
            auto access = dynamic_cast<const data_flow::AccessNode*>(node);
            if (access && access->data() == container) {
                return memlet.subset().at(0);
            }
        }
    }
    return SymEngine::null;
}

} // namespace

// The merge of attention heads, [4][128][64] to [128][256], fused into one flat copy.
TEST(MapDelinearizeTest, NestsAFlatCopyAlongItsDivisions) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);
    auto i = symbolic::symbol("i");
    auto head = symbolic::mod(symbolic::div(i, symbolic::integer(64)), symbolic::integer(4));
    auto read = symbolic::
        add(symbolic::
                add(symbolic::mul(symbolic::integer(64), symbolic::div(i, symbolic::integer(256))),
                    symbolic::mod(i, symbolic::integer(64))),
            symbolic::mul(symbolic::integer(8192), head));
    auto& map = flat_copy(builder, 32768, read, i);

    analysis::AnalysisManager analysis_manager(builder.subject());
    transformations::MapDelinearize delinearize(map);
    ASSERT_TRUE(delinearize.can_be_applied(builder, analysis_manager));
    delinearize.apply(builder, analysis_manager);

    auto& maps = delinearize.maps();
    ASSERT_EQ(maps.size(), 3);
    ASSERT_EQ(builder.subject().root().size(), 1);
    EXPECT_EQ(&builder.subject().root().at(0), maps[0]);
    std::vector<int64_t> extents;
    for (auto* nested : maps) {
        extents.push_back(SymEngine::down_cast<const SymEngine::Integer&>(*nested->canonical_bound()).as_int());
    }
    EXPECT_EQ(extents, (std::vector<int64_t>{128, 4, 64}));
    EXPECT_EQ(&maps[0]->root().at(0), maps[1]);
    EXPECT_EQ(&maps[1]->root().at(0), maps[2]);

    auto token = maps[0]->indvar();
    auto chunk = maps[1]->indvar();
    auto lane = maps[2]->indvar();
    ASSERT_EQ(maps[2]->root().size(), 1);
    auto& block = dynamic_cast<structured_control_flow::Block&>(maps[2]->root().at(0));
    auto expected_read = symbolic::
        add(symbolic::add(symbolic::mul(symbolic::integer(64), token), lane),
            symbolic::mul(symbolic::integer(8192), chunk));
    auto expected_write = symbolic::
        add(symbolic::add(symbolic::mul(symbolic::integer(256), token), symbolic::mul(symbolic::integer(64), chunk)),
            lane);
    EXPECT_TRUE(symbolic::eq(symbolic::expand(subset_of(block, "A")), symbolic::expand(expected_read)))
        << subset_of(block, "A")->__str__();
    EXPECT_TRUE(symbolic::eq(symbolic::expand(subset_of(block, "B")), symbolic::expand(expected_write)))
        << subset_of(block, "B")->__str__();
}

TEST(MapDelinearizeTest, KeepsTheInductionVariableTaskletsRead) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);
    auto i = symbolic::symbol("i");
    auto& map = flat_copy(builder, 16, symbolic::mod(i, symbolic::integer(4)), i);
    auto& block = dynamic_cast<structured_control_flow::Block&>(map.root().at(0));
    auto& index = builder.add_access(block, "i");
    auto& b = builder.add_access(block, "B");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, index, tasklet, "_in", {});
    builder.add_computational_memlet(block, tasklet, "_out", b, {symbolic::div(i, symbolic::integer(4))});

    analysis::AnalysisManager analysis_manager(builder.subject());
    transformations::MapDelinearize delinearize(map);
    ASSERT_TRUE(delinearize.can_be_applied(builder, analysis_manager));
    delinearize.apply(builder, analysis_manager);

    auto& maps = delinearize.maps();
    ASSERT_EQ(maps.size(), 2);
    auto& body = maps[1]->root();
    ASSERT_EQ(body.size(), 2);
    auto assignments = dynamic_cast<structured_control_flow::AssignmentBlock*>(&body.at(0));
    ASSERT_NE(assignments, nullptr);
    auto linear = symbolic::add(symbolic::mul(symbolic::integer(4), maps[0]->indvar()), maps[1]->indvar());
    ASSERT_TRUE(assignments->assignments().count(i));
    EXPECT_TRUE(symbolic::eq(symbolic::expand(assignments->assignments().at(i)), symbolic::expand(linear)));
}

TEST(MapDelinearizeTest, RejectsDivisionsThatDoNotNest) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);
    auto i = symbolic::symbol("i");
    auto& map = flat_copy(builder, 12, symbolic::div(i, symbolic::integer(4)), symbolic::mod(i, symbolic::integer(3)));

    analysis::AnalysisManager analysis_manager(builder.subject());
    transformations::MapDelinearize delinearize(map);
    EXPECT_FALSE(delinearize.can_be_applied(builder, analysis_manager));
}

TEST(MapDelinearizeTest, RejectsDivisionsThatDoNotDivideTheExtent) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);
    auto i = symbolic::symbol("i");
    auto& map = flat_copy(builder, 10, symbolic::mod(i, symbolic::integer(4)), i);

    analysis::AnalysisManager analysis_manager(builder.subject());
    transformations::MapDelinearize delinearize(map);
    EXPECT_FALSE(delinearize.can_be_applied(builder, analysis_manager));
}

TEST(MapDelinearizeTest, RejectsAffineMaps) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);
    auto i = symbolic::symbol("i");
    auto& map = flat_copy(builder, 16, symbolic::mul(symbolic::integer(2), i), i);

    analysis::AnalysisManager analysis_manager(builder.subject());
    transformations::MapDelinearize delinearize(map);
    EXPECT_FALSE(delinearize.can_be_applied(builder, analysis_manager));
}
