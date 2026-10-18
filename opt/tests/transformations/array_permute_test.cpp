#include "sdfg/transformations/array_permute.h"

#include <gtest/gtest.h>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_nodes/math/tensor/copy_node.h"
#include "sdfg/data_flow/library_nodes/stdlib/free.h"
#include "sdfg/data_flow/library_nodes/stdlib/malloc.h"
#include "sdfg/structured_control_flow/map.h"

using namespace sdfg;

namespace {

types::Pointer float_pointer() {
    return types::Pointer(types::Scalar(types::PrimitiveType::Float));
}

structured_control_flow::Map& add_map(
    builder::StructuredSDFGBuilder& builder,
    structured_control_flow::Sequence& scope,
    const std::string& name,
    int64_t extent
) {
    if (!builder.subject().exists(name)) {
        builder.add_container(name, types::Scalar(types::PrimitiveType::Int64));
    }
    auto indvar = symbolic::symbol(name);
    return builder.add_map(
        scope,
        indvar,
        symbolic::Lt(indvar, symbolic::integer(extent)),
        symbolic::zero(),
        symbolic::add(indvar, symbolic::one()),
        structured_control_flow::ScheduleType_Sequential::create()
    );
}

/// A nest over i < 4, j < 8 copying `source[read]` to `target[write]`.
structured_control_flow::Map& copy_nest(
    builder::StructuredSDFGBuilder& builder,
    const std::string& source,
    const symbolic::Expression& read,
    const std::string& target,
    const symbolic::Expression& write
) {
    auto& outer = add_map(builder, builder.subject().root(), "i", 4);
    auto& inner = add_map(builder, outer.root(), "j", 8);
    auto& block = builder.add_block(inner.root());
    auto& in = builder.add_access(block, source);
    auto& out = builder.add_access(block, target);
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, in, tasklet, "_in", {read}, float_pointer());
    builder.add_computational_memlet(block, tasklet, "_out", out, {write}, float_pointer());
    return outer;
}

void bookkeeping(builder::StructuredSDFGBuilder& builder, const std::string& name, bool allocate) {
    auto& block = builder.add_block(builder.subject().root());
    auto& access = builder.add_access(block, name);
    if (allocate) {
        auto& malloc = builder.add_library_node<stdlib::MallocNode>(block, DebugInfo(), symbolic::integer(128));
        builder.add_computational_memlet(block, malloc, "_ret", access, {}, float_pointer());
    } else {
        auto& free = builder.add_library_node<stdlib::FreeNode>(block, DebugInfo());
        builder.add_computational_memlet(
            block, access, free, "_ptr", {}, types::Pointer(types::Scalar(types::PrimitiveType::Void))
        );
    }
}

/// The subset of the memlet between @p container and a tasklet in the nest at @p index of the root.
symbolic::Expression subset_in(builder::StructuredSDFGBuilder& builder, size_t index, const std::string& container) {
    auto& outer = dynamic_cast<structured_control_flow::Map&>(builder.subject().root().at(index));
    auto& inner = dynamic_cast<structured_control_flow::Map&>(outer.root().at(0));
    auto& block = dynamic_cast<structured_control_flow::Block&>(inner.root().at(0));
    for (auto& memlet : block.dataflow().edges()) {
        for (auto* node : {&memlet.src(), &memlet.dst()}) {
            if (auto access = dynamic_cast<data_flow::AccessNode*>(node); access && access->data() == container) {
                return memlet.subset().at(0);
            }
        }
    }
    return SymEngine::null;
}

const math::tensor::TensorCopyNode* copy_at(builder::StructuredSDFGBuilder& builder, size_t index) {
    auto block = dynamic_cast<structured_control_flow::Block*>(&builder.subject().root().at(index));
    if (!block) {
        return nullptr;
    }
    auto nodes = block->dataflow().library_nodes();
    return nodes.size() == 1 ? dynamic_cast<const math::tensor::TensorCopyNode*>(*nodes.begin()) : nullptr;
}

symbolic::Expression row_major(const symbolic::Expression& i, const symbolic::Expression& j, int64_t columns) {
    return symbolic::add(symbolic::mul(symbolic::integer(columns), i), j);
}

} // namespace

TEST(ArrayPermuteTest, TransposesATransientInPlace) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);
    builder.add_container("A", float_pointer(), true);
    builder.add_container("B", float_pointer(), true);
    builder.add_container("T", float_pointer());
    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");
    bookkeeping(builder, "T", true);
    auto& produce = copy_nest(builder, "A", row_major(i, j, 8), "T", row_major(i, j, 8));
    auto& consume = copy_nest(builder, "T", row_major(i, j, 8), "B", row_major(i, j, 8));
    bookkeeping(builder, "T", false);

    analysis::AnalysisManager analysis_manager(builder.subject());
    transformations::ArrayPermute permute(builder.subject().root(), produce, consume, "T", {4, 8}, {1, 0});
    ASSERT_TRUE(permute.can_be_applied(builder, analysis_manager));
    permute.apply(builder, analysis_manager);

    EXPECT_EQ(permute.permuted_container(), "T");
    EXPECT_EQ(builder.subject().root().size(), 4);
    EXPECT_TRUE(symbolic::eq(subset_in(builder, 1, "T"), row_major(j, i, 4)));
    EXPECT_TRUE(symbolic::eq(subset_in(builder, 2, "T"), row_major(j, i, 4)));
    EXPECT_TRUE(symbolic::eq(subset_in(builder, 1, "A"), row_major(i, j, 8)));
}

TEST(ArrayPermuteTest, CopiesAnArgumentIntoTheRegion) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);
    builder.add_container("A", float_pointer(), true);
    builder.add_container("B", float_pointer(), true);
    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");
    auto& nest = copy_nest(builder, "A", row_major(i, j, 8), "B", row_major(i, j, 8));

    analysis::AnalysisManager analysis_manager(builder.subject());
    transformations::ArrayPermute permute(builder.subject().root(), nest, nest, "A", {4, 8}, {1, 0});
    ASSERT_TRUE(permute.can_be_applied(builder, analysis_manager));
    permute.apply(builder, analysis_manager);

    auto& copy = permute.permuted_container();
    EXPECT_NE(copy, "A");
    // malloc, copy in, the nest, free: the region only reads A.
    ASSERT_EQ(builder.subject().root().size(), 4);
    auto copy_in = copy_at(builder, 1);
    ASSERT_NE(copy_in, nullptr);
    EXPECT_EQ(
        copy_in->layout_x().toStr(),
        math::tensor::TensorLayout({symbolic::integer(8), symbolic::integer(4)}, {symbolic::one(), symbolic::integer(8)})
            .toStr()
    );
    EXPECT_EQ(
        copy_in->layout_y().toStr(), math::tensor::TensorLayout({symbolic::integer(8), symbolic::integer(4)}).toStr()
    );
    EXPECT_TRUE(symbolic::eq(subset_in(builder, 2, copy), row_major(j, i, 4)));
    EXPECT_TRUE(symbolic::eq(subset_in(builder, 2, "B"), row_major(i, j, 8)));
}

TEST(ArrayPermuteTest, CopiesBackWhatTheRegionWrites) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);
    builder.add_container("A", float_pointer(), true);
    builder.add_container("B", float_pointer(), true);
    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");
    auto& nest = copy_nest(builder, "A", row_major(i, j, 8), "B", row_major(i, j, 8));

    analysis::AnalysisManager analysis_manager(builder.subject());
    transformations::ArrayPermute permute(builder.subject().root(), nest, nest, "B", {4, 8}, {1, 0});
    ASSERT_TRUE(permute.can_be_applied(builder, analysis_manager));
    permute.apply(builder, analysis_manager);

    // malloc, the nest, copy out, free: the region writes all of B, so nothing needs copying in.
    ASSERT_EQ(builder.subject().root().size(), 4);
    auto copy_out = copy_at(builder, 2);
    ASSERT_NE(copy_out, nullptr);
    EXPECT_EQ(
        copy_out->layout_x().toStr(),
        math::tensor::TensorLayout({symbolic::integer(4), symbolic::integer(8)}, {symbolic::one(), symbolic::integer(4)})
            .toStr()
    );
    EXPECT_EQ(
        copy_out->layout_y().toStr(), math::tensor::TensorLayout({symbolic::integer(4), symbolic::integer(8)}).toStr()
    );
    EXPECT_TRUE(symbolic::eq(subset_in(builder, 1, permute.permuted_container()), row_major(j, i, 4)));
}

TEST(ArrayPermuteTest, CopiesInWhatTheRegionWritesInPart) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);
    builder.add_container("A", float_pointer(), true);
    builder.add_container("B", float_pointer(), true);
    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");
    // The nest writes the left half of each row of B, 4 x 16.
    auto& nest = copy_nest(builder, "A", row_major(i, j, 8), "B", row_major(i, j, 16));

    analysis::AnalysisManager analysis_manager(builder.subject());
    transformations::ArrayPermute permute(builder.subject().root(), nest, nest, "B", {4, 16}, {1, 0});
    ASSERT_TRUE(permute.can_be_applied(builder, analysis_manager));
    permute.apply(builder, analysis_manager);

    // malloc, copy in, the nest, copy out, free: the right halves come back unchanged.
    ASSERT_EQ(builder.subject().root().size(), 5);
    EXPECT_NE(copy_at(builder, 1), nullptr);
    EXPECT_NE(copy_at(builder, 3), nullptr);
}

TEST(ArrayPermuteTest, TransposesEachMatrixOfABatch) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);
    builder.add_container("A", float_pointer(), true);
    builder.add_container("B", float_pointer(), true);
    builder.add_container("T", float_pointer());
    auto b = symbolic::symbol("b");
    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");
    // T[b][i][j] = A[i][8b + j], the heads of A one after the other: a batch of two 4 x 8 matrices.
    auto& heads = add_map(builder, builder.subject().root(), "b", 2);
    auto& rows = add_map(builder, heads.root(), "i", 4);
    auto& columns = add_map(builder, rows.root(), "j", 8);
    auto& block = builder.add_block(columns.root());
    auto& in = builder.add_access(block, "A");
    auto& out = builder.add_access(block, "T");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    auto read = symbolic::add(row_major(i, j, 16), symbolic::mul(symbolic::integer(8), b));
    auto write = symbolic::add(symbolic::mul(symbolic::integer(32), b), row_major(i, j, 8));
    builder.add_computational_memlet(block, in, tasklet, "_in", {read}, float_pointer());
    builder.add_computational_memlet(block, tasklet, "_out", out, {write}, float_pointer());

    analysis::AnalysisManager analysis_manager(builder.subject());
    transformations::ArrayPermute permute(builder.subject().root(), heads, heads, "T", {2, 4, 8}, {0, 2, 1});
    ASSERT_TRUE(permute.can_be_applied(builder, analysis_manager));
    permute.apply(builder, analysis_manager);

    // T only lives in the region, so it is stored transposed in place: T[b][j][i].
    auto expected = symbolic::add(symbolic::mul(symbolic::integer(32), b), row_major(j, i, 4));
    for (auto& memlet : block.dataflow().edges()) {
        if (auto access = dynamic_cast<data_flow::AccessNode*>(&memlet.dst()); access && access->data() == "T") {
            EXPECT_TRUE(symbolic::eq(memlet.subset().at(0), symbolic::expand(expected)))
                << memlet.subset().at(0)->__str__();
        }
    }
}

TEST(ArrayPermuteTest, RejectsAccessesAcrossRows) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);
    builder.add_container("A", float_pointer(), true);
    builder.add_container("B", float_pointer(), true);
    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");
    auto& nest =
        copy_nest(builder, "A", symbolic::add(row_major(i, j, 8), symbolic::integer(4)), "B", row_major(i, j, 8));

    analysis::AnalysisManager analysis_manager(builder.subject());
    transformations::ArrayPermute permute(builder.subject().root(), nest, nest, "A", {5, 8}, {1, 0});
    EXPECT_FALSE(permute.can_be_applied(builder, analysis_manager));
}

TEST(ArrayPermuteTest, HandsOnAContainerTheRegionAllocates) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);
    builder.add_container("A", float_pointer(), true);
    builder.add_container("B", float_pointer(), true);
    builder.add_container("T", float_pointer());
    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");
    bookkeeping(builder, "T", true);
    auto& produce = copy_nest(builder, "A", row_major(i, j, 8), "T", row_major(i, j, 8));
    copy_nest(builder, "T", row_major(i, j, 8), "B", row_major(i, j, 8));
    auto& root = builder.subject().root();

    analysis::AnalysisManager analysis_manager(builder.subject());
    transformations::ArrayPermute permute(root, root.at(0), produce, "T", {4, 8}, {1, 0});
    ASSERT_TRUE(permute.can_be_applied(builder, analysis_manager));
    permute.apply(builder, analysis_manager);

    // malloc of the copy, malloc of T, the nest, copy out, free of the copy, the nest reading T.
    ASSERT_EQ(root.size(), 6);
    EXPECT_NE(copy_at(builder, 3), nullptr);
    EXPECT_TRUE(symbolic::eq(subset_in(builder, 5, "T"), row_major(i, j, 8)));
}
