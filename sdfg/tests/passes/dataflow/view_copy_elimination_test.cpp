#include "sdfg/passes/dataflow/view_copy_elimination.h"

#include <gtest/gtest.h>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_nodes/math/math.h"
#include "sdfg/data_flow/library_nodes/stdlib/stdlib.h"
#include "sdfg/types/pointer.h"

using namespace sdfg;
using types::PrimitiveType;

namespace {

const types::Scalar FLOAT(PrimitiveType::Float);
const types::Pointer POINTER(FLOAT);

math::tensor::TensorLayout row_major(std::vector<int64_t> shape) {
    symbolic::MultiExpression dims;
    for (auto extent : shape) {
        dims.push_back(symbolic::integer(extent));
    }
    return math::tensor::TensorLayout(dims);
}

void add_allocation(builder::StructuredSDFGBuilder& builder, const std::string& name, int64_t bytes) {
    auto& block = builder.add_block(builder.subject().root());
    auto& malloc = builder.add_library_node<stdlib::MallocNode>(block, DebugInfo(), symbolic::integer(bytes));
    builder.add_computational_memlet(block, malloc, "_ret", builder.add_access(block, name), {}, POINTER);
}

void add_free(builder::StructuredSDFGBuilder& builder, const std::string& name) {
    auto& block = builder.add_block(builder.subject().root());
    auto& free = builder.add_library_node<stdlib::FreeNode>(block, DebugInfo());
    builder.add_computational_memlet(block, builder.add_access(block, name), free, "_ptr", {}, POINTER);
}

/// `to = relu(from)`, standing for what computes the intermediate; the node writes through its input Y.
void add_compute(builder::StructuredSDFGBuilder& builder, const std::string& from, const std::string& to) {
    auto& block = builder.add_block(builder.subject().root());
    std::vector<symbolic::Expression> shape{symbolic::integer(64)};
    auto& relu = builder.add_library_node<math::tensor::ReLUNode>(block, DebugInfo(), shape);
    types::Tensor tensor(FLOAT, math::tensor::TensorLayout(shape));
    builder.add_computational_memlet(block, builder.add_access(block, from), relu, "X", {}, tensor);
    builder.add_computational_memlet(block, builder.add_access(block, to), relu, "Y", {}, tensor);
}

void add_copy(
    builder::StructuredSDFGBuilder& builder,
    const std::string& from,
    const std::string& to,
    const math::tensor::TensorLayout& from_layout,
    const math::tensor::TensorLayout& to_layout
) {
    auto& block = builder.add_block(builder.subject().root());
    auto& copy = builder.add_library_node<math::tensor::TensorCopyNode>(block, DebugInfo(), from_layout, to_layout);
    builder.add_computational_memlet(
        block, builder.add_access(block, from), copy, "X", {}, types::Tensor(FLOAT, from_layout)
    );
    builder
        .add_computational_memlet(block, builder.add_access(block, to), copy, "Y", {}, types::Tensor(FLOAT, to_layout));
}

/// x -> t (allocated, computed, copied into the view y, freed).
void add_view(
    builder::StructuredSDFGBuilder& builder,
    const math::tensor::TensorLayout& from_layout = row_major({1, 64}),
    const math::tensor::TensorLayout& to_layout = row_major({1, 1, 64})
) {
    builder.add_container("x", POINTER, true);
    builder.add_container("y", POINTER, true);
    builder.add_container("t", POINTER);
    add_allocation(builder, "t", 256);
    add_compute(builder, "x", "t");
    add_copy(builder, "t", "y", from_layout, to_layout);
    add_free(builder, "t");
}

bool eliminate(builder::StructuredSDFGBuilder& builder) {
    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::ViewCopyElimination pass;
    return pass.run(builder, analysis_manager);
}

std::set<std::string> accessed(structured_control_flow::Block& block) {
    std::set<std::string> names;
    for (auto* access : block.dataflow().data_nodes()) {
        names.insert(access->data());
    }
    return names;
}

} // namespace

TEST(ViewCopyElimination, Reshape) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    add_view(builder);

    EXPECT_TRUE(eliminate(builder));
    auto& root = builder.subject().root();
    ASSERT_EQ(root.size(), 1);
    EXPECT_EQ(accessed(static_cast<structured_control_flow::Block&>(root.at(0))), (std::set<std::string>{"x", "y"}));
    EXPECT_FALSE(builder.subject().exists("t"));
}

TEST(ViewCopyElimination, Transposed) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    math::tensor::TensorLayout
        transposed({symbolic::integer(8), symbolic::integer(8)}, {symbolic::one(), symbolic::integer(8)});
    add_view(builder, row_major({8, 8}), transposed);

    EXPECT_FALSE(eliminate(builder));
    EXPECT_EQ(builder.subject().root().size(), 4);
}

TEST(ViewCopyElimination, ReadAfter) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    builder.add_container("z", POINTER, true);
    add_view(builder);
    add_compute(builder, "t", "z");

    EXPECT_FALSE(eliminate(builder));
    EXPECT_TRUE(builder.subject().exists("t"));
}

TEST(ViewCopyElimination, TargetUsedBefore) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    builder.add_container("x", POINTER, true);
    builder.add_container("y", POINTER, true);
    builder.add_container("t", POINTER);
    add_compute(builder, "x", "y");
    add_allocation(builder, "t", 256);
    add_compute(builder, "y", "t");
    add_copy(builder, "t", "y", row_major({64}), row_major({64}));
    add_free(builder, "t");

    EXPECT_FALSE(eliminate(builder));
}

TEST(ViewCopyElimination, TransientTarget) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    builder.add_container("x", POINTER, true);
    builder.add_container("y", POINTER);
    builder.add_container("t", POINTER);
    add_allocation(builder, "t", 256);
    add_compute(builder, "x", "t");
    add_copy(builder, "t", "y", row_major({64}), row_major({64}));
    add_free(builder, "t");

    EXPECT_FALSE(eliminate(builder));
}
