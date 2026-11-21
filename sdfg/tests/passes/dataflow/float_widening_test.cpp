#include "sdfg/passes/dataflow/float_widening.h"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_nodes/math/math.h"
#include "sdfg/types/array.h"
#include "sdfg/types/pointer.h"
#include "sdfg/types/tensor.h"

using namespace sdfg;
using types::PrimitiveType;

namespace {

const std::set<PrimitiveType> NARROW = {PrimitiveType::Half, PrimitiveType::BFloat};

bool widen(builder::StructuredSDFGBuilder& builder, const std::set<PrimitiveType>& widened = NARROW) {
    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::FloatWidening pass(widened);
    return pass.run(builder, analysis_manager);
}

/// `y = x` in a new block of @p parent; returns the block.
structured_control_flow::Block& add_copy(
    builder::StructuredSDFGBuilder& builder, structured_control_flow::Sequence& parent, const types::Scalar& type
) {
    auto& block = builder.add_block(parent);
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, builder.add_access(block, "x"), tasklet, "_in", {}, type);
    builder.add_computational_memlet(block, tasklet, "_out", builder.add_access(block, "y"), {}, type);
    return block;
}

/// Connects every input and output of @p node to a container of its own, by memlets of @p type over @p element.
void connect(
    builder::StructuredSDFGBuilder& builder,
    structured_control_flow::Block& block,
    data_flow::LibraryNode& node,
    const types::IType& type,
    PrimitiveType element
) {
    auto container = [&](const std::string& connector) -> data_flow::AccessNode& {
        auto name = "c_" + connector;
        if (!builder.subject().exists(name)) {
            builder.add_container(name, types::Pointer(types::Scalar(element)), true);
        }
        return builder.add_access(block, name);
    };
    for (auto& input : node.inputs()) {
        builder.add_computational_memlet(block, container(input), node, input, {}, type);
    }
    for (auto& output : node.outputs()) {
        builder.add_computational_memlet(block, node, output, container(output), {}, type);
    }
}

} // namespace

TEST(FloatWidening, Scalar) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    builder.add_container("x", types::Scalar(PrimitiveType::BFloat));

    EXPECT_TRUE(widen(builder));
    EXPECT_EQ(builder.subject().type("x").primitive_type(), PrimitiveType::Float);
}

TEST(FloatWidening, Pointer) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    builder.add_container("x", types::Pointer(types::Scalar(PrimitiveType::Half)));

    EXPECT_TRUE(widen(builder));
    auto& pointer = static_cast<const types::Pointer&>(builder.subject().type("x"));
    EXPECT_EQ(pointer.pointee_type().primitive_type(), PrimitiveType::Float);
}

TEST(FloatWidening, Array) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    builder.add_container("x", types::Array(types::Scalar(PrimitiveType::BFloat), symbolic::integer(4)));

    EXPECT_TRUE(widen(builder));
    auto& array = static_cast<const types::Array&>(builder.subject().type("x"));
    EXPECT_EQ(array.element_type().primitive_type(), PrimitiveType::Float);
    EXPECT_TRUE(symbolic::eq(array.num_elements(), symbolic::integer(4)));
}

TEST(FloatWidening, Tensor) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& block = builder.add_block(builder.subject().root());
    std::vector<symbolic::Expression> shape{symbolic::integer(2), symbolic::integer(3)};
    auto& cast = builder.add_library_node<math::tensor::CastNode>(block, DebugInfo(), shape, PrimitiveType::BFloat);
    types::Tensor tensor(types::Scalar(PrimitiveType::BFloat), math::tensor::TensorLayout(shape));
    connect(builder, block, cast, tensor, PrimitiveType::BFloat);

    EXPECT_TRUE(widen(builder));
    for (auto& memlet : block.dataflow().edges()) {
        auto& widened = static_cast<const types::Tensor&>(memlet.base_type());
        EXPECT_EQ(widened.element_type().primitive_type(), PrimitiveType::Float);
        EXPECT_EQ(widened.shape().size(), 2);
    }
}

TEST(FloatWidening, Unlisted) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    builder.add_container("f", types::Scalar(PrimitiveType::Double));
    builder.add_container("i", types::Pointer(types::Scalar(PrimitiveType::Int8)));
    builder.add_container("e", types::Scalar(PrimitiveType::Float8E4M3));

    EXPECT_FALSE(widen(builder));
    EXPECT_EQ(builder.subject().type("f").primitive_type(), PrimitiveType::Double);
    EXPECT_EQ(builder.subject().type("i").primitive_type(), PrimitiveType::Int8);
    EXPECT_EQ(builder.subject().type("e").primitive_type(), PrimitiveType::Float8E4M3);
}

TEST(FloatWidening, EmptySet) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    builder.add_container("x", types::Scalar(PrimitiveType::Half), true);

    EXPECT_FALSE(widen(builder, {}));
    EXPECT_EQ(builder.subject().type("x").primitive_type(), PrimitiveType::Half);
    EXPECT_FALSE(builder.subject().metadata().contains(passes::FloatWidening::ARGUMENT_TYPES));
}

TEST(FloatWidening, Memlet) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    types::Scalar bfloat(PrimitiveType::BFloat);
    builder.add_container("x", bfloat);
    builder.add_container("y", bfloat);
    auto& block = add_copy(builder, builder.subject().root(), bfloat);

    EXPECT_TRUE(widen(builder));
    for (auto& memlet : block.dataflow().edges()) {
        EXPECT_EQ(memlet.base_type().primitive_type(), PrimitiveType::Float);
    }
}

TEST(FloatWidening, Constant) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    types::Scalar half(PrimitiveType::Half);
    builder.add_container("y", half);
    auto& block = builder.add_block(builder.subject().root());
    auto& one = builder.add_constant(block, "1.5", half);
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, one, tasklet, "_in", {}, half);
    builder.add_computational_memlet(block, tasklet, "_out", builder.add_access(block, "y"), {}, half);

    EXPECT_TRUE(widen(builder));
    size_t constants = 0;
    for (auto& node : block.dataflow().nodes()) {
        if (auto* constant = dynamic_cast<data_flow::ConstantNode*>(&node)) {
            ++constants;
            EXPECT_EQ(constant->data(), "1.5");
            EXPECT_EQ(constant->type().primitive_type(), PrimitiveType::Float);
            EXPECT_EQ(block.dataflow().out_degree(*constant), 1);
        }
    }
    EXPECT_EQ(constants, 1);
    EXPECT_EQ(block.dataflow().in_degree(tasklet), 1);
}

TEST(FloatWidening, While) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    types::Scalar bfloat(PrimitiveType::BFloat);
    builder.add_container("x", bfloat);
    builder.add_container("y", bfloat);
    auto& loop = builder.add_while(builder.subject().root());
    auto& block = add_copy(builder, loop.root(), bfloat);

    EXPECT_TRUE(widen(builder));
    for (auto& memlet : block.dataflow().edges()) {
        EXPECT_EQ(memlet.base_type().primitive_type(), PrimitiveType::Float);
    }
}

TEST(FloatWidening, CastNode) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& block = builder.add_block(builder.subject().root());
    std::vector<symbolic::Expression> shape{symbolic::integer(4)};
    auto& cast =
        static_cast<math::tensor::CastNode&>(builder.add_library_node<
                                             math::tensor::CastNode>(block, DebugInfo(), shape, PrimitiveType::BFloat));
    types::Tensor tensor(types::Scalar(PrimitiveType::BFloat), math::tensor::TensorLayout(shape));
    connect(builder, block, cast, tensor, PrimitiveType::BFloat);

    EXPECT_TRUE(widen(builder));
    EXPECT_EQ(cast.target_type(), PrimitiveType::Float);
}

TEST(FloatWidening, MatMulNode) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& block = builder.add_block(builder.subject().root());
    math::tensor::TensorLayout layout({symbolic::integer(2), symbolic::integer(2)});
    auto& matmul = static_cast<math::tensor::MatMulNode&>(builder.add_library_node<math::tensor::MatMulNode>(
        block,
        DebugInfo(),
        layout,
        layout,
        math::tensor::ContractionTypes{PrimitiveType::Half, PrimitiveType::Double, PrimitiveType::Half}
    ));
    types::Pointer pointer(types::Scalar(PrimitiveType::Half));
    connect(builder, block, matmul, pointer, PrimitiveType::Half);

    EXPECT_TRUE(widen(builder));
    EXPECT_EQ(matmul.types().input, PrimitiveType::Float);
    EXPECT_EQ(matmul.types().accumulator, PrimitiveType::Double);
    EXPECT_EQ(matmul.types().output, PrimitiveType::Float);
}

TEST(FloatWidening, CMathNode) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    types::Scalar half(PrimitiveType::Half);
    builder.add_container("x", half);
    builder.add_container("y", half);
    auto& block = builder.add_block(builder.subject().root());
    auto& exp = static_cast<math::cmath::CMathNode&>(builder.add_library_node<math::cmath::CMathNode>(
        block, DebugInfo(), math::cmath::CMathFunction::exp, PrimitiveType::Half
    ));
    builder.add_computational_memlet(block, builder.add_access(block, "x"), exp, "_in1", {}, half);
    builder.add_computational_memlet(block, exp, "_out", builder.add_access(block, "y"), {}, half);

    EXPECT_TRUE(widen(builder));
    EXPECT_EQ(exp.primitive_type(), PrimitiveType::Float);
}

TEST(FloatWidening, ArgumentTypes) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    builder.add_container("a", types::Pointer(types::Scalar(PrimitiveType::BFloat)), true);
    builder.add_container("b", types::Scalar(PrimitiveType::Half), true);
    builder.add_container("c", types::Pointer(types::Scalar(PrimitiveType::Float)), true);
    builder.add_container("t", types::Scalar(PrimitiveType::BFloat));

    EXPECT_TRUE(widen(builder));
    auto recorded = nlohmann::json::parse(builder.subject().metadata(passes::FloatWidening::ARGUMENT_TYPES));
    EXPECT_EQ(recorded, (nlohmann::json{{"a", "BFloat"}, {"b", "Half"}}));
}

TEST(FloatWidening, Idempotent) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    builder.add_container("a", types::Scalar(PrimitiveType::BFloat), true);

    EXPECT_TRUE(widen(builder));
    EXPECT_FALSE(widen(builder));
    auto recorded = nlohmann::json::parse(builder.subject().metadata(passes::FloatWidening::ARGUMENT_TYPES));
    EXPECT_EQ(recorded, (nlohmann::json{{"a", "BFloat"}}));
}
