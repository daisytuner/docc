#include "sdfg/data_flow/library_nodes/math/tensor/upsample_node.h"

#include <memory>
#include <vector>

#include <gtest/gtest.h>
#include <nlohmann/json_fwd.hpp>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/passes/expansion/library_node_expansion_pass.h"
#include "sdfg/serializer/json_serializer.h"
#include "sdfg/structured_sdfg.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/scalar.h"
#include "sdfg/types/tensor.h"
#include "sdfg_debug_dump.h"

using namespace sdfg;

// ── Helpers ───────────────────────────────────────────────────────────────────

static math::tensor::UpsampleBilinear2DNode& make_upsample_node(
    builder::StructuredSDFGBuilder& builder,
    structured_control_flow::Block& block,
    const std::vector<symbolic::Expression>& input_shape,
    const std::vector<symbolic::Expression>& output_shape,
    bool align_corners,
    const std::vector<double>& scale_factors,
    types::PrimitiveType prim = types::PrimitiveType::Float
) {
    types::Tensor x_tensor(prim, input_shape);
    types::Tensor y_tensor(prim, output_shape);

    auto& x_node = builder.add_access(block, "x");
    auto& y_node = builder.add_access(block, "y");

    auto& node = static_cast<
        math::tensor::UpsampleBilinear2DNode&>(builder.add_library_node<math::tensor::UpsampleBilinear2DNode>(
        block, DebugInfo(), input_shape, output_shape, align_corners, scale_factors
    ));

    builder.add_computational_memlet(block, y_node, node, "Y", {}, y_tensor, block.debug_info());
    builder.add_computational_memlet(block, x_node, node, "X", {}, x_tensor, block.debug_info());

    return node;
}

static std::vector<symbolic::Expression> shape4(long n, long c, long h, long w) {
    return {symbolic::integer(n), symbolic::integer(c), symbolic::integer(h), symbolic::integer(w)};
}

// ── Basic property tests ──────────────────────────────────────────────────────

TEST(UpsampleBilinear2DNodeTest, SizeBased_BasicProperties) {
    builder::StructuredSDFGBuilder builder("sdfg_ups_size_basic", FunctionType_CPU);
    auto& sdfg = builder.subject();

    types::Pointer ptr((types::Scalar(types::PrimitiveType::Float)));
    builder.add_container("x", ptr);
    builder.add_container("y", ptr);

    auto& block = builder.add_block(sdfg.root());

    auto in_shape = shape4(1, 2, 4, 3);
    auto out_shape = shape4(1, 2, 7, 5);

    auto& node = make_upsample_node(builder, block, in_shape, out_shape, false, {});

    EXPECT_EQ(node.input_shape().size(), 4u);
    EXPECT_EQ(node.output_shape().size(), 4u);
    EXPECT_TRUE(symbolic::eq(node.input_shape()[2], symbolic::integer(4)));
    EXPECT_TRUE(symbolic::eq(node.output_shape()[2], symbolic::integer(7)));
    EXPECT_FALSE(node.align_corners());
    EXPECT_TRUE(node.scale_factors().empty());
    EXPECT_FALSE(node.supports_integer_types());

    EXPECT_EQ(node.inputs().size(), 2u);
    EXPECT_EQ(node.input(0), "Y");
    EXPECT_EQ(node.input(1), "X");
    EXPECT_EQ(node.outputs().size(), 0u);

    EXPECT_NO_THROW(sdfg.validate());
}

TEST(UpsampleBilinear2DNodeTest, ScaleBased_BasicProperties) {
    builder::StructuredSDFGBuilder builder("sdfg_ups_scale_basic", FunctionType_CPU);
    auto& sdfg = builder.subject();

    types::Pointer ptr((types::Scalar(types::PrimitiveType::Float)));
    builder.add_container("x", ptr);
    builder.add_container("y", ptr);

    auto& block = builder.add_block(sdfg.root());

    auto in_shape = shape4(1, 3, 4, 4);
    auto out_shape = shape4(1, 3, 8, 8);

    auto& node = make_upsample_node(builder, block, in_shape, out_shape, true, {2.0, 2.0});

    EXPECT_TRUE(node.align_corners());
    ASSERT_EQ(node.scale_factors().size(), 2u);
    EXPECT_DOUBLE_EQ(node.scale_factors()[0], 2.0);
    EXPECT_DOUBLE_EQ(node.scale_factors()[1], 2.0);

    EXPECT_NO_THROW(sdfg.validate());
}

TEST(UpsampleBilinear2DNodeTest, ToStr_ContainsAlignCorners) {
    builder::StructuredSDFGBuilder builder("sdfg_ups_tostr", FunctionType_CPU);
    auto& sdfg = builder.subject();

    types::Pointer ptr((types::Scalar(types::PrimitiveType::Float)));
    builder.add_container("x", ptr);
    builder.add_container("y", ptr);

    auto& block = builder.add_block(sdfg.root());
    auto& node = make_upsample_node(builder, block, shape4(1, 1, 2, 2), shape4(1, 1, 4, 4), true, {2.0, 2.0});

    auto str = node.toStr();
    EXPECT_NE(str.find("UpsampleBilinear2D"), std::string::npos);
    EXPECT_NE(str.find("align_corners=true"), std::string::npos);
}

// ── symbols() tests ───────────────────────────────────────────────────────────

TEST(UpsampleBilinear2DNodeTest, Symbols_StaticDimensions_Empty) {
    builder::StructuredSDFGBuilder builder("sdfg_ups_sym_static", FunctionType_CPU);
    auto& sdfg = builder.subject();

    types::Pointer ptr((types::Scalar(types::PrimitiveType::Float)));
    builder.add_container("x", ptr);
    builder.add_container("y", ptr);

    auto& block = builder.add_block(sdfg.root());
    auto& node = make_upsample_node(builder, block, shape4(1, 4, 8, 8), shape4(1, 4, 16, 16), false, {});

    EXPECT_TRUE(node.symbols().empty());
}

TEST(UpsampleBilinear2DNodeTest, Symbols_SymbolicDimensions) {
    builder::StructuredSDFGBuilder builder("sdfg_ups_sym", FunctionType_CPU);
    auto& sdfg = builder.subject();

    types::Pointer ptr((types::Scalar(types::PrimitiveType::Float)));
    builder.add_container("x", ptr);
    builder.add_container("y", ptr);
    builder.add_container("N", types::Scalar(types::PrimitiveType::UInt64));
    builder.add_container("H", types::Scalar(types::PrimitiveType::UInt64));
    builder.add_container("HO", types::Scalar(types::PrimitiveType::UInt64));

    auto& block = builder.add_block(sdfg.root());

    auto N = symbolic::symbol("N");
    auto H = symbolic::symbol("H");
    auto HO = symbolic::symbol("HO");

    std::vector<symbolic::Expression> in_shape = {N, symbolic::integer(4), H, symbolic::integer(8)};
    std::vector<symbolic::Expression> out_shape = {N, symbolic::integer(4), HO, symbolic::integer(16)};

    auto& node = make_upsample_node(builder, block, in_shape, out_shape, false, {});

    auto syms = node.symbols();
    EXPECT_TRUE(syms.find(N) != syms.end());
    EXPECT_TRUE(syms.find(H) != syms.end());
    EXPECT_TRUE(syms.find(HO) != syms.end());
}

// ── replace() test ────────────────────────────────────────────────────────────

TEST(UpsampleBilinear2DNodeTest, Replace_SymbolicDimension) {
    builder::StructuredSDFGBuilder builder("sdfg_ups_replace", FunctionType_CPU);
    auto& sdfg = builder.subject();

    types::Pointer ptr((types::Scalar(types::PrimitiveType::Float)));
    builder.add_container("x", ptr);
    builder.add_container("y", ptr);
    builder.add_container("N", types::Scalar(types::PrimitiveType::UInt64));

    auto& block = builder.add_block(sdfg.root());

    auto N = symbolic::symbol("N");
    std::vector<symbolic::Expression> in_shape = {N, symbolic::integer(4), symbolic::integer(4), symbolic::integer(4)};
    std::vector<symbolic::Expression> out_shape = {N, symbolic::integer(4), symbolic::integer(8), symbolic::integer(8)};

    auto& node = make_upsample_node(builder, block, in_shape, out_shape, false, {});

    EXPECT_TRUE(symbolic::eq(node.input_shape()[0], N));
    EXPECT_TRUE(symbolic::eq(node.output_shape()[0], N));

    node.replace(N, symbolic::integer(2));
    EXPECT_TRUE(symbolic::eq(node.input_shape()[0], symbolic::integer(2)));
    EXPECT_TRUE(symbolic::eq(node.output_shape()[0], symbolic::integer(2)));
}

// ── clone() test ──────────────────────────────────────────────────────────────

TEST(UpsampleBilinear2DNodeTest, Clone_PreservesProperties) {
    builder::StructuredSDFGBuilder builder("sdfg_ups_clone", FunctionType_CPU);
    auto& sdfg = builder.subject();

    types::Pointer ptr((types::Scalar(types::PrimitiveType::Float)));
    builder.add_container("x", ptr);
    builder.add_container("y", ptr);

    auto& block = builder.add_block(sdfg.root());
    auto& node = make_upsample_node(builder, block, shape4(1, 2, 4, 4), shape4(1, 2, 8, 8), true, {2.0, 2.0});

    auto cloned = node.clone(node.element_id(), node.vertex(), node.get_parent());
    auto* cloned_node = dynamic_cast<math::tensor::UpsampleBilinear2DNode*>(cloned.get());
    ASSERT_NE(cloned_node, nullptr);
    EXPECT_TRUE(cloned_node->align_corners());
    ASSERT_EQ(cloned_node->scale_factors().size(), 2u);
    EXPECT_DOUBLE_EQ(cloned_node->scale_factors()[0], 2.0);
    EXPECT_EQ(cloned_node->input_shape().size(), 4u);
    EXPECT_EQ(cloned_node->output_shape().size(), 4u);
}

// ── validate() error tests ────────────────────────────────────────────────────

TEST(UpsampleBilinear2DNodeTest, Validate_IntegerType_Throws) {
    builder::StructuredSDFGBuilder builder("sdfg_ups_int", FunctionType_CPU);
    auto& sdfg = builder.subject();

    types::Scalar scalar(types::PrimitiveType::Int32);
    types::Pointer ptr(scalar);
    builder.add_container("x", ptr);
    builder.add_container("y", ptr);

    auto& block = builder.add_block(sdfg.root());
    make_upsample_node(builder, block, shape4(1, 2, 4, 4), shape4(1, 2, 8, 8), false, {}, types::PrimitiveType::Int32);

    EXPECT_THROW(sdfg.validate(), InvalidSDFGException);
}

// ── Expansion tests ───────────────────────────────────────────────────────────

TEST(UpsampleBilinear2DNodeTest, SizeUpscale_Expand_ProducesNestedMaps) {
    builder::StructuredSDFGBuilder builder("sdfg_ups_up_expand", FunctionType_CPU);
    auto& sdfg = builder.subject();

    types::Pointer ptr((types::Scalar(types::PrimitiveType::Float)));
    builder.add_container("x", ptr);
    builder.add_container("y", ptr);

    auto& block = builder.add_block(sdfg.root());
    auto& node = make_upsample_node(builder, block, shape4(1, 2, 4, 4), shape4(1, 2, 8, 8), false, {});

    sdfg.validate();

    auto outcome = passes::expansion::expand_single_math_node(builder, block, node);
    EXPECT_TRUE(outcome.expanded);
    EXPECT_TRUE(outcome.block_removed);

    ASSERT_GE(sdfg.root().size(), 1u);
    auto& outer = dyn_cast<structured_control_flow::Sequence&>(sdfg.root().at(0));

    bool found_map = false;
    for (size_t i = 0; i < outer.size(); ++i) {
        if (dyn_cast<structured_control_flow::Map*>(&outer.at(i))) {
            found_map = true;
            break;
        }
    }
    EXPECT_TRUE(found_map);

    EXPECT_NO_THROW(sdfg.validate());
}

TEST(UpsampleBilinear2DNodeTest, SizeDownscale_Expand_Succeeds) {
    builder::StructuredSDFGBuilder builder("sdfg_ups_down_expand", FunctionType_CPU);
    auto& sdfg = builder.subject();

    types::Pointer ptr((types::Scalar(types::PrimitiveType::Float)));
    builder.add_container("x", ptr);
    builder.add_container("y", ptr);

    auto& block = builder.add_block(sdfg.root());
    auto& node = make_upsample_node(builder, block, shape4(1, 2, 8, 8), shape4(1, 2, 4, 4), false, {});

    sdfg.validate();

    auto outcome = passes::expansion::expand_single_math_node(builder, block, node);
    EXPECT_TRUE(outcome.expanded);
    EXPECT_TRUE(outcome.block_removed);

    EXPECT_NO_THROW(sdfg.validate());
}

TEST(UpsampleBilinear2DNodeTest, AlignCorners_Expand_Succeeds) {
    builder::StructuredSDFGBuilder builder("sdfg_ups_align_expand", FunctionType_CPU);
    auto& sdfg = builder.subject();

    types::Pointer ptr((types::Scalar(types::PrimitiveType::Float)));
    builder.add_container("x", ptr);
    builder.add_container("y", ptr);

    auto& block = builder.add_block(sdfg.root());
    auto& node = make_upsample_node(builder, block, shape4(1, 2, 4, 4), shape4(1, 2, 7, 5), true, {});

    sdfg.validate();

    auto outcome = passes::expansion::expand_single_math_node(builder, block, node);
    EXPECT_TRUE(outcome.expanded);
    EXPECT_TRUE(outcome.block_removed);

    EXPECT_NO_THROW(sdfg.validate());
}

TEST(UpsampleBilinear2DNodeTest, ScaleFactors_Expand_Succeeds) {
    builder::StructuredSDFGBuilder builder("sdfg_ups_scale_expand", FunctionType_CPU);
    auto& sdfg = builder.subject();

    types::Pointer ptr((types::Scalar(types::PrimitiveType::Float)));
    builder.add_container("x", ptr);
    builder.add_container("y", ptr);

    auto& block = builder.add_block(sdfg.root());
    // scale_factor 1.5 → 4 -> 6
    auto& node = make_upsample_node(builder, block, shape4(1, 2, 4, 4), shape4(1, 2, 6, 6), false, {1.5, 1.5});

    sdfg.validate();

    auto outcome = passes::expansion::expand_single_math_node(builder, block, node);
    EXPECT_TRUE(outcome.expanded);
    EXPECT_TRUE(outcome.block_removed);

    EXPECT_NO_THROW(sdfg.validate());
}

TEST(UpsampleBilinear2DNodeTest, AsymmetricScaleFactors_Expand_Succeeds) {
    builder::StructuredSDFGBuilder builder("sdfg_ups_asym_expand", FunctionType_CPU);
    auto& sdfg = builder.subject();

    types::Pointer ptr((types::Scalar(types::PrimitiveType::Float)));
    builder.add_container("x", ptr);
    builder.add_container("y", ptr);

    auto& block = builder.add_block(sdfg.root());
    // scale (2.0, 3.0) → 4x4 -> 8x12
    auto& node = make_upsample_node(builder, block, shape4(1, 2, 4, 4), shape4(1, 2, 8, 12), false, {2.0, 3.0});

    sdfg.validate();

    auto outcome = passes::expansion::expand_single_math_node(builder, block, node);
    EXPECT_TRUE(outcome.expanded);
    EXPECT_TRUE(outcome.block_removed);

    EXPECT_NO_THROW(sdfg.validate());
}

TEST(UpsampleBilinear2DNodeTest, BatchDim_Expand_ContainsOuterMap) {
    builder::StructuredSDFGBuilder builder("sdfg_ups_batch", FunctionType_CPU);
    auto& sdfg = builder.subject();

    types::Pointer ptr((types::Scalar(types::PrimitiveType::Float)));
    builder.add_container("x", ptr);
    builder.add_container("y", ptr);

    auto& block = builder.add_block(sdfg.root());
    auto& node = make_upsample_node(builder, block, shape4(4, 2, 4, 4), shape4(4, 2, 8, 8), false, {});

    sdfg.validate();

    auto outcome = passes::expansion::expand_single_math_node(builder, block, node);
    EXPECT_TRUE(outcome.expanded);
    EXPECT_TRUE(outcome.block_removed);

    ASSERT_GE(sdfg.root().size(), 1u);
    auto& outer = dyn_cast<structured_control_flow::Sequence&>(sdfg.root().at(0));
    bool found_map = false;
    for (size_t i = 0; i < outer.size(); ++i) {
        if (dyn_cast<structured_control_flow::Map*>(&outer.at(i))) {
            found_map = true;
            break;
        }
    }
    EXPECT_TRUE(found_map);
}

TEST(UpsampleBilinear2DNodeTest, Identity_Expand_Succeeds) {
    // output_size == input_size (identity resize)
    builder::StructuredSDFGBuilder builder("sdfg_ups_identity", FunctionType_CPU);
    auto& sdfg = builder.subject();

    types::Pointer ptr((types::Scalar(types::PrimitiveType::Float)));
    builder.add_container("x", ptr);
    builder.add_container("y", ptr);

    auto& block = builder.add_block(sdfg.root());
    auto& node = make_upsample_node(builder, block, shape4(1, 2, 5, 5), shape4(1, 2, 5, 5), false, {});

    sdfg.validate();

    auto outcome = passes::expansion::expand_single_math_node(builder, block, node);
    EXPECT_TRUE(outcome.expanded);
    EXPECT_TRUE(outcome.block_removed);

    EXPECT_NO_THROW(sdfg.validate());
}

TEST(UpsampleBilinear2DNodeTest, SymbolicDims_Expand_Succeeds) {
    builder::StructuredSDFGBuilder builder("sdfg_ups_symbolic_expand", FunctionType_CPU);
    auto& sdfg = builder.subject();

    types::Pointer ptr((types::Scalar(types::PrimitiveType::Float)));
    builder.add_container("x", ptr);
    builder.add_container("y", ptr);
    builder.add_container("N", types::Scalar(types::PrimitiveType::UInt64));

    auto& block = builder.add_block(sdfg.root());

    auto N = symbolic::symbol("N");
    std::vector<symbolic::Expression> in_shape = {N, symbolic::integer(2), symbolic::integer(4), symbolic::integer(4)};
    std::vector<symbolic::Expression> out_shape = {N, symbolic::integer(2), symbolic::integer(8), symbolic::integer(8)};

    auto& node = make_upsample_node(builder, block, in_shape, out_shape, false, {});

    sdfg.validate();

    auto outcome = passes::expansion::expand_single_math_node(builder, block, node);
    EXPECT_TRUE(outcome.expanded);
    EXPECT_TRUE(outcome.block_removed);

    EXPECT_NO_THROW(sdfg.validate());
}

// ── Serialization round-trip test ─────────────────────────────────────────────

TEST(UpsampleBilinear2DNodeTest, Serialization_RoundTrip_SizeBased) {
    builder::StructuredSDFGBuilder builder("sdfg_ups_serial_size", FunctionType_CPU);
    auto& sdfg = builder.subject();

    types::Pointer ptr((types::Scalar(types::PrimitiveType::Float)));
    builder.add_container("x", ptr, true);
    builder.add_container("y", ptr, true);

    auto& block = builder.add_block(sdfg.root());
    make_upsample_node(builder, block, shape4(1, 2, 4, 3), shape4(1, 2, 7, 5), false, {});

    ASSERT_NO_THROW(sdfg.validate());

    serializer::JSONSerializer serializer;
    nlohmann::json j;
    ASSERT_NO_THROW(j = serializer.serialize(sdfg));

    std::unique_ptr<StructuredSDFG> new_sdfg;
    ASSERT_NO_THROW(new_sdfg = serializer.deserialize(j));
    ASSERT_NE(new_sdfg, nullptr);
    EXPECT_NO_THROW(new_sdfg->validate());
}

TEST(UpsampleBilinear2DNodeTest, Serialization_RoundTrip_ScaleBased) {
    builder::StructuredSDFGBuilder builder("sdfg_ups_serial_scale", FunctionType_CPU);
    auto& sdfg = builder.subject();

    types::Pointer ptr((types::Scalar(types::PrimitiveType::Float)));
    builder.add_container("x", ptr, true);
    builder.add_container("y", ptr, true);

    auto& block = builder.add_block(sdfg.root());
    make_upsample_node(builder, block, shape4(1, 2, 4, 4), shape4(1, 2, 8, 8), true, {2.0, 2.0});

    ASSERT_NO_THROW(sdfg.validate());

    serializer::JSONSerializer serializer;
    nlohmann::json j;
    ASSERT_NO_THROW(j = serializer.serialize(sdfg));

    std::unique_ptr<StructuredSDFG> new_sdfg;
    ASSERT_NO_THROW(new_sdfg = serializer.deserialize(j));
    ASSERT_NE(new_sdfg, nullptr);

    // Locate the deserialized node and confirm its fields survived the round-trip.
    bool found = false;
    auto& new_block = dyn_cast<structured_control_flow::Block&>(new_sdfg->root().at(0));
    for (auto& node : new_block.dataflow().nodes()) {
        if (auto* ups = dynamic_cast<math::tensor::UpsampleBilinear2DNode*>(&node)) {
            found = true;
            EXPECT_TRUE(ups->align_corners());
            ASSERT_EQ(ups->scale_factors().size(), 2u);
            EXPECT_DOUBLE_EQ(ups->scale_factors()[0], 2.0);
            EXPECT_DOUBLE_EQ(ups->scale_factors()[1], 2.0);
        }
    }
    EXPECT_TRUE(found);
}
