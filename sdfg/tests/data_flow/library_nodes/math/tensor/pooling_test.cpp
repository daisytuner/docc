#include <gtest/gtest.h>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_nodes/math/cmath/cmath_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/pooling_node.h"
#include "sdfg/passes/expansion/library_node_expansion_pass.h"
#include "sdfg_debug_dump.h"

using namespace sdfg;

// ── Helpers ───────────────────────────────────────────────────────────────────

static math::tensor::PoolingNode& make_pooling_node(
    builder::StructuredSDFGBuilder& builder,
    structured_control_flow::Block& block,
    math::tensor::PoolingMode mode,
    const std::vector<symbolic::Expression>& shape,
    const std::vector<symbolic::Expression>& kernel_shape,
    const std::vector<symbolic::Expression>& strides = {},
    const std::vector<symbolic::Expression>& pads = {},
    const std::vector<symbolic::Expression>& dilations = {}
) {
    types::Scalar scalar(types::PrimitiveType::Float);
    types::Pointer ptr(scalar);

    // Input tensor type
    types::Tensor x_tensor(scalar.primitive_type(), shape);

    // Output shape: [N, C, ...output_spatial...]
    size_t spatial_dims = kernel_shape.size();
    std::vector<symbolic::Expression> out_shape;
    out_shape.push_back(shape[0]);
    out_shape.push_back(shape[1]);
    for (size_t i = 0; i < spatial_dims; ++i) {
        out_shape.push_back(symbolic::integer(1)); // placeholder
    }
    types::Tensor y_tensor(scalar.primitive_type(), out_shape);

    auto& x_node = builder.add_access(block, "x");
    auto& y_node = builder.add_access(block, "y");

    auto& pool = static_cast<math::tensor::PoolingNode&>(builder.add_library_node<math::tensor::PoolingNode>(
        block, DebugInfo(), mode, shape, kernel_shape, strides, pads, dilations
    ));

    builder.add_computational_memlet(block, x_node, pool, "X", {}, x_tensor, block.debug_info());
    builder.add_computational_memlet(block, y_node, pool, "Y", {}, y_tensor, block.debug_info());

    return pool;
}

// ── Basic property tests ──────────────────────────────────────────────────────

TEST(PoolingNodeTest, MaxPool_BasicProperties) {
    builder::StructuredSDFGBuilder builder("sdfg_pool_basic", FunctionType_CPU);
    auto& sdfg = builder.subject();

    types::Pointer ptr((types::Scalar(types::PrimitiveType::Float)));
    builder.add_container("x", ptr);
    builder.add_container("y", ptr);

    auto& block = builder.add_block(sdfg.root());

    std::vector<symbolic::Expression> shape = {
        symbolic::integer(1), symbolic::integer(4), symbolic::integer(16), symbolic::integer(16)
    };
    std::vector<symbolic::Expression> kernel = {symbolic::integer(2), symbolic::integer(2)};
    std::vector<symbolic::Expression> strides = {symbolic::integer(2), symbolic::integer(2)};

    auto& pool = make_pooling_node(builder, block, math::tensor::PoolingMode::Max, shape, kernel, strides);

    EXPECT_EQ(pool.mode(), math::tensor::PoolingMode::Max);
    EXPECT_EQ(pool.kernel_shape().size(), 2u);
    EXPECT_TRUE(symbolic::eq(pool.kernel_shape()[0], symbolic::integer(2)));
    EXPECT_TRUE(symbolic::eq(pool.kernel_shape()[1], symbolic::integer(2)));
    EXPECT_EQ(pool.strides().size(), 2u);
    EXPECT_TRUE(symbolic::eq(pool.strides()[0], symbolic::integer(2)));
    EXPECT_EQ(pool.inputs().size(), 2u);
    EXPECT_EQ(pool.input(0), "Y");
    EXPECT_EQ(pool.input(1), "X");
    EXPECT_EQ(pool.outputs().size(), 0u);

    EXPECT_NO_THROW(sdfg.validate());
}

TEST(PoolingNodeTest, SumPool_BasicProperties) {
    builder::StructuredSDFGBuilder builder("sdfg_pool_sum_basic", FunctionType_CPU);
    auto& sdfg = builder.subject();

    types::Pointer ptr((types::Scalar(types::PrimitiveType::Float)));
    builder.add_container("x", ptr);
    builder.add_container("y", ptr);

    auto& block = builder.add_block(sdfg.root());

    std::vector<symbolic::Expression> shape = {
        symbolic::integer(1), symbolic::integer(8), symbolic::integer(10), symbolic::integer(10)
    };
    std::vector<symbolic::Expression> kernel = {symbolic::integer(3), symbolic::integer(3)};

    auto& pool = make_pooling_node(builder, block, math::tensor::PoolingMode::Sum, shape, kernel);

    EXPECT_EQ(pool.mode(), math::tensor::PoolingMode::Sum);
    EXPECT_EQ(pool.pads().size(), 0u);
    EXPECT_EQ(pool.dilations().size(), 0u);

    EXPECT_NO_THROW(sdfg.validate());
}

TEST(PoolingNodeTest, AvgPool_BasicProperties) {
    builder::StructuredSDFGBuilder builder("sdfg_pool_avg_basic", FunctionType_CPU);
    auto& sdfg = builder.subject();

    types::Pointer ptr((types::Scalar(types::PrimitiveType::Float)));
    builder.add_container("x", ptr);
    builder.add_container("y", ptr);

    auto& block = builder.add_block(sdfg.root());

    std::vector<symbolic::Expression> shape = {
        symbolic::integer(2), symbolic::integer(3), symbolic::integer(8), symbolic::integer(8)
    };
    std::vector<symbolic::Expression> kernel = {symbolic::integer(2), symbolic::integer(2)};
    std::vector<symbolic::Expression> strides = {symbolic::integer(2), symbolic::integer(2)};
    // Pads: [top, left, bottom, right] = [1, 1, 1, 1]
    std::vector<symbolic::Expression> pads = {
        symbolic::integer(1), symbolic::integer(1), symbolic::integer(1), symbolic::integer(1)
    };

    auto& pool = make_pooling_node(builder, block, math::tensor::PoolingMode::Avg, shape, kernel, strides, pads);

    EXPECT_EQ(pool.mode(), math::tensor::PoolingMode::Avg);
    EXPECT_EQ(pool.pads().size(), 4u);
    EXPECT_TRUE(symbolic::eq(pool.pads()[0], symbolic::integer(1)));

    EXPECT_NO_THROW(sdfg.validate());
}

TEST(PoolingNodeTest, ModeToString_RoundTrip) {
    EXPECT_EQ(math::tensor::PoolingNode::mode_to_string(math::tensor::PoolingMode::Max), "max");
    EXPECT_EQ(math::tensor::PoolingNode::mode_to_string(math::tensor::PoolingMode::Sum), "sum");
    EXPECT_EQ(math::tensor::PoolingNode::mode_to_string(math::tensor::PoolingMode::Avg), "avg");

    EXPECT_EQ(math::tensor::PoolingNode::string_to_mode("max"), math::tensor::PoolingMode::Max);
    EXPECT_EQ(math::tensor::PoolingNode::string_to_mode("sum"), math::tensor::PoolingMode::Sum);
    EXPECT_EQ(math::tensor::PoolingNode::string_to_mode("avg"), math::tensor::PoolingMode::Avg);
}

// ── symbols() tests ───────────────────────────────────────────────────────────

TEST(PoolingNodeTest, Symbols_StaticDimensions_Empty) {
    builder::StructuredSDFGBuilder builder("sdfg_pool_sym_static", FunctionType_CPU);
    auto& sdfg = builder.subject();

    types::Pointer ptr((types::Scalar(types::PrimitiveType::Float)));
    builder.add_container("x", ptr);
    builder.add_container("y", ptr);

    auto& block = builder.add_block(sdfg.root());

    std::vector<symbolic::Expression> shape = {
        symbolic::integer(1), symbolic::integer(4), symbolic::integer(8), symbolic::integer(8)
    };
    std::vector<symbolic::Expression> kernel = {symbolic::integer(2), symbolic::integer(2)};

    auto& pool = make_pooling_node(builder, block, math::tensor::PoolingMode::Max, shape, kernel);

    auto syms = pool.symbols();
    EXPECT_TRUE(syms.empty());
}

TEST(PoolingNodeTest, Symbols_SymbolicDimensions) {
    builder::StructuredSDFGBuilder builder("sdfg_pool_sym", FunctionType_CPU);
    auto& sdfg = builder.subject();

    types::Pointer ptr((types::Scalar(types::PrimitiveType::Float)));
    builder.add_container("x", ptr);
    builder.add_container("y", ptr);
    builder.add_container("N", types::Scalar(types::PrimitiveType::UInt64));
    builder.add_container("H", types::Scalar(types::PrimitiveType::UInt64));
    builder.add_container("W", types::Scalar(types::PrimitiveType::UInt64));

    auto& block = builder.add_block(sdfg.root());

    auto N = symbolic::symbol("N");
    auto H = symbolic::symbol("H");
    auto W = symbolic::symbol("W");

    std::vector<symbolic::Expression> shape = {N, symbolic::integer(4), H, W};
    std::vector<symbolic::Expression> kernel = {symbolic::integer(3), symbolic::integer(3)};

    auto& pool = make_pooling_node(builder, block, math::tensor::PoolingMode::Max, shape, kernel);

    auto syms = pool.symbols();
    EXPECT_TRUE(syms.find(N) != syms.end());
    EXPECT_TRUE(syms.find(H) != syms.end());
    EXPECT_TRUE(syms.find(W) != syms.end());
}

// ── Expansion tests ───────────────────────────────────────────────────────────

TEST(PoolingNodeTest, MaxPool2D_Expand_ProducesNestedMaps) {
    // MaxPool2d 4x4 input, 2x2 kernel, stride 2 → 2x2 output
    builder::StructuredSDFGBuilder builder("sdfg_pool_max_expand", FunctionType_CPU);
    auto& sdfg = builder.subject();

    types::Pointer ptr((types::Scalar(types::PrimitiveType::Float)));
    builder.add_container("x", ptr);
    builder.add_container("y", ptr);

    auto& block = builder.add_block(sdfg.root());

    std::vector<symbolic::Expression> shape = {
        symbolic::integer(1), symbolic::integer(2), symbolic::integer(4), symbolic::integer(4)
    };
    std::vector<symbolic::Expression> kernel = {symbolic::integer(2), symbolic::integer(2)};
    std::vector<symbolic::Expression> strides = {symbolic::integer(2), symbolic::integer(2)};

    auto& pool = make_pooling_node(builder, block, math::tensor::PoolingMode::Max, shape, kernel, strides);

    sdfg.validate();

    auto outcome = passes::expansion::expand_single_math_node(builder, block, pool);
    EXPECT_TRUE(outcome.expanded);
    EXPECT_TRUE(outcome.block_removed);

    // After expansion the original block is removed; a new sequence was inserted
    // The root should contain the new sequence
    EXPECT_GE(sdfg.root().size(), 1u);
    auto& outer = dyn_cast<structured_control_flow::Sequence&>(sdfg.root().at(0).first);

    // Outer sequence should contain a nested map structure (N → C → od0 → od1)
    EXPECT_GE(outer.size(), 1u);
    bool found_map = false;
    for (size_t i = 0; i < outer.size(); ++i) {
        if (dyn_cast<structured_control_flow::Map*>(&outer.at(i).first)) {
            found_map = true;
            break;
        }
    }
    EXPECT_TRUE(found_map);
}

TEST(PoolingNodeTest, SumPool2D_Expand_Succeeds) {
    builder::StructuredSDFGBuilder builder("sdfg_pool_sum_expand", FunctionType_CPU);
    auto& sdfg = builder.subject();

    types::Pointer ptr((types::Scalar(types::PrimitiveType::Float)));
    builder.add_container("x", ptr);
    builder.add_container("y", ptr);

    auto& block = builder.add_block(sdfg.root());

    std::vector<symbolic::Expression> shape = {
        symbolic::integer(1), symbolic::integer(4), symbolic::integer(8), symbolic::integer(8)
    };
    std::vector<symbolic::Expression> kernel = {symbolic::integer(2), symbolic::integer(2)};
    std::vector<symbolic::Expression> strides = {symbolic::integer(2), symbolic::integer(2)};

    auto& pool = make_pooling_node(builder, block, math::tensor::PoolingMode::Sum, shape, kernel, strides);

    sdfg.validate();

    auto outcome = passes::expansion::expand_single_math_node(builder, block, pool);
    EXPECT_TRUE(outcome.expanded);
    EXPECT_TRUE(outcome.block_removed);

    EXPECT_GE(sdfg.root().size(), 1u);
}

TEST(PoolingNodeTest, AvgPool2D_Expand_Succeeds) {
    builder::StructuredSDFGBuilder builder("sdfg_pool_avg_expand", FunctionType_CPU);
    auto& sdfg = builder.subject();

    types::Pointer ptr((types::Scalar(types::PrimitiveType::Float)));
    builder.add_container("x", ptr);
    builder.add_container("y", ptr);

    auto& block = builder.add_block(sdfg.root());

    std::vector<symbolic::Expression> shape = {
        symbolic::integer(1), symbolic::integer(4), symbolic::integer(8), symbolic::integer(8)
    };
    std::vector<symbolic::Expression> kernel = {symbolic::integer(3), symbolic::integer(3)};
    std::vector<symbolic::Expression> strides = {symbolic::integer(1), symbolic::integer(1)};

    auto& pool = make_pooling_node(builder, block, math::tensor::PoolingMode::Avg, shape, kernel, strides);

    sdfg.validate();

    auto outcome = passes::expansion::expand_single_math_node(builder, block, pool);
    EXPECT_TRUE(outcome.expanded);
    EXPECT_TRUE(outcome.block_removed);

    EXPECT_GE(sdfg.root().size(), 1u);
}

TEST(PoolingNodeTest, MaxPool_IntegerType_Expand_Succeeds) {
    // Integer types are supported for pooling (unlike matmul/GEMM)
    builder::StructuredSDFGBuilder builder("sdfg_pool_int", FunctionType_CPU);
    auto& sdfg = builder.subject();

    types::Scalar scalar(types::PrimitiveType::Int32);
    types::Pointer ptr(scalar);
    builder.add_container("x", ptr);
    builder.add_container("y", ptr);

    auto& block = builder.add_block(sdfg.root());

    std::vector<symbolic::Expression> shape = {
        symbolic::integer(1), symbolic::integer(2), symbolic::integer(4), symbolic::integer(4)
    };
    std::vector<symbolic::Expression> kernel = {symbolic::integer(2), symbolic::integer(2)};
    std::vector<symbolic::Expression> strides = {symbolic::integer(2), symbolic::integer(2)};

    types::Tensor x_tensor(scalar.primitive_type(), shape);
    std::vector<symbolic::Expression> out_shape = {
        symbolic::integer(1), symbolic::integer(2), symbolic::integer(1), symbolic::integer(1)
    };
    types::Tensor y_tensor(scalar.primitive_type(), out_shape);

    auto& x_node = builder.add_access(block, "x");
    auto& y_node = builder.add_access(block, "y");
    auto& pool = static_cast<math::tensor::PoolingNode&>(builder.add_library_node<math::tensor::PoolingNode>(
        block,
        DebugInfo(),
        math::tensor::PoolingMode::Max,
        shape,
        kernel,
        strides,
        std::vector<symbolic::Expression>{}, // pads
        std::vector<symbolic::Expression>{} // dilations
    ));
    builder.add_computational_memlet(block, x_node, pool, "X", {}, x_tensor, block.debug_info());
    builder.add_computational_memlet(block, y_node, pool, "Y", {}, y_tensor, block.debug_info());

    sdfg.validate();

    auto outcome = passes::expansion::expand_single_math_node(builder, block, pool);
    EXPECT_TRUE(outcome.expanded);
    EXPECT_TRUE(outcome.block_removed);
}

TEST(PoolingNodeTest, MaxPool2D_WithPadding_Expand_Succeeds) {
    // MaxPool with padding
    builder::StructuredSDFGBuilder builder("sdfg_pool_padded", FunctionType_CPU);
    auto& sdfg = builder.subject();

    types::Pointer ptr((types::Scalar(types::PrimitiveType::Float)));
    builder.add_container("x", ptr);
    builder.add_container("y", ptr);

    auto& block = builder.add_block(sdfg.root());

    std::vector<symbolic::Expression> shape = {
        symbolic::integer(1), symbolic::integer(2), symbolic::integer(6), symbolic::integer(6)
    };
    std::vector<symbolic::Expression> kernel = {symbolic::integer(3), symbolic::integer(3)};
    std::vector<symbolic::Expression> strides = {symbolic::integer(1), symbolic::integer(1)};
    // Symmetric padding of 1 on each side → [top, left, bottom, right]
    std::vector<symbolic::Expression> pads = {
        symbolic::integer(1), symbolic::integer(1), symbolic::integer(1), symbolic::integer(1)
    };

    auto& pool = make_pooling_node(builder, block, math::tensor::PoolingMode::Max, shape, kernel, strides, pads);

    EXPECT_TRUE(symbolic::eq(pool.pads()[0], symbolic::integer(1)));
    EXPECT_TRUE(symbolic::eq(pool.pads()[3], symbolic::integer(1)));

    sdfg.validate();

    auto outcome = passes::expansion::expand_single_math_node(builder, block, pool);
    EXPECT_TRUE(outcome.expanded);
    EXPECT_TRUE(outcome.block_removed);
}

TEST(PoolingNodeTest, MaxPool2D_WithDilations_Expand_Succeeds) {
    builder::StructuredSDFGBuilder builder("sdfg_pool_dilated", FunctionType_CPU);
    auto& sdfg = builder.subject();

    types::Pointer ptr((types::Scalar(types::PrimitiveType::Float)));
    builder.add_container("x", ptr);
    builder.add_container("y", ptr);

    auto& block = builder.add_block(sdfg.root());

    std::vector<symbolic::Expression> shape = {
        symbolic::integer(1), symbolic::integer(2), symbolic::integer(8), symbolic::integer(8)
    };
    std::vector<symbolic::Expression> kernel = {symbolic::integer(2), symbolic::integer(2)};
    std::vector<symbolic::Expression> strides = {symbolic::integer(1), symbolic::integer(1)};
    std::vector<symbolic::Expression> pads = {};
    std::vector<symbolic::Expression> dilations = {symbolic::integer(2), symbolic::integer(2)};

    auto& pool =
        make_pooling_node(builder, block, math::tensor::PoolingMode::Max, shape, kernel, strides, pads, dilations);

    EXPECT_TRUE(symbolic::eq(pool.dilations()[0], symbolic::integer(2)));
    EXPECT_TRUE(symbolic::eq(pool.dilations()[1], symbolic::integer(2)));

    sdfg.validate();

    auto outcome = passes::expansion::expand_single_math_node(builder, block, pool);
    EXPECT_TRUE(outcome.expanded);
    EXPECT_TRUE(outcome.block_removed);
}

TEST(PoolingNodeTest, MaxPool2D_BatchDim_Expand_ContainsOuterMap) {
    // Batch size > 1 — the outer map should loop over N
    builder::StructuredSDFGBuilder builder("sdfg_pool_batch", FunctionType_CPU);
    auto& sdfg = builder.subject();

    types::Pointer ptr((types::Scalar(types::PrimitiveType::Float)));
    builder.add_container("x", ptr);
    builder.add_container("y", ptr);

    auto& block = builder.add_block(sdfg.root());

    std::vector<symbolic::Expression> shape = {
        symbolic::integer(4), // N=4 (a batch)
        symbolic::integer(2),
        symbolic::integer(8),
        symbolic::integer(8)
    };
    std::vector<symbolic::Expression> kernel = {symbolic::integer(2), symbolic::integer(2)};
    std::vector<symbolic::Expression> strides = {symbolic::integer(2), symbolic::integer(2)};

    auto& pool = make_pooling_node(builder, block, math::tensor::PoolingMode::Max, shape, kernel, strides);

    sdfg.validate();

    auto outcome = passes::expansion::expand_single_math_node(builder, block, pool);
    EXPECT_TRUE(outcome.expanded);
    EXPECT_TRUE(outcome.block_removed);

    // After expansion:  root → new_sequence → map(n) → ...
    EXPECT_GE(sdfg.root().size(), 1u);
    auto& outer = dyn_cast<structured_control_flow::Sequence&>(sdfg.root().at(0).first);

    bool found_map = false;
    for (size_t i = 0; i < outer.size(); ++i) {
        if (dyn_cast<structured_control_flow::Map*>(&outer.at(i).first)) {
            found_map = true;
            break;
        }
    }
    EXPECT_TRUE(found_map);
}

TEST(PoolingNodeTest, MaxPool2D_SymbolicDims_Expand_Succeeds) {
    // Symbolic N/H/W dimensions
    builder::StructuredSDFGBuilder builder("sdfg_pool_symbolic", FunctionType_CPU);
    auto& sdfg = builder.subject();

    types::Pointer ptr((types::Scalar(types::PrimitiveType::Float)));
    builder.add_container("x", ptr);
    builder.add_container("y", ptr);
    builder.add_container("N", types::Scalar(types::PrimitiveType::UInt64));
    builder.add_container("H", types::Scalar(types::PrimitiveType::UInt64));
    builder.add_container("W", types::Scalar(types::PrimitiveType::UInt64));

    auto& block = builder.add_block(sdfg.root());

    auto N = symbolic::symbol("N");
    auto H = symbolic::symbol("H");
    auto W = symbolic::symbol("W");

    std::vector<symbolic::Expression> shape = {N, symbolic::integer(4), H, W};
    std::vector<symbolic::Expression> kernel = {symbolic::integer(2), symbolic::integer(2)};
    std::vector<symbolic::Expression> strides = {symbolic::integer(2), symbolic::integer(2)};

    auto& pool = make_pooling_node(builder, block, math::tensor::PoolingMode::Max, shape, kernel, strides);

    // symbols() should contain all symbolic variables
    auto syms = pool.symbols();
    EXPECT_TRUE(syms.find(N) != syms.end());
    EXPECT_TRUE(syms.find(H) != syms.end());
    EXPECT_TRUE(syms.find(W) != syms.end());

    sdfg.validate();

    auto outcome = passes::expansion::expand_single_math_node(builder, block, pool);
    EXPECT_TRUE(outcome.expanded);
    EXPECT_TRUE(outcome.block_removed);
}

TEST(PoolingNodeTest, MaxPool1D_Expand_Succeeds) {
    // 1-D pooling: shape [N, C, L], kernel [k]
    builder::StructuredSDFGBuilder builder("sdfg_pool_1d", FunctionType_CPU);
    auto& sdfg = builder.subject();

    types::Pointer ptr((types::Scalar(types::PrimitiveType::Float)));
    builder.add_container("x", ptr);
    builder.add_container("y", ptr);

    auto& block = builder.add_block(sdfg.root());

    std::vector<symbolic::Expression> shape = {symbolic::integer(1), symbolic::integer(4), symbolic::integer(16)};
    std::vector<symbolic::Expression> kernel = {symbolic::integer(3)};
    std::vector<symbolic::Expression> strides = {symbolic::integer(1)};

    types::Tensor x_tensor(types::PrimitiveType::Float, shape);
    std::vector<symbolic::Expression> out_shape = {symbolic::integer(1), symbolic::integer(4), symbolic::integer(1)};
    types::Tensor y_tensor(types::PrimitiveType::Float, out_shape);

    auto& x_node = builder.add_access(block, "x");
    auto& y_node = builder.add_access(block, "y");
    auto& pool = static_cast<math::tensor::PoolingNode&>(builder.add_library_node<math::tensor::PoolingNode>(
        block,
        DebugInfo(),
        math::tensor::PoolingMode::Max,
        shape,
        kernel,
        strides,
        std::vector<symbolic::Expression>{},
        std::vector<symbolic::Expression>{}
    ));
    builder.add_computational_memlet(block, x_node, pool, "X", {}, x_tensor, block.debug_info());
    builder.add_computational_memlet(block, y_node, pool, "Y", {}, y_tensor, block.debug_info());

    dump_sdfg(builder.subject(), "0.init");

    sdfg.validate();

    auto outcome = passes::expansion::expand_single_math_node(builder, block, pool);
    EXPECT_TRUE(outcome.expanded);
    EXPECT_TRUE(outcome.block_removed);

    dump_sdfg(builder.subject(), "1.expanded");

    EXPECT_GE(sdfg.root().size(), 1u);
}

// ── replace() test ────────────────────────────────────────────────────────────

TEST(PoolingNodeTest, Replace_SymbolicDimension) {
    builder::StructuredSDFGBuilder builder("sdfg_pool_replace", FunctionType_CPU);
    auto& sdfg = builder.subject();

    types::Pointer ptr((types::Scalar(types::PrimitiveType::Float)));
    builder.add_container("x", ptr);
    builder.add_container("y", ptr);
    builder.add_container("N", types::Scalar(types::PrimitiveType::UInt64));

    auto& block = builder.add_block(sdfg.root());

    auto N = symbolic::symbol("N");
    std::vector<symbolic::Expression> shape = {N, symbolic::integer(4), symbolic::integer(8), symbolic::integer(8)};
    std::vector<symbolic::Expression> kernel = {symbolic::integer(2), symbolic::integer(2)};

    auto& pool = make_pooling_node(builder, block, math::tensor::PoolingMode::Max, shape, kernel);

    EXPECT_TRUE(symbolic::eq(pool.shape()[0], N));

    pool.replace(N, symbolic::integer(2));
    EXPECT_TRUE(symbolic::eq(pool.shape()[0], symbolic::integer(2)));
}
