#include "sdfg/data_flow/library_nodes/math/tensor/tensor_layout.h"

#include <optional>
#include <vector>

#include <gtest/gtest.h>

#include "sdfg/helpers/helpers.h"
#include "sdfg/symbolic/symbolic.h"

using namespace sdfg;

TEST(TensorLayoutTest, PartitionOffsetIntoDimensions) {
    // shape=[16, 16], strides=[32, 1], off = _j1_tile0 + 32*tile_k0 + 16*wave_col0
    auto j1_tile0 = symbolic::symbol("_j1_tile0");
    auto tile_k0 = symbolic::symbol("tile_k0");
    auto wave_col0 = symbolic::symbol("wave_col0");

    auto offset = symbolic::add(
        j1_tile0,
        symbolic::add(symbolic::mul(symbolic::integer(32), tile_k0), symbolic::mul(symbolic::integer(16), wave_col0))
    );

    math::tensor::TensorLayout
        layout({symbolic::integer(16), symbolic::integer(16)}, {symbolic::integer(32), symbolic::integer(1)}, offset);

    auto result = layout.partition_offset_into_dimensions();
    ASSERT_TRUE(result.has_value());
    ASSERT_EQ(result->size(), 2);

    // 32*tile_k0 -> dim 0 (stride 32): coord tile_k0
    EXPECT_TRUE(symbolic::eq(result->at(0), tile_k0));

    // _j1_tile0 (stride 1) + 16*wave_col0 (stride 1) -> dim 1: coord _j1_tile0 + 16*wave_col0
    auto expected_dim1 = symbolic::add(j1_tile0, symbolic::mul(symbolic::integer(16), wave_col0));
    EXPECT_TRUE(symbolic::eq(symbolic::expand(result->at(1)), symbolic::expand(expected_dim1)));

    // Reconstruction must match the original offset.
    auto reconstructed = symbolic::
        add(symbolic::mul(layout.strides().at(0), result->at(0)), symbolic::mul(layout.strides().at(1), result->at(1)));
    EXPECT_TRUE(symbolic::eq(symbolic::expand(reconstructed), symbolic::expand(offset)));
}

TEST(TensorLayoutTest, PartitionOffsetIntoDimensions2) {
    // shape=[16, 16], strides=[32, 1], off = 32*_i1_tile0 + tile_k0 + 512*wave_row0
    auto i1_tile0 = symbolic::symbol("_i1_tile0");
    auto tile_k0 = symbolic::symbol("tile_k0");
    auto wave_row0 = symbolic::symbol("wave_row0");

    auto offset = SymEngine::add(
        {symbolic::mul(symbolic::integer(32), i1_tile0), tile_k0, symbolic::mul(symbolic::integer(512), wave_row0)}
    );

    math::tensor::TensorLayout
        layout({symbolic::integer(16), symbolic::integer(16)}, {symbolic::integer(32), symbolic::integer(1)}, offset);

    auto result = layout.partition_offset_into_dimensions();
    ASSERT_TRUE(result.has_value());
    ASSERT_EQ(result->size(), 2);

    DEBUG_PRINTLN("DimOffset: " << result->at(0)->__str__() << " x " << result->at(1)->__str__());

    EXPECT_TRUE(symbolic::eq(result->at(1), tile_k0));

    // 32*_i1_tile0 (stride 32) + 512*wave_row0 (stride 32) -> dim 0: coord _i1_tile0 + 16*wave_row0
    auto expected_dim0 = symbolic::add(i1_tile0, symbolic::mul(symbolic::integer(16), wave_row0));
    EXPECT_TRUE(symbolic::eq(symbolic::expand(result->at(0)), symbolic::expand(expected_dim0)));

    // Reconstruction must match the original offset.
    auto reconstructed = symbolic::
        add(symbolic::mul(layout.strides().at(0), result->at(0)), symbolic::mul(layout.strides().at(1), result->at(1)));
    EXPECT_TRUE(symbolic::eq(symbolic::expand(reconstructed), symbolic::expand(offset)));
}

TEST(TensorLayoutTest, PartitionZeroOffset) {
    math::tensor::TensorLayout
        layout({symbolic::integer(16), symbolic::integer(16)}, {symbolic::integer(32), symbolic::integer(1)});

    auto result = layout.partition_offset_into_dimensions();
    ASSERT_TRUE(result.has_value());
    ASSERT_EQ(result->size(), 2);
    EXPECT_TRUE(symbolic::eq(result->at(0), symbolic::zero()));
    EXPECT_TRUE(symbolic::eq(result->at(1), symbolic::zero()));
}

TEST(TensorLayoutTest, PartitionNonImmediateStrideFails) {
    auto ld = symbolic::symbol("ld");
    math::tensor::TensorLayout
        layout({symbolic::integer(16), symbolic::integer(16)}, {ld, symbolic::integer(1)}, symbolic::symbol("x"));

    auto result = layout.partition_offset_into_dimensions();
    EXPECT_FALSE(result.has_value());
}

TEST(TensorLayoutTest, BroadcastKeepsStrideOfMatchingUnitDimension) {
    // [1, 8960] slice of a [1, 17920] row against a [1, 8960] result (e.g. a single token): dimension 0 matches, so
    // it keeps its stride instead of being folded to a broadcast stride of 0.
    math::tensor::TensorLayout
        layout({symbolic::one(), symbolic::integer(8960)}, {symbolic::integer(17920), symbolic::one()});

    auto result = layout.broadcast({symbolic::one(), symbolic::integer(8960)});
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(symbolic::eq(result->strides().at(0), symbolic::integer(17920)));
    EXPECT_TRUE(symbolic::eq(result->strides().at(1), symbolic::one()));
}

TEST(TensorLayoutTest, BroadcastZeroesStrideOfGenuineBroadcast) {
    auto n = symbolic::symbol("N");
    math::tensor::TensorLayout layout({symbolic::one(), symbolic::integer(8)}, {symbolic::integer(8), symbolic::one()});

    // Extent 1 against a larger extent and a missing leading dimension are broadcast with stride 0.
    auto result = layout.broadcast({symbolic::integer(4), n, symbolic::integer(8)});
    ASSERT_NE(result, nullptr);
    ASSERT_EQ(result->shape().size(), 3);
    EXPECT_TRUE(symbolic::eq(result->shape().at(0), symbolic::integer(4)));
    EXPECT_TRUE(symbolic::eq(result->shape().at(1), n));
    EXPECT_TRUE(symbolic::eq(result->strides().at(0), symbolic::zero()));
    EXPECT_TRUE(symbolic::eq(result->strides().at(1), symbolic::zero()));
    EXPECT_TRUE(symbolic::eq(result->strides().at(2), symbolic::one()));

    // Incompatible extents cannot be broadcast.
    EXPECT_EQ(layout.broadcast({symbolic::integer(2), symbolic::integer(4)}), nullptr);
}
