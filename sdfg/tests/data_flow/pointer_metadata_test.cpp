#include <gtest/gtest.h>

#include "sdfg/builder/sdfg_builder.h"
#include "sdfg/data_flow/library_nodes/math/tensor/tensor_layout.h"
#include "sdfg/data_flow/pointer_metadata.h"
using namespace sdfg;

static math::tensor::TensorLayout flat_region(int n) {
    return math::tensor::TensorLayout(symbolic::MultiExpression{symbolic::integer(n)});
}

TEST(PointerMetadataTest, CreateReadOnly) {
    auto meta = data_flow::PointerAccessMeta::create_read_only(symbolic::integer(7), true);
    EXPECT_TRUE(meta->no_capture());
    EXPECT_TRUE(meta->may_contain_reads());
    EXPECT_FALSE(meta->may_contain_writes());
    ASSERT_NE(meta->read_layout(), nullptr);
    EXPECT_TRUE(symbolic::eq(meta->read_layout()->memory_span(), symbolic::integer(7)));
    EXPECT_FALSE(meta->read_covers_all());
    EXPECT_EQ(meta->write_layout(), nullptr);
}

TEST(PointerMetadataTest, CreateFullWriteOnly) {
    auto meta = data_flow::PointerAccessMeta::create_full_write_only(symbolic::integer(7), false);
    EXPECT_FALSE(meta->no_capture());
    EXPECT_FALSE(meta->may_contain_reads());
    EXPECT_TRUE(meta->may_contain_writes());
    EXPECT_EQ(meta->read_layout(), nullptr);
    ASSERT_NE(meta->write_layout(), nullptr);
    EXPECT_TRUE(meta->write_covers_all());
    EXPECT_TRUE(symbolic::eq(meta->write_layout()->memory_span(), symbolic::integer(7)));
}

TEST(PointerMetadataTest, CreateGeneric) {
    data_flow::AccessRegion read{true, flat_region(7), false};
    data_flow::AccessRegion write{true, flat_region(3), true};
    auto meta = data_flow::PointerAccessMeta::create_generic(read, write, true);
    EXPECT_TRUE(meta->no_capture());
    EXPECT_TRUE(meta->may_contain_reads());
    EXPECT_TRUE(meta->may_contain_writes());
    ASSERT_NE(meta->read_layout(), nullptr);
    EXPECT_FALSE(meta->read_covers_all());
    EXPECT_TRUE(symbolic::eq(meta->read_layout()->memory_span(), symbolic::integer(7)));
    ASSERT_NE(meta->write_layout(), nullptr);
    EXPECT_TRUE(meta->write_covers_all());
    EXPECT_TRUE(symbolic::eq(meta->write_layout()->memory_span(), symbolic::integer(3)));
}

// A null size means the access is unbounded: no region is reported.
TEST(PointerMetadataTest, UnboundedWhenNullSize) {
    auto ro = data_flow::PointerAccessMeta::create_read_only(SymEngine::null, true);
    EXPECT_TRUE(ro->may_contain_reads());
    EXPECT_EQ(ro->read_layout(), nullptr);
    auto wo = data_flow::PointerAccessMeta::create_full_write_only(SymEngine::null, true);
    EXPECT_TRUE(wo->may_contain_writes());
    EXPECT_EQ(wo->write_layout(), nullptr);
    auto inv = data_flow::PointerAccessMeta::create_invalidate();
    EXPECT_EQ(inv->read_layout(), nullptr);
    EXPECT_EQ(inv->write_layout(), nullptr);
}

TEST(PointerMetadataTest, SerializationTest) {
    data_flow::AccessRegion gread{true, flat_region(3), false};
    data_flow::AccessRegion gwrite{false, std::nullopt, false};
    std::vector<data_flow::PointerAccessType> meta;
    meta.push_back(data_flow::PointerAccessMeta::create_invalidate());
    meta.push_back(data_flow::PointerAccessMeta::create_read_only(symbolic::integer(5), false));
    meta.push_back(data_flow::PointerAccessMeta::create_full_write_only(symbolic::integer(8), true));
    meta.push_back(data_flow::PointerAccessMeta::create_generic(gread, gwrite, true));

    auto j = data_flow::PointerAccessMetaSerializer::serialize(meta);
    auto deserialized = data_flow::PointerAccessMetaSerializer::deserialize_list(j);

    EXPECT_EQ(deserialized.size(), meta.size());
    EXPECT_TRUE(dynamic_cast<data_flow::PointerInvalidate*>(deserialized.at(0).get()));

    auto* ro = dynamic_cast<data_flow::PointerReadOnly*>(deserialized.at(1).get());
    ASSERT_NE(ro, nullptr);
    ASSERT_NE(ro->read_layout(), nullptr);
    EXPECT_TRUE(symbolic::eq(ro->read_layout()->memory_span(), symbolic::integer(5)));
    EXPECT_EQ(ro->write_layout(), nullptr);
    EXPECT_FALSE(ro->no_capture());

    auto* wr = dynamic_cast<data_flow::PointerFullWriteOnly*>(deserialized.at(2).get());
    ASSERT_NE(wr, nullptr);
    EXPECT_EQ(wr->read_layout(), nullptr);
    ASSERT_NE(wr->write_layout(), nullptr);
    EXPECT_TRUE(symbolic::eq(wr->write_layout()->memory_span(), symbolic::integer(8)));

    auto* gen = dynamic_cast<data_flow::PointerGenericAccess*>(deserialized.at(3).get());
    ASSERT_NE(gen, nullptr);
    EXPECT_TRUE(gen->may_contain_reads());
    EXPECT_FALSE(gen->may_contain_writes());
    ASSERT_NE(gen->read_layout(), nullptr);
    EXPECT_TRUE(symbolic::eq(gen->read_layout()->memory_span(), symbolic::integer(3)));
    EXPECT_EQ(gen->write_layout(), nullptr);
}

TEST(PointerMetadataTest, StructuredRegion) {
    // A 4x8 row-major tile: shape [4, 8], strides [8, 1], offset 0.
    symbolic::MultiExpression shape = {symbolic::integer(4), symbolic::integer(8)};
    math::tensor::TensorLayout layout(shape);

    auto ro = data_flow::PointerAccessMeta::create_read_only(symbolic::integer(32), true, layout);
    ASSERT_NE(ro->read_layout(), nullptr);
    EXPECT_EQ(ro->read_layout()->dims(), 2);
    EXPECT_TRUE(symbolic::eq(ro->read_layout()->get_dim(0), symbolic::integer(4)));
    EXPECT_TRUE(symbolic::eq(ro->read_layout()->get_dim(1), symbolic::integer(8)));
    EXPECT_TRUE(symbolic::eq(ro->read_layout()->get_stride(0), symbolic::integer(8)));
    EXPECT_TRUE(symbolic::eq(ro->read_layout()->get_stride(1), symbolic::integer(1)));

    auto wo = data_flow::PointerAccessMeta::create_full_write_only(symbolic::integer(32), true, layout);
    ASSERT_NE(wo->write_layout(), nullptr);
    EXPECT_TRUE(symbolic::eq(wo->write_layout()->get_dim(0), symbolic::integer(4)));

    // Cloning preserves the region.
    auto cloned = ro->clone();
    ASSERT_NE(cloned->read_layout(), nullptr);
    EXPECT_TRUE(symbolic::eq(cloned->read_layout()->get_dim(0), symbolic::integer(4)));
}

TEST(PointerMetadataTest, ReplaceSymbols) {
    auto M = symbolic::symbol("M");
    auto K = symbolic::symbol("K");
    math::tensor::TensorLayout layout(symbolic::MultiExpression{M, K});
    auto ro = data_flow::PointerAccessMeta::create_read_only(symbolic::mul(M, K), true, layout);

    ro->replace(M, symbolic::integer(16));
    ASSERT_NE(ro->read_layout(), nullptr);
    EXPECT_TRUE(symbolic::eq(ro->read_layout()->get_dim(0), symbolic::integer(16)));
    EXPECT_TRUE(symbolic::eq(ro->read_layout()->get_dim(1), K));

    // The caller's layout is copied in, so mutating the meta leaves it untouched.
    EXPECT_TRUE(symbolic::eq(layout.get_dim(0), M));
}

TEST(PointerMetadataTest, RegionSurvivesSerialization) {
    symbolic::MultiExpression shape = {symbolic::integer(3), symbolic::integer(5)};
    symbolic::MultiExpression strides = {symbolic::integer(5), symbolic::integer(1)};
    math::tensor::TensorLayout layout(shape, strides, symbolic::integer(2));

    std::vector<data_flow::PointerAccessType> meta;
    meta.push_back(data_flow::PointerAccessMeta::create_read_only(symbolic::integer(15), true, layout));
    meta.push_back(data_flow::PointerAccessMeta::create_full_write_only(symbolic::integer(15), false, layout));

    auto j = data_flow::PointerAccessMetaSerializer::serialize(meta);
    auto deserialized = data_flow::PointerAccessMetaSerializer::deserialize_list(j);
    ASSERT_EQ(deserialized.size(), 2u);

    auto check = [](const math::tensor::TensorLayout* l) {
        ASSERT_NE(l, nullptr);
        EXPECT_TRUE(symbolic::eq(l->get_dim(0), symbolic::integer(3)));
        EXPECT_TRUE(symbolic::eq(l->get_dim(1), symbolic::integer(5)));
        EXPECT_TRUE(symbolic::eq(l->get_stride(0), symbolic::integer(5)));
        EXPECT_TRUE(symbolic::eq(l->offset(), symbolic::integer(2)));
    };
    check(deserialized.at(0)->read_layout());
    check(deserialized.at(1)->write_layout());
}
