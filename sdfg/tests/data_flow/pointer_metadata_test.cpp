#include <gtest/gtest.h>

#include "sdfg/builder/sdfg_builder.h"
#include "sdfg/data_flow/library_nodes/math/tensor/tensor_layout.h"
#include "sdfg/data_flow/pointer_metadata.h"
using namespace sdfg;

// A structured pattern exposes its affine layout; a flat convex span does not.
static const math::tensor::TensorLayout* pattern_layout(const data_flow::MemoryAccessPatternType& pattern) {
    return pattern ? pattern->layout() : nullptr;
}

TEST(PointerMetadataTest, CreateReadOnly) {
    auto meta = data_flow::PointerAccessMeta::create_read_only(symbolic::integer(7), true);
    EXPECT_TRUE(meta->no_capture());
    EXPECT_TRUE(meta->may_contain_reads());
    EXPECT_FALSE(meta->may_contain_writes());

    auto read = meta->access_read_pattern();
    ASSERT_NE(read, nullptr);
    EXPECT_FALSE(read->every_element_accessed());
    auto* convex = dynamic_cast<data_flow::ConvexAccessPattern*>(read.get());
    ASSERT_NE(convex, nullptr);
    EXPECT_TRUE(symbolic::eq(convex->size(), symbolic::integer(7)));

    EXPECT_TRUE(meta->access_write_pattern()->empty());
}

TEST(PointerMetadataTest, CreateFullWriteOnly) {
    auto meta = data_flow::PointerAccessMeta::create_full_write_only(symbolic::integer(7), false);
    EXPECT_FALSE(meta->no_capture());
    EXPECT_FALSE(meta->may_contain_reads());
    EXPECT_TRUE(meta->may_contain_writes());

    EXPECT_TRUE(meta->access_read_pattern()->empty());
    auto write = meta->access_write_pattern();
    ASSERT_NE(write, nullptr);
    EXPECT_TRUE(write->every_element_accessed());
    auto* convex = dynamic_cast<data_flow::ConvexAccessPattern*>(write.get());
    ASSERT_NE(convex, nullptr);
    EXPECT_TRUE(symbolic::eq(convex->size(), symbolic::integer(7)));
}

TEST(PointerMetadataTest, CreateGeneric) {
    auto meta = data_flow::PointerAccessMeta::create_generic(
        data_flow::ConvexAccessPattern::create(symbolic::integer(7), false),
        data_flow::ConvexAccessPattern::create(symbolic::integer(3), true),
        true
    );
    EXPECT_TRUE(meta->no_capture());
    EXPECT_TRUE(meta->may_contain_reads());
    EXPECT_TRUE(meta->may_contain_writes());

    auto read = meta->access_read_pattern();
    ASSERT_NE(read, nullptr);
    EXPECT_FALSE(read->every_element_accessed());
    EXPECT_TRUE(symbolic::eq(dynamic_cast<data_flow::ConvexAccessPattern&>(*read).size(), symbolic::integer(7)));

    auto write = meta->access_write_pattern();
    ASSERT_NE(write, nullptr);
    EXPECT_TRUE(write->every_element_accessed());
    EXPECT_TRUE(symbolic::eq(dynamic_cast<data_flow::ConvexAccessPattern&>(*write).size(), symbolic::integer(3)));
}

// A null size means the access is unbounded: no pattern is reported.
TEST(PointerMetadataTest, UnboundedWhenNullSize) {
    auto ro = data_flow::PointerAccessMeta::create_read_only(SymEngine::null, true);
    EXPECT_TRUE(ro->may_contain_reads());
    EXPECT_EQ(ro->access_read_pattern(), nullptr);
    auto wo = data_flow::PointerAccessMeta::create_full_write_only(SymEngine::null, true);
    EXPECT_TRUE(wo->may_contain_writes());
    EXPECT_EQ(wo->access_write_pattern(), nullptr);
    auto inv = data_flow::PointerAccessMeta::create_invalidate();
    EXPECT_TRUE(inv->access_read_pattern()->empty());
    EXPECT_TRUE(inv->access_write_pattern()->empty());
}

TEST(PointerMetadataTest, SerializationTest) {
    std::vector<data_flow::PointerAccessType> meta;
    meta.push_back(data_flow::PointerAccessMeta::create_invalidate());
    meta.push_back(data_flow::PointerAccessMeta::create_read_only(symbolic::integer(5), false));
    meta.push_back(data_flow::PointerAccessMeta::create_full_write_only(symbolic::integer(8), true));
    meta.push_back(
        data_flow::PointerAccessMeta::create_generic(
            data_flow::ConvexAccessPattern::create(symbolic::integer(3), false),
            data_flow::NoAccessPattern::instance(),
            true
        )
    );

    auto j = data_flow::PointerAccessMetaSerializer::serialize(meta);
    auto deserialized = data_flow::PointerAccessMetaSerializer::deserialize_list(j);

    EXPECT_EQ(deserialized.size(), meta.size());
    EXPECT_TRUE(dynamic_cast<data_flow::PointerInvalidate*>(deserialized.at(0).get()));

    auto* ro = dynamic_cast<data_flow::PointerReadOnly*>(deserialized.at(1).get());
    ASSERT_NE(ro, nullptr);
    auto ro_read = ro->access_read_pattern();
    ASSERT_NE(ro_read, nullptr);
    EXPECT_TRUE(symbolic::eq(dynamic_cast<data_flow::ConvexAccessPattern&>(*ro_read).size(), symbolic::integer(5)));
    EXPECT_FALSE(ro->no_capture());

    auto* wr = dynamic_cast<data_flow::PointerFullWriteOnly*>(deserialized.at(2).get());
    ASSERT_NE(wr, nullptr);
    auto wr_write = wr->access_write_pattern();
    ASSERT_NE(wr_write, nullptr);
    EXPECT_TRUE(symbolic::eq(dynamic_cast<data_flow::ConvexAccessPattern&>(*wr_write).size(), symbolic::integer(8)));

    auto* gen = dynamic_cast<data_flow::PointerGenericAccess*>(deserialized.at(3).get());
    ASSERT_NE(gen, nullptr);
    EXPECT_TRUE(gen->may_contain_reads());
    EXPECT_FALSE(gen->may_contain_writes());
    auto gen_read = gen->access_read_pattern();
    ASSERT_NE(gen_read, nullptr);
    EXPECT_TRUE(symbolic::eq(dynamic_cast<data_flow::ConvexAccessPattern&>(*gen_read).size(), symbolic::integer(3)));
    EXPECT_TRUE(gen->access_write_pattern()->empty());
}

// A structured operand reports its full affine layout through a TensorLayoutPattern.
TEST(PointerMetadataTest, StructuredRegion) {
    // A 4x8 row-major tile: shape [4, 8], strides [8, 1], offset 0.
    symbolic::MultiExpression shape = {symbolic::integer(4), symbolic::integer(8)};
    math::tensor::TensorLayout layout(shape);

    auto ro = data_flow::PointerAccessMeta::create_read_only(symbolic::integer(32), true, layout);
    const auto* l = pattern_layout(ro->access_read_pattern());
    ASSERT_NE(l, nullptr);
    EXPECT_EQ(l->dims(), 2);
    EXPECT_TRUE(symbolic::eq(l->get_dim(0), symbolic::integer(4)));
    EXPECT_TRUE(symbolic::eq(l->get_dim(1), symbolic::integer(8)));
    EXPECT_TRUE(symbolic::eq(l->get_stride(0), symbolic::integer(8)));
    EXPECT_TRUE(symbolic::eq(l->get_stride(1), symbolic::integer(1)));

    auto wo = data_flow::PointerAccessMeta::create_full_write_only(symbolic::integer(32), true, layout);
    const auto* wl = pattern_layout(wo->access_write_pattern());
    ASSERT_NE(wl, nullptr);
    EXPECT_TRUE(wo->access_write_pattern()->every_element_accessed());
    EXPECT_TRUE(symbolic::eq(wl->get_dim(0), symbolic::integer(4)));

    // Cloning preserves the region.
    auto cloned = ro->clone();
    const auto* cl = pattern_layout(cloned->access_read_pattern());
    ASSERT_NE(cl, nullptr);
    EXPECT_TRUE(symbolic::eq(cl->get_dim(0), symbolic::integer(4)));
}

TEST(PointerMetadataTest, ReplaceSymbols) {
    auto M = symbolic::symbol("M");
    auto K = symbolic::symbol("K");
    math::tensor::TensorLayout layout(symbolic::MultiExpression{M, K});
    auto ro = data_flow::PointerAccessMeta::create_read_only(symbolic::mul(M, K), true, layout);

    ro->replace(M, symbolic::integer(16));
    const auto* l = pattern_layout(ro->access_read_pattern());
    ASSERT_NE(l, nullptr);
    EXPECT_TRUE(symbolic::eq(l->get_dim(0), symbolic::integer(16)));
    EXPECT_TRUE(symbolic::eq(l->get_dim(1), K));

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
    check(pattern_layout(deserialized.at(0)->access_read_pattern()));
    check(pattern_layout(deserialized.at(1)->access_write_pattern()));
}
