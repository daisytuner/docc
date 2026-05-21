#include <gtest/gtest.h>

#include "sdfg/builder/sdfg_builder.h"
#include "sdfg/data_flow/pointer_metadata.h"
using namespace sdfg;

TEST(PointerMetadataTest, CreateReadOnly) {
    auto meta = data_flow::PointerAccessMeta::create_read_only(symbolic::integer(7), true);
    EXPECT_TRUE(meta->no_capture());
    EXPECT_TRUE(meta->may_contain_reads());
    EXPECT_FALSE(meta->may_contain_writes());
    auto rd_pattern = meta->access_read_pattern();
    EXPECT_FALSE(rd_pattern->empty());
    EXPECT_FALSE(rd_pattern->every_element_accessed());

    auto* convex = dynamic_cast<data_flow::ConvexAccessPattern*>(rd_pattern.get());
    EXPECT_TRUE(convex);
    EXPECT_TRUE(symbolic::eq(convex->size(), symbolic::integer(7)));
    auto wr_pattern = meta->access_write_pattern();
    auto* no_access = dynamic_cast<data_flow::NoAccessPattern*>(wr_pattern.get());
    EXPECT_TRUE(no_access);
    EXPECT_TRUE(wr_pattern->empty());
    EXPECT_TRUE(wr_pattern->every_element_accessed());
}

TEST(PointerMetadataTest, CreateFullWriteOnly) {
    auto meta = data_flow::PointerAccessMeta::create_full_write_only(symbolic::integer(7), false);
    EXPECT_FALSE(meta->no_capture());
    EXPECT_FALSE(meta->may_contain_reads());
    EXPECT_TRUE(meta->may_contain_writes());
    auto rd_pattern = meta->access_read_pattern();
    EXPECT_TRUE(rd_pattern->empty());
    EXPECT_TRUE(rd_pattern->every_element_accessed());

    auto* no_access = dynamic_cast<data_flow::NoAccessPattern*>(rd_pattern.get());
    EXPECT_TRUE(no_access);

    auto wr_pattern = meta->access_write_pattern();
    EXPECT_TRUE(wr_pattern->every_element_accessed());
    EXPECT_FALSE(wr_pattern->empty());
    auto* convex = dynamic_cast<data_flow::ConvexAccessPattern*>(wr_pattern.get());
    EXPECT_TRUE(convex);
    EXPECT_TRUE(symbolic::eq(convex->size(), symbolic::integer(7)));
}

TEST(PointerMetadataTest, CreateGeneric) {
    auto meta = data_flow::PointerAccessMeta::create_generic(
        data_flow::ConvexAccessPattern::create(symbolic::integer(7)),
        data_flow::ConvexAccessPattern::create(symbolic::integer(3), true),
        true
    );
    EXPECT_TRUE(meta->no_capture());
    EXPECT_TRUE(meta->may_contain_reads());
    EXPECT_TRUE(meta->may_contain_writes());
    auto rd_pattern = meta->access_read_pattern();
    EXPECT_FALSE(rd_pattern->empty());
    EXPECT_FALSE(rd_pattern->every_element_accessed());

    auto* convex_rd = dynamic_cast<data_flow::ConvexAccessPattern*>(rd_pattern.get());
    EXPECT_TRUE(convex_rd);
    EXPECT_TRUE(symbolic::eq(convex_rd->size(), symbolic::integer(7)));

    auto wr_pattern = meta->access_write_pattern();
    EXPECT_TRUE(wr_pattern->every_element_accessed());
    EXPECT_FALSE(wr_pattern->empty());
    auto* convex_wr = dynamic_cast<data_flow::ConvexAccessPattern*>(wr_pattern.get());
    EXPECT_TRUE(convex_wr);
    EXPECT_TRUE(symbolic::eq(convex_wr->size(), symbolic::integer(3)));
}

TEST(PointerMetadataTest, SerializationTest) {
    std::vector<data_flow::PointerAccessType> meta;
    meta.push_back(data_flow::PointerAccessMeta::create_invalidate());
    meta.push_back(data_flow::PointerAccessMeta::create_read_only(symbolic::integer(5), false));
    meta.push_back(data_flow::PointerAccessMeta::create_full_write_only(symbolic::integer(8), true));
    meta.push_back(data_flow::PointerAccessMeta::create_generic(
        data_flow::ConvexAccessPattern::create(symbolic::integer(3)), data_flow::NoAccessPattern::instance(), true
    ));

    auto j = data_flow::PointerAccessMetaSerializer::serialize(meta);
    auto deserialized = data_flow::PointerAccessMetaSerializer::deserialize_list(j);

    EXPECT_EQ(deserialized.size(), meta.size());
    auto* inv = dynamic_cast<data_flow::PointerInvalidate*>(deserialized.at(0).get());
    EXPECT_TRUE(inv);
    auto* ro = dynamic_cast<data_flow::PointerReadOnly*>(deserialized.at(1).get());
    EXPECT_TRUE(ro);
    EXPECT_TRUE(symbolic::
                    eq(dynamic_cast<data_flow::ConvexAccessPattern*>(ro->access_read_pattern().get())->size(),
                       symbolic::integer(5)));
    EXPECT_TRUE(dynamic_cast<data_flow::NoAccessPattern*>(ro->access_write_pattern().get()));
    EXPECT_FALSE(ro->no_capture());
    auto* wr = dynamic_cast<data_flow::PointerFullWriteOnly*>(deserialized.at(2).get());
    EXPECT_TRUE(wr);
    EXPECT_TRUE(dynamic_cast<data_flow::NoAccessPattern*>(wr->access_read_pattern().get()));
    EXPECT_TRUE(symbolic::
                    eq(dynamic_cast<data_flow::ConvexAccessPattern*>(wr->access_write_pattern().get())->size(),
                       symbolic::integer(8)));
    auto* gen = dynamic_cast<data_flow::PointerGenericAccess*>(deserialized.at(3).get());
    EXPECT_TRUE(gen);
}
