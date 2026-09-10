#include <gtest/gtest.h>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/serializer/json_serializer.h"
#include "sdfg/tiles/library_nodes/async_copy_node.h"

using namespace sdfg;

namespace {
builder::StructuredSDFGBuilder make_builder() { return builder::StructuredSDFGBuilder("async_test", FunctionType_CPU); }
} // namespace

inline data_flow::ImplementationType ImplementationType_DUMMY{"DUMMY"};

TEST(AsyncCopyNodeTest, ConstructAndProperties) {
    auto builder = make_builder();
    auto& block = builder.add_block(builder.subject().root());

    auto& copy =
        static_cast<tiles::CpAsyncCopyNode&>(builder.add_library_node<
                                             tiles::CpAsyncCopyNode>(block, DebugInfo(), ImplementationType_DUMMY, 16));
    EXPECT_EQ(copy.code().value(), "cp_async_copy");
    EXPECT_EQ(copy.bytes(), 16u);
    EXPECT_EQ(copy.inputs().size(), 2u); // {_dst, _src}
    EXPECT_EQ(copy.outputs().size(), 0u);

    auto& commit =
        static_cast<tiles::PipelineCommitNode&>(builder.add_library_node<
                                                tiles::PipelineCommitNode>(block, DebugInfo(), ImplementationType_DUMMY)
        );
    EXPECT_EQ(commit.code().value(), "pipeline_commit");

    auto& wait =
        static_cast<tiles::PipelineWaitNode&>(builder.add_library_node<
                                              tiles::PipelineWaitNode>(block, DebugInfo(), ImplementationType_DUMMY, 1)
        );
    EXPECT_EQ(wait.code().value(), "pipeline_wait");
    EXPECT_EQ(wait.keep_outstanding(), 1u);
}

TEST(AsyncCopyNodeTest, Clone) {
    auto builder = make_builder();
    auto& block = builder.add_block(builder.subject().root());
    auto& copy =
        static_cast<tiles::CpAsyncCopyNode&>(builder.add_library_node<
                                             tiles::CpAsyncCopyNode>(block, DebugInfo(), ImplementationType_DUMMY, 8));
    auto cloned = copy.clone(copy.element_id(), copy.vertex(), copy.get_parent());
    auto* copy_clone = dynamic_cast<tiles::CpAsyncCopyNode*>(cloned.get());
    ASSERT_NE(copy_clone, nullptr);
    EXPECT_EQ(copy_clone->bytes(), 8u);

    auto& wait =
        static_cast<tiles::PipelineWaitNode&>(builder.add_library_node<
                                              tiles::PipelineWaitNode>(block, DebugInfo(), ImplementationType_DUMMY, 3)
        );
    auto wcloned = wait.clone(wait.element_id(), wait.vertex(), wait.get_parent());
    auto* wait_clone = dynamic_cast<tiles::PipelineWaitNode*>(wcloned.get());
    ASSERT_NE(wait_clone, nullptr);
    EXPECT_EQ(wait_clone->keep_outstanding(), 3u);
}

TEST(AsyncCopyNodeTest, SerializeRoundTrip) {
    auto builder = make_builder();
    auto& block = builder.add_block(builder.subject().root());
    builder.add_library_node<tiles::CpAsyncCopyNode>(block, DebugInfo(), ImplementationType_DUMMY, 16);
    builder.add_library_node<tiles::PipelineCommitNode>(block, DebugInfo(), ImplementationType_DUMMY);
    builder.add_library_node<tiles::PipelineWaitNode>(block, DebugInfo(), ImplementationType_DUMMY, 2);

    serializer::JSONSerializer serializer;
    auto j = serializer.serialize(builder.subject());
    auto restored = serializer.deserialize(j);

    auto& rblock = static_cast<structured_control_flow::Block&>(restored->root().at(0));
    size_t n_copy = 0, n_commit = 0, n_wait = 0;
    for (auto& node : rblock.dataflow().nodes()) {
        if (auto* c = dynamic_cast<tiles::CpAsyncCopyNode*>(&node)) {
            n_copy++;
            EXPECT_EQ(c->bytes(), 16u);
        } else if (dynamic_cast<tiles::PipelineCommitNode*>(&node)) {
            n_commit++;
        } else if (auto* w = dynamic_cast<tiles::PipelineWaitNode*>(&node)) {
            n_wait++;
            EXPECT_EQ(w->keep_outstanding(), 2u);
        }
    }
    EXPECT_EQ(n_copy, 1u);
    EXPECT_EQ(n_commit, 1u);
    EXPECT_EQ(n_wait, 1u);
}

TEST(AsyncCopyNodeTest, VectorCopyConstructAndProperties) {
    auto builder = make_builder();
    auto& block = builder.add_block(builder.subject().root());
    auto& copy =
        static_cast<tiles::VectorCopyNode&>(builder.add_library_node<
                                            tiles::VectorCopyNode>(block, DebugInfo(), ImplementationType_DUMMY, 16));
    EXPECT_EQ(copy.code().value(), "vector_copy");
    EXPECT_EQ(copy.bytes(), 16u);
    EXPECT_EQ(copy.inputs().size(), 2u); // {_dst, _src}
    EXPECT_EQ(copy.outputs().size(), 0u);
}

TEST(AsyncCopyNodeTest, VectorCopyCloneAndSerialize) {
    auto builder = make_builder();
    auto& block = builder.add_block(builder.subject().root());
    auto& copy =
        static_cast<tiles::VectorCopyNode&>(builder.add_library_node<
                                            tiles::VectorCopyNode>(block, DebugInfo(), ImplementationType_DUMMY, 8));
    auto cloned = copy.clone(copy.element_id(), copy.vertex(), copy.get_parent());
    auto* copy_clone = dynamic_cast<tiles::VectorCopyNode*>(cloned.get());
    ASSERT_NE(copy_clone, nullptr);
    EXPECT_EQ(copy_clone->bytes(), 8u);

    serializer::JSONSerializer serializer;
    auto j = serializer.serialize(builder.subject());
    auto restored = serializer.deserialize(j);
    auto& rblock = static_cast<structured_control_flow::Block&>(restored->root().at(0));
    size_t n = 0;
    for (auto& node : rblock.dataflow().nodes()) {
        if (auto* c = dynamic_cast<tiles::VectorCopyNode*>(&node)) {
            n++;
            EXPECT_EQ(c->bytes(), 8u);
        }
    }
    EXPECT_EQ(n, 1u);
}

// {"_dst","_src"}: the copy writes _dst and reads _src through no-capture pointers,
// so a container staged through it is not flagged aliased by the tile analysis and a
// later LocalStorage (of any container) still applies — vectorization stays
// order-independent w.r.t. other transformations.
TEST(AsyncCopyNodeTest, PointerAccessTypeNoCapture) {
    auto builder = make_builder();
    auto& block = builder.add_block(builder.subject().root());

    auto& vec =
        static_cast<tiles::VectorCopyNode&>(builder.add_library_node<
                                            tiles::VectorCopyNode>(block, DebugInfo(), ImplementationType_DUMMY, 16));
    auto vdst = vec.pointer_access_type(0);
    auto vsrc = vec.pointer_access_type(1);
    ASSERT_NE(vdst, nullptr);
    ASSERT_NE(vsrc, nullptr);
    EXPECT_TRUE(vdst->no_capture());
    EXPECT_TRUE(vsrc->no_capture());

    auto& cp =
        static_cast<tiles::CpAsyncCopyNode&>(builder.add_library_node<
                                             tiles::CpAsyncCopyNode>(block, DebugInfo(), ImplementationType_DUMMY, 16));
    auto cdst = cp.pointer_access_type(0);
    auto csrc = cp.pointer_access_type(1);
    ASSERT_NE(cdst, nullptr);
    ASSERT_NE(csrc, nullptr);
    EXPECT_TRUE(cdst->no_capture());
    EXPECT_TRUE(csrc->no_capture());
}
