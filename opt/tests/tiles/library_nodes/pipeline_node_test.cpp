#include <gtest/gtest.h>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/serializer/json_serializer.h"
#include "sdfg/tiles/library_nodes/pipeline_node.h"

using namespace sdfg;

namespace {
builder::StructuredSDFGBuilder make_builder() {
    return builder::StructuredSDFGBuilder("async_test", FunctionType_CPU);
}
} // namespace

inline data_flow::ImplementationType ImplementationType_DUMMY{"DUMMY"};

TEST(PipelineNodeTest, ConstructAndProperties) {
    auto builder = make_builder();
    auto& block = builder.add_block(builder.subject().root());

    auto& commit = static_cast<
        tiles::PipelineCommitNode&>(builder.add_library_node<
                                    tiles::PipelineCommitNode>(block, DebugInfo(), ImplementationType_DUMMY));
    EXPECT_EQ(commit.code().value(), "pipeline_commit");

    auto& wait =
        static_cast<tiles::PipelineWaitNode&>(builder.add_library_node<
                                              tiles::PipelineWaitNode>(block, DebugInfo(), ImplementationType_DUMMY, 1));
    EXPECT_EQ(wait.code().value(), "pipeline_wait");
    EXPECT_EQ(wait.keep_outstanding(), 1u);
}

TEST(PipelineNodeTest, Clone) {
    auto builder = make_builder();
    auto& block = builder.add_block(builder.subject().root());

    auto& wait =
        static_cast<tiles::PipelineWaitNode&>(builder.add_library_node<
                                              tiles::PipelineWaitNode>(block, DebugInfo(), ImplementationType_DUMMY, 3));
    auto wcloned = wait.clone(wait.element_id(), wait.vertex(), wait.get_parent());
    auto* wait_clone = dynamic_cast<tiles::PipelineWaitNode*>(wcloned.get());
    ASSERT_NE(wait_clone, nullptr);
    EXPECT_EQ(wait_clone->keep_outstanding(), 3u);
}

TEST(PipelineNodeTest, SerializeRoundTrip) {
    auto builder = make_builder();
    auto& block = builder.add_block(builder.subject().root());
    builder.add_library_node<tiles::PipelineCommitNode>(block, DebugInfo(), ImplementationType_DUMMY);
    builder.add_library_node<tiles::PipelineWaitNode>(block, DebugInfo(), ImplementationType_DUMMY, 2);

    serializer::JSONSerializer serializer;
    auto j = serializer.serialize(builder.subject());
    auto restored = serializer.deserialize(j);

    auto& rblock = static_cast<structured_control_flow::Block&>(restored->root().at(0));
    size_t n_commit = 0, n_wait = 0;
    for (auto& node : rblock.dataflow().nodes()) {
        if (dynamic_cast<tiles::PipelineCommitNode*>(&node)) {
            n_commit++;
        } else if (auto* w = dynamic_cast<tiles::PipelineWaitNode*>(&node)) {
            n_wait++;
            EXPECT_EQ(w->keep_outstanding(), 2u);
        }
    }
    EXPECT_EQ(n_commit, 1u);
    EXPECT_EQ(n_wait, 1u);
}

namespace {

std::vector<tiles::ScheduleGroup> sample_schedule(size_t steps) {
    std::vector<tiles::ScheduleGroup> groups{{tiles::InstructionClass::LdsRead, 2}};
    for (size_t k = 0; k < steps; ++k) {
        groups.push_back({tiles::InstructionClass::Mfma, 1});
        groups.push_back({tiles::InstructionClass::GlobalRead, 1});
        groups.push_back({tiles::InstructionClass::LdsWrite, 1});
    }
    return groups;
}

} // namespace

// Base: a schedule round-trips through clone and JSON with its groups unchanged.
TEST(PipelineNodeTest, ScheduleBaseRoundTrip) {
    auto builder = make_builder();
    auto& block = builder.add_block(builder.subject().root());
    auto& node = static_cast<tiles::PipelineScheduleNode&>(builder.add_library_node<tiles::PipelineScheduleNode>(
        block, DebugInfo(), ImplementationType_DUMMY, sample_schedule(1)
    ));
    EXPECT_EQ(node.code().value(), "pipeline_schedule");
    EXPECT_NO_THROW(builder.subject().validate());
    auto cloned = node.clone(node.element_id(), node.vertex(), node.get_parent());
    EXPECT_EQ(static_cast<tiles::PipelineScheduleNode&>(*cloned).groups(), sample_schedule(1));

    serializer::JSONSerializer serializer;
    auto j = serializer.serialize(builder.subject());
    auto restored = serializer.deserialize(j);
    auto& rblock = static_cast<structured_control_flow::Block&>(restored->root().at(0));
    size_t found = 0;
    for (auto& n : rblock.dataflow().nodes()) {
        if (auto* s = dynamic_cast<tiles::PipelineScheduleNode*>(&n)) {
            ++found;
            EXPECT_EQ(s->groups(), sample_schedule(1));
        }
    }
    EXPECT_EQ(found, 1u);
}

// Step: longer schedules keep every group in order through JSON; a group claiming no instruction is invalid.
TEST(PipelineNodeTest, ScheduleStepRoundTripAndValidation) {
    for (size_t steps = 2; steps <= 4; ++steps) {
        auto builder = make_builder();
        auto& block = builder.add_block(builder.subject().root());
        builder.add_library_node<
            tiles::PipelineScheduleNode>(block, DebugInfo(), ImplementationType_DUMMY, sample_schedule(steps));
        serializer::JSONSerializer serializer;
        auto j = serializer.serialize(builder.subject());
        auto restored = serializer.deserialize(j);
        auto& rblock = static_cast<structured_control_flow::Block&>(restored->root().at(0));
        for (auto& n : rblock.dataflow().nodes()) {
            if (auto* s = dynamic_cast<tiles::PipelineScheduleNode*>(&n)) {
                EXPECT_EQ(s->groups(), sample_schedule(steps)) << steps;
            }
        }
    }
    auto builder = make_builder();
    auto& block = builder.add_block(builder.subject().root());
    builder.add_library_node<tiles::PipelineScheduleNode>(
        block,
        DebugInfo(),
        ImplementationType_DUMMY,
        std::vector<tiles::ScheduleGroup>{{tiles::InstructionClass::Mfma, 0}}
    );
    EXPECT_THROW(builder.subject().validate(), InvalidSDFGException);
}
