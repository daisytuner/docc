#include "sdfg/deepcopy/structured_sdfg_deep_copy.h"

#include <gtest/gtest.h>

#include "sdfg/data_flow/library_nodes/barrier_local_node.h"

using namespace sdfg;

TEST(StructuredSDFGDeepCopy, Block) {
    builder::StructuredSDFGBuilder builder_source("sdfg_source", FunctionType_CPU);
    auto& sdfg_source = builder_source.subject();
    auto& root_source = sdfg_source.root();

    auto& block = builder_source.add_block(root_source);

    builder::StructuredSDFGBuilder builder_target("sdfg_target", FunctionType_CPU);
    auto& sdfg_target = builder_target.subject();
    auto& root_target = sdfg_target.root();

    deepcopy::StructuredSDFGDeepCopy deep_copy(builder_target, root_target, root_source);
    deep_copy.copy();

    EXPECT_EQ(root_target.size(), 1);
    auto inserted = root_target.at(0);
    EXPECT_EQ(inserted.second.size(), 0);
    auto inserted_root = dyn_cast<structured_control_flow::Sequence*>(&inserted.first);
    EXPECT_TRUE(inserted_root);

    EXPECT_EQ(inserted_root->size(), 1);
    EXPECT_TRUE(dyn_cast<structured_control_flow::Block*>(&inserted_root->at(0).first));
    EXPECT_EQ(inserted_root->at(0).second.size(), 0);
}

TEST(StructuredSDFGDeepCopy, Block_WithAssignments) {
    builder::StructuredSDFGBuilder builder_source("sdfg_source", FunctionType_CPU);
    auto& sdfg_source = builder_source.subject();
    auto& root_source = sdfg_source.root();

    builder_source.add_container("a", types::Scalar(types::PrimitiveType::Int32));
    builder_source.add_container("b", types::Scalar(types::PrimitiveType::Int32));
    auto& block = builder_source.add_block(root_source, {{symbolic::symbol("a"), symbolic::symbol("b")}});

    builder::StructuredSDFGBuilder builder_target("sdfg_target", FunctionType_CPU);
    auto& sdfg_target = builder_target.subject();
    auto& root_target = sdfg_target.root();

    deepcopy::StructuredSDFGDeepCopy deep_copy(builder_target, root_target, root_source);
    deep_copy.copy();

    EXPECT_EQ(root_target.size(), 1);
    auto inserted = root_target.at(0);
    EXPECT_EQ(inserted.second.size(), 0);
    auto inserted_root = dyn_cast<structured_control_flow::Sequence*>(&inserted.first);
    EXPECT_TRUE(inserted_root);

    EXPECT_EQ(inserted_root->size(), 1);
    EXPECT_TRUE(dyn_cast<structured_control_flow::Block*>(&inserted_root->at(0).first));
    EXPECT_EQ(inserted_root->at(0).second.size(), 1);
}

TEST(StructuredSDFGDeepCopy, Block_WithLibraryNodebarrier_local) {
    builder::StructuredSDFGBuilder builder_source("sdfg_source", FunctionType_CPU);
    auto& sdfg_source = builder_source.subject();
    auto& root_source = sdfg_source.root();

    auto& block = builder_source.add_block(root_source);
    auto& barrier = builder_source.add_library_node<data_flow::BarrierLocalNode>(block, DebugInfo());

    builder::StructuredSDFGBuilder builder_target("sdfg_target", FunctionType_CPU);
    auto& sdfg_target = builder_target.subject();
    auto& root_target = sdfg_target.root();

    deepcopy::StructuredSDFGDeepCopy deep_copy(builder_target, root_target, root_source);
    deep_copy.copy();

    EXPECT_EQ(root_target.size(), 1);
    auto inserted = root_target.at(0);
    EXPECT_EQ(inserted.second.size(), 0);
    auto inserted_root = dyn_cast<structured_control_flow::Sequence*>(&inserted.first);
    EXPECT_TRUE(inserted_root);

    EXPECT_EQ(inserted_root->size(), 1);
    EXPECT_TRUE(dyn_cast<structured_control_flow::Block*>(&inserted_root->at(0).first));
    EXPECT_EQ(inserted_root->at(0).second.size(), 0);
    auto inserted_block = dyn_cast<structured_control_flow::Block*>(&inserted_root->at(0).first);
    EXPECT_EQ(inserted_block->dataflow().nodes().size(), 1);
    EXPECT_TRUE(dynamic_cast<data_flow::LibraryNode*>(&(*inserted_block->dataflow().nodes().begin())));
    auto inserted_barrier = dynamic_cast<data_flow::LibraryNode*>(&(*inserted_block->dataflow().nodes().begin()));
    EXPECT_EQ(inserted_barrier->code(), data_flow::LibraryNodeType_BarrierLocal);
    EXPECT_EQ(inserted_barrier->side_effect(), barrier.side_effect());
    EXPECT_TRUE(dynamic_cast<data_flow::BarrierLocalNode*>(inserted_barrier));
}

TEST(StructuredSDFGDeepCopy, Sequence) {
    builder::StructuredSDFGBuilder builder_source("sdfg_source", FunctionType_CPU);
    auto& sdfg_source = builder_source.subject();
    auto& root_source = sdfg_source.root();

    auto& sequence = builder_source.add_sequence(root_source);

    builder::StructuredSDFGBuilder builder_target("sdfg_target", FunctionType_CPU);
    auto& sdfg_target = builder_target.subject();
    auto& root_target = sdfg_target.root();

    deepcopy::StructuredSDFGDeepCopy deep_copy(builder_target, root_target, root_source);
    deep_copy.copy();

    EXPECT_EQ(root_target.size(), 1);
    auto inserted = root_target.at(0);
    EXPECT_EQ(inserted.second.size(), 0);
    auto inserted_root = dyn_cast<structured_control_flow::Sequence*>(&inserted.first);
    EXPECT_TRUE(inserted_root);

    EXPECT_EQ(inserted_root->size(), 1);
    EXPECT_TRUE(dyn_cast<structured_control_flow::Sequence*>(&inserted_root->at(0).first));
    EXPECT_EQ(inserted_root->at(0).second.size(), 0);
}

TEST(StructuredSDFGDeepCopy, Sequence_WithAssignments) {
    builder::StructuredSDFGBuilder builder_source("sdfg_source", FunctionType_CPU);
    auto& sdfg_source = builder_source.subject();
    auto& root_source = sdfg_source.root();

    builder_source.add_container("a", types::Scalar(types::PrimitiveType::Int32));
    builder_source.add_container("b", types::Scalar(types::PrimitiveType::Int32));
    auto& sequence = builder_source.add_sequence(root_source, {{symbolic::symbol("a"), symbolic::symbol("b")}});

    builder::StructuredSDFGBuilder builder_target("sdfg_target", FunctionType_CPU);
    auto& sdfg_target = builder_target.subject();
    auto& root_target = sdfg_target.root();

    deepcopy::StructuredSDFGDeepCopy deep_copy(builder_target, root_target, root_source);
    deep_copy.copy();

    EXPECT_EQ(root_target.size(), 1);
    auto inserted = root_target.at(0);
    EXPECT_EQ(inserted.second.size(), 0);
    auto inserted_root = dyn_cast<structured_control_flow::Sequence*>(&inserted.first);
    EXPECT_TRUE(inserted_root);

    EXPECT_EQ(inserted_root->size(), 1);
    EXPECT_TRUE(dyn_cast<structured_control_flow::Sequence*>(&inserted_root->at(0).first));
    EXPECT_EQ(inserted_root->at(0).second.size(), 1);
}

TEST(StructuredSDFGDeepCopy, Return) {
    builder::StructuredSDFGBuilder builder_source("sdfg_source", FunctionType_CPU);
    auto& sdfg_source = builder_source.subject();
    auto& root_source = sdfg_source.root();

    auto& sequence = builder_source.add_return(root_source, "");

    builder::StructuredSDFGBuilder builder_target("sdfg_target", FunctionType_CPU);
    auto& sdfg_target = builder_target.subject();
    auto& root_target = sdfg_target.root();

    deepcopy::StructuredSDFGDeepCopy deep_copy(builder_target, root_target, root_source);
    deep_copy.copy();

    EXPECT_EQ(root_target.size(), 1);
    auto inserted = root_target.at(0);
    EXPECT_EQ(inserted.second.size(), 0);
    auto inserted_root = dyn_cast<structured_control_flow::Sequence*>(&inserted.first);
    EXPECT_TRUE(inserted_root);

    EXPECT_EQ(inserted_root->size(), 1);
    EXPECT_TRUE(dyn_cast<structured_control_flow::Return*>(&inserted_root->at(0).first));
    EXPECT_EQ(inserted_root->at(0).second.size(), 0);
}

TEST(StructuredSDFGDeepCopy, While) {
    builder::StructuredSDFGBuilder builder_source("sdfg_source", FunctionType_CPU);
    auto& sdfg_source = builder_source.subject();
    auto& root_source = sdfg_source.root();

    auto& loop = builder_source.add_while(root_source);

    builder::StructuredSDFGBuilder builder_target("sdfg_target", FunctionType_CPU);
    auto& sdfg_target = builder_target.subject();
    auto& root_target = sdfg_target.root();

    deepcopy::StructuredSDFGDeepCopy deep_copy(builder_target, root_target, root_source);
    deep_copy.copy();

    EXPECT_EQ(root_target.size(), 1);
    auto inserted = root_target.at(0);
    EXPECT_EQ(inserted.second.size(), 0);
    auto inserted_root = dyn_cast<structured_control_flow::Sequence*>(&inserted.first);
    EXPECT_TRUE(inserted_root);

    EXPECT_EQ(inserted_root->size(), 1);
    EXPECT_TRUE(dyn_cast<structured_control_flow::While*>(&inserted_root->at(0).first));
    EXPECT_EQ(inserted_root->at(0).second.size(), 0);
}

TEST(StructuredSDFGDeepCopy, Break) {
    builder::StructuredSDFGBuilder builder_source("sdfg_source", FunctionType_CPU);
    auto& sdfg_source = builder_source.subject();
    auto& root_source = sdfg_source.root();

    auto& loop = builder_source.add_while(root_source);
    auto& break_node = builder_source.add_break(loop.root());

    builder::StructuredSDFGBuilder builder_target("sdfg_target", FunctionType_CPU);
    auto& sdfg_target = builder_target.subject();
    auto& root_target = sdfg_target.root();

    deepcopy::StructuredSDFGDeepCopy deep_copy(builder_target, root_target, root_source);
    deep_copy.copy();

    EXPECT_EQ(root_target.size(), 1);
    auto inserted = root_target.at(0);
    EXPECT_EQ(inserted.second.size(), 0);
    auto inserted_root = dyn_cast<structured_control_flow::Sequence*>(&inserted.first);
    EXPECT_TRUE(inserted_root);

    EXPECT_EQ(inserted_root->size(), 1);
    EXPECT_TRUE(dyn_cast<structured_control_flow::While*>(&inserted_root->at(0).first));
    EXPECT_EQ(inserted_root->at(0).second.size(), 0);

    auto inserted_loop = dyn_cast<structured_control_flow::While*>(&inserted_root->at(0).first);
    EXPECT_TRUE(inserted_loop);
    EXPECT_EQ(inserted_loop->root().size(), 1);
    EXPECT_TRUE(dyn_cast<structured_control_flow::Break*>(&inserted_loop->root().at(0).first));
}

TEST(StructuredSDFGDeepCopy, Continue) {
    builder::StructuredSDFGBuilder builder_source("sdfg_source", FunctionType_CPU);
    auto& sdfg_source = builder_source.subject();
    auto& root_source = sdfg_source.root();

    auto& loop = builder_source.add_while(root_source);
    auto& continue_node = builder_source.add_continue(loop.root());

    builder::StructuredSDFGBuilder builder_target("sdfg_target", FunctionType_CPU);
    auto& sdfg_target = builder_target.subject();
    auto& root_target = sdfg_target.root();

    deepcopy::StructuredSDFGDeepCopy deep_copy(builder_target, root_target, root_source);
    deep_copy.copy();

    EXPECT_EQ(root_target.size(), 1);
    auto inserted = root_target.at(0);
    EXPECT_EQ(inserted.second.size(), 0);
    auto inserted_root = dyn_cast<structured_control_flow::Sequence*>(&inserted.first);
    EXPECT_TRUE(inserted_root);

    EXPECT_EQ(inserted_root->size(), 1);
    EXPECT_TRUE(dyn_cast<structured_control_flow::While*>(&inserted_root->at(0).first));
    EXPECT_EQ(inserted_root->at(0).second.size(), 0);

    auto inserted_loop = dyn_cast<structured_control_flow::While*>(&inserted_root->at(0).first);
    EXPECT_TRUE(inserted_loop);
    EXPECT_EQ(inserted_loop->root().size(), 1);
    EXPECT_TRUE(dyn_cast<structured_control_flow::Continue*>(&inserted_loop->root().at(0).first));
}

TEST(StructuredSDFGDeepCopy, For) {
    builder::StructuredSDFGBuilder builder_source("sdfg_source", FunctionType_CPU);
    auto& sdfg_source = builder_source.subject();
    auto& root_source = sdfg_source.root();

    auto loopvar = symbolic::symbol("i");
    auto bound = symbolic::Lt(symbolic::symbol("i"), symbolic::symbol("N"));
    auto update = symbolic::add(symbolic::symbol("i"), symbolic::integer(1));
    auto init = symbolic::integer(0);

    auto& loop = builder_source.add_for(root_source, loopvar, bound, init, update);

    builder::StructuredSDFGBuilder builder_target("sdfg_target", FunctionType_CPU);
    auto& sdfg_target = builder_target.subject();
    auto& root_target = sdfg_target.root();

    deepcopy::StructuredSDFGDeepCopy deep_copy(builder_target, root_target, root_source);
    deep_copy.copy();

    EXPECT_EQ(root_target.size(), 1);
    auto inserted = root_target.at(0);
    EXPECT_EQ(inserted.second.size(), 0);
    auto inserted_root = dyn_cast<structured_control_flow::Sequence*>(&inserted.first);
    EXPECT_TRUE(inserted_root);

    EXPECT_EQ(inserted_root->size(), 1);
    EXPECT_TRUE(dyn_cast<structured_control_flow::For*>(&inserted_root->at(0).first));

    auto inserted_loop = dyn_cast<structured_control_flow::For*>(&inserted_root->at(0).first);
    EXPECT_TRUE(inserted_loop);
    EXPECT_EQ(inserted_loop->root().size(), 0);
    EXPECT_TRUE(symbolic::eq(inserted_loop->indvar(), loopvar));
    EXPECT_TRUE(symbolic::eq(inserted_loop->condition(), bound));
    EXPECT_TRUE(symbolic::eq(inserted_loop->update(), update));
    EXPECT_TRUE(symbolic::eq(inserted_loop->init(), init));
}

TEST(StructuredSDFGDeepCopy, Map) {
    builder::StructuredSDFGBuilder builder_source("sdfg_source", FunctionType_CPU);
    auto& sdfg_source = builder_source.subject();
    auto& root_source = sdfg_source.root();

    auto& map = builder_source.add_map(
        root_source,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::integer(10)),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );

    builder::StructuredSDFGBuilder builder_target("sdfg_target", FunctionType_CPU);
    auto& sdfg_target = builder_target.subject();
    auto& root_target = sdfg_target.root();

    deepcopy::StructuredSDFGDeepCopy deep_copy(builder_target, root_target, root_source);
    deep_copy.copy();

    EXPECT_EQ(root_target.size(), 1);
    auto inserted = root_target.at(0);
    EXPECT_EQ(inserted.second.size(), 0);
    auto inserted_root = dyn_cast<structured_control_flow::Sequence*>(&inserted.first);
    EXPECT_TRUE(inserted_root);

    EXPECT_EQ(inserted_root->size(), 1);
    EXPECT_TRUE(dyn_cast<structured_control_flow::Map*>(&inserted_root->at(0).first));

    auto inserted_map = dyn_cast<structured_control_flow::Map*>(&inserted_root->at(0).first);
    EXPECT_TRUE(inserted_map);
    EXPECT_EQ(inserted_map->root().size(), 0);

    EXPECT_TRUE(symbolic::eq(inserted_map->indvar(), symbolic::symbol("i")));
    EXPECT_TRUE(symbolic::eq(inserted_map->condition(), symbolic::Lt(symbolic::symbol("i"), symbolic::integer(10))));
    EXPECT_TRUE(symbolic::eq(inserted_map->init(), symbolic::integer(0)));
    EXPECT_TRUE(symbolic::eq(inserted_map->update(), symbolic::add(symbolic::symbol("i"), symbolic::integer(1))));
    EXPECT_EQ(inserted_map->schedule_type().value(), structured_control_flow::ScheduleType_Sequential::value());
}

TEST(StructuredSDFGDeepCopy, Reduce) {
    builder::StructuredSDFGBuilder builder_source("sdfg_source", FunctionType_CPU);
    auto& sdfg_source = builder_source.subject();
    auto& root_source = sdfg_source.root();

    auto& reduce = builder_source.add_reduce(
        root_source,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::integer(10)),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1)),
        {structured_control_flow::ReductionInfo{structured_control_flow::ReductionOperation::Add, "i"}},
        structured_control_flow::ScheduleType_Sequential::create()
    );

    builder::StructuredSDFGBuilder builder_target("sdfg_target", FunctionType_CPU);
    auto& sdfg_target = builder_target.subject();
    auto& root_target = sdfg_target.root();

    deepcopy::StructuredSDFGDeepCopy deep_copy(builder_target, root_target, root_source);
    deep_copy.copy();

    EXPECT_EQ(root_target.size(), 1);
    auto inserted = root_target.at(0);
    EXPECT_EQ(inserted.second.size(), 0);
    auto inserted_root = dyn_cast<structured_control_flow::Sequence*>(&inserted.first);
    EXPECT_TRUE(inserted_root);

    EXPECT_EQ(inserted_root->size(), 1);
    EXPECT_TRUE(dyn_cast<structured_control_flow::Reduce*>(&inserted_root->at(0).first));

    auto inserted_reduce = dyn_cast<structured_control_flow::Reduce*>(&inserted_root->at(0).first);
    EXPECT_TRUE(inserted_reduce);
    EXPECT_EQ(inserted_reduce->root().size(), 0);

    EXPECT_TRUE(symbolic::eq(inserted_reduce->indvar(), symbolic::symbol("i")));
    EXPECT_TRUE(symbolic::eq(inserted_reduce->condition(), symbolic::Lt(symbolic::symbol("i"), symbolic::integer(10))));
    EXPECT_TRUE(symbolic::eq(inserted_reduce->init(), symbolic::integer(0)));
    EXPECT_TRUE(symbolic::eq(inserted_reduce->update(), symbolic::add(symbolic::symbol("i"), symbolic::integer(1))));
    ASSERT_EQ(inserted_reduce->reductions().size(), 1u);
    EXPECT_EQ(inserted_reduce->reductions()[0].operation, structured_control_flow::ReductionOperation::Add);
    EXPECT_EQ(inserted_reduce->reductions()[0].container, "i");
    EXPECT_EQ(inserted_reduce->schedule_type().value(), structured_control_flow::ScheduleType_Sequential::value());
}
