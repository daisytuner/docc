#include "sdfg/transformations/offloading/gpu_tiling.h"

#include <gtest/gtest.h>
#include <nlohmann/json_fwd.hpp>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/library_node.h"
#include "sdfg/data_flow/library_nodes/barrier_local_node.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/element.h"
#include "sdfg/function.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/if_else.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/targets/cuda/cuda.h"
#include "sdfg/types/array.h"
#include "sdfg/types/pointer.h"
#include "sdfg/types/scalar.h"
#include "sdfg/types/type.h"

using namespace sdfg;

TEST(GPUTilingTest, json_serialization) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);
    auto& seq = builder.subject().root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Array array_desc(base_desc, symbolic::integer(500));
    types::Pointer pointer_desc(array_desc);
    types::Array array_desc2(base_desc, symbolic::integer(500));
    types::Pointer pointer_desc2(array_desc2);
    types::Scalar loop_var_type(types::PrimitiveType::Int32);

    builder.add_container("A", pointer_desc);
    builder.add_container("B", pointer_desc2);
    builder.add_container("C", pointer_desc2);
    builder.add_container("i", loop_var_type);
    builder.add_container("j", loop_var_type);
    builder.add_container("k", loop_var_type);


    auto init = symbolic::zero();
    auto condition = symbolic::Le(symbolic::symbol("i"), symbolic::integer(500));
    auto update = symbolic::add(symbolic::symbol("i"), symbolic::one());

    auto& map = builder.add_map(seq, symbolic::symbol("i"), condition, init, update, cuda::ScheduleType_CUDA::create());

    auto init2 = symbolic::zero();
    auto condition2 = symbolic::Le(symbolic::symbol("j"), symbolic::integer(500));
    auto update2 = symbolic::add(symbolic::symbol("j"), symbolic::one());
    auto schedule_type = cuda::ScheduleType_CUDA::create();
    cuda::ScheduleType_CUDA::dimension(schedule_type, cuda::CUDADimension::Y);

    auto& map2 = builder.add_map(map.root(), symbolic::symbol("j"), condition2, init2, update2, schedule_type);

    auto init_loop = symbolic::zero();
    auto condition_loop = symbolic::Le(symbolic::symbol("k"), symbolic::integer(500));
    auto update_loop = symbolic::add(symbolic::symbol("k"), symbolic::one());

    auto& loop = builder.add_for(map2.root(), symbolic::symbol("k"), condition_loop, init_loop, update_loop);

    auto& block = builder.add_block(loop.root());

    auto& access_in = builder.add_access(block, "A");
    auto& access_in2 = builder.add_access(block, "B");
    auto& access_in3 = builder.add_access(block, "C");
    auto& access_out = builder.add_access(block, "C");

    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_fma, "out_", {"in1_", "in2_", "in3_"});

    builder.add_computational_memlet(block, access_in, tasklet, "in1_", {symbolic::symbol("i"), symbolic::symbol("k")});
    builder.add_computational_memlet(block, access_in2, tasklet, "in2_", {symbolic::symbol("k"), symbolic::symbol("j")});
    builder.add_computational_memlet(block, access_in3, tasklet, "in3_", {symbolic::symbol("i"), symbolic::symbol("j")});
    builder.add_computational_memlet(block, tasklet, "out_", access_out, {symbolic::symbol("i"), symbolic::symbol("j")});

    // Transformation
    transformations::GPUTiling gpu_tiling(loop, 8);

    nlohmann::json j;
    gpu_tiling.to_json(j);
    auto copy_transformation = transformations::GPUTiling::from_json(builder, j);
    EXPECT_EQ(gpu_tiling.name(), copy_transformation.name());

    // New serialization: embedding-compatible description plus legacy fields.
    ASSERT_TRUE(j.contains("transformation_type"));
    EXPECT_EQ(j["transformation_type"], "GPUTiling");

    // Embedding-style subgraph description
    ASSERT_TRUE(j.contains("subgraph"));
    ASSERT_TRUE(j["subgraph"].is_object());
    ASSERT_TRUE(j["subgraph"].contains("0"));
    const auto& node = j["subgraph"]["0"];
    ASSERT_TRUE(node.contains("element_id"));
    ASSERT_TRUE(node["element_id"].is_number_unsigned());
    EXPECT_EQ(node["element_id"].get<size_t>(), loop.element_id());
    ASSERT_TRUE(node.contains("type"));
    ASSERT_TRUE(node["type"].is_string());

    // Parameters section
    ASSERT_TRUE(j.contains("parameters"));
    ASSERT_TRUE(j["parameters"].is_object());
    ASSERT_TRUE(j["parameters"].contains("size"));
    EXPECT_EQ(j["parameters"]["size"].get<size_t>(), 8u);
}

TEST(GPUTilingTest, WithOffset) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);
    auto& seq = builder.subject().root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Array array_desc(base_desc, symbolic::integer(500));
    types::Pointer pointer_desc(array_desc);
    types::Scalar loop_var_type(types::PrimitiveType::Int32);

    builder.add_container("A", pointer_desc);
    builder.add_container("B", pointer_desc);
    builder.add_container("C", pointer_desc);
    builder.add_container("i", loop_var_type);
    builder.add_container("j", loop_var_type);
    builder.add_container("k", loop_var_type);


    auto init = symbolic::zero();
    auto condition = symbolic::Le(symbolic::symbol("i"), symbolic::integer(500));
    auto update = symbolic::add(symbolic::symbol("i"), symbolic::one());

    auto& map = builder.add_map(seq, symbolic::symbol("i"), condition, init, update, cuda::ScheduleType_CUDA::create());

    auto init2 = symbolic::zero();
    auto condition2 = symbolic::Le(symbolic::symbol("j"), symbolic::integer(500));
    auto update2 = symbolic::add(symbolic::symbol("j"), symbolic::one());
    auto schedule_type = cuda::ScheduleType_CUDA::create();
    cuda::ScheduleType_CUDA::dimension(schedule_type, cuda::CUDADimension::Y);

    auto& map2 = builder.add_map(map.root(), symbolic::symbol("j"), condition2, init2, update2, schedule_type);

    auto init_loop = symbolic::zero();
    auto condition_loop = symbolic::Lt(symbolic::symbol("k"), symbolic::integer(500));
    auto update_loop = symbolic::add(symbolic::symbol("k"), symbolic::one());

    auto& loop = builder.add_for(map2.root(), symbolic::symbol("k"), condition_loop, init_loop, update_loop);

    auto& block = builder.add_block(loop.root());

    auto& access_in = builder.add_access(block, "A");
    auto& access_in2 = builder.add_access(block, "B");
    auto& access_in3 = builder.add_access(block, "C");
    auto& access_out = builder.add_access(block, "C");

    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_fma, "out_", {"in1_", "in2_", "in3_"});

    builder.add_computational_memlet(block, access_in, tasklet, "in1_", {symbolic::symbol("i"), symbolic::symbol("k")});
    builder.add_computational_memlet(block, access_in2, tasklet, "in2_", {symbolic::symbol("k"), symbolic::symbol("j")});
    builder.add_computational_memlet(block, access_in3, tasklet, "in3_", {symbolic::symbol("i"), symbolic::symbol("j")});
    builder.add_computational_memlet(block, tasklet, "out_", access_out, {symbolic::symbol("i"), symbolic::symbol("j")});

    // Transformation
    analysis::AnalysisManager analysis_manager(builder.subject());
    transformations::GPUTiling tiling(loop, 8);

    EXPECT_TRUE(tiling.can_be_applied(builder, analysis_manager));
    tiling.apply(builder, analysis_manager);

    auto inner_loop = tiling.inner_loop();
    auto outer_loop = tiling.outer_loop();

    auto& sdfg = builder.subject();
    bool found_container = false;
    for (auto container : sdfg.containers()) {
        if (container == "__daisy_shared_A") {
            found_container = true;
            auto& type = sdfg.type(container);
            EXPECT_EQ(type.type_id(), types::TypeID::Array);
            EXPECT_EQ(type.storage_type(), types::StorageType::NV_Generic());

            auto& array_type = static_cast<const types::Array&>(type);
            EXPECT_TRUE(symbolic::eq(array_type.num_elements(), symbolic::integer(32)));

            auto& nested_type = array_type.element_type();
            EXPECT_EQ(nested_type.type_id(), types::TypeID::Array);
            EXPECT_EQ(nested_type.storage_type(), types::StorageType::NV_Shared());

            auto& nested_array_type = static_cast<const types::Array&>(nested_type);
            EXPECT_TRUE(symbolic::eq(nested_array_type.num_elements(), symbolic::integer(8)));

            auto& innermost_type = nested_array_type.element_type();
            EXPECT_EQ(innermost_type.type_id(), types::TypeID::Scalar);
            EXPECT_EQ(innermost_type.primitive_type(), types::PrimitiveType::Float);
        }
    }
    EXPECT_TRUE(found_container);

    EXPECT_EQ(outer_loop->root().size(), 7);
    auto sync1 = dynamic_cast<Block*>(&outer_loop->root().at(0).first);
    auto if_else = dynamic_cast<IfElse*>(&outer_loop->root().at(1).first);
    auto sync2 = dynamic_cast<Block*>(&outer_loop->root().at(2).first);

    EXPECT_TRUE(sync1);
    EXPECT_TRUE(if_else);
    EXPECT_TRUE(sync2);

    EXPECT_EQ(sync1->dataflow().nodes().size(), 1);
    EXPECT_EQ(sync2->dataflow().nodes().size(), 1);

    bool found1 = false;
    for (auto& node : sync1->dataflow().nodes()) {
        if (auto lib_node = dynamic_cast<data_flow::LibraryNode*>(&node)) {
            EXPECT_EQ(lib_node->code(), data_flow::LibraryNodeType_BarrierLocal);
            found1 = true;
        }
    }
    EXPECT_TRUE(found1);

    bool found2 = false;
    for (auto& node : sync2->dataflow().nodes()) {
        if (auto lib_node = dynamic_cast<data_flow::LibraryNode*>(&node)) {
            EXPECT_EQ(lib_node->code(), data_flow::LibraryNodeType_BarrierLocal);
            found2 = true;
        }
    }
    EXPECT_TRUE(found2);

    EXPECT_EQ(if_else->size(), 1);

    auto& branch = if_else->at(0).first;
    EXPECT_EQ(branch.size(), 1);

    auto inner_if_else = dynamic_cast<IfElse*>(&branch.at(0).first);
    EXPECT_TRUE(inner_if_else);

    EXPECT_EQ(inner_if_else->size(), 1);

    auto& inner_branch = inner_if_else->at(0).first;
    EXPECT_EQ(inner_branch.size(), 1);

    auto if_else_block = dynamic_cast<Block*>(&inner_branch.at(0).first);
    EXPECT_TRUE(if_else_block);

    EXPECT_EQ(if_else_block->dataflow().nodes().size(), 3);

    bool found_in, found_out, found_tasklet = false;
    for (auto& node : if_else_block->dataflow().nodes()) {
        if (auto access_node = dynamic_cast<data_flow::AccessNode*>(&node)) {
            if (access_node->data() == "A") {
                EXPECT_EQ(if_else_block->dataflow().out_degree(*access_node), 1);
                EXPECT_EQ(if_else_block->dataflow().in_degree(*access_node), 0);
                found_in = true;
            } else if (access_node->data() == "__daisy_shared_A") {
                EXPECT_EQ(if_else_block->dataflow().out_degree(*access_node), 0);
                EXPECT_EQ(if_else_block->dataflow().in_degree(*access_node), 1);
                found_out = true;
            }
        } else if (auto tasklet = dynamic_cast<data_flow::Tasklet*>(&node)) {
            found_tasklet = true;
            EXPECT_EQ(tasklet->code(), data_flow::TaskletCode::assign);
        }
    }

    EXPECT_TRUE(found_in);
    EXPECT_TRUE(found_out);
    EXPECT_TRUE(found_tasklet);

    auto sync3 = dynamic_cast<Block*>(&outer_loop->root().at(3).first);
    auto if_else2 = dynamic_cast<IfElse*>(&outer_loop->root().at(4).first);
    auto sync4 = dynamic_cast<Block*>(&outer_loop->root().at(5).first);

    EXPECT_TRUE(sync3);
    EXPECT_TRUE(if_else2);
    EXPECT_TRUE(sync4);

    EXPECT_EQ(sync3->dataflow().nodes().size(), 1);
    EXPECT_EQ(sync4->dataflow().nodes().size(), 1);
    bool found3 = false;
    for (auto& node : sync3->dataflow().nodes()) {
        if (auto lib_node = dynamic_cast<data_flow::LibraryNode*>(&node)) {
            EXPECT_EQ(lib_node->code(), data_flow::LibraryNodeType_BarrierLocal);
            found3 = true;
        }
    }
    EXPECT_TRUE(found3);

    bool found4 = false;
    for (auto& node : sync4->dataflow().nodes()) {
        if (auto lib_node = dynamic_cast<data_flow::LibraryNode*>(&node)) {
            EXPECT_EQ(lib_node->code(), data_flow::LibraryNodeType_BarrierLocal);
            found4 = true;
        }
    }
    EXPECT_TRUE(found4);

    EXPECT_EQ(if_else2->size(), 1);

    auto& branch2 = if_else2->at(0).first;
    EXPECT_EQ(branch2.size(), 1);

    auto inner_if_else2 = dynamic_cast<IfElse*>(&branch2.at(0).first);
    EXPECT_TRUE(inner_if_else2);

    EXPECT_EQ(inner_if_else2->size(), 1);

    auto& inner_branch2 = inner_if_else2->at(0).first;
    EXPECT_EQ(inner_branch2.size(), 1);

    auto if_else_block2 = dynamic_cast<Block*>(&inner_branch2.at(0).first);
    EXPECT_TRUE(if_else_block2);

    EXPECT_EQ(if_else_block2->dataflow().nodes().size(), 3);

    bool found_in2, found_out2, found_tasklet2 = false;
    for (auto& node : if_else_block2->dataflow().nodes()) {
        if (auto access_node = dynamic_cast<data_flow::AccessNode*>(&node)) {
            if (access_node->data() == "B") {
                EXPECT_EQ(if_else_block2->dataflow().out_degree(*access_node), 1);
                EXPECT_EQ(if_else_block2->dataflow().in_degree(*access_node), 0);
                found_in2 = true;
            } else if (access_node->data() == "__daisy_shared_B") {
                EXPECT_EQ(if_else_block2->dataflow().out_degree(*access_node), 0);
                EXPECT_EQ(if_else_block2->dataflow().in_degree(*access_node), 1);
                found_out2 = true;
            }
        } else if (auto tasklet = dynamic_cast<data_flow::Tasklet*>(&node)) {
            found_tasklet2 = true;
            EXPECT_EQ(tasklet->code(), data_flow::TaskletCode::assign);
        }
    }

    EXPECT_TRUE(found_in2);
    EXPECT_TRUE(found_out2);
    EXPECT_TRUE(found_tasklet2);
}
