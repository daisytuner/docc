#include "sdfg/transformations/offloading/kernel_local_storage.h"

#include <gtest/gtest.h>
#include <nlohmann/json_fwd.hpp>
#include <string>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/library_node.h"
#include "sdfg/data_flow/library_nodes/barrier_local_node.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/element.h"
#include "sdfg/function.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/for.h"
#include "sdfg/structured_control_flow/if_else.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/targets/cuda/cuda.h"
#include "sdfg/transformations/loop_tiling.h"
#include "sdfg/types/array.h"
#include "sdfg/types/pointer.h"
#include "sdfg/types/scalar.h"
#include "sdfg/types/type.h"

using namespace sdfg;

TEST(KernelLocalStorageTest, json_serialization) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);
    auto& seq = builder.subject().root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Array array_desc(base_desc, symbolic::integer(8));
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
    auto condition_loop = symbolic::Le(symbolic::symbol("k"), symbolic::integer(8));
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
    transformations::KernelLocalStorage kernel_local_storage(loop, symbolic::zero(), access_in);

    nlohmann::json j;
    kernel_local_storage.to_json(j);
    auto copy_transformation = transformations::KernelLocalStorage::from_json(builder, j);

    EXPECT_EQ(kernel_local_storage.name(), copy_transformation.name());

    // New serialization: embedding-compatible description plus legacy fields.
    ASSERT_TRUE(j.contains("transformation_type"));
    EXPECT_EQ(j["transformation_type"], "KernelLocalStorage");

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
    ASSERT_TRUE(j["parameters"].contains("offset"));
    // Offsets are serialized as string expressions
    EXPECT_EQ(j["parameters"]["offset"].get<std::string>(), "0");
}

TEST(KernelLocalStorageTest, NoOffset) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);
    auto& seq = builder.subject().root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Array array_desc(base_desc, symbolic::integer(8));
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
    auto condition_loop = symbolic::Lt(symbolic::symbol("k"), symbolic::integer(8));
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
    transformations::KernelLocalStorage kernel_local_storage(loop, symbolic::zero(), access_in);

    analysis::AnalysisManager analysis_manager(builder.subject());
    EXPECT_TRUE(kernel_local_storage.can_be_applied(builder, analysis_manager));

    kernel_local_storage.apply(builder, analysis_manager);

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

    EXPECT_EQ(map2.root().size(), 4);
    auto sync1 = dynamic_cast<Block*>(&map2.root().at(0).first);
    auto if_else = dynamic_cast<IfElse*>(&map2.root().at(1).first);
    auto sync2 = dynamic_cast<Block*>(&map2.root().at(2).first);

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
    auto if_else_block = dynamic_cast<Block*>(&branch.at(0).first);

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
}

TEST(KernelLocalStorageTest, WithOffset) {
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
    transformations::LoopTiling tiling(loop, 8);

    EXPECT_TRUE(tiling.can_be_applied(builder, analysis_manager));
    tiling.apply(builder, analysis_manager);

    auto inner_loop = tiling.inner_loop();
    auto outer_loop = tiling.outer_loop();

    transformations::KernelLocalStorage kernel_local_storage(*inner_loop, outer_loop->indvar(), access_in);

    EXPECT_TRUE(kernel_local_storage.can_be_applied(builder, analysis_manager));

    kernel_local_storage.apply(builder, analysis_manager);

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

    EXPECT_EQ(outer_loop->root().size(), 4);
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
    auto if_else_block = dynamic_cast<Block*>(&branch.at(0).first);

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
}

// Helper that builds the standard two-GPU-map + inner for-loop scaffold used by the
// negative-criterion tests.  Returns refs to the two GPU maps and the inner for-loop.
// The for-loop body is left empty; callers add accesses as needed.
static inline std::tuple<structured_control_flow::Map*, structured_control_flow::Map*, structured_control_flow::For*>
build_standard_gpu_scaffold(builder::StructuredSDFGBuilder& builder) {
    auto& seq = builder.subject().root();

    types::Scalar loop_var_type(types::PrimitiveType::Int32);
    builder.add_container("i", loop_var_type);
    builder.add_container("j", loop_var_type);
    builder.add_container("k", loop_var_type);

    // Outer CUDA-X map  (indvar i, 32 threads, condition: i <= 31)
    auto& map_x = builder.add_map(
        seq,
        symbolic::symbol("i"),
        symbolic::Le(symbolic::symbol("i"), symbolic::integer(31)),
        symbolic::zero(),
        symbolic::add(symbolic::symbol("i"), symbolic::one()),
        cuda::ScheduleType_CUDA::create()
    );

    // Inner CUDA-Y map  (indvar j, 32 threads, condition: j <= 31)
    auto schedule_y = cuda::ScheduleType_CUDA::create();
    cuda::ScheduleType_CUDA::dimension(schedule_y, cuda::CUDADimension::Y);
    auto& map_y = builder.add_map(
        map_x.root(),
        symbolic::symbol("j"),
        symbolic::Le(symbolic::symbol("j"), symbolic::integer(31)),
        symbolic::zero(),
        symbolic::add(symbolic::symbol("j"), symbolic::one()),
        schedule_y
    );

    // Sequential for-loop  k = 0 .. 7   (k < 8 → iteration count = 8)
    auto& loop = builder.add_for(
        map_y.root(),
        symbolic::symbol("k"),
        symbolic::Lt(symbolic::symbol("k"), symbolic::integer(8)),
        symbolic::zero(),
        symbolic::add(symbolic::symbol("k"), symbolic::one())
    );

    return {&map_x, &map_y, &loop};
}

// -------------------------------------------------------------------------
// 1. Criterion: must NOT be a GPU map itself
// -------------------------------------------------------------------------
TEST(KernelLocalStorageTest, CannotApply_TargetIsGPUMap) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Array array_desc(base_desc, symbolic::integer(8));
    types::Pointer pointer_desc(array_desc);
    types::Array array_desc2(base_desc, symbolic::integer(500));
    types::Pointer pointer_desc2(array_desc2);

    builder.add_container("A", pointer_desc);
    builder.add_container("C", pointer_desc2);

    auto [map_x, map_y, loop] = build_standard_gpu_scaffold(builder);

    auto& block = builder.add_block(loop->root());
    auto& access_in = builder.add_access(block, "A");
    auto& access_out = builder.add_access(block, "C");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "out_", {"in_"});
    builder.add_computational_memlet(block, access_in, tasklet, "in_", {symbolic::symbol("i"), symbolic::symbol("k")});
    builder.add_computational_memlet(block, tasklet, "out_", access_out, {symbolic::symbol("i"), symbolic::symbol("j")});

    // Passing map_y (a GPU map) as the "loop" must be rejected.
    transformations::KernelLocalStorage kls(*map_y, symbolic::zero(), access_in);
    analysis::AnalysisManager am(builder.subject());
    EXPECT_FALSE(kls.can_be_applied(builder, am));
}

// -------------------------------------------------------------------------
// 2. Criterion: must be nested inside a GPU schedule
// -------------------------------------------------------------------------
TEST(KernelLocalStorageTest, CannotApply_NotInGPUScope) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);
    auto& seq = builder.subject().root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Array array_desc(base_desc, symbolic::integer(8));
    types::Pointer pointer_desc(array_desc);
    types::Array array_desc2(base_desc, symbolic::integer(500));
    types::Pointer pointer_desc2(array_desc2);
    types::Scalar loop_var_type(types::PrimitiveType::Int32);

    builder.add_container("A", pointer_desc);
    builder.add_container("C", pointer_desc2);
    builder.add_container("i", loop_var_type);
    builder.add_container("j", loop_var_type);
    builder.add_container("k", loop_var_type);

    // For-loop placed directly in the root — no GPU ancestor at all.
    auto& loop = builder.add_for(
        seq,
        symbolic::symbol("k"),
        symbolic::Lt(symbolic::symbol("k"), symbolic::integer(8)),
        symbolic::zero(),
        symbolic::add(symbolic::symbol("k"), symbolic::one())
    );

    auto& block = builder.add_block(loop.root());
    auto& access_in = builder.add_access(block, "A");
    auto& access_out = builder.add_access(block, "C");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "out_", {"in_"});
    builder.add_computational_memlet(block, access_in, tasklet, "in_", {symbolic::symbol("i"), symbolic::symbol("k")});
    builder.add_computational_memlet(block, tasklet, "out_", access_out, {symbolic::symbol("i"), symbolic::symbol("j")});

    transformations::KernelLocalStorage kls(loop, symbolic::zero(), access_in);
    analysis::AnalysisManager am(builder.subject());
    EXPECT_FALSE(kls.can_be_applied(builder, am));
}

// -------------------------------------------------------------------------
// 3. Criterion: container's innermost element must NOT be a Pointer type
//    (i.e., the effective element must be contiguous)
// -------------------------------------------------------------------------
TEST(KernelLocalStorageTest, CannotApply_ContainerIsPointerToPointer) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Array array_desc(base_desc, symbolic::integer(8));
    // Pointer → Pointer → Array: after peeling the outer pointer, the remaining
    // type is still a Pointer, which fails the contiguity check.
    types::Pointer inner_ptr(array_desc);
    types::Pointer outer_ptr(types::StorageType::NV_Generic(), 8, "", inner_ptr);

    builder.add_container("A", outer_ptr);

    auto [map_x, map_y, loop] = build_standard_gpu_scaffold(builder);

    data_flow::AccessNode& access_in = builder.add_access(builder.add_block(loop->root()), "A");
    transformations::KernelLocalStorage kls(*loop, symbolic::zero(), access_in);
    analysis::AnalysisManager am(builder.subject());
    EXPECT_FALSE(kls.can_be_applied(builder, am));
}

// -------------------------------------------------------------------------
// 4. Criterion: iteration count must be a known integer
//    Using ≤ instead of <: get_iteration_count returns null for LessThan.
// -------------------------------------------------------------------------
TEST(KernelLocalStorageTest, CannotApply_UnknownIterationCount) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Array array_desc(base_desc, symbolic::integer(8));
    types::Pointer pointer_desc(array_desc);
    types::Array array_desc2(base_desc, symbolic::integer(500));
    types::Pointer pointer_desc2(array_desc2);

    builder.add_container("A", pointer_desc);
    builder.add_container("C", pointer_desc2);

    auto& seq = builder.subject().root();
    types::Scalar loop_var_type(types::PrimitiveType::Int32);
    builder.add_container("i", loop_var_type);
    builder.add_container("j", loop_var_type);
    builder.add_container("k", loop_var_type);

    auto& map_x = builder.add_map(
        seq,
        symbolic::symbol("i"),
        symbolic::Le(symbolic::symbol("i"), symbolic::integer(31)),
        symbolic::zero(),
        symbolic::add(symbolic::symbol("i"), symbolic::one()),
        cuda::ScheduleType_CUDA::create()
    );
    auto schedule_y = cuda::ScheduleType_CUDA::create();
    cuda::ScheduleType_CUDA::dimension(schedule_y, cuda::CUDADimension::Y);
    auto& map_y = builder.add_map(
        map_x.root(),
        symbolic::symbol("j"),
        symbolic::Le(symbolic::symbol("j"), symbolic::integer(31)),
        symbolic::zero(),
        symbolic::add(symbolic::symbol("j"), symbolic::one()),
        schedule_y
    );

    auto& loop = builder.add_for(
        map_y.root(),
        symbolic::symbol("k"),
        symbolic::Le(symbolic::symbol("k"), symbolic::symbol("j")),
        symbolic::zero(),
        symbolic::add(symbolic::symbol("k"), symbolic::one())
    );

    auto& block = builder.add_block(loop.root());
    auto& access_in = builder.add_access(block, "A");
    auto& access_out = builder.add_access(block, "C");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "out_", {"in_"});
    builder.add_computational_memlet(block, access_in, tasklet, "in_", {symbolic::symbol("i"), symbolic::symbol("k")});
    builder.add_computational_memlet(block, tasklet, "out_", access_out, {symbolic::symbol("i"), symbolic::symbol("j")});

    std::string container = "A";
    transformations::KernelLocalStorage kls(loop, symbolic::zero(), access_in);
    analysis::AnalysisManager am(builder.subject());
    EXPECT_FALSE(kls.can_be_applied(builder, am));
}

// -------------------------------------------------------------------------
// 5. Criterion: container must be read-only inside the loop
//    (here A is also written → rejected)
// -------------------------------------------------------------------------
TEST(KernelLocalStorageTest, CannotApply_ContainerIsWritten) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Array array_desc(base_desc, symbolic::integer(8));
    types::Pointer pointer_desc(array_desc);
    types::Array array_desc2(base_desc, symbolic::integer(500));
    types::Pointer pointer_desc2(array_desc2);

    builder.add_container("A", pointer_desc);
    builder.add_container("C", pointer_desc2);

    auto [map_x, map_y, loop] = build_standard_gpu_scaffold(builder);

    auto& block = builder.add_block(loop->root());

    // Read A[i, k] → one input connector
    auto& access_in = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "out_", {"in_"});
    builder.add_computational_memlet(block, access_in, tasklet, "in_", {symbolic::symbol("i"), symbolic::symbol("k")});

    // Write back to A[i, j] → A is both read and written → rejected.
    auto& access_write = builder.add_access(block, "A");
    builder
        .add_computational_memlet(block, tasklet, "out_", access_write, {symbolic::symbol("i"), symbolic::symbol("j")});

    transformations::KernelLocalStorage kls(*loop, symbolic::zero(), access_in);
    analysis::AnalysisManager am(builder.subject());
    EXPECT_FALSE(kls.can_be_applied(builder, am));
}

// -------------------------------------------------------------------------
// 6. Criterion: container must have at least one read inside the loop
// -------------------------------------------------------------------------
TEST(KernelLocalStorageTest, CannotApply_ContainerNotRead) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Array array_desc(base_desc, symbolic::integer(8));
    types::Pointer pointer_desc(array_desc);
    types::Array array_desc2(base_desc, symbolic::integer(500));
    types::Pointer pointer_desc2(array_desc2);

    builder.add_container("A", pointer_desc);
    builder.add_container("B", pointer_desc2);
    builder.add_container("C", pointer_desc2);

    auto [map_x, map_y, loop] = build_standard_gpu_scaffold(builder);

    // Place an access to A outside the loop (before it, in the map body).
    auto& outer_block = builder.add_block(map_y->root());
    auto& access_A_outside = builder.add_access(outer_block, "A");
    auto& access_C_outside = builder.add_access(outer_block, "C");
    auto& tasklet_outside = builder.add_tasklet(outer_block, data_flow::TaskletCode::assign, "out_", {"in_"});
    builder.add_computational_memlet(
        outer_block, access_A_outside, tasklet_outside, "in_", {symbolic::symbol("i"), symbolic::symbol("j")}
    );
    builder.add_computational_memlet(
        outer_block, tasklet_outside, "out_", access_C_outside, {symbolic::symbol("i"), symbolic::symbol("j")}
    );

    // Inside the loop, only B and C are accessed; A is never read here.
    auto& block = builder.add_block(loop->root());
    auto& access_B = builder.add_access(block, "B");
    auto& access_C = builder.add_access(block, "C");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "out_", {"in_"});
    builder.add_computational_memlet(block, access_B, tasklet, "in_", {symbolic::symbol("i"), symbolic::symbol("k")});
    builder.add_computational_memlet(block, tasklet, "out_", access_C, {symbolic::symbol("i"), symbolic::symbol("j")});

    // Pass the outside access node — A has no reads in the loop body.
    transformations::KernelLocalStorage kls(*loop, symbolic::zero(), access_A_outside);
    analysis::AnalysisManager am(builder.subject());
    EXPECT_FALSE(kls.can_be_applied(builder, am));
}

// -------------------------------------------------------------------------
// 7. Criterion: exactly one read of the container (limitation)
//    Two separate reads of A → rejected.
// -------------------------------------------------------------------------
TEST(KernelLocalStorageTest, CannotApply_MultipleReads) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Array array_desc(base_desc, symbolic::integer(8));
    types::Pointer pointer_desc(array_desc);
    types::Array array_desc2(base_desc, symbolic::integer(500));
    types::Pointer pointer_desc2(array_desc2);

    builder.add_container("A", pointer_desc);
    builder.add_container("C", pointer_desc2);

    auto [map_x, map_y, loop] = build_standard_gpu_scaffold(builder);

    auto& block = builder.add_block(loop->root());

    // First read: A[i, k]
    auto& access_in1 = builder.add_access(block, "A");
    auto& tasklet1 = builder.add_tasklet(block, data_flow::TaskletCode::assign, "out_", {"in_"});
    builder.add_computational_memlet(block, access_in1, tasklet1, "in_", {symbolic::symbol("i"), symbolic::symbol("k")});
    auto& access_out1 = builder.add_access(block, "C");
    builder
        .add_computational_memlet(block, tasklet1, "out_", access_out1, {symbolic::symbol("i"), symbolic::symbol("j")});

    // Second read: A[j, k] — creates a second read user for A.
    auto& access_in2 = builder.add_access(block, "A");
    auto& tasklet2 = builder.add_tasklet(block, data_flow::TaskletCode::assign, "out_", {"in_"});
    builder.add_computational_memlet(block, access_in2, tasklet2, "in_", {symbolic::symbol("j"), symbolic::symbol("k")});
    auto& access_out2 = builder.add_access(block, "C");
    builder
        .add_computational_memlet(block, tasklet2, "out_", access_out2, {symbolic::symbol("i"), symbolic::symbol("j")});

    auto& access_in = builder.add_access(block, "A");
    transformations::KernelLocalStorage kls(*loop, symbolic::zero(), access_in);
    analysis::AnalysisManager am(builder.subject());
    EXPECT_FALSE(kls.can_be_applied(builder, am));
}

// -------------------------------------------------------------------------
// 8. Criterion: the inner loop's induction variable must appear in the access
//    A[i, j] does not involve k → rejected.
// -------------------------------------------------------------------------
TEST(KernelLocalStorageTest, CannotApply_InnerIndvarNotUsed) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Array array_desc(base_desc, symbolic::integer(8));
    types::Pointer pointer_desc(array_desc);
    types::Array array_desc2(base_desc, symbolic::integer(500));
    types::Pointer pointer_desc2(array_desc2);

    builder.add_container("A", pointer_desc);
    builder.add_container("C", pointer_desc2);

    auto [map_x, map_y, loop] = build_standard_gpu_scaffold(builder);

    auto& block = builder.add_block(loop->root());
    auto& access_in = builder.add_access(block, "A");
    auto& access_out = builder.add_access(block, "C");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "out_", {"in_"});
    // Access A[i, j] — neither dimension uses the inner loop variable k.
    builder.add_computational_memlet(block, access_in, tasklet, "in_", {symbolic::symbol("i"), symbolic::symbol("j")});
    builder.add_computational_memlet(block, tasklet, "out_", access_out, {symbolic::symbol("i"), symbolic::symbol("j")});

    transformations::KernelLocalStorage kls(*loop, symbolic::zero(), access_in);
    analysis::AnalysisManager am(builder.subject());
    EXPECT_FALSE(kls.can_be_applied(builder, am));
}

// -------------------------------------------------------------------------
// 9. Criterion: at least two GPU dimensions must be present in ancestor maps
//    Only one GPU map (X) → indvars.size() == 1 → rejected.
// -------------------------------------------------------------------------
TEST(KernelLocalStorageTest, CannotApply_OnlyOneGPUDimension) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);
    auto& seq = builder.subject().root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Array array_desc(base_desc, symbolic::integer(8));
    types::Pointer pointer_desc(array_desc);
    types::Array array_desc2(base_desc, symbolic::integer(500));
    types::Pointer pointer_desc2(array_desc2);
    types::Scalar loop_var_type(types::PrimitiveType::Int32);

    builder.add_container("A", pointer_desc);
    builder.add_container("C", pointer_desc2);
    builder.add_container("i", loop_var_type);
    builder.add_container("k", loop_var_type);

    // Only a single CUDA-X map — no Y or Z dimension available.
    auto& map_x = builder.add_map(
        seq,
        symbolic::symbol("i"),
        symbolic::Le(symbolic::symbol("i"), symbolic::integer(31)),
        symbolic::zero(),
        symbolic::add(symbolic::symbol("i"), symbolic::one()),
        cuda::ScheduleType_CUDA::create()
    );

    auto& loop = builder.add_for(
        map_x.root(),
        symbolic::symbol("k"),
        symbolic::Lt(symbolic::symbol("k"), symbolic::integer(8)),
        symbolic::zero(),
        symbolic::add(symbolic::symbol("k"), symbolic::one())
    );

    auto& block = builder.add_block(loop.root());
    auto& access_in = builder.add_access(block, "A");
    auto& access_out = builder.add_access(block, "C");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "out_", {"in_"});
    builder.add_computational_memlet(block, access_in, tasklet, "in_", {symbolic::symbol("i"), symbolic::symbol("k")});
    builder.add_computational_memlet(block, tasklet, "out_", access_out, {symbolic::symbol("i"), symbolic::symbol("k")});

    transformations::KernelLocalStorage kls(loop, symbolic::zero(), access_in);
    analysis::AnalysisManager am(builder.subject());
    EXPECT_FALSE(kls.can_be_applied(builder, am));
}

// -------------------------------------------------------------------------
// 10. Criterion: symbols appearing in the subset of the target container read
//     must not be written inside the loop body.
//     Access A[n, k] where container n is also written → rejected.
// -------------------------------------------------------------------------
TEST(KernelLocalStorageTest, CannotApply_SubsetVariableWritten) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Array array_desc(base_desc, symbolic::integer(8));
    types::Pointer pointer_desc(array_desc);
    types::Array array_desc2(base_desc, symbolic::integer(500));
    types::Pointer pointer_desc2(array_desc2);

    builder.add_container("A", pointer_desc);
    builder.add_container("C", pointer_desc2);
    // n is an integer index used inside A's access subset and also written in the loop.
    builder.add_container("n", types::Scalar(types::PrimitiveType::Int32));

    auto [map_x, map_y, loop] = build_standard_gpu_scaffold(builder);

    auto& block = builder.add_block(loop->root());

    // Read A[n, k] — n appears in the subset.
    auto& access_A = builder.add_access(block, "A");
    auto& tasklet_use = builder.add_tasklet(block, data_flow::TaskletCode::assign, "out_", {"in_"});
    builder
        .add_computational_memlet(block, access_A, tasklet_use, "in_", {symbolic::symbol("n"), symbolic::symbol("k")});
    auto& access_C_out = builder.add_access(block, "C");
    builder.add_computational_memlet(
        block, tasklet_use, "out_", access_C_out, {symbolic::symbol("i"), symbolic::symbol("j")}
    );

    // Write to n inside the loop body — this violates the criterion.
    auto& access_n_in = builder.add_access(block, "n");
    auto& access_n_out = builder.add_access(block, "n");
    auto& tasklet_n = builder.add_tasklet(block, data_flow::TaskletCode::assign, "out_", {"in_"});
    builder.add_computational_memlet(block, access_n_in, tasklet_n, "in_", {});
    builder.add_computational_memlet(block, tasklet_n, "out_", access_n_out, {});

    transformations::KernelLocalStorage kls(*loop, symbolic::zero(), access_A);
    analysis::AnalysisManager am(builder.subject());
    EXPECT_FALSE(kls.can_be_applied(builder, am));
}

// -------------------------------------------------------------------------
// 10b. Criterion: subset of the container access must not depend on a symbol
//      that is written inside the loop (nested-loop indvar variant).
//      A nested for-loop with indvar m is placed inside the target loop k.
//      Accessing A[i, m] makes the read subset depend on m; since the nested
//      for-loop writes m on each iteration, the transformation is rejected.
// -------------------------------------------------------------------------
TEST(KernelLocalStorageTest, CannotApply_SubsetDependsOnNestedLoopIndvar) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Array array_desc(base_desc, symbolic::integer(8));
    types::Pointer pointer_desc(array_desc);
    types::Array array_desc2(base_desc, symbolic::integer(500));
    types::Pointer pointer_desc2(array_desc2);

    builder.add_container("A", pointer_desc);
    builder.add_container("C", pointer_desc2);
    builder.add_container("m", types::Scalar(types::PrimitiveType::Int32));

    auto [map_x, map_y, loop] = build_standard_gpu_scaffold(builder);

    // Nested for-loop inside the target loop k: m = 0 .. 7
    auto& inner_loop = builder.add_for(
        loop->root(),
        symbolic::symbol("m"),
        symbolic::Lt(symbolic::symbol("m"), symbolic::integer(8)),
        symbolic::zero(),
        symbolic::add(symbolic::symbol("m"), symbolic::one())
    );

    auto& block = builder.add_block(inner_loop.root());
    auto& access_in = builder.add_access(block, "A");
    auto& access_out = builder.add_access(block, "C");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "out_", {"in_"});
    // A[i, m]: the subset depends on m, the nested loop's indvar.
    // The nested for-loop writes m on every iteration, so the
    // subset-depends-on-written-variable criterion rejects the transformation.
    builder.add_computational_memlet(block, access_in, tasklet, "in_", {symbolic::symbol("i"), symbolic::symbol("m")});
    builder.add_computational_memlet(block, tasklet, "out_", access_out, {symbolic::symbol("i"), symbolic::symbol("j")});

    transformations::KernelLocalStorage kls(*loop, symbolic::zero(), access_in);
    analysis::AnalysisManager am(builder.subject());
    EXPECT_FALSE(kls.can_be_applied(builder, am));
}

// -------------------------------------------------------------------------
// 11. Criterion: at least one GPU dimension must remain free (not appear in
//     the access subset and hence available for tiling).
//     A[i+j, k] uses both X-indvar (i) and Y-indvar (j) in its subset,
//     leaving no free dimension → rejected.
// -------------------------------------------------------------------------
TEST(KernelLocalStorageTest, CannotApply_NoFreeDimension) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Array array_desc(base_desc, symbolic::integer(8));
    types::Pointer pointer_desc(array_desc);
    types::Array array_desc2(base_desc, symbolic::integer(500));
    types::Pointer pointer_desc2(array_desc2);

    builder.add_container("A", pointer_desc);
    builder.add_container("C", pointer_desc2);

    auto [map_x, map_y, loop] = build_standard_gpu_scaffold(builder);

    auto& block = builder.add_block(loop->root());
    auto& access_in = builder.add_access(block, "A");
    auto& access_out = builder.add_access(block, "C");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "out_", {"in_"});
    // Access A[i+j, k]: both the X-dimension indvar (i) and Y-dimension indvar (j)
    // appear in the first subset element, so neither dimension is free.
    builder.add_computational_memlet(
        block,
        access_in,
        tasklet,
        "in_",
        {symbolic::add(symbolic::symbol("i"), symbolic::symbol("j")), symbolic::symbol("k")}
    );
    builder.add_computational_memlet(block, tasklet, "out_", access_out, {symbolic::symbol("i"), symbolic::symbol("j")});

    transformations::KernelLocalStorage kls(*loop, symbolic::zero(), access_in);
    analysis::AnalysisManager am(builder.subject());
    EXPECT_FALSE(kls.can_be_applied(builder, am));
}
