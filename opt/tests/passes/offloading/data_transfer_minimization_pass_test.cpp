#include "sdfg/passes/offloading/data_transfer_minimization_pass.h"

#include <gtest/gtest.h>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/codegen/code_generator.h"
#include "sdfg/codegen/code_generators/cpp_code_generator.h"
#include "sdfg/data_flow/library_nodes/call_node.h"
#include "sdfg/data_flow/library_nodes/stdlib/free.h"
#include "sdfg/element.h"
#include "sdfg/function.h"
#include "sdfg/passes/dataflow/dead_data_elimination.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/targets/cuda/cuda.h"
#include "sdfg/targets/cuda/cuda_data_offloading_node.h"
#include "sdfg/targets/offloading/data_offloading_node.h"
#include "sdfg_debug_dump.h"

using namespace sdfg;

TEST(DataTransferMinimizationPassTest, SingleTransferTest) {
    sdfg::builder::StructuredSDFGBuilder builder("dot", sdfg::FunctionType_CPU);
    sdfg::analysis::AnalysisManager analysis_manager(builder.subject());

    auto& root = builder.subject().root();

    // Add containers
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("A", desc, true);
    builder.add_container("__daisy_offload_A", desc);

    auto [block, memcpy_nod] = offloading::add_offloading_block<cuda::CUDADataOffloadingNode>(
        builder,
        root,
        "A",
        "__daisy_offload_A",
        offloading::DataTransferDirection::D2H,
        offloading::BufferLifecycle::NO_CHANGE,
        desc,
        {},
        symbolic::integer(400),
        symbolic::integer(0)
    );

    dump_sdfg(builder.subject(), "0.init");

    passes::DataTransferMinimizationPass pass;
    EXPECT_FALSE(pass.run(builder, analysis_manager));
};

TEST(DataTransferMinimizationPassTest, MultiMapTest) {
    sdfg::builder::StructuredSDFGBuilder builder("dot", sdfg::FunctionType_CPU);
    sdfg::analysis::AnalysisManager analysis_manager(builder.subject());

    auto& root = builder.subject().root();

    // Add containers
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("A", desc, true);
    builder.add_container("__daisy_offload_A", desc);

    auto [block_d2h, memcpy_d2h] = offloading::add_offloading_block<cuda::CUDADataOffloadingNode>(
        builder,
        root,
        "A",
        "__daisy_offload_A",
        offloading::DataTransferDirection::D2H,
        offloading::BufferLifecycle::FREE,
        desc,
        {},
        symbolic::integer(400),
        symbolic::integer(0)
    );

    auto [block_h2d, memcpy_h2d] = offloading::add_offloading_block<cuda::CUDADataOffloadingNode>(
        builder,
        root,
        "A",
        "__daisy_offload_A",
        offloading::DataTransferDirection::H2D,
        offloading::BufferLifecycle::ALLOC,
        desc,
        {},
        symbolic::integer(400),
        symbolic::integer(0)
    );

    dump_sdfg(builder.subject(), "0-before");

    passes::DataTransferMinimizationPass pass;
    EXPECT_TRUE(pass.run(builder, analysis_manager));

    dump_sdfg(builder.subject(), "1-after");

    EXPECT_EQ(block_d2h.dataflow().nodes().size(), 3);
    EXPECT_EQ(
        dynamic_cast<const cuda::CUDADataOffloadingNode&>(memcpy_d2h).transfer_direction(),
        offloading::DataTransferDirection::D2H
    );
    EXPECT_EQ(
        dynamic_cast<const cuda::CUDADataOffloadingNode&>(memcpy_d2h).buffer_lifecycle(),
        offloading::BufferLifecycle::NO_CHANGE
    );
    EXPECT_EQ(block_h2d.dataflow().nodes().size(), 0);
};

TEST(DataTransferMinimizationPassTest, MultiMapWithLatterUseTest) {
    sdfg::builder::StructuredSDFGBuilder builder("dot", sdfg::FunctionType_CPU);
    sdfg::analysis::AnalysisManager analysis_manager(builder.subject());

    auto& root = builder.subject().root();

    // Add containers
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("A", desc, true);
    builder.add_container("B", desc, true);
    builder.add_container("__daisy_offload_A", desc);

    auto [block, memcpy_node] = offloading::add_offloading_block<cuda::CUDADataOffloadingNode>(
        builder,
        root,
        "A",
        "__daisy_offload_A",
        offloading::DataTransferDirection::D2H,
        offloading::BufferLifecycle::FREE,
        desc,
        {},
        symbolic::integer(400),
        symbolic::integer(0)
    );

    auto [block2, memcpy_node2] = offloading::add_offloading_block<cuda::CUDADataOffloadingNode>(
        builder,
        root,
        "A",
        "__daisy_offload_A",
        offloading::DataTransferDirection::H2D,
        offloading::BufferLifecycle::ALLOC,
        desc,
        {},
        symbolic::integer(400),
        symbolic::integer(0)
    );

    // Add another use of C after the second map
    auto& block3 = builder.add_block(root);
    auto& C3 = builder.add_access(block3, "A");
    auto& B = builder.add_access(block3, "B");
    auto& tasklet3 = builder.add_tasklet(block3, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block3, C3, tasklet3, "_in", {symbolic::zero()});
    builder.add_computational_memlet(block3, tasklet3, "_out", B, {symbolic::zero()});

    dump_sdfg(builder.subject(), "0-before");

    passes::DataTransferMinimizationPass pass;
    EXPECT_TRUE(pass.run(builder, analysis_manager));

    dump_sdfg(builder.subject(), "1-after");

    // Check that there is exactly two H2D and one D2H transfer for C
    int h2d_count = 0;
    int d2h_count = 0;
    for (int i = 0; i < root.size(); i++) {
        auto& cf_node = root.at(i);
        if (auto* block = dyn_cast<structured_control_flow::Block*>(&cf_node)) {
            for (auto& node : block->dataflow().nodes()) {
                if (auto* cuda_offload = dynamic_cast<cuda::CUDADataOffloadingNode*>(&node)) {
                    if (cuda_offload->is_h2d()) {
                        h2d_count++;
                    } else if (cuda_offload->is_d2h()) {
                        d2h_count++;
                    }
                }
            }
        }
    }
    EXPECT_EQ(h2d_count, 0);
    EXPECT_EQ(d2h_count, 1);
};

TEST(DataTransferMinimizationPassTest, UselessMallocTest) {
    sdfg::builder::StructuredSDFGBuilder builder("dot", sdfg::FunctionType_CPU);
    sdfg::analysis::AnalysisManager analysis_manager(builder.subject());

    auto& root = builder.subject().root();

    // Add containers
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("A", desc, true);
    builder.add_container("__daisy_offload_A", desc);

    auto& block_malloc = builder.add_block(root);
    auto& access_node_malloc = builder.add_access(block_malloc, "A");

    auto& malloc_node = builder.add_library_node<stdlib::MallocNode>(block_malloc, DebugInfo(), symbolic::integer(400));

    builder.add_computational_memlet(block_malloc, malloc_node, "_ret", access_node_malloc, {}, desc);


    auto [block_h2d, memcpy_node_h2d] = offloading::add_offloading_block<cuda::CUDADataOffloadingNode>(
        builder,
        root,
        "A",
        "__daisy_offload_A",
        offloading::DataTransferDirection::H2D,
        offloading::BufferLifecycle::ALLOC,
        desc,
        {},
        symbolic::integer(400),
        symbolic::integer(0)
    );

    auto [block_d2h, memcpy_node_d2h] = offloading::add_offloading_block<cuda::CUDADataOffloadingNode>(
        builder,
        root,
        "A",
        "__daisy_offload_A",
        offloading::DataTransferDirection::D2H,
        offloading::BufferLifecycle::FREE,
        desc,
        {},
        symbolic::integer(400),
        symbolic::integer(0)
    );

    dump_sdfg(builder.subject(), "0-before");

    passes::DataTransferMinimizationPass pass;
    EXPECT_TRUE(pass.run(builder, analysis_manager));

    dump_sdfg(builder.subject(), "1-after");

    EXPECT_EQ(block_malloc.dataflow().nodes().size(), 2);
    EXPECT_EQ(block_h2d.dataflow().nodes().size(), 2);
    EXPECT_EQ(
        dynamic_cast<cuda::CUDADataOffloadingNode&>(memcpy_node_h2d).transfer_direction(),
        offloading::DataTransferDirection::NONE
    );
    EXPECT_EQ(
        dynamic_cast<cuda::CUDADataOffloadingNode&>(memcpy_node_h2d).buffer_lifecycle(),
        offloading::BufferLifecycle::ALLOC
    );
    EXPECT_EQ(block_d2h.dataflow().nodes().size(), 3);

    auto instrumentation_plan = codegen::InstrumentationPlan::none(builder.subject());
    auto arg_capture_plan = codegen::ArgCapturePlan::none(builder.subject());
    codegen::CPPCodeGenerator
        code_generator(builder.subject(), analysis_manager, *instrumentation_plan, *arg_capture_plan);
    code_generator.generate();
};

TEST(DataTransferMinimizationPassTest, ReadOnlyDataReuseTest) {
    sdfg::builder::StructuredSDFGBuilder builder("dot", sdfg::FunctionType_CPU);
    sdfg::analysis::AnalysisManager analysis_manager(builder.subject());

    auto& root = builder.subject().root();

    // Add containers
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    types::Scalar void_type(types::Void);
    builder.add_container("A", desc, true);
    builder.add_container("__daisy_offload_A", desc);

    auto& in_type = builder.subject().type("A");
    auto& out_type = builder.subject().type("A");

    // -- Malloc
    auto& block_malloc = builder.add_block(root);
    auto& access_node_malloc = builder.add_access(block_malloc, "A");

    auto& malloc_node = builder.add_library_node<stdlib::MallocNode>(block_malloc, DebugInfo(), symbolic::integer(400));

    builder.add_computational_memlet(block_malloc, malloc_node, "_ret", access_node_malloc, {}, desc);

    // -- host-dirty data
    auto& block_dirty = builder.add_block(root);
    auto& access_node_dirty = builder.add_access(block_dirty, "A");
    std::vector<std::string> blackblox_ins = {"_ptr"};
    std::vector<std::string> blackbox_outs = {};
    types::Function func_type(void_type);
    func_type.add_param(desc);
    builder.add_external("blackbox", func_type, LinkageType_External);
    auto& dirty_blackbox =
        builder
            .add_library_node<data_flow::CallNode>(block_dirty, DebugInfo(), "blackbox", blackbox_outs, blackblox_ins);
    builder.add_computational_memlet(block_dirty, access_node_dirty, dirty_blackbox, "_ptr", {}, desc);


    // --- Init H2D
    auto [block_h2d, memcpy_node_h2d] = offloading::add_offloading_block<cuda::CUDADataOffloadingNode>(
        builder,
        root,
        "A",
        "__daisy_offload_A",
        offloading::DataTransferDirection::H2D,
        offloading::BufferLifecycle::ALLOC,
        desc,
        {},
        symbolic::integer(400),
        symbolic::integer(0)
    );

    // --- Init D2H
    auto [block_d2h, memcpy_node_d2h] = offloading::add_offloading_block<cuda::CUDADataOffloadingNode>(
        builder,
        root,
        "A",
        "__daisy_offload_A",
        offloading::DataTransferDirection::D2H,
        offloading::BufferLifecycle::FREE,
        desc,
        {},
        symbolic::integer(400),
        symbolic::integer(0)
    );

    // ------- reuse H2D

    auto [block_h2d_reuse, memcpy_node_reuse_h2d] = offloading::add_offloading_block<cuda::CUDADataOffloadingNode>(
        builder,
        root,
        "A",
        "__daisy_offload_A",
        offloading::DataTransferDirection::H2D,
        offloading::BufferLifecycle::ALLOC,
        desc,
        {},
        symbolic::integer(400),
        symbolic::integer(0)
    );

    // --- reuse D2H
    auto [block_d2h_reuse, memcpy_node_reuse_d2h] = offloading::add_offloading_block<cuda::CUDADataOffloadingNode>(
        builder,
        root,
        "A",
        "__daisy_offload_A",
        offloading::DataTransferDirection::NONE,
        offloading::BufferLifecycle::FREE,
        desc,
        {},
        symbolic::integer(400),
        symbolic::integer(0)
    );

    // ------- reuse2 H2D

    auto [block_h2d_reuse2, memcpy_node_reuse2_h2d] = offloading::add_offloading_block<cuda::CUDADataOffloadingNode>(
        builder,
        root,
        "A",
        "__daisy_offload_A",
        offloading::DataTransferDirection::H2D,
        offloading::BufferLifecycle::ALLOC,
        desc,
        {},
        symbolic::integer(400),
        symbolic::integer(0)
    );

    // --- reuse2 D2H
    auto [block_d2h_reuse2, memcpy_node_reuse2_d2h] = offloading::add_offloading_block<cuda::CUDADataOffloadingNode>(
        builder,
        root,
        "A",
        "__daisy_offload_A",
        offloading::DataTransferDirection::D2H,
        offloading::BufferLifecycle::FREE,
        desc,
        {},
        symbolic::integer(400),
        symbolic::integer(0)
    );

    // --- Free
    auto& block_free = builder.add_block(root);
    auto& access_node_free_in = builder.add_access(block_free, "A");

    auto& free_node = builder.add_library_node<stdlib::FreeNode>(block_free, DebugInfo());

    builder.add_computational_memlet(block_free, access_node_free_in, free_node, "_ptr", {}, desc);

    dump_sdfg(builder.subject(), "0-before");

    passes::DataTransferMinimizationPass pass;
    EXPECT_TRUE(pass.run(builder, analysis_manager));

    dump_sdfg(builder.subject(), "1-after");

    EXPECT_EQ(block_malloc.dataflow().nodes().size(), 2);
    EXPECT_EQ(block_h2d.dataflow().nodes().size(), 3);
    EXPECT_EQ(
        dynamic_cast<cuda::CUDADataOffloadingNode&>(memcpy_node_h2d).transfer_direction(),
        offloading::DataTransferDirection::H2D
    );
    EXPECT_EQ(
        dynamic_cast<cuda::CUDADataOffloadingNode&>(memcpy_node_h2d).buffer_lifecycle(),
        offloading::BufferLifecycle::ALLOC
    );
    EXPECT_EQ(block_d2h.dataflow().nodes().size(), 0);
    EXPECT_EQ(block_h2d_reuse.dataflow().nodes().size(), 0);
    EXPECT_EQ(block_d2h_reuse.dataflow().nodes().size(), 0);
    EXPECT_EQ(block_h2d_reuse2.dataflow().nodes().size(), 0);
    EXPECT_EQ(block_d2h_reuse2.dataflow().nodes().size(), 2);

    passes::DeadDataElimination dde_pass(false);
    dde_pass.run(builder, analysis_manager);

    EXPECT_EQ(block_malloc.dataflow().nodes().size(), 2);
    EXPECT_EQ(block_h2d.dataflow().nodes().size(), 3);
    EXPECT_EQ(
        dynamic_cast<cuda::CUDADataOffloadingNode&>(memcpy_node_h2d).transfer_direction(),
        offloading::DataTransferDirection::H2D
    );
    EXPECT_EQ(
        dynamic_cast<cuda::CUDADataOffloadingNode&>(memcpy_node_h2d).buffer_lifecycle(),
        offloading::BufferLifecycle::ALLOC
    );
    EXPECT_EQ(block_d2h.dataflow().nodes().size(), 0);
    EXPECT_EQ(block_h2d_reuse.dataflow().nodes().size(), 0);
    EXPECT_EQ(block_d2h_reuse.dataflow().nodes().size(), 0);
    EXPECT_EQ(block_h2d_reuse2.dataflow().nodes().size(), 0);
    EXPECT_EQ(block_d2h_reuse2.dataflow().nodes().size(), 2);

    dump_sdfg(builder.subject(), "2-cleanup");
};

TEST(DataTransferMinimizationPassTest, ReadOnlyDataPureDeviceTest) {
    sdfg::builder::StructuredSDFGBuilder builder("dot", sdfg::FunctionType_CPU);
    sdfg::analysis::AnalysisManager analysis_manager(builder.subject());

    auto& root = builder.subject().root();

    // Add containers
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    types::Scalar void_type(types::Void);
    builder.add_container("A", desc);
    builder.add_container("__daisy_offload_A", desc);

    auto& in_type = builder.subject().type("A");
    auto& out_type = builder.subject().type("A");

    // -- Malloc
    auto& block_malloc = builder.add_block(root);
    auto& access_node_malloc = builder.add_access(block_malloc, "A");

    auto& malloc_node = builder.add_library_node<stdlib::MallocNode>(block_malloc, DebugInfo(), symbolic::integer(400));

    builder.add_computational_memlet(block_malloc, malloc_node, "_ret", access_node_malloc, {}, desc);


    // --- Init H2D
    auto [block_h2d, memcpy_node_h2d] = offloading::add_offloading_block<cuda::CUDADataOffloadingNode>(
        builder,
        root,
        "A",
        "__daisy_offload_A",
        offloading::DataTransferDirection::H2D,
        offloading::BufferLifecycle::ALLOC,
        desc,
        {},
        symbolic::integer(400),
        symbolic::integer(0)
    );

    // --- Init D2H
    auto [block_d2h, memcpy_node_d2h] = offloading::add_offloading_block<cuda::CUDADataOffloadingNode>(
        builder,
        root,
        "A",
        "__daisy_offload_A",
        offloading::DataTransferDirection::D2H,
        offloading::BufferLifecycle::FREE,
        desc,
        {},
        symbolic::integer(400),
        symbolic::integer(0)
    );

    // ------- reuse H2D

    auto [block_h2d_reuse, memcpy_node_reuse_h2d] = offloading::add_offloading_block<cuda::CUDADataOffloadingNode>(
        builder,
        root,
        "A",
        "__daisy_offload_A",
        offloading::DataTransferDirection::H2D,
        offloading::BufferLifecycle::ALLOC,
        desc,
        {},
        symbolic::integer(400),
        symbolic::integer(0)
    );

    // --- reuse D2H
    auto [block_d2h_reuse, memcpy_node_reuse_d2h] = offloading::add_offloading_block<cuda::CUDADataOffloadingNode>(
        builder,
        root,
        "A",
        "__daisy_offload_A",
        offloading::DataTransferDirection::NONE,
        offloading::BufferLifecycle::FREE,
        desc,
        {},
        symbolic::integer(400),
        symbolic::integer(0)
    );

    // ------- reuse2 H2D

    auto [block_h2d_reuse2, memcpy_node_reuse2_h2d] = offloading::add_offloading_block<cuda::CUDADataOffloadingNode>(
        builder,
        root,
        "A",
        "__daisy_offload_A",
        offloading::DataTransferDirection::H2D,
        offloading::BufferLifecycle::ALLOC,
        desc,
        {},
        symbolic::integer(400),
        symbolic::integer(0)
    );

    // --- reuse2 D2H
    auto [block_d2h_reuse2, memcpy_node_reuse2_d2h] = offloading::add_offloading_block<cuda::CUDADataOffloadingNode>(
        builder,
        root,
        "A",
        "__daisy_offload_A",
        offloading::DataTransferDirection::D2H,
        offloading::BufferLifecycle::FREE,
        desc,
        {},
        symbolic::integer(400),
        symbolic::integer(0)
    );

    // --- Free
    auto& block_free = builder.add_block(root);
    auto& access_node_free_in = builder.add_access(block_free, "A");

    auto& free_node = builder.add_library_node<stdlib::FreeNode>(block_free, DebugInfo());

    builder.add_computational_memlet(block_free, access_node_free_in, free_node, "_ptr", {}, desc);

    dump_sdfg(builder.subject(), "0-before");

    passes::DataTransferMinimizationPass pass;
    EXPECT_TRUE(pass.run(builder, analysis_manager));

    dump_sdfg(builder.subject(), "1-after");

    EXPECT_EQ(block_malloc.dataflow().nodes().size(), 2);
    EXPECT_EQ(block_h2d.dataflow().nodes().size(), 2);
    EXPECT_EQ(
        dynamic_cast<cuda::CUDADataOffloadingNode&>(memcpy_node_h2d).transfer_direction(),
        offloading::DataTransferDirection::NONE
    );
    EXPECT_EQ(
        dynamic_cast<cuda::CUDADataOffloadingNode&>(memcpy_node_h2d).buffer_lifecycle(),
        offloading::BufferLifecycle::ALLOC
    );
    EXPECT_EQ(block_d2h.dataflow().nodes().size(), 3);
    EXPECT_EQ(
        dynamic_cast<cuda::CUDADataOffloadingNode&>(memcpy_node_d2h).transfer_direction(),
        offloading::DataTransferDirection::D2H
    );
    EXPECT_EQ(
        dynamic_cast<cuda::CUDADataOffloadingNode&>(memcpy_node_d2h).buffer_lifecycle(),
        offloading::BufferLifecycle::NO_CHANGE
    );
    EXPECT_EQ(block_h2d_reuse.dataflow().nodes().size(), 0);
    EXPECT_EQ(block_d2h_reuse.dataflow().nodes().size(), 0);
    EXPECT_EQ(block_h2d_reuse2.dataflow().nodes().size(), 0);
    EXPECT_EQ(block_d2h_reuse2.dataflow().nodes().size(), 2);
    EXPECT_EQ(
        dynamic_cast<cuda::CUDADataOffloadingNode&>(memcpy_node_reuse2_d2h).transfer_direction(),
        offloading::DataTransferDirection::NONE
    );
    EXPECT_EQ(
        dynamic_cast<cuda::CUDADataOffloadingNode&>(memcpy_node_reuse2_d2h).buffer_lifecycle(),
        offloading::BufferLifecycle::FREE
    );
    EXPECT_EQ(block_free.dataflow().nodes().size(), 2);

    passes::DeadDataElimination dde_pass(false);
    dde_pass.run(builder, analysis_manager);

    EXPECT_EQ(block_malloc.dataflow().nodes().size(), 0);
    EXPECT_EQ(block_h2d.dataflow().nodes().size(), 2);
    EXPECT_EQ(
        dynamic_cast<cuda::CUDADataOffloadingNode&>(memcpy_node_h2d).transfer_direction(),
        offloading::DataTransferDirection::NONE
    );
    EXPECT_EQ(
        dynamic_cast<cuda::CUDADataOffloadingNode&>(memcpy_node_h2d).buffer_lifecycle(),
        offloading::BufferLifecycle::ALLOC
    );
    EXPECT_EQ(block_d2h.dataflow().nodes().size(), 0);
    EXPECT_EQ(block_h2d_reuse.dataflow().nodes().size(), 0);
    EXPECT_EQ(block_d2h_reuse.dataflow().nodes().size(), 0);
    EXPECT_EQ(block_h2d_reuse2.dataflow().nodes().size(), 0);
    EXPECT_EQ(block_d2h_reuse2.dataflow().nodes().size(), 2);
    EXPECT_EQ(
        dynamic_cast<cuda::CUDADataOffloadingNode&>(memcpy_node_reuse2_d2h).transfer_direction(),
        offloading::DataTransferDirection::NONE
    );
    EXPECT_EQ(
        dynamic_cast<cuda::CUDADataOffloadingNode&>(memcpy_node_reuse2_d2h).buffer_lifecycle(),
        offloading::BufferLifecycle::FREE
    );
    EXPECT_EQ(block_free.dataflow().nodes().size(), 0);

    dump_sdfg(builder.subject(), "2-cleanup");
};

TEST(DataTransferMinimizationPassTest, NotReadOnlyDataReuseTest) {
    sdfg::builder::StructuredSDFGBuilder builder("dot", sdfg::FunctionType_CPU);
    sdfg::analysis::AnalysisManager analysis_manager(builder.subject());

    auto& root = builder.subject().root();

    // Add containers
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("A", desc, true);
    builder.add_container("__daisy_offload_A", desc);
    types::Scalar indvar_desc(types::PrimitiveType::Int32);
    builder.add_container("i", indvar_desc);

    auto& in_type = builder.subject().type("A");
    auto& out_type = builder.subject().type("A");

    auto arr_size = symbolic::integer(400);
    auto device_id = symbolic::integer(0);

    // -- Malloc
    auto& block_malloc = builder.add_block(root);
    auto& access_node_malloc = builder.add_access(block_malloc, "A");

    auto& malloc_node = builder.add_library_node<stdlib::MallocNode>(block_malloc, DebugInfo(), symbolic::integer(400));

    builder.add_computational_memlet(block_malloc, malloc_node, "_ret", access_node_malloc, {}, desc);


    // --- Init H2D
    auto [block_h2d, memcpy_node_h2d] = offloading::add_offloading_block<cuda::CUDADataOffloadingNode>(
        builder,
        root,
        "A",
        "__daisy_offload_A",
        offloading::DataTransferDirection::H2D,
        offloading::BufferLifecycle::ALLOC,
        desc,
        {},
        arr_size,
        device_id
    );

    auto indvar = symbolic::symbol("i");
    // --- modify on device loop
    auto& map_modify = builder.add_map(
        root,
        indvar,
        symbolic::Lt(indvar, arr_size),
        symbolic::zero(),
        symbolic::add(indvar, symbolic::one()),
        cuda::ScheduleType_CUDA::create()
    );
    auto& block_modify = builder.add_block(map_modify.root());

    auto& access_modify_in = builder.add_access(block_modify, "__daisy_offload_A");
    auto& const_1 = builder.add_constant(block_modify, "1.0", base_desc);
    auto& access_modify_out = builder.add_access(block_modify, "__daisy_offload_A");
    auto& modify_add = builder.add_tasklet(block_modify, data_flow::fp_add, "out", {"a", "b"}, {});
    builder.add_computational_memlet(block_modify, access_modify_in, modify_add, "a", {indvar}, desc);
    builder.add_computational_memlet(block_modify, const_1, modify_add, "b", {}, base_desc);
    builder.add_computational_memlet(block_modify, modify_add, "out", access_modify_out, {indvar}, desc);


    // --- Init D2H
    auto [block_d2h, memcpy_node_d2h] = offloading::add_offloading_block<cuda::CUDADataOffloadingNode>(
        builder,
        root,
        "A",
        "__daisy_offload_A",
        offloading::DataTransferDirection::NONE,
        offloading::BufferLifecycle::FREE,
        desc,
        {},
        arr_size,
        device_id
    );

    // ------- reuse H2D

    auto [block_h2d_reuse, memcpy_node_reuse_h2d] = offloading::add_offloading_block<cuda::CUDADataOffloadingNode>(
        builder,
        root,
        "A",
        "__daisy_offload_A",
        offloading::DataTransferDirection::H2D,
        offloading::BufferLifecycle::ALLOC,
        desc,
        {},
        arr_size,
        device_id
    );

    // --- reuse D2H
    auto [block_d2h_reuse, memcpy_node_reuse_d2h] = offloading::add_offloading_block<cuda::CUDADataOffloadingNode>(
        builder,
        root,
        "A",
        "__daisy_offload_A",
        offloading::DataTransferDirection::NONE,
        offloading::BufferLifecycle::FREE,
        desc,
        {},
        arr_size,
        device_id
    );

    // --- Free
    auto& block_free = builder.add_block(root);
    auto& access_node_free_in = builder.add_access(block_free, "A");

    auto& free_node = builder.add_library_node<stdlib::FreeNode>(block_free, DebugInfo());

    builder.add_computational_memlet(block_free, access_node_free_in, free_node, "_ptr", {}, desc);

    dump_sdfg(builder.subject(), "0-before");

    passes::DataTransferMinimizationPass pass;
    EXPECT_TRUE(pass.run(builder, analysis_manager));

    dump_sdfg(builder.subject(), "1-after");

    passes::DeadDataElimination dde_pass(false);
    dde_pass.run(builder, analysis_manager);

    dump_sdfg(builder.subject(), "2-cleanup");

    EXPECT_EQ(block_malloc.dataflow().nodes().size(), 2);
    EXPECT_EQ(block_h2d.dataflow().nodes().size(), 2);
    EXPECT_EQ(
        dynamic_cast<cuda::CUDADataOffloadingNode&>(memcpy_node_h2d).transfer_direction(),
        offloading::DataTransferDirection::NONE
    );
    EXPECT_EQ(
        dynamic_cast<cuda::CUDADataOffloadingNode&>(memcpy_node_h2d).buffer_lifecycle(),
        offloading::BufferLifecycle::ALLOC
    );
    EXPECT_EQ(block_d2h.dataflow().nodes().size(), 2);
    EXPECT_EQ(
        dynamic_cast<cuda::CUDADataOffloadingNode&>(memcpy_node_d2h).transfer_direction(),
        offloading::DataTransferDirection::NONE
    );
    EXPECT_EQ(
        dynamic_cast<cuda::CUDADataOffloadingNode&>(memcpy_node_d2h).buffer_lifecycle(),
        offloading::BufferLifecycle::FREE
    );
    EXPECT_EQ(block_h2d_reuse.dataflow().nodes().size(), 3);
    EXPECT_EQ(
        dynamic_cast<cuda::CUDADataOffloadingNode&>(memcpy_node_reuse_h2d).transfer_direction(),
        offloading::DataTransferDirection::H2D
    );
    EXPECT_EQ(
        dynamic_cast<cuda::CUDADataOffloadingNode&>(memcpy_node_h2d).buffer_lifecycle(),
        offloading::BufferLifecycle::ALLOC
    );
    EXPECT_EQ(block_d2h_reuse.dataflow().nodes().size(), 2);
    EXPECT_EQ(
        dynamic_cast<cuda::CUDADataOffloadingNode&>(memcpy_node_reuse_d2h).transfer_direction(),
        offloading::DataTransferDirection::NONE
    );
    EXPECT_EQ(
        dynamic_cast<cuda::CUDADataOffloadingNode&>(memcpy_node_reuse_d2h).buffer_lifecycle(),
        offloading::BufferLifecycle::FREE
    );
    EXPECT_EQ(block_free.dataflow().nodes().size(), 2);
};

TEST(DataTransferMinimizationPassTest, RemoveRedundantD2HTest) {
    sdfg::builder::StructuredSDFGBuilder builder("dot", sdfg::FunctionType_CPU);
    sdfg::analysis::AnalysisManager analysis_manager(builder.subject());

    auto& root = builder.subject().root();

    // Add containers
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("A", desc, true);
    builder.add_container("__daisy_offload_A", desc);
    builder.add_container("i", types::Scalar(types::PrimitiveType::Int32));
    auto indvar = symbolic::symbol("i");

    auto device_id = symbolic::integer(0);
    auto arr_size = symbolic::integer(400);

    // --- H2D initial

    auto [block_h2d, memcpy_h2d] = offloading::add_offloading_block<cuda::CUDADataOffloadingNode>(
        builder,
        root,
        "A",
        "__daisy_offload_A",
        offloading::DataTransferDirection::H2D,
        offloading::BufferLifecycle::ALLOC,
        desc,
        {},
        arr_size,
        device_id
    );

    // --- D2H initial

    auto [block_d2h, memcpy_d2h] = offloading::add_offloading_block<cuda::CUDADataOffloadingNode>(
        builder,
        root,
        "A",
        "__daisy_offload_A",
        offloading::DataTransferDirection::D2H,
        offloading::BufferLifecycle::FREE,
        desc,
        {},
        arr_size,
        device_id
    );

    // --- H2D reuse

    auto [block_h2d_reuse, memcpy_h2d_reuse] = offloading::add_offloading_block<cuda::CUDADataOffloadingNode>(
        builder,
        root,
        "A",
        "__daisy_offload_A",
        offloading::DataTransferDirection::H2D,
        offloading::BufferLifecycle::ALLOC,
        desc,
        {},
        arr_size,
        device_id
    );

    // --- modify on device loop
    auto& map_modify = builder.add_map(
        root,
        indvar,
        symbolic::Lt(indvar, arr_size),
        symbolic::zero(),
        symbolic::add(indvar, symbolic::one()),
        cuda::ScheduleType_CUDA::create()
    );
    auto& block_modify = builder.add_block(map_modify.root());

    auto& access_modify_in = builder.add_access(block_modify, "__daisy_offload_A");
    auto& const_1 = builder.add_constant(block_modify, "1.0", base_desc);
    auto& access_modify_out = builder.add_access(block_modify, "__daisy_offload_A");
    auto& modify_add = builder.add_tasklet(block_modify, data_flow::fp_add, "out", {"a", "b"}, {});
    builder.add_computational_memlet(block_modify, access_modify_in, modify_add, "a", {indvar}, desc);
    builder.add_computational_memlet(block_modify, const_1, modify_add, "b", {}, base_desc);
    builder.add_computational_memlet(block_modify, modify_add, "out", access_modify_out, {indvar}, desc);

    // --- D2H reuse

    auto [block_d2h_reuse, memcpy_d2h_reuse] = offloading::add_offloading_block<cuda::CUDADataOffloadingNode>(
        builder,
        root,
        "A",
        "__daisy_offload_A",
        offloading::DataTransferDirection::D2H,
        offloading::BufferLifecycle::FREE,
        desc,
        {},
        arr_size,
        device_id
    );

    dump_sdfg(builder.subject(), "0-before");

    passes::DataTransferMinimizationPass pass;
    EXPECT_TRUE(pass.run(builder, analysis_manager));

    dump_sdfg(builder.subject(), "1-after");

    EXPECT_EQ(block_h2d.dataflow().nodes().size(), 3);
    EXPECT_EQ(
        dynamic_cast<cuda::CUDADataOffloadingNode&>(memcpy_h2d).transfer_direction(),
        offloading::DataTransferDirection::H2D
    );
    EXPECT_EQ(
        dynamic_cast<cuda::CUDADataOffloadingNode&>(memcpy_h2d).buffer_lifecycle(), offloading::BufferLifecycle::ALLOC
    );
    EXPECT_EQ(block_d2h.dataflow().nodes().size(), 0);
    EXPECT_EQ(block_h2d_reuse.dataflow().nodes().size(), 0);
    EXPECT_EQ(block_d2h_reuse.dataflow().nodes().size(), 3);
    EXPECT_EQ(
        dynamic_cast<cuda::CUDADataOffloadingNode&>(memcpy_d2h_reuse).transfer_direction(),
        offloading::DataTransferDirection::D2H
    );
    EXPECT_EQ(
        dynamic_cast<cuda::CUDADataOffloadingNode&>(memcpy_d2h_reuse).buffer_lifecycle(),
        offloading::BufferLifecycle::FREE
    );
};
