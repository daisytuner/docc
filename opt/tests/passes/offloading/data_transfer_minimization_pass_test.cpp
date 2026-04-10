#include "sdfg/passes/offloading/data_transfer_minimization_pass.h"

#include <gtest/gtest.h>

#include "../../../../sdfg/tests/sdfg_debug_dump.h"
#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/element.h"
#include "sdfg/function.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/targets/cuda/cuda_data_offloading_node.h"
#include "sdfg/targets/offloading/data_offloading_node.h"

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

    auto& block = builder.add_block(root);
    auto& access_node_in = builder.add_access(block, "__daisy_offload_A");
    auto& access_node_out = builder.add_access(block, "A");

    auto& memcpy_node = builder.add_library_node<cuda::CUDADataOffloadingNode>(
        block,
        DebugInfo(),
        symbolic::integer(400),
        symbolic::integer(0),
        offloading::DataTransferDirection::D2H,
        offloading::BufferLifecycle::NO_CHANGE
    );

    auto& in_type = builder.subject().type("A");
    builder.add_computational_memlet(block, access_node_in, memcpy_node, "_src", {}, in_type);

    auto& out_type = builder.subject().type("A");
    builder.add_computational_memlet(block, memcpy_node, "_dst", access_node_out, {}, out_type);

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

    auto& block = builder.add_block(root);
    auto& access_node_in = builder.add_access(block, "__daisy_offload_A");
    auto& access_node_out = builder.add_access(block, "A");

    auto& memcpy_node = builder.add_library_node<cuda::CUDADataOffloadingNode>(
        block,
        DebugInfo(),
        symbolic::integer(400),
        symbolic::integer(0),
        offloading::DataTransferDirection::D2H,
        offloading::BufferLifecycle::FREE
    );

    auto& in_type = builder.subject().type("A");
    builder.add_computational_memlet(block, access_node_in, memcpy_node, "_src", {}, in_type);

    auto& out_type = builder.subject().type("A");
    builder.add_computational_memlet(block, memcpy_node, "_dst", access_node_out, {}, out_type);

    auto& block2 = builder.add_block(root);

    auto& access_node_in2 = builder.add_access(block2, "A");
    auto& access_node_out2 = builder.add_access(block2, "__daisy_offload_A");

    auto& memcpy_node2 = builder.add_library_node<cuda::CUDADataOffloadingNode>(
        block2,
        DebugInfo(),
        symbolic::integer(400),
        symbolic::integer(0),
        offloading::DataTransferDirection::H2D,
        offloading::BufferLifecycle::ALLOC
    );

    auto& in_type2 = builder.subject().type("A");
    builder.add_computational_memlet(block2, access_node_in2, memcpy_node2, "_src", {}, in_type2);

    auto& out_type2 = builder.subject().type("A");
    builder.add_computational_memlet(block2, memcpy_node2, "_dst", access_node_out2, {}, out_type2);

    dump_sdfg(builder.subject(), "0-before");

    passes::DataTransferMinimizationPass pass;
    EXPECT_TRUE(pass.run(builder, analysis_manager));

    dump_sdfg(builder.subject(), "1-after");

    EXPECT_EQ(block.dataflow().nodes().size(), 0);
    EXPECT_EQ(block2.dataflow().nodes().size(), 0);
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

    auto& block = builder.add_block(root);
    auto& access_node_in = builder.add_access(block, "__daisy_offload_A");
    auto& access_node_out = builder.add_access(block, "A");

    auto& memcpy_node = builder.add_library_node<cuda::CUDADataOffloadingNode>(
        block,
        DebugInfo(),
        symbolic::integer(400),
        symbolic::integer(0),
        offloading::DataTransferDirection::D2H,
        offloading::BufferLifecycle::FREE
    );

    auto& in_type = builder.subject().type("A");
    builder.add_computational_memlet(block, access_node_in, memcpy_node, "_src", {}, in_type);

    auto& out_type = builder.subject().type("A");
    builder.add_computational_memlet(block, memcpy_node, "_dst", access_node_out, {}, out_type);

    auto& block2 = builder.add_block(root);

    auto& access_node_in2 = builder.add_access(block2, "A");
    auto& access_node_out2 = builder.add_access(block2, "__daisy_offload_A");

    auto& memcpy_node2 = builder.add_library_node<cuda::CUDADataOffloadingNode>(
        block2,
        DebugInfo(),
        symbolic::integer(400),
        symbolic::integer(0),
        offloading::DataTransferDirection::H2D,
        offloading::BufferLifecycle::ALLOC
    );

    auto& in_type2 = builder.subject().type("A");
    builder.add_computational_memlet(block2, access_node_in2, memcpy_node2, "_src", {}, in_type2);

    auto& out_type2 = builder.subject().type("A");
    builder.add_computational_memlet(block2, memcpy_node2, "_dst", access_node_out2, {}, out_type2);

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
        auto& cf_node = root.at(i).first;
        if (auto* block = dynamic_cast<structured_control_flow::Block*>(&cf_node)) {
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
