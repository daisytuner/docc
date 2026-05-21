#include <gtest/gtest.h>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/element.h"
#include "sdfg/function.h"
#include "sdfg/passes/offloading/remove_redundant_transfers_pass.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/targets/cuda/cuda_data_offloading_node.h"
#include "sdfg/targets/offloading/data_offloading_node.h"
#include "sdfg_debug_dump.h"

using namespace sdfg;

TEST(RemoveRedundantTransfersPassTest, SingleTransferTest) {
    sdfg::builder::StructuredSDFGBuilder builder("dot", sdfg::FunctionType_CPU);
    sdfg::analysis::AnalysisManager analysis_manager(builder.subject());

    auto& root = builder.subject().root();

    // Add containers
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("A", desc, true);
    builder.add_container("__daisy_offload_A", desc);

    auto [block, memcpy_node] = offloading::add_offloading_block<cuda::CUDADataOffloadingNode>(
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

    passes::RemoveRedundantTransfersPass pass;
    EXPECT_FALSE(pass.run_pass(builder, analysis_manager));
};

TEST(RemoveRedundantTransfersPassTest, MultiMapTest) {
    sdfg::builder::StructuredSDFGBuilder builder("dot", sdfg::FunctionType_CPU);
    sdfg::analysis::AnalysisManager analysis_manager(builder.subject());

    auto& root = builder.subject().root();

    // Add containers
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("A", desc, true);
    builder.add_container("__daisy_offload_A", desc);

    auto [block, memcpy_node] = offloading::add_offloading_block<cuda::CUDADataOffloadingNode>(
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

    auto [block2, memcpy_node2] = offloading::add_offloading_block<cuda::CUDADataOffloadingNode>(
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

    passes::RemoveRedundantTransfersPass pass;
    EXPECT_TRUE(pass.run_pass(builder, analysis_manager));

    dump_sdfg(builder.subject(), "1.after");

    int d2h_count = 0;
    for (int i = 0; i < root.size(); i++) {
        auto& cf_node = root.at(i).first;
        if (auto* block = dynamic_cast<structured_control_flow::Block*>(&cf_node)) {
            for (auto& node : block->dataflow().nodes()) {
                if (auto* data_transfer = dynamic_cast<offloading::DataOffloadingNode*>(&node)) {
                    if (data_transfer->is_d2h()) {
                        d2h_count++;
                    }
                }
            }
        }
    }
    EXPECT_EQ(d2h_count, 1);
};

TEST(RemoveRedundantTransfersPassTest, MultiMapWithLatterUseTest) {
    sdfg::builder::StructuredSDFGBuilder builder("dot", sdfg::FunctionType_CPU);
    sdfg::analysis::AnalysisManager analysis_manager(builder.subject());

    auto& root = builder.subject().root();

    // Add containers
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("A", desc, true);
    builder.add_container("__daisy_offload_A", desc);

    auto [block, memcpy_node] = offloading::add_offloading_block<cuda::CUDADataOffloadingNode>(
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

    auto [block2, memcpy_node2] = offloading::add_offloading_block<cuda::CUDADataOffloadingNode>(
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

    // Add another use of C after the second map
    auto& block3 = builder.add_block(root);
    auto& C3 = builder.add_access(block3, "A");
    auto& tasklet3 = builder.add_tasklet(block3, data_flow::TaskletCode::assign, "_out", {"_in"});
    auto& memlet_c3 = builder.add_computational_memlet(block3, C3, tasklet3, "_in", {symbolic::zero()});

    dump_sdfg(builder.subject(), "0.init");

    passes::RemoveRedundantTransfersPass pass;
    EXPECT_TRUE(pass.run_pass(builder, analysis_manager));

    dump_sdfg(builder.subject(), "1.after");

    // Check that there is exactly two H2D and one D2H transfer for C
    int h2d_count = 0;
    int d2h_count = 0;
    for (int i = 0; i < root.size(); i++) {
        auto& cf_node = root.at(i).first;
        if (auto* block = dynamic_cast<structured_control_flow::Block*>(&cf_node)) {
            for (auto& node : block->dataflow().nodes()) {
                if (auto* data_transfer = dynamic_cast<offloading::DataOffloadingNode*>(&node)) {
                    if (data_transfer->is_h2d()) {
                        h2d_count++;
                    } else if (data_transfer->is_d2h()) {
                        d2h_count++;
                    }
                }
            }
        }
    }
    EXPECT_EQ(h2d_count, 0);
    EXPECT_EQ(d2h_count, 1);
};
