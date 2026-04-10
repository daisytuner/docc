#include <gtest/gtest.h>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_nodes/stdlib/memset.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/targets/cuda/cuda.h"
#include "sdfg/targets/cuda/cuda_data_offloading_node.h"
#include "sdfg/transformations/offloading/cuda_stdlib_data_transfer_extraction.h"

using namespace sdfg;

TEST(CUDAStdlibDataTransferExtractionTest, MemsetCanBeApplied) {
    builder::StructuredSDFGBuilder builder("sdfg_memset", FunctionType_CPU);
    auto& sdfg = builder.subject();

    auto num = symbolic::integer(1024);
    auto value = symbolic::zero();

    types::Scalar desc(types::PrimitiveType::Float);
    types::Pointer ptr_type(desc);

    builder.add_container("buf", ptr_type);

    auto& block = builder.add_block(sdfg.root());
    auto& buf_node = builder.add_access(block, "buf");

    auto& memset_node =
        static_cast<stdlib::MemsetNode&>(builder.add_library_node<stdlib::MemsetNode>(block, DebugInfo(), value, num));
    // Set implementation type as the rewriter would
    memset_node.implementation_type() = cuda::ImplementationType_CUDAWithTransfers;

    builder.add_computational_memlet(block, memset_node, "_ptr", buf_node, {}, ptr_type);

    analysis::AnalysisManager analysis_manager(sdfg);

    cuda::CUDAStdlibDataTransferExtraction expansion(memset_node);
    EXPECT_TRUE(expansion.can_be_applied(builder, analysis_manager));
}

TEST(CUDAStdlibDataTransferExtractionTest, MemsetApply) {
    builder::StructuredSDFGBuilder builder("sdfg_memset", FunctionType_CPU);
    auto& sdfg = builder.subject();

    auto num = symbolic::integer(1024);
    auto value = symbolic::zero();

    types::Scalar desc(types::PrimitiveType::Float);
    types::Pointer ptr_type(desc);

    builder.add_container("buf", ptr_type);

    auto& block = builder.add_block(sdfg.root());
    auto& buf_node = builder.add_access(block, "buf");

    auto& memset_node =
        static_cast<stdlib::MemsetNode&>(builder.add_library_node<stdlib::MemsetNode>(block, DebugInfo(), value, num));
    memset_node.implementation_type() = cuda::ImplementationType_CUDAWithTransfers;

    builder.add_computational_memlet(block, memset_node, "_ptr", buf_node, {}, ptr_type);

    analysis::AnalysisManager analysis_manager(sdfg);

    cuda::CUDAStdlibDataTransferExtraction expansion(memset_node);
    ASSERT_TRUE(expansion.can_be_applied(builder, analysis_manager));
    expansion.apply(builder, analysis_manager);

    // After apply: implementation type should be WithoutTransfers
    EXPECT_EQ(memset_node.implementation_type().value(), cuda::ImplementationType_CUDAWithoutTransfers.value());

    // The root sequence should now have 3 blocks:
    // alloc_device, memset_block, copy_from_device_and_dealloc
    EXPECT_EQ(sdfg.root().size(), 3);

    // The output access node should now reference a device container
    EXPECT_NE(buf_node.data().find(cuda::CUDA_DEVICE_PREFIX), std::string::npos);
}

TEST(CUDAStdlibDataTransferExtractionTest, MemsetWrongImplType) {
    builder::StructuredSDFGBuilder builder("sdfg_memset", FunctionType_CPU);
    auto& sdfg = builder.subject();

    auto num = symbolic::integer(1024);
    auto value = symbolic::zero();

    types::Scalar desc(types::PrimitiveType::Float);
    types::Pointer ptr_type(desc);

    builder.add_container("buf", ptr_type);

    auto& block = builder.add_block(sdfg.root());
    auto& buf_node = builder.add_access(block, "buf");

    auto& memset_node =
        static_cast<stdlib::MemsetNode&>(builder.add_library_node<stdlib::MemsetNode>(block, DebugInfo(), value, num));
    // Use WithoutTransfers — expansion should not apply
    memset_node.implementation_type() = cuda::ImplementationType_CUDAWithoutTransfers;

    builder.add_computational_memlet(block, memset_node, "_ptr", buf_node, {}, ptr_type);

    analysis::AnalysisManager analysis_manager(sdfg);

    cuda::CUDAStdlibDataTransferExtraction expansion(memset_node);
    EXPECT_FALSE(expansion.can_be_applied(builder, analysis_manager));
}

TEST(CUDAStdlibDataTransferExtractionTest, MemsetNoneImplType) {
    builder::StructuredSDFGBuilder builder("sdfg_memset", FunctionType_CPU);
    auto& sdfg = builder.subject();

    auto num = symbolic::integer(1024);
    auto value = symbolic::zero();

    types::Scalar desc(types::PrimitiveType::Float);
    types::Pointer ptr_type(desc);

    builder.add_container("buf", ptr_type);

    auto& block = builder.add_block(sdfg.root());
    auto& buf_node = builder.add_access(block, "buf");

    auto& memset_node =
        static_cast<stdlib::MemsetNode&>(builder.add_library_node<stdlib::MemsetNode>(block, DebugInfo(), value, num));
    // Default ImplementationType_NONE — expansion should not apply

    builder.add_computational_memlet(block, memset_node, "_ptr", buf_node, {}, ptr_type);

    analysis::AnalysisManager analysis_manager(sdfg);

    cuda::CUDAStdlibDataTransferExtraction expansion(memset_node);
    EXPECT_FALSE(expansion.can_be_applied(builder, analysis_manager));
}

TEST(CUDAStdlibDataTransferExtractionTest, MemsetSerialization) {
    builder::StructuredSDFGBuilder builder("sdfg_memset", FunctionType_CPU);
    auto& sdfg = builder.subject();

    auto num = symbolic::integer(1024);
    auto value = symbolic::zero();

    types::Scalar desc(types::PrimitiveType::Float);
    types::Pointer ptr_type(desc);

    builder.add_container("buf", ptr_type);

    auto& block = builder.add_block(sdfg.root());
    auto& buf_node = builder.add_access(block, "buf");

    auto& memset_node =
        static_cast<stdlib::MemsetNode&>(builder.add_library_node<stdlib::MemsetNode>(block, DebugInfo(), value, num));
    memset_node.implementation_type() = cuda::ImplementationType_CUDAWithTransfers;

    builder.add_computational_memlet(block, memset_node, "_ptr", buf_node, {}, ptr_type);

    cuda::CUDAStdlibDataTransferExtraction expansion(memset_node);

    nlohmann::json j;
    expansion.to_json(j);

    EXPECT_EQ(j["transformation_type"], "CUDAStdlibDataTransferExtraction");
    EXPECT_TRUE(j.contains("subgraph"));
    EXPECT_EQ(j["subgraph"]["0"]["element_id"], memset_node.element_id());
}
