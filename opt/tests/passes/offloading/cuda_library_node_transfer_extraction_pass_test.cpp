#include <gtest/gtest.h>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_nodes/stdlib/memset.h"
#include "sdfg/passes/offloading/cuda_library_node_transfer_extraction_pass.h"
#include "sdfg/targets/cuda/cuda.h"

using namespace sdfg;

TEST(CudaLibraryNodeTransferExtractionPassTest, MemsetExpansion) {
    builder::StructuredSDFGBuilder builder("sdfg_memset", FunctionType_CPU);
    auto& sdfg = builder.subject();

    auto num = symbolic::integer(1024);
    auto value = symbolic::zero();

    types::Scalar desc(types::PrimitiveType::Float);
    types::Pointer ptr_type(desc);

    builder.add_container("buf", ptr_type);

    auto [block, memset_node] = stdlib::add_memset_block(builder, sdfg.root(), "buf", value, num, ptr_type);

    memset_node.implementation_type() = cuda::ImplementationType_CUDAWithTransfers;

    analysis::AnalysisManager analysis_manager(sdfg);

    cuda::CudaLibraryNodeTransferExtractionPass pass;
    bool changed = pass.run(builder, analysis_manager);
    EXPECT_TRUE(changed);

    // After pass: root should have 3 blocks (alloc, memset, copy+dealloc)
    EXPECT_EQ(sdfg.root().size(), 3);

    // The memset node should now be WithoutTransfers
    EXPECT_EQ(memset_node.implementation_type().value(), cuda::ImplementationType_CUDAWithoutTransfers.value());
}

TEST(CudaLibraryNodeTransferExtractionPassTest, MemsetNoExpansionWhenNone) {
    builder::StructuredSDFGBuilder builder("sdfg_memset", FunctionType_CPU);
    auto& sdfg = builder.subject();

    auto num = symbolic::integer(1024);
    auto value = symbolic::zero();

    types::Scalar desc(types::PrimitiveType::Float);
    types::Pointer ptr_type(desc);

    builder.add_container("buf", ptr_type);

    auto [block, memset_node] = stdlib::add_memset_block(builder, sdfg.root(), "buf", value, num, ptr_type);
    // Leave as NONE — pass should not expand

    analysis::AnalysisManager analysis_manager(sdfg);

    cuda::CudaLibraryNodeTransferExtractionPass pass;
    bool changed = pass.run(builder, analysis_manager);
    EXPECT_FALSE(changed);

    // Should remain 1 block
    EXPECT_EQ(sdfg.root().size(), 1);
}
