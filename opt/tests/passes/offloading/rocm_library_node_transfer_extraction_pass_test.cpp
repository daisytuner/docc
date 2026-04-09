#include <gtest/gtest.h>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_nodes/stdlib/memset.h"
#include "sdfg/passes/offloading/rocm_library_node_transfer_extraction_pass.h"
#include "sdfg/targets/rocm/rocm.h"

using namespace sdfg;

TEST(RocmLibraryNodeTransferExtractionPassTest, MemsetExpansion) {
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
    memset_node.implementation_type() = rocm::ImplementationType_ROCMWithTransfers;

    builder.add_computational_memlet(block, memset_node, "_ptr", buf_node, {}, ptr_type);

    analysis::AnalysisManager analysis_manager(sdfg);

    rocm::RocmLibraryNodeTransferExtractionPass pass;
    bool changed = pass.run(builder, analysis_manager);
    EXPECT_TRUE(changed);

    // After pass: root should have 3 blocks (alloc, memset, copy+dealloc)
    EXPECT_EQ(sdfg.root().size(), 3);

    // The memset node should now be WithoutTransfers
    EXPECT_EQ(memset_node.implementation_type().value(), rocm::ImplementationType_ROCMWithoutTransfers.value());
}

TEST(RocmLibraryNodeTransferExtractionPassTest, MemsetNoExpansionWhenNone) {
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

    builder.add_computational_memlet(block, memset_node, "_ptr", buf_node, {}, ptr_type);

    analysis::AnalysisManager analysis_manager(sdfg);

    rocm::RocmLibraryNodeTransferExtractionPass pass;
    bool changed = pass.run(builder, analysis_manager);
    EXPECT_FALSE(changed);

    EXPECT_EQ(sdfg.root().size(), 1);
}
