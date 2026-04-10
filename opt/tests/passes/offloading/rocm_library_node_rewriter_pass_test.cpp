#include <gtest/gtest.h>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_nodes/stdlib/memset.h"
#include "sdfg/passes/offloading/rocm_library_node_rewriter_pass.h"
#include "sdfg/targets/rocm/rocm.h"

using namespace sdfg;

TEST(RocmLibraryNodeRewriterPassTest, MemsetRewrite) {
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

    // Before rewriter: implementation type should be NONE
    EXPECT_EQ(memset_node.implementation_type().value(), data_flow::ImplementationType_NONE.value());

    analysis::AnalysisManager analysis_manager(sdfg);

    rocm::RocmLibraryNodeRewriterPass pass;
    pass.run(builder, analysis_manager);

    // After rewriter: implementation type should be ROCMWithTransfers
    EXPECT_EQ(memset_node.implementation_type().value(), rocm::ImplementationType_ROCMWithTransfers.value());
}

TEST(RocmLibraryNodeRewriterPassTest, MemsetRewriteDoublePrecision) {
    builder::StructuredSDFGBuilder builder("sdfg_memset", FunctionType_CPU);
    auto& sdfg = builder.subject();

    auto num = symbolic::integer(2048);
    auto value = symbolic::integer(255);

    types::Scalar desc(types::PrimitiveType::Double);
    types::Pointer ptr_type(desc);

    builder.add_container("buf", ptr_type);

    auto& block = builder.add_block(sdfg.root());
    auto& buf_node = builder.add_access(block, "buf");

    auto& memset_node =
        static_cast<stdlib::MemsetNode&>(builder.add_library_node<stdlib::MemsetNode>(block, DebugInfo(), value, num));

    builder.add_computational_memlet(block, memset_node, "_ptr", buf_node, {}, ptr_type);

    analysis::AnalysisManager analysis_manager(sdfg);

    rocm::RocmLibraryNodeRewriterPass pass;
    pass.run(builder, analysis_manager);

    EXPECT_EQ(memset_node.implementation_type().value(), rocm::ImplementationType_ROCMWithTransfers.value());
}
