#include <gtest/gtest.h>

#include <vector>

#include "sdfg/analysis/analysis.h"
#include "sdfg/analysis/data_dependency_analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_node.h"
#include "sdfg/tiles/layout.h"
#include "sdfg/tiles/library_nodes/tile_copy_node.h"
#include "sdfg/tiles/tiled_copy.h"
#include "sdfg/types/array.h"
#include "sdfg/types/pointer.h"
#include "sdfg/types/scalar.h"

using namespace sdfg;

// Regression: an Array shared buffer written *through* a TileCopyNode's _dst
// pointer input is classified WRITE by analysis::Users (ungated by element type),
// but DataDependencyAnalysis used to gate that classification on Pointer types and
// fell back to READ for the Array buffer, then called Users::get_user(READ) for a
// user Users never registered -> std::out_of_range ("unordered_map::at"). This
// surfaced as "[Runtime] device-residency promotion failed: unordered_map::at".
TEST(DataDependencyTileCopyTest, ArrayBufferWrittenByTileCopyNodeNoThrow) {
    builder::StructuredSDFGBuilder builder("k", FunctionType_CPU);
    auto& root = builder.subject().root();

    types::Scalar elem(types::PrimitiveType::Float);
    types::Pointer ptr(elem);
    types::Array arr(elem, symbolic::integer(16)); // shared staging buffer (Array, not Pointer)
    builder.add_container("A", ptr, true); // global source (pointer argument)
    builder.add_container("sA", arr); // shared staging buffer

    auto& block = builder.add_block(root);
    auto& dst_acc = builder.add_access(block, "sA");
    auto& src_acc = builder.add_access(block, "A");
    tiles::TiledCopy plan;
    plan.src = tiles::Layout({symbolic::integer(4)}, {symbolic::integer(1)}, symbolic::integer(0));
    plan.dst = tiles::Layout({symbolic::integer(4)}, {symbolic::integer(1)}, symbolic::integer(0));
    plan.atom = tiles::CopyAtom::ScalarSync;
    data_flow::ImplementationType impl_cuda{"CUDA"};
    auto& node = builder.add_library_node<tiles::TileCopyNode>(
        block, DebugInfo(), impl_cuda, plan, tiles::CopyDirection::In, 4u, tiles::TileGuard{}, std::vector<int>{}
    );
    builder.add_computational_memlet(block, dst_acc, node, "_dst", {}, ptr);
    builder.add_computational_memlet(block, src_acc, node, "_src", {}, ptr);

    analysis::AnalysisManager analysis_manager(builder.subject());
    EXPECT_NO_THROW({ analysis_manager.get<analysis::DataDependencyAnalysis>(); });
}
