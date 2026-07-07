#include "sdfg/passes/offloading/device_buffer_reuse_pass.h"

#include <gtest/gtest.h>

#include <set>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/data_flow_graph.h"
#include "sdfg/data_flow/memlet.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/element.h"
#include "sdfg/function.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/if_else.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/targets/cuda/cuda_data_offloading_node.h"
#include "sdfg/targets/offloading/data_offloading_node.h"
#include "sdfg/targets/rocm/rocm_data_offloading_node.h"
#include "sdfg/types/pointer.h"
#include "sdfg/types/scalar.h"
#include "sdfg_debug_dump.h"

using namespace sdfg;

namespace {

// A pure device allocation: an offloading node with NO data transfer and an ALLOC lifecycle.
// In generated code this lowers to `deviceMalloc(&dev, size)`.
template<typename NodeT>
structured_control_flow::Block& add_pure_alloc(
    builder::StructuredSDFGBuilder& builder,
    structured_control_flow::Sequence& seq,
    const std::string& dev,
    const types::IType& dev_type,
    int64_t size
) {
    auto [block, node] = offloading::add_offloading_block<NodeT>(
        builder,
        seq,
        dev, // host container (unused for a pure alloc)
        dev, // device container that gets allocated
        offloading::DataTransferDirection::NONE,
        offloading::BufferLifecycle::ALLOC,
        dev_type,
        DebugInfo(),
        symbolic::integer(size),
        symbolic::integer(0)
    );
    return block;
}

// A pure device free: an offloading node with NO data transfer and a FREE lifecycle.
// In generated code this lowers to `deviceFree(dev)`.
template<typename NodeT>
structured_control_flow::Block& add_pure_free(
    builder::StructuredSDFGBuilder& builder,
    structured_control_flow::Sequence& seq,
    const std::string& dev,
    const types::IType& dev_type,
    int64_t size
) {
    auto [block, node] = offloading::add_offloading_block<NodeT>(
        builder,
        seq,
        dev, // host container (unused for a pure free)
        dev, // device container that gets freed
        offloading::DataTransferDirection::NONE,
        offloading::BufferLifecycle::FREE,
        dev_type,
        DebugInfo(),
        symbolic::integer(size),
        symbolic::integer(0)
    );
    return block;
}

// Symbolic-size variants of the alloc/free helpers. Symbolic sizes (e.g. `S`, `T`) are not totally
// ordered by `symbolic::Gt` (it answers false in both directions), which is exactly the case that
// must not break the largest-first heuristic nor the merged-allocation sizing.
template<typename NodeT>
structured_control_flow::Block& add_pure_alloc(
    builder::StructuredSDFGBuilder& builder,
    structured_control_flow::Sequence& seq,
    const std::string& dev,
    const types::IType& dev_type,
    const symbolic::Expression& size
) {
    auto [block, node] = offloading::add_offloading_block<NodeT>(
        builder,
        seq,
        dev,
        dev,
        offloading::DataTransferDirection::NONE,
        offloading::BufferLifecycle::ALLOC,
        dev_type,
        DebugInfo(),
        size,
        symbolic::integer(0)
    );
    return block;
}

template<typename NodeT>
structured_control_flow::Block& add_pure_free(
    builder::StructuredSDFGBuilder& builder,
    structured_control_flow::Sequence& seq,
    const std::string& dev,
    const types::IType& dev_type,
    const symbolic::Expression& size
) {
    auto [block, node] = offloading::add_offloading_block<NodeT>(
        builder,
        seq,
        dev,
        dev,
        offloading::DataTransferDirection::NONE,
        offloading::BufferLifecycle::FREE,
        dev_type,
        DebugInfo(),
        size,
        symbolic::integer(0)
    );
    return block;
}

// A use of the device buffer (read + write) standing in for a kernel / map nest that operates on
// it. This gives the container a live range between its alloc and its free.
void add_use(builder::StructuredSDFGBuilder& builder, structured_control_flow::Sequence& seq, const std::string& dev) {
    auto& block = builder.add_block(seq);
    auto& in = builder.add_access(block, dev);
    auto& out = builder.add_access(block, dev);
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, in, tasklet, "_in", {symbolic::zero()});
    builder.add_computational_memlet(block, tasklet, "_out", out, {symbolic::zero()});
}

// A block that reads `src` and writes `dst`, creating a data dependency from `src` to `dst`. Used
// to chain two device buffers through a persistent container so that the second use provably
// happens-after the first (the dataflow counterpart of a control-flow ordering).
void add_copy(
    builder::StructuredSDFGBuilder& builder,
    structured_control_flow::Sequence& seq,
    const std::string& src,
    const std::string& dst
) {
    auto& block = builder.add_block(seq);
    auto& in = builder.add_access(block, src);
    auto& out = builder.add_access(block, dst);
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, in, tasklet, "_in", {symbolic::zero()});
    builder.add_computational_memlet(block, tasklet, "_out", out, {symbolic::zero()});
}

// Recursively collect every pure ALLOC / FREE offloading node together with its parent block.
using OffloadSite = std::pair<structured_control_flow::Block*, offloading::DataOffloadingNode*>;

void collect_offloads(
    structured_control_flow::ControlFlowNode& node, std::vector<OffloadSite>& allocs, std::vector<OffloadSite>& frees
) {
    if (auto* block = dyn_cast<structured_control_flow::Block*>(&node)) {
        for (auto& n : block->dataflow().nodes()) {
            if (auto* off = dynamic_cast<offloading::DataOffloadingNode*>(&n)) {
                if (off->is_alloc()) {
                    allocs.emplace_back(block, off);
                } else if (off->is_free()) {
                    frees.emplace_back(block, off);
                }
            }
        }
    } else if (auto* seq = dyn_cast<structured_control_flow::Sequence*>(&node)) {
        for (size_t i = 0; i < seq->size(); i++) {
            collect_offloads(seq->at(i).first, allocs, frees);
        }
    } else if (auto* if_else = dyn_cast<structured_control_flow::IfElse*>(&node)) {
        for (size_t i = 0; i < if_else->size(); i++) {
            collect_offloads(if_else->at(i).first, allocs, frees);
        }
    } else if (auto* loop = dyn_cast<structured_control_flow::StructuredLoop*>(&node)) {
        collect_offloads(loop->root(), allocs, frees);
    }
}

// The device container produced by an ALLOC node (the destination access node of its `_dev` edge).
std::string alloc_container(const OffloadSite& site) {
    auto& dataflow = site.first->dataflow();
    for (auto& memlet : dataflow.out_edges(*site.second)) {
        if (auto* access = dynamic_cast<const data_flow::AccessNode*>(&memlet.dst())) {
            return access->data();
        }
    }
    return "";
}

struct OffloadSummary {
    std::vector<OffloadSite> allocs;
    std::vector<OffloadSite> frees;
    std::set<std::string> alloc_containers;
};

OffloadSummary summarize(StructuredSDFG& sdfg) {
    OffloadSummary summary;
    collect_offloads(sdfg.root(), summary.allocs, summary.frees);
    for (auto& site : summary.allocs) {
        summary.alloc_containers.insert(alloc_container(site));
    }
    return summary;
}

} // namespace

// The pass operates purely through the base offloading::DataOffloadingNode interface, so every
// scenario is exercised with both concrete target nodes (CUDA and ROCm).
template<typename NodeT>
class DeviceBufferReusePassTest : public ::testing::Test {};

using OffloadNodeTypes = ::testing::Types<cuda::CUDADataOffloadingNode, rocm::ROCMDataOffloadingNode>;

class OffloadNodeTypeNames {
public:
    template<typename T>
    static std::string GetName(int i) {
        if (std::is_same<T, cuda::CUDADataOffloadingNode>::value) {
            return "CUDA";
        }
        if (std::is_same<T, rocm::ROCMDataOffloadingNode>::value) {
            return "ROCm";
        }
        return std::to_string(i);
    }
};

TYPED_TEST_SUITE(DeviceBufferReusePassTest, OffloadNodeTypes, OffloadNodeTypeNames);

// ===========================================================================================
// Linear control flow
// ===========================================================================================

// Two same-typed device buffers with strictly sequential, non-overlapping live ranges
// (d0 is freed before d1 is allocated) must be coalesced onto a single, max-sized allocation.
TYPED_TEST(DeviceBufferReusePassTest, ReuseTwoSequentialSameType) {
    builder::StructuredSDFGBuilder builder("reuse_seq", FunctionType_CPU);
    analysis::AnalysisManager analysis_manager(builder.subject());
    auto& root = builder.subject().root();

    types::Scalar element(types::PrimitiveType::Float);
    types::Pointer desc(element);
    builder.add_container("d0", desc);
    builder.add_container("d1", desc);

    add_pure_alloc<TypeParam>(builder, root, "d0", desc, 512);
    add_use(builder, root, "d0");
    add_pure_free<TypeParam>(builder, root, "d0", desc, 512);

    add_pure_alloc<TypeParam>(builder, root, "d1", desc, 1024);
    add_use(builder, root, "d1");
    add_pure_free<TypeParam>(builder, root, "d1", desc, 1024);

    dump_sdfg(builder.subject(), "0-before");

    passes::DeviceBufferReusePass pass;
    EXPECT_TRUE(pass.run(builder, analysis_manager));

    dump_sdfg(builder.subject(), "1-after");

    auto summary = summarize(builder.subject());
    EXPECT_EQ(summary.allocs.size(), 1u);
    EXPECT_EQ(summary.frees.size(), 1u);
    EXPECT_EQ(summary.alloc_containers.size(), 1u);
    // The shared allocation is sized to the larger of the two buffers.
    ASSERT_EQ(summary.allocs.size(), 1u);
    EXPECT_TRUE(symbolic::eq(summary.allocs.front().second->size(), symbolic::integer(1024)));
}

// Overlapping live ranges (both buffers alive simultaneously) must NOT be coalesced.
TYPED_TEST(DeviceBufferReusePassTest, NoReuseOverlappingLiveRanges) {
    builder::StructuredSDFGBuilder builder("no_reuse_overlap", FunctionType_CPU);
    analysis::AnalysisManager analysis_manager(builder.subject());
    auto& root = builder.subject().root();

    types::Scalar element(types::PrimitiveType::Float);
    types::Pointer desc(element);
    builder.add_container("d0", desc);
    builder.add_container("d1", desc);

    add_pure_alloc<TypeParam>(builder, root, "d0", desc, 512);
    add_pure_alloc<TypeParam>(builder, root, "d1", desc, 1024);
    add_use(builder, root, "d0"); // both d0 and d1 are live here
    add_use(builder, root, "d1");
    add_pure_free<TypeParam>(builder, root, "d0", desc, 512);
    add_pure_free<TypeParam>(builder, root, "d1", desc, 1024);

    dump_sdfg(builder.subject(), "0-before");

    passes::DeviceBufferReusePass pass;
    EXPECT_FALSE(pass.run(builder, analysis_manager));

    auto summary = summarize(builder.subject());
    EXPECT_EQ(summary.allocs.size(), 2u);
    EXPECT_EQ(summary.frees.size(), 2u);
    EXPECT_EQ(summary.alloc_containers.size(), 2u);
}

// Buffers of different element data types must NOT be coalesced even when non-overlapping.
TYPED_TEST(DeviceBufferReusePassTest, NoReuseDifferentDtype) {
    builder::StructuredSDFGBuilder builder("no_reuse_dtype", FunctionType_CPU);
    analysis::AnalysisManager analysis_manager(builder.subject());
    auto& root = builder.subject().root();

    types::Scalar float_element(types::PrimitiveType::Float);
    types::Scalar int_element(types::PrimitiveType::Int64);
    types::Pointer float_desc(float_element);
    types::Pointer int_desc(int_element);
    builder.add_container("d0", float_desc);
    builder.add_container("d1", int_desc);

    add_pure_alloc<TypeParam>(builder, root, "d0", float_desc, 512);
    add_use(builder, root, "d0");
    add_pure_free<TypeParam>(builder, root, "d0", float_desc, 512);

    add_pure_alloc<TypeParam>(builder, root, "d1", int_desc, 512);
    add_use(builder, root, "d1");
    add_pure_free<TypeParam>(builder, root, "d1", int_desc, 512);

    dump_sdfg(builder.subject(), "0-before");

    passes::DeviceBufferReusePass pass;
    EXPECT_FALSE(pass.run(builder, analysis_manager));

    auto summary = summarize(builder.subject());
    EXPECT_EQ(summary.allocs.size(), 2u);
    EXPECT_EQ(summary.alloc_containers.size(), 2u);
}

// Three sequential same-typed buffers all collapse onto one allocation; the largest-allocation-
// first heuristic sizes the shared buffer to the maximum (2048).
TYPED_TEST(DeviceBufferReusePassTest, LargerFirstHeuristic) {
    builder::StructuredSDFGBuilder builder("larger_first", FunctionType_CPU);
    analysis::AnalysisManager analysis_manager(builder.subject());
    auto& root = builder.subject().root();

    types::Scalar element(types::PrimitiveType::Float);
    types::Pointer desc(element);
    builder.add_container("d0", desc);
    builder.add_container("d1", desc);
    builder.add_container("d2", desc);

    add_pure_alloc<TypeParam>(builder, root, "d0", desc, 256);
    add_use(builder, root, "d0");
    add_pure_free<TypeParam>(builder, root, "d0", desc, 256);

    add_pure_alloc<TypeParam>(builder, root, "d1", desc, 2048);
    add_use(builder, root, "d1");
    add_pure_free<TypeParam>(builder, root, "d1", desc, 2048);

    add_pure_alloc<TypeParam>(builder, root, "d2", desc, 512);
    add_use(builder, root, "d2");
    add_pure_free<TypeParam>(builder, root, "d2", desc, 512);

    dump_sdfg(builder.subject(), "0-before");

    passes::DeviceBufferReusePass pass;
    EXPECT_TRUE(pass.run(builder, analysis_manager));

    auto summary = summarize(builder.subject());
    EXPECT_EQ(summary.alloc_containers.size(), 1u);
    ASSERT_EQ(summary.allocs.size(), 1u);
    EXPECT_TRUE(symbolic::eq(summary.allocs.front().second->size(), symbolic::integer(2048)));
}

// A buffer carrying an actual host/device transfer (not a pure alloc) must be left untouched and
// never merged with a pure scratch buffer.
TYPED_TEST(DeviceBufferReusePassTest, PureOnlyIgnoreTransferBuffers) {
    builder::StructuredSDFGBuilder builder("pure_only", FunctionType_CPU);
    analysis::AnalysisManager analysis_manager(builder.subject());
    auto& root = builder.subject().root();

    types::Scalar element(types::PrimitiveType::Float);
    types::Pointer desc(element);
    builder.add_container("host", desc, true);
    builder.add_container("transfer_dev", desc);
    builder.add_container("d0", desc);

    // A transfer buffer: H2D copy with an ALLOC lifecycle. Not a pure alloc.
    offloading::add_offloading_block<TypeParam>(
        builder,
        root,
        "host",
        "transfer_dev",
        offloading::DataTransferDirection::H2D,
        offloading::BufferLifecycle::ALLOC,
        desc,
        DebugInfo(),
        symbolic::integer(512),
        symbolic::integer(0)
    );
    add_use(builder, root, "transfer_dev");

    // A pure scratch buffer, non-overlapping with the transfer buffer.
    add_pure_alloc<TypeParam>(builder, root, "d0", desc, 512);
    add_use(builder, root, "d0");
    add_pure_free<TypeParam>(builder, root, "d0", desc, 512);

    dump_sdfg(builder.subject(), "0-before");

    passes::DeviceBufferReusePass pass;
    // The transfer buffer cannot be reused, so the pure buffer has no merge partner.
    EXPECT_FALSE(pass.run(builder, analysis_manager));

    auto summary = summarize(builder.subject());
    // transfer_dev (alloc) + d0 (alloc) both remain distinct.
    EXPECT_EQ(summary.alloc_containers.size(), 2u);
}

// ===========================================================================================
// Dataflow branching (asynchronous kernels)
// ===========================================================================================

// Two same-typed buffers are each fully scoped (alloc/use/free) in the same sequence and the two
// uses share no data. Their control-flow live ranges do not overlap, so under a sequential
// execution assumption they may be coalesced. But device kernels launch asynchronously: with no
// data dependency forcing an order, the two uses form independent dataflow branches that may run
// concurrently, so aliasing their storage would race. The `consider_dataflow_branching` flag
// selects between these two interpretations.
TYPED_TEST(DeviceBufferReusePassTest, DataflowBranchingFlagGovernsIndependentUses) {
    // Builds two independent, non data-sharing buffers in a single sequence.
    auto build = [](builder::StructuredSDFGBuilder& builder) {
        auto& root = builder.subject().root();
        types::Scalar element(types::PrimitiveType::Float);
        types::Pointer desc(element);
        builder.add_container("d0", desc);
        builder.add_container("d1", desc);

        add_pure_alloc<TypeParam>(builder, root, "d0", desc, 512);
        add_use(builder, root, "d0"); // kernel A: touches only d0
        add_pure_free<TypeParam>(builder, root, "d0", desc, 512);

        add_pure_alloc<TypeParam>(builder, root, "d1", desc, 1024);
        add_use(builder, root, "d1"); // kernel B: touches only d1 (no data shared with A)
        add_pure_free<TypeParam>(builder, root, "d1", desc, 1024);
    };

    // Ignoring dataflow branching (assume the independent kernels are serialized): coalesce.
    {
        builder::StructuredSDFGBuilder builder("dataflow_branch_off", FunctionType_CPU);
        analysis::AnalysisManager analysis_manager(builder.subject());
        build(builder);

        passes::DeviceBufferReusePass pass(/*consider_dataflow_branching=*/false);
        EXPECT_TRUE(pass.run(builder, analysis_manager));

        auto summary = summarize(builder.subject());
        EXPECT_EQ(summary.alloc_containers.size(), 1u);
    }

    // Respecting dataflow branching (the independent kernels may run concurrently): do NOT merge.
    {
        builder::StructuredSDFGBuilder builder("dataflow_branch_on", FunctionType_CPU);
        analysis::AnalysisManager analysis_manager(builder.subject());
        build(builder);

        passes::DeviceBufferReusePass pass(/*consider_dataflow_branching=*/true);
        EXPECT_FALSE(pass.run(builder, analysis_manager));

        auto summary = summarize(builder.subject());
        EXPECT_EQ(summary.alloc_containers.size(), 2u);
    }
}

// Three same-typed buffers, each scoped by an independent kernel that shares no data with the
// others. With dataflow branching respected all three may run concurrently and none may share
// storage; ignoring it collapses all three onto one allocation.
TYPED_TEST(DeviceBufferReusePassTest, DataflowBranchingThreeIndependentUses) {
    auto build = [](builder::StructuredSDFGBuilder& builder) {
        auto& root = builder.subject().root();
        types::Scalar element(types::PrimitiveType::Float);
        types::Pointer desc(element);
        builder.add_container("d0", desc);
        builder.add_container("d1", desc);
        builder.add_container("d2", desc);

        add_pure_alloc<TypeParam>(builder, root, "d0", desc, 256);
        add_use(builder, root, "d0");
        add_pure_free<TypeParam>(builder, root, "d0", desc, 256);

        add_pure_alloc<TypeParam>(builder, root, "d1", desc, 2048);
        add_use(builder, root, "d1");
        add_pure_free<TypeParam>(builder, root, "d1", desc, 2048);

        add_pure_alloc<TypeParam>(builder, root, "d2", desc, 512);
        add_use(builder, root, "d2");
        add_pure_free<TypeParam>(builder, root, "d2", desc, 512);
    };

    {
        builder::StructuredSDFGBuilder builder("three_independent_off", FunctionType_CPU);
        analysis::AnalysisManager analysis_manager(builder.subject());
        build(builder);

        passes::DeviceBufferReusePass pass(/*consider_dataflow_branching=*/false);
        EXPECT_TRUE(pass.run(builder, analysis_manager));
        EXPECT_EQ(summarize(builder.subject()).alloc_containers.size(), 1u);
    }

    {
        builder::StructuredSDFGBuilder builder("three_independent_on", FunctionType_CPU);
        analysis::AnalysisManager analysis_manager(builder.subject());
        build(builder);

        passes::DeviceBufferReusePass pass(/*consider_dataflow_branching=*/true);
        EXPECT_FALSE(pass.run(builder, analysis_manager));
        EXPECT_EQ(summarize(builder.subject()).alloc_containers.size(), 3u);
    }
}

// Counterpart to the independent-uses test: here the two device buffers are chained through a
// persistent container (`acc`), so the second kernel provably happens-after the first via a data
// dependency. Such a chain is NOT a dataflow branch, so the buffers may be coalesced even when
// dataflow branching is respected.
TYPED_TEST(DeviceBufferReusePassTest, DataflowChainSharedDataAllowsReuse) {
    builder::StructuredSDFGBuilder builder("dataflow_chain", FunctionType_CPU);
    analysis::AnalysisManager analysis_manager(builder.subject());
    auto& root = builder.subject().root();

    types::Scalar element(types::PrimitiveType::Float);
    types::Pointer desc(element);
    builder.add_container("acc", desc, true); // persistent: carries the dependency across buffers
    builder.add_container("d0", desc);
    builder.add_container("d1", desc);

    // Kernel A produces into d0, then stores its result into the persistent acc.
    add_pure_alloc<TypeParam>(builder, root, "d0", desc, 512);
    add_use(builder, root, "d0");
    add_copy(builder, root, "d0", "acc");
    add_pure_free<TypeParam>(builder, root, "d0", desc, 512);

    // Kernel B seeds d1 from acc (depending on A) and operates on it.
    add_pure_alloc<TypeParam>(builder, root, "d1", desc, 1024);
    add_copy(builder, root, "acc", "d1");
    add_use(builder, root, "d1");
    add_pure_free<TypeParam>(builder, root, "d1", desc, 1024);

    dump_sdfg(builder.subject(), "0-before");

    // Even respecting dataflow branching, the data dependency through acc serializes A before B.
    passes::DeviceBufferReusePass pass(/*consider_dataflow_branching=*/true);
    EXPECT_TRUE(pass.run(builder, analysis_manager));

    auto summary = summarize(builder.subject());
    EXPECT_EQ(summary.alloc_containers.size(), 1u);
    ASSERT_EQ(summary.allocs.size(), 1u);
    EXPECT_TRUE(symbolic::eq(summary.allocs.front().second->size(), symbolic::integer(1024)));
}

// ===========================================================================================
// Branching control flow
// ===========================================================================================

// Buffers isolated in sibling branches of an if/else are never concurrent at runtime, but they
// live on divergent (not totally ordered) paths. The pass is conservative and does NOT merge.
TYPED_TEST(DeviceBufferReusePassTest, NoReuseAcrossDivergentBranches) {
    builder::StructuredSDFGBuilder builder("divergent_branches", FunctionType_CPU);
    analysis::AnalysisManager analysis_manager(builder.subject());
    auto& root = builder.subject().root();

    types::Scalar element(types::PrimitiveType::Float);
    types::Pointer desc(element);
    builder.add_container("p", types::Scalar(types::PrimitiveType::Int32), true);
    builder.add_container("d0", desc);
    builder.add_container("d1", desc);

    auto& if_else = builder.add_if_else(root);
    auto cond = symbolic::Eq(symbolic::symbol("p"), symbolic::integer(0));
    auto& then_case = builder.add_case(if_else, cond);
    auto& else_case = builder.add_case(if_else, symbolic::Not(cond));

    add_pure_alloc<TypeParam>(builder, then_case, "d0", desc, 512);
    add_use(builder, then_case, "d0");
    add_pure_free<TypeParam>(builder, then_case, "d0", desc, 512);

    add_pure_alloc<TypeParam>(builder, else_case, "d1", desc, 1024);
    add_use(builder, else_case, "d1");
    add_pure_free<TypeParam>(builder, else_case, "d1", desc, 1024);

    dump_sdfg(builder.subject(), "0-before");

    passes::DeviceBufferReusePass pass;
    EXPECT_FALSE(pass.run(builder, analysis_manager));

    auto summary = summarize(builder.subject());
    EXPECT_EQ(summary.alloc_containers.size(), 2u);
}

// A buffer fully scoped before a branch and another fully scoped inside that branch are totally
// ordered (the first's free dominates the second's alloc) and may be merged.
TYPED_TEST(DeviceBufferReusePassTest, ReuseSequentialBeforeBranch) {
    builder::StructuredSDFGBuilder builder("seq_before_branch", FunctionType_CPU);
    analysis::AnalysisManager analysis_manager(builder.subject());
    auto& root = builder.subject().root();

    types::Scalar element(types::PrimitiveType::Float);
    types::Pointer desc(element);
    builder.add_container("p", types::Scalar(types::PrimitiveType::Int32), true);
    builder.add_container("d0", desc);
    builder.add_container("d1", desc);

    // d0 fully alloc/use/free before the branch.
    add_pure_alloc<TypeParam>(builder, root, "d0", desc, 512);
    add_use(builder, root, "d0");
    add_pure_free<TypeParam>(builder, root, "d0", desc, 512);

    // d1 fully alloc/use/free inside a (single) branch that runs strictly after d0 is freed.
    auto& if_else = builder.add_if_else(root);
    auto& then_case = builder.add_case(if_else, symbolic::Eq(symbolic::symbol("p"), symbolic::integer(0)));
    add_pure_alloc<TypeParam>(builder, then_case, "d1", desc, 1024);
    add_use(builder, then_case, "d1");
    add_pure_free<TypeParam>(builder, then_case, "d1", desc, 1024);

    dump_sdfg(builder.subject(), "0-before");

    passes::DeviceBufferReusePass pass;
    EXPECT_TRUE(pass.run(builder, analysis_manager));

    auto summary = summarize(builder.subject());
    EXPECT_EQ(summary.alloc_containers.size(), 1u);
    ASSERT_EQ(summary.allocs.size(), 1u);
    EXPECT_TRUE(symbolic::eq(summary.allocs.front().second->size(), symbolic::integer(1024)));
}

// A buffer that is live across a branch (allocated before, used inside both branches, freed after)
// overlaps a buffer allocated inside one of the branches, so they must NOT be merged.
TYPED_TEST(DeviceBufferReusePassTest, NoReuseLiveAcrossBranch) {
    builder::StructuredSDFGBuilder builder("live_across_branch", FunctionType_CPU);
    analysis::AnalysisManager analysis_manager(builder.subject());
    auto& root = builder.subject().root();

    types::Scalar element(types::PrimitiveType::Float);
    types::Pointer desc(element);
    builder.add_container("p", types::Scalar(types::PrimitiveType::Int32), true);
    builder.add_container("d0", desc);
    builder.add_container("d1", desc);

    add_pure_alloc<TypeParam>(builder, root, "d0", desc, 512);

    auto& if_else = builder.add_if_else(root);
    auto cond = symbolic::Eq(symbolic::symbol("p"), symbolic::integer(0));
    auto& then_case = builder.add_case(if_else, cond);
    auto& else_case = builder.add_case(if_else, symbolic::Not(cond));

    add_use(builder, then_case, "d0");
    // d1 lives entirely inside the else branch, overlapping d0's live range.
    add_pure_alloc<TypeParam>(builder, else_case, "d1", desc, 1024);
    add_use(builder, else_case, "d1");
    add_use(builder, else_case, "d0");
    add_pure_free<TypeParam>(builder, else_case, "d1", desc, 1024);

    add_pure_free<TypeParam>(builder, root, "d0", desc, 512);

    dump_sdfg(builder.subject(), "0-before");

    passes::DeviceBufferReusePass pass;
    EXPECT_FALSE(pass.run(builder, analysis_manager));

    auto summary = summarize(builder.subject());
    EXPECT_EQ(summary.alloc_containers.size(), 2u);
}

// Two buffers allocated/used/freed sequentially within a loop body are non-overlapping within a
// single iteration and may share an allocation.
TYPED_TEST(DeviceBufferReusePassTest, ReuseSequentialInLoopBody) {
    builder::StructuredSDFGBuilder builder("loop_body_reuse", FunctionType_CPU);
    analysis::AnalysisManager analysis_manager(builder.subject());
    auto& root = builder.subject().root();

    types::Scalar element(types::PrimitiveType::Float);
    types::Pointer desc(element);
    builder.add_container("i", types::Scalar(types::PrimitiveType::Int64), true);
    builder.add_container("d0", desc);
    builder.add_container("d1", desc);

    auto& loop = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::integer(10)),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1))
    );
    auto& body = loop.root();

    add_pure_alloc<TypeParam>(builder, body, "d0", desc, 512);
    add_use(builder, body, "d0");
    add_pure_free<TypeParam>(builder, body, "d0", desc, 512);

    add_pure_alloc<TypeParam>(builder, body, "d1", desc, 1024);
    add_use(builder, body, "d1");
    add_pure_free<TypeParam>(builder, body, "d1", desc, 1024);

    dump_sdfg(builder.subject(), "0-before");

    passes::DeviceBufferReusePass pass;
    EXPECT_TRUE(pass.run(builder, analysis_manager));

    auto summary = summarize(builder.subject());
    EXPECT_EQ(summary.alloc_containers.size(), 1u);
    ASSERT_EQ(summary.allocs.size(), 1u);
    EXPECT_TRUE(symbolic::eq(summary.allocs.front().second->size(), symbolic::integer(1024)));
}

// ===========================================================================================
// Symbolic allocation sizes
// ===========================================================================================

// Two sequential buffers whose sizes are symbols `S` and `T`. `symbolic::Gt` cannot order S and T
// (it answers false in both directions), so the colouring order is arbitrary and `members.front()`
// is not necessarily the largest. The merged allocation must therefore be sized to the symbolic
// maximum `max(S, T)`, never to whichever member happened to sort first.
TYPED_TEST(DeviceBufferReusePassTest, ReuseSymbolicSizesUsesSymbolicMax) {
    builder::StructuredSDFGBuilder builder("symbolic_sizes_max", FunctionType_CPU);
    analysis::AnalysisManager analysis_manager(builder.subject());
    auto& root = builder.subject().root();

    types::Scalar element(types::PrimitiveType::Float);
    types::Pointer desc(element);
    builder.add_container("S", types::Scalar(types::PrimitiveType::Int64), true);
    builder.add_container("T", types::Scalar(types::PrimitiveType::Int64), true);
    builder.add_container("d0", desc);
    builder.add_container("d1", desc);

    auto s = symbolic::symbol("S");
    auto t = symbolic::symbol("T");

    add_pure_alloc<TypeParam>(builder, root, "d0", desc, s);
    add_use(builder, root, "d0");
    add_pure_free<TypeParam>(builder, root, "d0", desc, s);

    add_pure_alloc<TypeParam>(builder, root, "d1", desc, t);
    add_use(builder, root, "d1");
    add_pure_free<TypeParam>(builder, root, "d1", desc, t);

    dump_sdfg(builder.subject(), "0-before");

    passes::DeviceBufferReusePass pass;
    EXPECT_TRUE(pass.run(builder, analysis_manager));

    dump_sdfg(builder.subject(), "1-after");

    auto summary = summarize(builder.subject());
    EXPECT_EQ(summary.alloc_containers.size(), 1u);
    ASSERT_EQ(summary.allocs.size(), 1u);
    // The surviving allocation must cover both buffers: max(S, T), not just S or T.
    EXPECT_TRUE(symbolic::eq(summary.allocs.front().second->size(), symbolic::max(s, t)));
}

// Three sequential buffers with mutually incomparable symbolic sizes `S`, `T`, `U`. They all
// collapse onto a single allocation, which must be sized to the maximum over ALL members, not the
// (unstable) first element of the colouring order.
TYPED_TEST(DeviceBufferReusePassTest, SymbolicSizesMaxOverAllMembers) {
    builder::StructuredSDFGBuilder builder("symbolic_sizes_all", FunctionType_CPU);
    analysis::AnalysisManager analysis_manager(builder.subject());
    auto& root = builder.subject().root();

    types::Scalar element(types::PrimitiveType::Float);
    types::Pointer desc(element);
    builder.add_container("S", types::Scalar(types::PrimitiveType::Int64), true);
    builder.add_container("T", types::Scalar(types::PrimitiveType::Int64), true);
    builder.add_container("U", types::Scalar(types::PrimitiveType::Int64), true);
    builder.add_container("d0", desc);
    builder.add_container("d1", desc);
    builder.add_container("d2", desc);

    auto s = symbolic::symbol("S");
    auto t = symbolic::symbol("T");
    auto u = symbolic::symbol("U");

    add_pure_alloc<TypeParam>(builder, root, "d0", desc, s);
    add_use(builder, root, "d0");
    add_pure_free<TypeParam>(builder, root, "d0", desc, s);

    add_pure_alloc<TypeParam>(builder, root, "d1", desc, t);
    add_use(builder, root, "d1");
    add_pure_free<TypeParam>(builder, root, "d1", desc, t);

    add_pure_alloc<TypeParam>(builder, root, "d2", desc, u);
    add_use(builder, root, "d2");
    add_pure_free<TypeParam>(builder, root, "d2", desc, u);

    dump_sdfg(builder.subject(), "0-before");

    passes::DeviceBufferReusePass pass;
    EXPECT_TRUE(pass.run(builder, analysis_manager));

    auto summary = summarize(builder.subject());
    EXPECT_EQ(summary.alloc_containers.size(), 1u);
    ASSERT_EQ(summary.allocs.size(), 1u);
    EXPECT_TRUE(symbolic::eq(summary.allocs.front().second->size(), symbolic::max(symbolic::max(s, t), u)));
}

// A symbolic buffer `S` and a concrete buffer of 4096 elements, sequential. The symbolic size is
// not comparable with the integer, so the merged allocation must take the symbolic maximum of the
// two rather than assuming either dominates.
TYPED_TEST(DeviceBufferReusePassTest, SymbolicAndConcreteSizeMax) {
    builder::StructuredSDFGBuilder builder("symbolic_and_concrete", FunctionType_CPU);
    analysis::AnalysisManager analysis_manager(builder.subject());
    auto& root = builder.subject().root();

    types::Scalar element(types::PrimitiveType::Float);
    types::Pointer desc(element);
    builder.add_container("S", types::Scalar(types::PrimitiveType::Int64), true);
    builder.add_container("d0", desc);
    builder.add_container("d1", desc);

    auto s = symbolic::symbol("S");

    add_pure_alloc<TypeParam>(builder, root, "d0", desc, s);
    add_use(builder, root, "d0");
    add_pure_free<TypeParam>(builder, root, "d0", desc, s);

    add_pure_alloc<TypeParam>(builder, root, "d1", desc, 4096);
    add_use(builder, root, "d1");
    add_pure_free<TypeParam>(builder, root, "d1", desc, 4096);

    dump_sdfg(builder.subject(), "0-before");

    passes::DeviceBufferReusePass pass;
    EXPECT_TRUE(pass.run(builder, analysis_manager));

    auto summary = summarize(builder.subject());
    EXPECT_EQ(summary.alloc_containers.size(), 1u);
    ASSERT_EQ(summary.allocs.size(), 1u);
    EXPECT_TRUE(symbolic::eq(summary.allocs.front().second->size(), symbolic::max(s, symbolic::integer(4096))));
}
