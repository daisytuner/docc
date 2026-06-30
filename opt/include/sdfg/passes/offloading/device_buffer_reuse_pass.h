#pragma once

#include <string>

#include "sdfg/analysis/analysis.h"
#include "sdfg/analysis/users.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/passes/pass.h"
#include "sdfg/targets/offloading/data_offloading_node.h"
#include "sdfg/types/type.h"
#include "sdfg/visitor/structured_sdfg_visitor.h"

namespace sdfg {
namespace passes {

/**
 * @brief Coalesces pure device buffer allocations (deviceMalloc/deviceFree without any host/device
 * transfer) so that non-overlapping buffers of the same element type share a single, max-sized
 * allocation.
 *
 * The pass works in a mark-then-sweep fashion:
 *   1. MARK   - collect pure ALLOC/FREE DataOffloadingNode pairs, compute their live ranges and
 *               build an interference graph (interfere if differing element type, overlapping
 *               live ranges, or not provably ordered on a single path).
 *   2. COLOR  - greedily assign buffers to shared "colors", processing the largest allocations
 *               first; a buffer reuses an existing color only if it is compatible and does not
 *               interfere. The color's allocation is sized to the largest member.
 *   3. SWEEP  - rename merged containers to their representative, keep one ALLOC (max size) and
 *               one FREE per color, and remove the now-redundant ALLOC/FREE nodes.
 *
 * NOTE: This pass operates purely through the base offloading::DataOffloadingNode interface and
 * is therefore target agnostic (deviceMalloc/deviceFree).
 *
 * Branching can occur in two ways and both prevent reuse:
 *   - control-flow branching: uses live on divergent (not totally ordered) control-flow paths.
 *   - dataflow branching: device kernels execute asynchronously, so two uses that are reachable
 *     from one another but share no data dependency may run concurrently. When
 *     `consider_dataflow_branching` is enabled such independent uses are treated as interfering
 *     (conservative, correct under unsynchronized async execution). When disabled (the default)
 *     the pass assumes consecutive uses are serialized and may coalesce their buffers.
 */
class DeviceBufferReusePass : public Pass {
public:
    explicit DeviceBufferReusePass(bool consider_dataflow_branching = false);

    std::string name() override;

    bool run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

private:
    /// Treat independent (non data-dependent) dataflow branches as potentially concurrent.
    bool consider_dataflow_branching_;
};


// A pure device-scratch buffer: it is allocated and freed without any host/device transfer, so its
// storage can in principle be shared with another buffer whose live range does not overlap.
struct Candidate {
    std::string container;
    types::PrimitiveType dtype;
    symbolic::Expression size;

    structured_control_flow::Block* alloc_block = nullptr;
    offloading::DataOffloadingNode* alloc_node = nullptr;
    analysis::User* alloc_user = nullptr;

    structured_control_flow::Block* free_block = nullptr;
    offloading::DataOffloadingNode* free_node = nullptr;
    analysis::User* free_user = nullptr;
};


// A pending allocation: a pure ALLOC seen during the traversal that has not yet been matched to a
// kernel that uses it.
struct AllocRec {
    structured_control_flow::Block* block;
    offloading::DataOffloadingNode* node;
    const data_flow::AccessNode* access;
};

// Single-traversal marker. Walks the SDFG in program order and builds both the candidate list and
// the interference graph on demand: allocations wait in `unassigned_` until a using kernel is seen,
// at which point they become live candidates interfering with everything currently live; frees end
// a live range directly; buffers created in sibling if/else branches cross-interfere.
class BufferReuseMarker : public visitor::StructuredSDFGVisitor {
public:
    BufferReuseMarker(
        builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager, analysis::Users& users
    );

    std::vector<Candidate> candidates;
    std::vector<std::unordered_set<size_t>> adjacency;

protected:
    bool visit_internal(structured_control_flow::Sequence& parent) override;

private:
    analysis::Users& users_;
    StructuredSDFG& sdfg_;
    std::unordered_map<std::string, AllocRec> unassigned_;
    std::unordered_set<std::string> disqualified_;
    std::vector<size_t> live_;
    std::unordered_map<std::string, size_t> live_index_;

    void add_edge(size_t a, size_t b);

    // Promote a pending allocation to a live candidate once a using kernel has been found.
    void assign(const std::string& container);

    // End a buffer's live range at its free.
    void close(
        const std::string& container,
        structured_control_flow::Block& block,
        offloading::DataOffloadingNode& node,
        const data_flow::AccessNode* access
    );

    void process_block(structured_control_flow::Block& block);
};

} // namespace passes
} // namespace sdfg
