#pragma once

#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "sdfg/analysis/analysis.h"
#include "sdfg/structured_control_flow/reduce.h"
#include "sdfg/targets/gpu/gpu_reduce_layout.h"

namespace sdfg {
namespace tiles {

/// How precisely the analysis can describe a reduction's storage requirements.
enum class ReductionBufferStatus {
    Exact, ///< Exact layout and allocation costs, suitable for materialization.
    ConservativeBound, ///< Allocation upper bounds only; not a materializable layout.
    Unsupported ///< No supported footprint; consult the diagnostic.
};

/// Proposed loop bounds and step, retaining the loop's induction variable.
struct ReductionLoopHeader {
    symbolic::Expression init;
    symbolic::Condition condition;
    symbolic::Expression update;
};

/// Target loops and the headers after swapping directly nested loops.
struct ReductionInterchangeProposal {
    structured_control_flow::StructuredLoop& outer;
    structured_control_flow::StructuredLoop& inner;
    ReductionLoopHeader new_outer;
    ReductionLoopHeader new_inner;
};

/// A schedule-only change; loop domains, access expressions, and nesting stay fixed.
struct ReductionScheduleProposal {
    structured_control_flow::StructuredLoop& loop;
    const structured_control_flow::ScheduleType& schedule;
};

/// Native source-to-copy correspondence; copied elements have independent IDs.
using ReductionNodeMapping =
    std::unordered_map<const structured_control_flow::ControlFlowNode*, const structured_control_flow::ControlFlowNode*>;

/**
 * @brief One accumulator's logical footprint and allocation contribution
 *
 * Missing costs mean unknown, not zero. Conservative bounds retain byte costs
 * but no layout. Shared storage is charged only to its owner; nested users of
 * that allocation contribute zero shared bytes.
 */
struct ReductionBufferInfo {
    ReductionBufferStatus status = ReductionBufferStatus::Unsupported;
    /// Explanation for a non-exact result.
    std::string diagnostic;
    /// Original-address to compact-slot mapping, present only for exact results.
    std::optional<gpu::ReductionLayout> layout;
    /// Original body index, retained after accesses are redirected to partials.
    symbolic::Expression accumulator_index = SymEngine::null;
    std::optional<types::PrimitiveType> primitive;
    /// Scalar element width in bytes, rounded up from the primitive bit width.
    std::optional<int64_t> element_bytes;
    /// Per-thread private allocation, or zero when no private partial is needed.
    std::optional<int64_t> private_bytes;
    /// Per-block shared allocation contributed by this reduction.
    std::optional<int64_t> shared_bytes;
    /// Element ID of the shared allocation's owner, including for nested users.
    std::optional<size_t> shared_owner;
    /// Flattened block-thread coordinate for a shared owner; null otherwise.
    symbolic::Expression linear_thread_index = SymEngine::null;
    /// Schedule references to materialized partials; empty before placement.
    std::string private_buffer;
    std::string shared_buffer;
    /// The reduction retains an original index and its packed buffers were validated.
    bool materialized = false;
    /// The original index has inner-loop output axes, including unit-count axes.
    bool multi_output = false;
};

/// Aggregate reduction-owned shared storage; excludes unrelated staging buffers.
struct ReductionKernelInfo {
    ReductionBufferStatus status = ReductionBufferStatus::Unsupported;
    /// Exact bytes or an upper bound, according to status; absent if unsupported.
    std::optional<int64_t> shared_bytes;
    std::string diagnostic;
};

/**
 * @brief Infer reduction footprints and verify explicitly materialized buffers
 *
 * Queries return independently owned results without caching or changing the SDFG.
 * Callers may retain results while their graph remains unchanged and must invalidate
 * dependent analyses after changing bounds, schedules, accesses, or container types.
 * Device capacity limits and strategy selection are outside this analysis.
 */
class ReductionBufferAnalysis : public analysis::Analysis {
    analysis::AnalysisManager* analysis_manager_ = nullptr;

    /// Infer one footprint, converting unsupported geometry or storage into a diagnostic result.
    ReductionBufferInfo compute(structured_control_flow::Reduce& reduction, const std::string& container) const;

protected:
    /// Bind the manager and options for dependency queries.
    void run(analysis::AnalysisManager& analysis_manager) override;

public:
    /// Bind the subject graph; AnalysisManager initializes dependencies through run().
    explicit ReductionBufferAnalysis(StructuredSDFG& sdfg);

    /// @return "ReductionBufferAnalysis".
    std::string name() const override;

    /// Infer an independent result, including bounded or unsupported results.
    ReductionBufferInfo buffer(structured_control_flow::Reduce& reduction, const std::string& container) const;

    /// Require an exact footprint, throwing InvalidSDFGException otherwise.
    /// Exactness alone does not imply materialization; check the result's materialized flag.
    ReductionBufferInfo require(structured_control_flow::Reduce& reduction, const std::string& container) const;

    /// Reprice a caller-owned exact footprint under a read-only schedule override.
    /// The footprint must describe this accumulator in the unchanged source graph.
    /// Does not visit body memlets, copy the graph, or validate materialized buffers.
    /// Returned ownership uses source node IDs; materialized is false for estimates.
    ReductionBufferInfo estimate_schedule(
        structured_control_flow::Reduce& reduction,
        const std::string& container,
        ReductionBufferInfo footprint,
        const ReductionScheduleProposal& proposal
    ) const;

    /// Copy only the enclosing nest and referenced declarations into a detached builder.
    /// The returned correspondence locates copied nodes without preserving element IDs.
    ReductionNodeMapping
    copy_nest(structured_control_flow::StructuredLoop& loop, builder::StructuredSDFGBuilder& destination) const;

    /// Check directly nested loop interchange on a native nest copy.
    /// Malformed nesting throws; moving materialized reductions is unsupported.
    bool supports_interchange(const ReductionInterchangeProposal& proposal) const;

    /// Validate a detached nest after a proposed rewrite, using native node correspondence
    /// to preserve materialized layouts, types, costs, and allocation ownership.
    bool supports(StructuredSDFG& proposal, const ReductionNodeMapping& nodes) const;

    /// Check exactness and materialized-buffer compatibility for a proposed schedule,
    /// including sibling reductions in the enclosing nest. Uses analytical allocation
    /// effects for exact source footprints, with native-copy fallback otherwise.
    bool supports_schedule(
        structured_control_flow::StructuredLoop& loop, const structured_control_flow::ScheduleType& schedule
    ) const;
    /// Return GPU reductions at, above, or below @p loop in the current graph.
    std::vector<structured_control_flow::Reduce*> affected_reductions(structured_control_flow::StructuredLoop& loop
    ) const;

    /// Sum shared allocation contributions at or below @p root, charging each owner
    /// once. Unknown costs, duplicate named allocations, or overflow are unsupported.
    ReductionKernelInfo kernel(const structured_control_flow::StructuredLoop& root) const;

    /// Whether a GPU reduction schedule references @p container as a private or shared partial.
    bool is_partial_buffer(const std::string& container) const;
};

} // namespace tiles
} // namespace sdfg
