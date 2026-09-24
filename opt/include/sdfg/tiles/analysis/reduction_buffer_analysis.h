#pragma once

#include <map>
#include <optional>
#include <string>
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

/// Original loop IDs and the headers after swapping directly nested loops.
struct ReductionInterchangeProposal {
    size_t outer_id;
    size_t inner_id;
    ReductionLoopHeader new_outer;
    ReductionLoopHeader new_inner;
};

/// Replacement schedule for a loop identified in the original SDFG.
struct ReductionScheduleProposal {
    size_t loop_id;
    structured_control_flow::ScheduleType schedule;
};

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

/// Proposed GPU reduction results keyed by reduction element ID and accumulator name.
using ReductionBufferEstimate = std::map<std::pair<size_t, std::string>, ReductionBufferInfo>;

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
 * Queries are cached per reduction and original accumulator without changing the
 * SDFG. Callers must invalidate the analysis after changing bounds, schedules,
 * accesses, or container types. Returned references expire with the analysis.
 * Device capacity limits and strategy selection are outside this analysis.
 */
class ReductionBufferAnalysis : public analysis::Analysis {
    analysis::AnalysisManager* analysis_manager_ = nullptr;
    mutable std::map<std::pair<const structured_control_flow::Reduce*, std::string>, ReductionBufferInfo> buffers_;

    /// Infer one footprint, converting unsupported geometry or storage into a diagnostic result.
    ReductionBufferInfo compute(structured_control_flow::Reduce& reduction, const std::string& container) const;

protected:
    /// Bind the manager and options, and reset the lazily populated result cache.
    void run(analysis::AnalysisManager& analysis_manager) override;

public:
    /// Bind the subject graph; AnalysisManager initializes dependencies through run().
    explicit ReductionBufferAnalysis(StructuredSDFG& sdfg);

    /// @return "ReductionBufferAnalysis".
    std::string name() const override;

    /// Return the cached or newly inferred result, including bounded or unsupported results.
    const ReductionBufferInfo& buffer(structured_control_flow::Reduce& reduction, const std::string& container) const;

    /// Require an exact footprint, throwing InvalidSDFGException otherwise.
    /// Exactness alone does not imply materialization; check the result's materialized flag.
    const ReductionBufferInfo& require(structured_control_flow::Reduce& reduction, const std::string& container) const;

    /// Preview directly nested loop interchange on a clone; malformed loop IDs or
    /// nesting throw InvalidSDFGException. Moving materialized reductions is unsupported.
    ReductionBufferEstimate estimate(const ReductionInterchangeProposal& proposal) const;

    /// Preview a schedule replacement on a clone; throws InvalidSDFGException if the loop is missing.
    ReductionBufferEstimate estimate(const ReductionScheduleProposal& proposal) const;

    /// Analyze a detached proposal with fresh analyses and the same assumptions/options.
    /// Materialized layouts, types, costs, and owners must match the original graph.
    /// Throws InvalidSDFGException if @p proposal is the original SDFG.
    ReductionBufferEstimate estimate(StructuredSDFG& proposal) const;

    /// Check exactness and materialized-buffer compatibility for a proposed schedule,
    /// including affected sibling reductions in the enclosing GPU root.
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
