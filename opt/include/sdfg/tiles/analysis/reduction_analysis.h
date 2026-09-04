#pragma once

#include <string>
#include <vector>

#include "sdfg/analysis/analysis.h"
#include "sdfg/structured_control_flow/reduce.h"
#include "sdfg/structured_control_flow/structured_loop.h"

namespace sdfg {
namespace tiles {

/**
 * @file reduction_analysis.h
 * @brief Reduction-ownership queries for localization: which @ref Reduce nodes
 *        accumulate into a container relative to a loop scope, and whether that
 *        reduction can be privatized into a local buffer.
 *
 * A reduction accumulator is combined either sequentially (safe to privatize +
 * writeback) or cooperatively across threads (owned by the reduce dispatcher —
 * not localizable). These queries encode that distinction; see the module README.
 */

/// Whether @p container is the accumulator of a Reduce enclosing, nested within,
/// or equal to @p loop.
bool is_reduction_accumulator(
    structured_control_flow::StructuredLoop& loop,
    const std::string& container,
    analysis::AnalysisManager& analysis_manager
);

/// Collect the non-cooperative Reduce nodes (@p loop or a descendant) that reduce
/// into @p container, for accumulator privatization.
///
/// @return false if a *cooperative* (GPU-combined) Reduce at/below @p loop, or a
///         GPU block/warp-cooperative ancestor Reduce, owns @p container (those are
///         combined by the reduce dispatcher). A sequential or grid-parallel
///         ancestor Reduce is permitted and *not* added to @p out. Otherwise fills
///         @p out with the owning non-cooperative Reduce nodes and returns true.
bool collect_reduction_owners(
    structured_control_flow::StructuredLoop& loop,
    const std::string& container,
    analysis::AnalysisManager& analysis_manager,
    std::vector<structured_control_flow::Reduce*>& out
);

/// Collect grid-parallel (GPU-offloaded) *ancestor* Reduce nodes reducing into
/// @p container. Their cross-block merge becomes an atomic writeback, so a
/// localizing transformation demotes them to plain Map nodes and emits an atomic
/// copy-out.
std::vector<structured_control_flow::Reduce*> collect_grid_reduction_owners(
    structured_control_flow::StructuredLoop& loop,
    const std::string& container,
    analysis::AnalysisManager& analysis_manager
);

} // namespace tiles
} // namespace sdfg
