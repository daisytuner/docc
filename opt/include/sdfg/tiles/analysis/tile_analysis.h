#pragma once

#include <map>
#include <string>
#include <utility>
#include <vector>

#include "sdfg/analysis/analysis.h"
#include "sdfg/analysis/memory_layout_analysis.h"
#include "sdfg/structured_control_flow/control_flow_node.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/tiles/tile.h"

namespace sdfg {
namespace tiles {

/// Schedule-aware tile analysis: for every `(loop scope, container)` that
/// MemoryLayoutAnalysis resolves to a bounded tile, produces a @ref tiles::Tile
/// (geometry, thread/value partition, copy direction) for memory-level
/// transformations to consume instead of re-deriving.
class TileAnalysis : public analysis::Analysis {
private:
    std::map<std::pair<const structured_control_flow::ControlFlowNode*, std::string>, tiles::Tile> tiles_;

protected:
    void run(analysis::AnalysisManager& analysis_manager) override;

public:
    TileAnalysis(StructuredSDFG& sdfg);

    std::string name() const override { return "TileAnalysis"; }

    /// The tile for @p container at loop scope @p scope, or nullptr if none.
    const tiles::Tile* tile(const structured_control_flow::ControlFlowNode& scope, const std::string& container) const;

    /// How a container is accessed within a loop, read straight off the dataflow
    /// (memlet types), independent of any pointer-weak Users analysis.
    struct AccessSummary {
        bool reads = false; ///< a computational read of the container
        bool writes = false; ///< a computational write of the container
        bool aliased = false; ///< the pointer escapes, is overwritten, or is captured by a library node
    };

    /// Classify @p container's accesses within @p loop via the pointer analyzers.
    /// `aliased` is set when the pointer escapes, is overwritten/swapped, or is
    /// passed to a library node that may capture it — any of which lets the
    /// container's memory be reached outside the memlets a transformation rewrites.
    static AccessSummary
    summarize(const StructuredSDFG& sdfg, structured_control_flow::StructuredLoop& loop, const std::string& container);
};

/// True iff @p group is a real tile whose extents are all compile-time integer
/// constants (the constant-bounded localization precondition). A null/empty group
/// or any symbolic/unbounded extent is false.
bool is_constant_bounded(const analysis::MemoryTileGroup* group);

/// All tile groups MemoryLayoutAnalysis formed for @p container at @p loop
/// (nullptr if none).
const std::vector<analysis::MemoryTileGroup>* tile_groups(
    structured_control_flow::StructuredLoop& loop,
    const std::string& container,
    analysis::AnalysisManager& analysis_manager
);

/// Scalar slots a packed buffer for @p group would occupy (product of extents):
/// a compile-time integer iff @ref is_constant_bounded, null if any extent is
/// unbounded, 0 for a null group.
symbolic::Expression tile_element_count(const analysis::MemoryTileGroup* group);

/// The single localizable tile group of @p container at @p loop, or nullptr.
/// Returns the sole group iff the container forms EXACTLY ONE group there AND
/// every one of its body memlets belongs to it — anchoring on the container so all
/// its access nodes localize together as one coherent tile, which makes wholesale
/// rewriting safe.
const analysis::MemoryTileGroup* localizable_tile(
    structured_control_flow::StructuredLoop& loop,
    const std::string& container,
    analysis::AnalysisManager& analysis_manager
);

} // namespace tiles
} // namespace sdfg
