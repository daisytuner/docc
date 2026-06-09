/**
 * @file memory_layout_analysis.h
 * @brief Analysis for inferring memory layouts of memlets
 *
 * This analysis attempts to infer the memory layout of memlets using
 * symbolic assumptions to interpret linearized subset expressions.
 */

#pragma once

#include <map>
#include <set>
#include <unordered_map>
#include <vector>

#include <string>

#include "sdfg/analysis/analysis.h"
#include "sdfg/data_flow/memlet.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/control_flow_node.h"
#include "sdfg/symbolic/extreme_values.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {
namespace analysis {

typedef math::tensor::TensorLayout MemoryLayout;

struct MemoryAccess {
    std::string container; // Container name
    data_flow::Subset subset; // Symbolic indices after delinearization
    MemoryLayout layout; // Inferred memory layout
    bool first_dim_bounded; // True if first dimension is bounded (Tensor/Array), false for unbounded pointers
};

struct MemoryTile {
    std::string container; // Container name
    data_flow::Subset min_subset; // Minimum accessed indices in this tile
    data_flow::Subset max_subset; // Maximum accessed indices in this tile
    MemoryLayout layout; // Inferred tile layout at this loop level
    bool first_dim_bounded; // True if first dimension is bounded (Tensor/Array), false for unbounded pointers

    /// Per-dimension bounding box extents: max[d] - min[d] + 1.
    /// Returns `SymEngine::null` in slot `d` if that extent would depend on an
    /// unbounded leading-dimension sentinel. Callers MUST check each entry for null
    /// before using it.
    symbolic::MultiExpression extents() const;

    /// Per-dimension extents with min/max resolved to upper bounds via overapproximation.
    /// Returns `SymEngine::null` in slot `d` if that extent would depend on an
    /// unbounded leading-dimension sentinel. Callers MUST check each entry for null.
    symbolic::MultiExpression extents_approx() const;

    /// First and last linear element addresses: offset + sum(stride[d] * idx[d]).
    /// Returns `{SymEngine::null, SymEngine::null}` if either endpoint would depend
    /// on an unbounded leading-dimension sentinel (e.g. a layout whose strides
    /// reference an unknown shape entry). Callers MUST check `.first.is_null()` /
    /// `.second.is_null()` before using the result.
    std::pair<symbolic::Expression, symbolic::Expression> contiguous_range() const;
};

struct MemoryTileGroup {
    MemoryTile tile; // Bounding box for this group
    std::vector<const data_flow::Memlet*> memlets; // Memlets belonging to this group
};

/**
 * @class MemoryLayoutAnalysis
 * @brief Analysis that infers the assumed memory layouts of memlets
 *
 * This analysis attempts to infer the memory layout of memlets using
 * symbolic assumptions to interpret linearized subset expressions.
 *
 * The technique is based on the algorithm described in:
 * "Optimistic Delinearization of Parametrically Sized Arrays"
 * by Grosser et al.
 * https://dl.acm.org/doi/10.1145/2751205.2751248
 */
class MemoryLayoutAnalysis : public Analysis {
private:
    std::unordered_map<const data_flow::Memlet*, MemoryAccess> accesses_;
    std::map<std::pair<const structured_control_flow::ControlFlowNode*, std::string>, MemoryTile> tiles_;
    std::map<std::pair<const structured_control_flow::ControlFlowNode*, std::string>, std::vector<MemoryTileGroup>>
        tile_groups_;

    void traverse(structured_control_flow::ControlFlowNode& node, analysis::AnalysisManager& analysis_manager);

    void process_block(structured_control_flow::Block& block, analysis::AnalysisManager& analysis_manager);

    void merge_scope_layouts(
        structured_control_flow::ControlFlowNode& scope,
        const std::vector<const data_flow::Memlet*>& memlets_before,
        const std::set<std::pair<const structured_control_flow::ControlFlowNode*, std::string>>& tiles_before,
        analysis::AnalysisManager& analysis_manager
    );

    void compute_tile_groups(
        structured_control_flow::ControlFlowNode& scope,
        const std::string& container,
        const std::vector<const data_flow::Memlet*>& memlets,
        const MemoryLayout& reference_layout,
        size_t ndims,
        symbolic::BoundAnalysis& ba_tight,
        symbolic::BoundAnalysis& ba_loose
    );

protected:
    void run(analysis::AnalysisManager& analysis_manager) override;

public:
    MemoryLayoutAnalysis(StructuredSDFG& sdfg);

    std::string name() const override { return "MemoryLayoutAnalysis"; }

    /**
     * @brief Get the inferred memory layout for a memlet
     * @param memlet The memlet to query
     * @return A pointer to the inferred memory layout information if inference was successful, nullptr otherwise
     */
    const MemoryAccess* access(const data_flow::Memlet& memlet) const;

    /**
     * @brief Get the inferred memory layout for a container at a specific scope
     * @param scope The structured control-flow scope to query (Sequence, IfElse, While, StructuredLoop)
     * @param container The container name
     * @return A pointer to the memory tile at that scope, nullptr if not available
     */
    const MemoryTile* tile(const structured_control_flow::ControlFlowNode& scope, const std::string& container) const;

    /**
     * @brief Get tile groups for a container at a specific scope
     * @param scope The structured control-flow scope to query
     * @param container The container name
     * @return A pointer to the vector of tile groups, nullptr if not available
     */
    const std::vector<MemoryTileGroup>*
    tile_groups(const structured_control_flow::ControlFlowNode& scope, const std::string& container) const;

    /**
     * @brief Get the tile group containing a specific memlet at a scope
     * @param scope The structured control-flow scope to query
     * @param memlet The memlet to find
     * @return A pointer to the tile group containing the memlet, nullptr if not found
     */
    const MemoryTileGroup*
    tile_group_for(const structured_control_flow::ControlFlowNode& scope, const data_flow::Memlet& memlet) const;
};

} // namespace analysis
} // namespace sdfg
