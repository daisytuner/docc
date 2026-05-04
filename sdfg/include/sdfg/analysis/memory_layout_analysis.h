/**
 * @file memlet_delinearization_analysis.h
 * @brief Analysis for delinearizing memlet subsets
 *
 * This analysis attempts to delinearize memlet subsets by recovering
 * multi-dimensional structure from linearized expressions using the
 * symbolic delinearize function with block-level assumptions.
 */

#pragma once

#include <map>
#include <set>
#include <unordered_map>
#include <vector>

#include <string>

#include "sdfg/analysis/analysis.h"
#include "sdfg/data_flow/library_nodes/math/tensor/tensor_layout.h"
#include "sdfg/data_flow/memlet.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/structured_loop.h"
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

    /// Per-dimension bounding box extents: max[d] - min[d] + 1
    symbolic::MultiExpression extents() const;

    /// Per-dimension extents with min/max resolved to upper bounds via overapproximation
    symbolic::MultiExpression extents_approx() const;

    /// First and last linear element addresses: offset + sum(stride[d] * idx[d])
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
    std::map<std::pair<const structured_control_flow::StructuredLoop*, std::string>, MemoryTile> tiles_;
    std::map<std::pair<const structured_control_flow::StructuredLoop*, std::string>, std::vector<MemoryTileGroup>>
        tile_groups_;

    void traverse(structured_control_flow::ControlFlowNode& node, analysis::AnalysisManager& analysis_manager);

    void process_block(structured_control_flow::Block& block, analysis::AnalysisManager& analysis_manager);

    void merge_loop_layouts(
        structured_control_flow::StructuredLoop& loop,
        const std::vector<const data_flow::Memlet*>& memlets_before,
        const std::set<std::pair<const structured_control_flow::StructuredLoop*, std::string>>& tiles_before,
        analysis::AnalysisManager& analysis_manager
    );

    void compute_tile_groups(
        structured_control_flow::StructuredLoop& loop,
        const std::string& container,
        const std::vector<const data_flow::Memlet*>& memlets,
        const MemoryLayout& reference_layout,
        size_t ndims,
        const symbolic::SymbolSet& parameters,
        const symbolic::Assumptions& assumptions
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
     * @brief Get the inferred memory layout for a container at a specific loop level
     * @param loop The loop to query
     * @param container The container name
     * @return A pointer to the memory layout at that loop level, nullptr if not available
     */
    const MemoryTile* tile(const structured_control_flow::StructuredLoop& loop, const std::string& container) const;

    /**
     * @brief Get tile groups for a container at a specific loop level
     * @param loop The loop to query
     * @param container The container name
     * @return A pointer to the vector of tile groups, nullptr if not available
     */
    const std::vector<MemoryTileGroup>*
    tile_groups(const structured_control_flow::StructuredLoop& loop, const std::string& container) const;

    /**
     * @brief Get the tile group containing a specific memlet at a loop level
     * @param loop The loop to query
     * @param memlet The memlet to find
     * @return A pointer to the tile group containing the memlet, nullptr if not found
     */
    const MemoryTileGroup*
    tile_group_for(const structured_control_flow::StructuredLoop& loop, const data_flow::Memlet& memlet) const;
};

} // namespace analysis
} // namespace sdfg
