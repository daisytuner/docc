#pragma once

#include <unordered_set>

#include "sdfg/data_flow/access_node.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/transformations/transformation.h"
#include "sdfg/types/type.h"

namespace sdfg {
namespace transformations {

/**
 * @brief Out-of-loop local storage transformation for write or read-write data
 *
 * This transformation creates a local tile buffer for data accessed within a loop.
 * It uses MemoryLayoutAnalysis to compute the bounding box of all accesses to the
 * container within the loop, then:
 *   - If read-write: copies tile from memory before loop, accumulates into tile, writes back after
 *   - If write-only: skips the init copy, only writes back after
 *
 * The tile dimensions are determined by `extents_approx()` which gives integer upper
 * bounds for each dimension of the accessed region.
 *
 * Before (read-write accumulator):
 *   for i = 0..N:
 *       for j = 0..M:
 *           C[j] += A[i][j]
 *
 * After OutLocalStorage(i_loop, "C"):
 *   for __d0 = 0..M: C_local[__d0] = C[__d0]    // init (copy-in)
 *   for i = 0..N:
 *       for j = 0..M:
 *           C_local[j] += A[i][j]                 // compute on tile
 *   for __d0 = 0..M: C[__d0] = C_local[__d0]    // writeback (copy-out)
 *
 * This subsumes the previous scalar/array modes and AccumulatorTile:
 *   - Constant access (e.g. C[5]) → tile extent = {1} → scalar
 *   - Target loop indvar access (e.g. C[i]) → tile extent = {N}
 *   - Inner loop indvar access (e.g. C[j]) → tile extent = {M}
 *   - Multi-dim (e.g. C[i*K+j]) → delinearized tile extents
 *
 * @note Container must have at least writes within the loop scope
 * @note Tile extents must be provably integer (via overapproximation)
 * @note At least one extent must be > 1 (otherwise the transformation is trivial)
 */
class OutLocalStorage : public Transformation {
public:
    /// Tile information populated by can_be_applied
    struct TileInfo {
        /// Per-dimension buffer extents (integer overapproximations)
        std::vector<symbolic::Expression> dimensions;
        /// Per-dimension base addresses (min_subset from tile)
        std::vector<symbolic::Expression> bases;
        /// Layout strides for re-linearization (from tile layout)
        std::vector<symbolic::Expression> strides;
        /// Layout offset for re-linearization
        symbolic::Expression offset;
        /// True if container is also read (read-write), false if write-only
        bool has_read = false;
    };

private:
    structured_control_flow::StructuredLoop& loop_;
    std::string container_;
    const data_flow::AccessNode& access_node_;
    std::string local_name_; ///< Name of the created local buffer
    TileInfo tile_info_; ///< Populated by can_be_applied
    types::StorageType storage_type_; ///< Storage type for the local buffer
    std::unordered_set<const data_flow::Memlet*> group_memlets_; ///< Memlets in the selected tile group

public:
    /**
     * @brief Construct an out-of-loop storage transformation
     * @param loop The loop to optimize
     * @param access_node The access node to optimize
     * @param storage_type Storage type for the local buffer (default: CPU_Stack)
     */
    OutLocalStorage(
        structured_control_flow::StructuredLoop& loop,
        const data_flow::AccessNode& access_node,
        const types::StorageType& storage_type = types::StorageType::CPU_Stack()
    );

    /**
     * @brief Get the name of this transformation
     * @return "OutLocalStorage"
     */
    virtual std::string name() const override;

    /**
     * @brief Check if this transformation can be applied
     * @param builder The SDFG builder
     * @param analysis_manager The analysis manager
     * @return true if the transformation can be applied safely
     */
    virtual bool can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager)
        override;

    /**
     * @brief Apply the out-of-loop storage transformation
     * @param builder The SDFG builder
     * @param analysis_manager The analysis manager
     */
    virtual void apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    /**
     * @brief Serialize this transformation to JSON
     * @param j JSON object to populate
     */
    virtual void to_json(nlohmann::json& j) const override;

    /**
     * @brief Deserialize an out-of-loop storage transformation from JSON
     * @param builder The SDFG builder
     * @param j JSON description of the transformation
     * @return The deserialized transformation
     */
    static OutLocalStorage from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& j);
};
} // namespace transformations
} // namespace sdfg
