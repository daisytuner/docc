#pragma once

#include "sdfg/data_flow/memlet.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/transformations/transformation.h"

namespace sdfg {
namespace transformations {

/**
 * @brief Accumulator tile transformation for read-modify-write patterns
 *
 * This transformation creates a local tile buffer for accumulator data accessed
 * within nested loops. It initializes the tile from memory before the computation,
 * accumulates into the local tile during the loop, and writes back after.
 *
 * This is useful for BLIS-style micro-kernels where C[MR×NR] is accumulated
 * in registers/cache before writing back to memory.
 *
 * Use case: Access pattern depends on INNER loop indvars (not the target loop).
 *
 * Before:
 *   for ir = 0 to MR:          // target loop (i_micro)
 *       for jr = 0 to NR:      // nested loops
 *           for k = 0 to KC:
 *               C[ir][jr] += A[ir][k] * B[k][jr]
 *
 * After AccumulatorTile(ir_loop, "C"):
 *   for ir = 0 to MR:
 *       for jr = 0 to NR:
 *           // Initialize tile from C
 *           for __i = 0 to MR:
 *               for __j = 0 to NR:
 *                   C_tile[__i][__j] = C[ir + __i][jr + __j]
 *
 *           for k = 0 to KC:
 *               for __i = 0 to MR:
 *                   for __j = 0 to NR:
 *                       C_tile[__i][__j] += A[...] * B[...]
 *
 *           // Write back tile to C
 *           for __i = 0 to MR:
 *               for __j = 0 to NR:
 *                   C[ir + __i][jr + __j] = C_tile[__i][__j]
 *
 * Comparison with OutLocalStorage:
 *   - OutLocalStorage: Access is constant or depends on TARGET loop indvar only
 *   - AccumulatorTile: Access depends on INNER loop indvars (creates multi-dim tile)
 *
 * @note The container must be read-write within the target loop scope
 * @note Access indices must be affine in inner loop variables
 * @note Inner loop iteration counts must be computable for buffer allocation
 */
class AccumulatorTile : public Transformation {
public:
    /// Analyzed access information for tile allocation
    struct TileInfo {
        /// Symbolic expressions for tile dimensions (from inner loop iteration counts)
        std::vector<symbolic::Expression> dimensions;
        /// Base expressions for each dimension (computed from target loop indvars)
        std::vector<symbolic::Expression> bases;
        /// The representative access subset (first encountered)
        data_flow::Subset representative_subset;
        /// Inner loops whose indvars appear in access indices
        std::vector<structured_control_flow::For*> inner_loops;
    };

private:
    structured_control_flow::StructuredLoop& loop_;
    std::string container_;
    std::string local_name_; ///< Name of the created tile buffer
    TileInfo tile_info_; ///< Populated by can_be_applied

public:
    /**
     * @brief Construct an accumulator tile transformation
     * @param loop The outer loop defining the tile scope
     * @param container Name of the container to tile
     */
    AccumulatorTile(structured_control_flow::StructuredLoop& loop, std::string container);

    /**
     * @brief Get the name of this transformation
     * @return "AccumulatorTile"
     */
    virtual std::string name() const override;

    /**
     * @brief Check if this transformation can be applied
     *
     * Criteria:
     * - Container exists and is an array type
     * - Container is read-write within the loop (has both reads and writes)
     * - Access indices depend on inner loop indvars (not just target loop)
     * - Inner loop iteration counts can be computed symbolically
     *
     * @param builder The SDFG builder
     * @param analysis_manager The analysis manager
     * @return true if the transformation can be applied safely
     */
    virtual bool can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager)
        override;

    /**
     * @brief Apply the accumulator tile transformation
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
     * @brief Deserialize an accumulator tile transformation from JSON
     * @param builder The SDFG builder
     * @param j JSON description of the transformation
     * @return The deserialized transformation
     */
    static AccumulatorTile from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& j);

    /**
     * @brief Get the name of the tile buffer container
     * @return The tile buffer container name (valid after apply())
     */
    const std::string& local_container() const { return local_name_; }

    /**
     * @brief Get the analyzed tile information
     * @return The tile info (valid after can_be_applied() returns true)
     */
    const TileInfo& tile_info() const { return tile_info_; }
};

} // namespace transformations
} // namespace sdfg
