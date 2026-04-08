#pragma once

#include "sdfg/data_flow/memlet.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/transformations/transformation.h"

namespace sdfg {
namespace transformations {

/**
 * @brief In-loop local storage transformation for read-only data
 *
 * This transformation creates a local buffer for read-only array data accessed
 * within a loop. It copies the accessed tile into a contiguous buffer before
 * the loop, then redirects all reads to the local buffer.
 *
 * This is the read-only counterpart to OutLocalStorage. While OutLocalStorage
 * handles accumulators (read-modify-write), InLocalStorage handles pure inputs.
 *
 * Before:
 *   for ic = 0 to M step MC:
 *       for k = 0 to K:
 *           ... A[ic + ir][k] ...   // strided access
 *
 * After InLocalStorage(ic_loop, "A"):
 *   for ic = 0 to M step MC:
 *       // Copy tile of A into local buffer
 *       for ir = 0 to MC:
 *           for k = 0 to K:
 *               A_local[ir][k] = A[ic + ir][k]
 *
 *       for k = 0 to K:
 *           ... A_local[ir][k] ...   // contiguous access
 *
 * The transformation analyzes the convex hull of all read accesses to determine
 * the buffer dimensions and index transformations.
 *
 * Composable with other transformations:
 *   1. LoopTiling creates tile loops (strip-mining)
 *   2. InLocalStorage copies tile into contiguous buffer
 *   3. LoopInterchange can reorder copy loops for micro-panel layout
 *
 * @note The container must be read-only within the loop scope (no writes)
 * @note Access pattern must be affine in loop indices
 * @note Accessed ranges must be computable for buffer allocation
 */
class InLocalStorage : public Transformation {
public:
    /// Analyzed access information for buffer allocation
    struct AccessInfo {
        /// Symbolic expressions for buffer dimensions
        std::vector<symbolic::Expression> dimensions;
        /// Index transformation: original_index[d] = base[d] + local_index[d]
        std::vector<symbolic::Expression> bases;
        /// The representative access subset (first encountered)
        data_flow::Subset representative_subset;
        /// Determines copy loop placement based on which loops contribute to buffer dimensions:
        ///
        /// - false (simple case): Target loop's indvar contributes to access indices.
        ///   Copy happens ONCE before target loop.
        ///   Example: for i: A[i] → copy A[0..N] before loop, then A_local[i]
        ///
        /// - true (tiled case): Only descendant loops contribute, not the target.
        ///   Copy happens PER ITERATION inside target loop.
        ///   Example: for i_tile: for i: A[i] → copy A[i_tile..i_tile+TILE] each iteration
        ///
        /// Logic: copy_inside_loop = descendant_loops_contribute && !target_loop_contributes
        bool copy_inside_loop = false;
    };

private:
    structured_control_flow::StructuredLoop& loop_;
    std::string container_;
    std::string local_name_; ///< Name of the created local buffer
    AccessInfo access_info_; ///< Populated by can_be_applied

public:
    /**
     * @brief Construct an in-local storage transformation
     * @param loop The loop defining the scope for localization
     * @param container Name of the container to localize
     */
    InLocalStorage(structured_control_flow::StructuredLoop& loop, std::string container);

    /**
     * @brief Get the name of this transformation
     * @return "InLocalStorage"
     */
    virtual std::string name() const override;

    /**
     * @brief Check if this transformation can be applied
     *
     * Criteria:
     * - Container exists and is an array type
     * - Container is read-only within the loop (no writes)
     * - All accesses have identical dimensionality
     * - Access indices are affine in loop variables
     * - Accessed ranges can be computed symbolically
     *
     * @param builder The SDFG builder
     * @param analysis_manager The analysis manager
     * @return true if the transformation can be applied safely
     */
    virtual bool can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager)
        override;

    /**
     * @brief Apply the in-local storage transformation
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
     * @brief Deserialize an in-local storage transformation from JSON
     * @param builder The SDFG builder
     * @param j JSON description of the transformation
     * @return The deserialized transformation
     */
    static InLocalStorage from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& j);

    /**
     * @brief Get the name of the local buffer container
     * @return The local buffer container name (valid after apply())
     */
    const std::string& local_container() const { return local_name_; }

    /**
     * @brief Get the analyzed access information
     * @return The access info (valid after can_be_applied() returns true)
     */
    const AccessInfo& access_info() const { return access_info_; }
};

} // namespace transformations
} // namespace sdfg
