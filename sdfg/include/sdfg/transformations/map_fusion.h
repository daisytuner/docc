#pragma once

#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/transformations/transformation.h"

namespace sdfg {
namespace transformations {

/**
 * @brief Map fusion transformation that fuses two sequential maps
 *
 * This transformation fuses two sequential maps (children of the same sequence)
 * when the second map reads from containers that are written by the first map.
 * The transformation inlines the computation from the first map into the second map,
 * eliminating intermediate storage and improving data locality.
 *
 * ## Preconditions
 * - The first node must be a Map instance
 * - The second node must be a StructuredLoop (Map or For)
 * - The loops must be sequential children of the same sequence (first_map directly before second_loop)
 * - The second loop must read from at least one output of the first map
 * - Memory accesses must be affine (to allow solving the index mapping equation)
 * - Both maps must have simple structure (single block body for now)
 *
 * ## Fusion Strategy
 * For each read in the second map that reads from a container written by the first map:
 * 1. Extract the subset (memory access indices) of the read
 * 2. Find the corresponding write in the first map with matching subset
 * 3. Solve the affine equation system to determine the index mapping
 * 4. Inline the producer subgraph from the first map into the second map
 * 5. Reconnect the dataflow to bypass the intermediate container
 *
 * @note Currently focuses on simple maps with single blocks (loop nests of size 1)
 * @note Only handles affine memory access patterns
 *
 * @see Map
 * @see Sequence
 * @see ArgumentsAnalysis
 */
class MapFusion : public Transformation {
    structured_control_flow::Map& first_map_;
    structured_control_flow::StructuredLoop& second_loop_;
    bool applied_ = false;

    // Cached analysis results used in both can_be_applied and apply
    struct FusionCandidate {
        std::string container; ///< Container being read/written
        data_flow::AccessNode* consumer_access; ///< Read access in second loop
        data_flow::AccessNode* producer_access; ///< Write access in first map
        data_flow::Memlet* consumer_memlet; ///< Memlet for the read in second loop
        data_flow::Memlet* producer_memlet; ///< Memlet for the write in first map
        structured_control_flow::Block* consumer_block; ///< Block containing the consumer access
        symbolic::Expression index_mapping; ///< Expression mapping second_indvar to first_indvar
    };
    std::vector<FusionCandidate> fusion_candidates_;

public:
    /**
     * @brief Construct a map fusion transformation
     * @param first_map The first map (producer) to be fused
     * @param second_loop The second loop (consumer, can be Map or For) to be fused
     */
    MapFusion(structured_control_flow::Map& first_map, structured_control_flow::StructuredLoop& second_loop);

    /**
     * @brief Get the name of this transformation
     * @return "MapFusion"
     */
    virtual std::string name() const override;

    /**
     * @brief Check if this transformation can be applied
     *
     * Validates preconditions:
     * - Both maps are sequential children of the same sequence
     * - The second map reads from outputs of the first map
     * - Memory accesses are affine and can be solved
     *
     * @param builder The SDFG builder
     * @param analysis_manager The analysis manager
     * @return true if the transformation can be applied safely
     */
    virtual bool can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager)
        override;

    /**
     * @brief Apply the map fusion transformation
     *
     * Inlines the producer computation from the first map into the second map,
     * eliminating intermediate storage accesses.
     *
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
     * @brief Deserialize a map fusion transformation from JSON
     * @param builder The SDFG builder
     * @param j JSON description of the transformation
     * @return The deserialized transformation
     */
    static MapFusion from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& j);
};

} // namespace transformations
} // namespace sdfg
