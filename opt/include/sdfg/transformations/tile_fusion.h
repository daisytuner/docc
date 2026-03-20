#pragma once

#include "sdfg/structured_control_flow/map.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/symbolic/assumptions.h"
#include "sdfg/transformations/transformation.h"

namespace sdfg {
namespace transformations {

/**
 * @brief Tile fusion transformation that merges two consecutive sibling tiled Maps
 *
 * Takes two consecutive tiled Maps (each the result of LoopTiling applied to a Map)
 * that communicate through an intermediate array and merges their outer tile loops
 * into a single For loop. The producer's inner iteration range is extended by the
 * computed stencil radius to cover cross-tile reads by the consumer.
 *
 * Before:
 *   map i_tile step S: map i: B[f(i)] = ...A...    // producer
 *   map j_tile step S: map j: A[g(j)] = ...B...    // consumer
 *
 * After:
 *   for tile step S:                                 // For (cross-tile deps)
 *     map i = max(0, tile-r)..min(tile+S+r, N): K1  // extended by radius r
 *     map j = tile..min(tile+S, N):              K2  // unchanged
 *
 * This is the structural enabler for diamond tiling on multi-kernel stencils.
 */
class TileFusion : public Transformation {
    structured_control_flow::Map& first_map_;
    structured_control_flow::Map& second_map_;
    bool applied_ = false;

    structured_control_flow::StructuredLoop* fused_loop_ = nullptr;
    int radius_ = 0;

    /// Shared containers and their computed radii
    struct TileFusionCandidate {
        std::string container;
        int radius;
    };
    std::vector<TileFusionCandidate> candidates_;

    /// Compute the radius for a shared container by analyzing producer write
    /// and consumer read subsets. Returns -1 if the radius cannot be determined.
    static int compute_radius(
        const data_flow::Subset& producer_write_subset,
        const std::vector<data_flow::Subset>& consumer_read_subsets,
        const std::vector<structured_control_flow::StructuredLoop*>& producer_loops,
        const std::vector<structured_control_flow::StructuredLoop*>& consumer_loops,
        const symbolic::Assumptions& producer_assumptions,
        const symbolic::Assumptions& consumer_assumptions
    );

public:
    /**
     * @brief Construct a tile fusion transformation
     * @param first_map The first (producer) outer tile Map
     * @param second_map The second (consumer) outer tile Map
     */
    TileFusion(structured_control_flow::Map& first_map, structured_control_flow::Map& second_map);

    virtual std::string name() const override;

    virtual bool can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager)
        override;

    virtual void apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    virtual void to_json(nlohmann::json& j) const override;

    static TileFusion from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& j);

    /// Get the fused For loop (only valid after apply)
    structured_control_flow::StructuredLoop* fused_loop() const;

    /// Get the computed radius (only valid after can_be_applied returns true)
    int radius() const;
};

} // namespace transformations
} // namespace sdfg
