#pragma once

#include "sdfg/passes/loop_fusion/fusion_types.h"
#include "sdfg/passes/loop_fusion/loop_fusion_by_accesses.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/symbolic/assumptions.h"
#include "sdfg/transformations/transformation.h"
#include "sdfg/visitor/structured_sdfg_visitor.h"

namespace sdfg {
namespace transformations {

class FusionConsumerUpdateVisitor;

/**
 * @brief Map fusion transformation that fuses two sequential maps
 *
 * This transformation fuses two sequential maps (children of the same sequence)
 * when the second map reads from containers that are written by the first map.
 *
 * Supports three patterns:
 * - Pattern 1 (ProducerIntoConsumer): Both perfectly nested. Inlines producer into consumer.
 * - Pattern 2 (ConsumerIntoProducer): Producer non-perfectly-nested, consumer perfectly nested.
 *   Inlines consumer blocks at the producer's write location to avoid replicating
 *   the producer's shallow sibling computations.
 * - Reverse Pattern 2 (ProducerIntoConsumer): Producer perfectly nested, consumer non-perfectly-nested.
 *   Inlines producer blocks at the consumer's read location.
 */
class MapFusion : public Transformation {
    friend class FusionConsumerUpdateVisitor;

    structured_control_flow::Map& first_map_;
    structured_control_flow::StructuredLoop& second_loop_;
    bool applied_ = false;
    bool require_consecutive_ = true; // Only fuse if maps are consecutive in the sequence
    // When false, Case 2 (init-into-reduction hoisting) is disabled and such fusions are
    // rejected. This lets the pipeline restrict hoisting to a single, final map-fusion run
    // so that loop distribution and earlier fusion runs do not fight each other.
    bool allow_init_hoist_ = true;
    // whether to consider ProducerIntoConsumer fusions. Those can now be handled by the LoopFusionByAccessWorker
    bool allow_prod_into_cons_;

    passes::loop_fusion::LoopFusionByAccessWorker::FusionDirection direction_;
    std::vector<passes::loop_fusion::FusionRegCandidate> fusion_candidates_;

    // Resolved locations populated during can_be_applied()
    std::vector<structured_control_flow::StructuredLoop*> producer_loops_;
    structured_control_flow::Sequence* producer_body_ = nullptr;
    structured_control_flow::Block* producer_block_ = nullptr;

    std::vector<structured_control_flow::StructuredLoop*> consumer_loops_;
    structured_control_flow::Sequence* consumer_body_ = nullptr;

    // Case 2 (init-into-reduction): when true, the producer is hoisted to the
    // reduction's outer parallel band (before the innermost sequential loop) and
    // keeps writing the accumulator array, instead of being scalarized and inlined
    // element-by-element inside the reduction loop (Case 1).
    bool init_hoist_ = false;
    // The outer parallel-band body that hosts the hoisted init (Case 2 only).
    structured_control_flow::Sequence* hoist_body_ = nullptr;

    /**
     * @brief Find the unique write location of a container in a loop nest
     *
     * Recursively walks the loop body to find the block that writes the given
     * container, collecting the enclosing loop chain.
     *
     * @return true if a unique write location was found
     */
    bool find_write_location(
        structured_control_flow::StructuredLoop& loop,
        const std::string& container,
        std::vector<structured_control_flow::StructuredLoop*>& loops,
        structured_control_flow::Sequence*& body,
        structured_control_flow::Block*& block
    );

    /**
     * @brief Find the read location of a container in a loop nest
     *
     * Recursively walks the loop body to find the sequence containing blocks
     * that read the given container, collecting the enclosing loop chain.
     *
     * @return true if a unique read location was found
     */
    bool find_read_location(
        structured_control_flow::StructuredLoop& loop,
        const std::string& container,
        std::vector<structured_control_flow::StructuredLoop*>& loops,
        structured_control_flow::Sequence*& body
    );

public:
    /**
     * @brief Construct a map fusion transformation
     * @param first_map The first map (producer) to be fused
     * @param second_loop The second loop (consumer, can be Map or For) to be fused
     * @param require_consecutive Whether the maps must be consecutive in the sequence for fusion to be applied
     * @param allow_init_hoist Whether Case 2 (init-into-reduction hoisting) may be applied
     * @param allow_prod_into_cons Allow ProducerIntoConsumer fusions
     */
    MapFusion(
        structured_control_flow::Map& first_map,
        structured_control_flow::StructuredLoop& second_loop,
        bool require_consecutive = true,
        bool allow_init_hoist = true,
        bool allow_prod_into_cons = true
    );

    passes::loop_fusion::LoopFusionByAccessWorker::FusionDirection last_fusion_direction() const;

    /**
     * @brief Get the name of this transformation
     * @return "MapFusion"
     */
    virtual std::string name() const override;

    /**
     * @brief Check if this transformation can be applied
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

class FusionConsumerSubsetVisitor : public visitor::ActualStructuredSDFGVisitor {
    friend MapFusion;

    std::unordered_map<std::string, const data_flow::Subset*>& target_containers_;
    std::unordered_map<std::string, std::vector<data_flow::Subset>> unique_subsets_per_container_;

protected:
    bool abort() { return true; }

public:
    FusionConsumerSubsetVisitor(std::unordered_map<std::string, const data_flow::Subset*>& target_containers);

    bool visit(sdfg::structured_control_flow::Block& block) override;

    bool visit(sdfg::structured_control_flow::Sequence& node) override;

    bool visit(IfElse& node) override;

    const std::unordered_map<std::string, std::vector<data_flow::Subset>>& unique_subsets_per_container();
};

class FusionConsumerUpdateVisitor : public visitor::ActualStructuredSDFGVisitor {
    friend MapFusion;

    builder::StructuredSDFGBuilder& builder_;
    const std::vector<passes::loop_fusion::FusionRegCandidate>& fusion_candidates_;
    const std::vector<std::string>& candidate_temps_;

public:
    FusionConsumerUpdateVisitor(
        builder::StructuredSDFGBuilder& builder,
        const std::vector<passes::loop_fusion::FusionRegCandidate>& fusion_candidates,
        const std::vector<std::string>& candidate_temps
    );

    bool dispatch_partial_sequence(Sequence& node, size_t first, size_t end);

    bool visit(sdfg::structured_control_flow::Block& block) override;

    bool visit(sdfg::structured_control_flow::Sequence& node) override;

    bool visit(IfElse& node) override;
};

} // namespace transformations
} // namespace sdfg
