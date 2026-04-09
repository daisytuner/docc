#pragma once

#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/symbolic/assumptions.h"
#include "sdfg/transformations/transformation.h"

namespace sdfg {
namespace transformations {

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
    structured_control_flow::Map& first_map_;
    structured_control_flow::StructuredLoop& second_loop_;
    bool applied_ = false;

    enum class FusionDirection {
        ProducerIntoConsumer,
        ConsumerIntoProducer,
    };
    FusionDirection direction_;

    struct FusionCandidate {
        std::string container;
        data_flow::Subset consumer_subset;
        std::vector<std::pair<symbolic::Symbol, symbolic::Expression>> index_mappings;
    };
    std::vector<FusionCandidate> fusion_candidates_;

    // Resolved locations populated during can_be_applied()
    std::vector<structured_control_flow::StructuredLoop*> producer_loops_;
    structured_control_flow::Sequence* producer_body_ = nullptr;
    structured_control_flow::Block* producer_block_ = nullptr;

    std::vector<structured_control_flow::StructuredLoop*> consumer_loops_;
    structured_control_flow::Sequence* consumer_body_ = nullptr;

    static std::vector<std::pair<symbolic::Symbol, symbolic::Expression>> solve_subsets(
        const data_flow::Subset& producer_subset,
        const data_flow::Subset& consumer_subset,
        const std::vector<structured_control_flow::StructuredLoop*>& producer_loops,
        const std::vector<structured_control_flow::StructuredLoop*>& consumer_loops,
        const symbolic::Assumptions& producer_assumptions,
        const symbolic::Assumptions& consumer_assumptions,
        bool invert_range_check = false
    );

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
