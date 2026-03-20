#pragma once

#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/transformations/transformation.h"

namespace sdfg {
namespace transformations {

/**
 * @brief Out-of-loop storage optimization transformation
 *
 * This transformation optimizes memory accesses within a loop by creating
 * local storage (scalar or array) outside the loop for frequently accessed
 * containers. This can reduce memory traffic and improve performance by
 * keeping data in registers or cache.
 *
 * @note All accesses to the container within the loop must be identical
 * @note Accesses must not depend on containers written in the loop
 * @note For array storage, loop iteration count must be known
 */
class OutLocalStorage : public Transformation {
private:
    structured_control_flow::StructuredLoop& loop_;
    std::string container_;
    bool requires_array_;

public:
    /**
     * @brief Construct an out-of-loop storage transformation
     * @param loop The loop to optimize
     * @param container The container name to optimize access for
     */
    OutLocalStorage(structured_control_flow::StructuredLoop& loop, std::string container);

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

private:
    /**
     * @brief Apply transformation using array storage
     * @param builder The SDFG builder
     * @param analysis_manager The analysis manager
     */
    void apply_array(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager);

    /**
     * @brief Apply transformation using scalar storage
     * @param builder The SDFG builder
     * @param analysis_manager The analysis manager
     */
    void apply_scalar(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager);
};
} // namespace transformations
} // namespace sdfg
