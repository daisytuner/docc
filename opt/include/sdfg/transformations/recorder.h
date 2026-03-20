#pragma once

#include <sdfg/analysis/analysis.h>
#include <sdfg/structured_sdfg.h>
#include <sdfg/transformations/transformation.h>

#include <concepts>
#include <nlohmann/json.hpp>
#include "sdfg/optimization_report/optimization_report.h"

namespace sdfg {
namespace transformations {

/**
 * @brief Concept for types that are transformations
 */
template<typename T>
concept transformation_concept = std::derived_from<T, sdfg::transformations::Transformation>;

/**
 * @brief Records transformation history for replay and analysis
 *
 * The Recorder class tracks all transformations applied to an SDFG and
 * serializes them to JSON. This enables:
 * - Replay of transformation sequences on different SDFGs
 * - Analysis of transformation impact
 * - Debugging and validation of transformation sequences
 */
class Recorder {
private:
    nlohmann::json history_;

public:
    /**
     * @brief Construct an empty recorder
     */
    Recorder();

    /**
     * @brief Apply a transformation and record it
     *
     * @tparam T The transformation type (must satisfy transformation_concept)
     * @tparam Args Argument types for transformation constructor
     * @param builder The SDFG builder
     * @param analysis_manager The analysis manager
     * @param skip_if_not_applicable If true, skip instead of throwing on failure
     * @param args Arguments forwarded to transformation constructor
     * @throws InvalidTransformationException if transformation cannot be applied and skip_if_not_applicable is false
     */
    template<typename T, typename... Args>
        requires transformation_concept<T>
    void apply(
        builder::StructuredSDFGBuilder& builder,
        analysis::AnalysisManager& analysis_manager,
        bool skip_if_not_applicable,
        Args&&... args
    ) {
        T transformation(std::forward<Args>(args)...);

        if (!transformation.can_be_applied(builder, analysis_manager)) {
            if (!skip_if_not_applicable) {
                throw transformations::
                    InvalidTransformationException("Transformation " + transformation.name() + " cannot be applied.");
            }
            return;
        }

        nlohmann::json desc;
        transformation.to_json(desc);
        history_.push_back(desc);

        transformation.apply(builder, analysis_manager);
    };

    /**
     * @brief Save recorded transformation history to a file
     * @param path The file path to save to
     */
    void save(std::filesystem::path path) const;

    /**
     * @brief Get the transformation history
     * @return JSON array of transformation descriptions
     */
    nlohmann::json get_history() const { return history_; }

    /**
     * @brief Get mutable reference to transformation history
     * @return JSON array of transformation descriptions
     */
    nlohmann::json& history() { return history_; }
};

} // namespace transformations
} // namespace sdfg
