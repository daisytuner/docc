#pragma once

#include <sdfg/analysis/analysis.h>
#include <sdfg/structured_sdfg.h>
#include <sdfg/transformations/recorder.h>
#include <sdfg/transformations/transformation.h>

#include <nlohmann/json.hpp>

namespace sdfg {
namespace transformations {

/**
 * @brief Replays recorded transformations on an SDFG
 *
 * The Replayer class takes transformation descriptions (typically from a Recorder)
 * and applies them to an SDFG. This enables:
 * - Reapplication of successful transformation sequences
 * - Testing transformation sequences on different inputs
 * - Automated optimization pipelines
 */
class Replayer {
public:
    /**
     * @brief Replay a sequence of transformations from JSON
     *
     * @param builder The SDFG builder
     * @param analysis_manager The analysis manager
     * @param desc JSON array or single transformation description
     * @param skip_if_not_applicable If true, skip transformations that cannot be applied
     * @param loopnest_index Starting index for loop nests
     * @throws InvalidTransformationException if transformation cannot be applied and skip_if_not_applicable is false
     */
    void replay(
        builder::StructuredSDFGBuilder& builder,
        analysis::AnalysisManager& analysis_manager,
        const nlohmann::json& desc,
        bool skip_if_not_applicable = true,
        size_t loopnest_index = 0
    );

    /**
     * @brief Apply a single transformation from JSON description
     *
     * @tparam T The transformation type (must satisfy transformation_concept)
     * @param builder The SDFG builder
     * @param analysis_manager The analysis manager
     * @param desc JSON transformation description
     * @param skip_if_not_applicable If true, skip if transformation cannot be applied
     * @throws InvalidTransformationException if transformation cannot be applied and skip_if_not_applicable is false
     */
    template<typename T>
        requires transformation_concept<T>
    void apply(
        builder::StructuredSDFGBuilder& builder,
        analysis::AnalysisManager& analysis_manager,
        const nlohmann::json& desc,
        bool skip_if_not_applicable = true
    ) {
        T transformation(T::from_json(builder, desc));
        if (!transformation.can_be_applied(builder, analysis_manager)) {
            if (!skip_if_not_applicable) {
                throw transformations::
                    InvalidTransformationException("Transformation " + transformation.name() + " cannot be applied.");
            }
            return;
        }

        transformation.apply(builder, analysis_manager);
    };
};

} // namespace transformations
} // namespace sdfg
