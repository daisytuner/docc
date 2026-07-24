#pragma once

#include <sdfg/analysis/analysis.h>
#include <sdfg/analysis/loop_analysis.h>
#include <sdfg/structured_control_flow/control_flow_node.h>
#include <sdfg/structured_sdfg.h>
#include <sdfg/transformations/transformation.h>
#include <sdfg/transformations/transformation_schema.h>

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

protected:
    /**
     * @brief Enrich a transformation descriptor after ``to_json`` and before it
     * is recorded and applied.
     *
     * The base implementation annotates each loop anchor of the descriptor's
     * ``subgraph`` with ``loop_info``.
     */
    virtual void
    enrich(nlohmann::json& desc, builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
        auto& loop_analysis = analysis_manager.get<analysis::LoopAnalysis>();
        for (auto& [key, value] : desc["subgraph"].items()) {
            if (!value.contains("element_id")) continue;
            auto element_id = value["element_id"].template get<size_t>();
            auto* elem = builder.find_element_by_id(element_id);
            if (dyn_cast<structured_control_flow::StructuredLoop*>(elem) == nullptr) continue;
            auto* loop = static_cast<structured_control_flow::ControlFlowNode*>(elem);
            auto loop_info = loop_analysis.loop_info(loop);
            value["loop_info"] = loop_info_to_json(loop_info);
        }
    }

public:
    /**
     * @brief Construct an empty recorder
     */
    Recorder();

    virtual ~Recorder() = default;

    /**
     * @brief Apply an already-constructed transformation and record it.
     *
     * The single implementation of the record-and-apply flow: check
     * applicability, serialize, run the virtual ``enrich`` hook, append to the history and
     * apply. The templated ``apply`` overload is a thin constructor-forwarding
     * wrapper around this.
     *
     * @return true if applied, false if skipped (not applicable).
     * @throws InvalidTransformationException if not applicable and not skipped.
     */
    bool record(
        Transformation& transformation,
        builder::StructuredSDFGBuilder& builder,
        analysis::AnalysisManager& analysis_manager,
        bool skip_if_not_applicable = false
    ) {
        if (!transformation.can_be_applied(builder, analysis_manager)) {
            if (!skip_if_not_applicable) {
                throw transformations::
                    InvalidTransformationException("Transformation " + transformation.name() + " cannot be applied.");
            }
            return false;
        }

        nlohmann::json desc;
        transformation.to_json(desc);

#ifndef NDEBUG
        std::string schema_error;
        if (!validate_transformation_schema(desc, schema_error)) {
            throw InvalidTransformationDescriptionException(
                "Transformation '" + transformation.name() + "' produced an invalid description: " + schema_error
            );
        }
#endif

        this->enrich(desc, builder, analysis_manager);
        history_.push_back(desc);
        transformation.apply(builder, analysis_manager);
        return true;
    }

    /**
     * @brief Construct a transformation from ``args`` and record it.
     *
     * Convenience wrapper around ``record`` for call sites that hold the
     * transformation's constructor arguments rather than a built object.
     *
     * @tparam T The transformation type (must satisfy transformation_concept)
     * @tparam Args Argument types for transformation constructor
     * @throws InvalidTransformationException if not applicable and not skipped.
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
        this->record(transformation, builder, analysis_manager, skip_if_not_applicable);
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
