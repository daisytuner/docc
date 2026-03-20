#pragma once

#include <exception>

#include "sdfg/analysis/analysis.h"
#include "sdfg/analysis/users.h"
#include "sdfg/builder/structured_sdfg_builder.h"

namespace sdfg {

class PassReportConsumer;

namespace transformations {

/**
 * @brief Base class for all SDFG transformations
 *
 * A transformation modifies the structure of an SDFG while preserving its
 * computational semantics. Each transformation implements can_be_applied()
 * to check preconditions and apply() to perform the modification.
 *
 * Transformations can be serialized to/from JSON for replay and analysis.
 */
class Transformation {
protected:
    PassReportConsumer* report_ = nullptr;

public:
    virtual ~Transformation() = default;

    /**
     * @brief Set the report consumer for transformation feedback
     * @param report The report consumer
     */
    virtual void set_report(PassReportConsumer* report) { report_ = report; }

    /**
     * @brief Get the name of this transformation
     * @return The transformation name (e.g., "LoopInterchange")
     */
    virtual std::string name() const = 0;

    /**
     * @brief Check if this transformation can be safely applied
     * @param builder The SDFG builder
     * @param analysis_manager The analysis manager providing necessary analyses
     * @return true if all preconditions are met
     */
    // TODO builder and probably analysis manager should be given via constructor, as in practice, the nodes which are
    // transformed are already constructor params, and therefore the SDFG is implied
    virtual bool can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) = 0;

    /**
     * @brief Apply this transformation to the SDFG
     * @param builder The SDFG builder
     * @param analysis_manager The analysis manager (may be invalidated)
     */
    virtual void apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) = 0;

    /**
     * @brief Attempt to apply the transformation if possible
     *
     * Bundles can_be_applied() and apply() into a single function so they can share state
     *
     * @param builder The SDFG builder
     * @param analysis_manager The analysis manager
     * @return true if the transformation was applied
     */
    virtual bool try_apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
        auto can_be = can_be_applied(builder, analysis_manager);

        if (!can_be) {
            return false;
        } else {
            apply(builder, analysis_manager);
            return true;
        }
    }

    /**
     * @brief Serialize this transformation to JSON
     * @param j JSON object to populate with transformation description
     * @throws std::logic_error if not implemented
     */
    virtual void to_json(nlohmann::json& j) const = 0;
};

/**
 * @brief Exception thrown when a transformation cannot be applied
 */
class InvalidTransformationException : public std::exception {
private:
    std::string message_;

public:
    explicit InvalidTransformationException(const std::string& message) : message_(message) {}

    const char* what() const noexcept override { return message_.c_str(); }
};

/**
 * @brief Exception thrown when a transformation description is invalid
 */
class InvalidTransformationDescriptionException : public std::exception {
private:
    std::string message_;

public:
    explicit InvalidTransformationDescriptionException(const std::string& message) : message_(message) {}

    const char* what() const noexcept override { return message_.c_str(); }
};

} // namespace transformations
} // namespace sdfg
