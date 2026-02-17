#include "sdfg/passes/pass.h"

#include <chrono>

#include "sdfg/helpers/helpers.h"
#include "sdfg/passes/statistics.h"

namespace sdfg {
namespace passes {

bool Pass::run(builder::SDFGBuilder& builder, bool create_report) {
    std::chrono::time_point<std::chrono::system_clock, std::chrono::nanoseconds> start;
    if (PassStatistics::instance().enabled()) {
        start = std::chrono::high_resolution_clock::now();
#ifndef NDEBUG
        DEBUG_PRINTLN("Started SDFG Pass '" << this->name() << "' on '" << builder.subject().name() << "'");
#endif
    }

    bool applied = this->run_pass(builder);

    if (PassStatistics::instance().enabled()) {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        PassStatistics::instance().add_sdfg_pass(this->name(), duration);
#ifndef NDEBUG
        DEBUG_PRINTLN("Finished SDFG Pass '" << this->name() << "' in " << duration << " ms");
#endif
    }

#ifndef NDEBUG
    builder.subject().validate();
#endif

    return applied;
};

bool Pass::run(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager, bool create_report) {
    std::chrono::time_point<std::chrono::system_clock, std::chrono::nanoseconds> start;
    if (PassStatistics::instance().enabled()) {
        start = std::chrono::high_resolution_clock::now();
#ifndef NDEBUG
        DEBUG_PRINTLN("Started Structured SDFG Pass '" << this->name() << "' on '" << builder.subject().name() << "'");
#endif
    }

    bool applied = this->run_pass(builder, analysis_manager);
    this->invalidates(analysis_manager, applied);

    if (PassStatistics::instance().enabled()) {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        PassStatistics::instance().add_structured_sdfg_pass(this->name(), duration);
#ifndef NDEBUG
        DEBUG_PRINTLN("Finished Structured SDFG Pass '" << this->name() << "' in " << duration << " ms");
#endif
    }

#ifndef NDEBUG
    builder.subject().validate();
#endif

    return applied;
};

bool Pass::run_pass(builder::SDFGBuilder& builder) { throw std::logic_error("Not implemented"); };

bool Pass::run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    throw std::logic_error("Not implemented");
};

void Pass::invalidates(analysis::AnalysisManager& analysis_manager, bool applied) {
    if (applied) {
        analysis_manager.invalidate_all();
    }
};

} // namespace passes
} // namespace sdfg
