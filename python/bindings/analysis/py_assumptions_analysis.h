#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sdfg/analysis/assumptions_analysis.h>

namespace py = pybind11;

/**
 * @brief Python wrapper for AssumptionsAnalysis
 */
class PyAssumptionsAnalysis {
private:
    sdfg::analysis::AnalysisManager& manager_;
    sdfg::analysis::AssumptionsAnalysis& analysis_;

public:
    PyAssumptionsAnalysis(sdfg::analysis::AnalysisManager& manager)
        : manager_(manager), analysis_(manager.get<sdfg::analysis::AssumptionsAnalysis>()) {
    }

    sdfg::analysis::AssumptionsAnalysis& analysis() {
        return analysis_;
    }
};
