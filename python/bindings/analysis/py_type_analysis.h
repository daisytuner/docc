#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sdfg/analysis/type_analysis.h>

namespace py = pybind11;

/**
 * @brief Python wrapper for TypeAnalysis
 */
class PyTypeAnalysis {
private:
    sdfg::analysis::AnalysisManager& manager_;
    sdfg::analysis::TypeAnalysis& analysis_;

public:
    PyTypeAnalysis(sdfg::analysis::AnalysisManager& manager)
        : manager_(manager), analysis_(manager.get<sdfg::analysis::TypeAnalysis>()) {
    }

    sdfg::analysis::TypeAnalysis& analysis() {
        return analysis_;
    }
};
