#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sdfg/analysis/dominance_analysis.h>

namespace py = pybind11;

/**
 * @brief Python wrapper for DominanceAnalysis
 */
class PyDominanceAnalysis {
private:
    sdfg::analysis::AnalysisManager& manager_;
    sdfg::analysis::DominanceAnalysis& analysis_;

public:
    PyDominanceAnalysis(sdfg::analysis::AnalysisManager& manager)
        : manager_(manager), analysis_(manager.get<sdfg::analysis::DominanceAnalysis>()) {
    }

    sdfg::analysis::DominanceAnalysis& analysis() {
        return analysis_;
    }
};
