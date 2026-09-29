#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sdfg/analysis/control_flow_analysis.h>

namespace py = pybind11;

/**
 * @brief Python wrapper for ControlFlowAnalysis
 */
class PyControlFlowAnalysis {
private:
    sdfg::analysis::AnalysisManager& manager_;
    sdfg::analysis::ControlFlowAnalysis& analysis_;

public:
    PyControlFlowAnalysis(sdfg::analysis::AnalysisManager& manager)
        : manager_(manager), analysis_(manager.get<sdfg::analysis::ControlFlowAnalysis>()) {
    }

    sdfg::analysis::ControlFlowAnalysis& analysis() {
        return analysis_;
    }
};
