#pragma once

#include <cmath>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sdfg/analysis/arguments_analysis.h>
#include <sdfg/structured_control_flow/control_flow_node.h>
#include <sdfg/symbolic/symbolic.h>
#include <sstream>
#include <symengine/eval_double.h>
#include <symengine/integer.h>

namespace py = pybind11;

/**
 * @brief Python wrapper for ArgumentsAnalysis
 */
class PyArgumentsAnalysis {
private:
    sdfg::analysis::AnalysisManager& manager_;
    sdfg::analysis::ArgumentsAnalysis& analysis_;

    static py::object expr_to_py(const sdfg::symbolic::Expression& expr) {
        if (expr.is_null()) {
            return py::none();
        }
        auto simplified = sdfg::symbolic::simplify(expr);
        if (SymEngine::is_a<SymEngine::Integer>(*simplified)) {
            return py::int_(SymEngine::rcp_static_cast<const SymEngine::Integer>(simplified)->as_int());
        }
        // Non-constant expression (dynamic size): return its string form.
        return py::str(simplified->__str__());
    }

public:
    PyArgumentsAnalysis(sdfg::analysis::AnalysisManager& manager)
        : manager_(manager), analysis_(manager.get<sdfg::analysis::ArgumentsAnalysis>()) {}

    sdfg::analysis::ArgumentsAnalysis& analysis() { return analysis_; }

    /// Total size in bytes of every argument at the given region, as a dict
    /// name -> int (or a string expression when the size is not constant).
    py::dict argument_sizes(sdfg::structured_control_flow::ControlFlowNode& node, bool allow_dynamic_sizes) {
        py::dict result;
        const auto& sizes = analysis_.argument_sizes(manager_, node, allow_dynamic_sizes);
        for (const auto& [name, expr] : sizes) {
            result[py::str(name)] = expr_to_py(expr);
        }
        return result;
    }

    /// Element size in bytes of every argument at the given region.
    py::dict argument_element_sizes(sdfg::structured_control_flow::ControlFlowNode& node, bool allow_dynamic_sizes) {
        py::dict result;
        const auto& sizes = analysis_.argument_element_sizes(manager_, node, allow_dynamic_sizes);
        for (const auto& [name, expr] : sizes) {
            result[py::str(name)] = expr_to_py(expr);
        }
        return result;
    }

    /// Read/write classification of every argument at the given region.
    py::dict arguments(sdfg::structured_control_flow::ControlFlowNode& node) {
        py::dict result;
        const auto& args = analysis_.arguments(manager_, node);
        for (const auto& [name, meta] : args) {
            py::dict info;
            info["is_scalar"] = meta.is_scalar;
            info["is_ptr"] = meta.is_ptr;
            info["is_input"] = meta.is_input;
            info["is_output"] = meta.is_output;
            info["is_explicit_input"] = meta.is_explicit_input;
            result[py::str(name)] = info;
        }
        return result;
    }
};
