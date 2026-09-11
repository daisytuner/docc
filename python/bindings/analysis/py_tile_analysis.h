#pragma once

#include <string>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <symengine/eval_double.h>

#include <sdfg/analysis/analysis.h>
#include <sdfg/structured_control_flow/control_flow_node.h>
#include <sdfg/structured_control_flow/structured_loop.h>
#include <sdfg/symbolic/symbolic.h>
#include <sdfg/tiles/analysis/tile_analysis.h>
#include <sdfg/tiles/locality.h>
#include <sdfg/tiles/tile.h>

namespace py = pybind11;

/// Constant value of a symbolic expression as a Python int, or None if it has free symbols.
inline py::object tile_eval_const(const sdfg::symbolic::Expression& expr) {
    if (expr.is_null() || !sdfg::symbolic::atoms(expr).empty()) {
        return py::none();
    }
    return py::int_(static_cast<long long>(SymEngine::eval_double(*expr)));
}

inline const char* tile_space_name(sdfg::tiles::Space space) {
    switch (space) {
        case sdfg::tiles::Space::Global:
            return "global";
        case sdfg::tiles::Space::Shared:
            return "shared";
        case sdfg::tiles::Space::Register:
            return "register";
    }
    return "unknown";
}

/**
 * @brief Python wrapper for the schedule-aware TileAnalysis (sdfg::tiles).
 */
class PyTileAnalysis {
private:
    sdfg::analysis::AnalysisManager& manager_;
    sdfg::tiles::TileAnalysis& analysis_;

public:
    PyTileAnalysis(sdfg::analysis::AnalysisManager& manager)
        : manager_(manager), analysis_(manager.get<sdfg::tiles::TileAnalysis>()) {}

    sdfg::tiles::TileAnalysis& analysis() { return analysis_; }

    /// The tile of @p container at loop @p loop (a loop node from LoopAnalysis), or None. The dict is
    /// {container, reads, writes, cooperative, space, num_axes, shape, size}; `space` is the required
    /// storage tier (register/shared/global) derived from the enclosing schedules, and `size` is the
    /// tile element count (None when not a compile-time constant).
    py::object tile(sdfg::structured_control_flow::ControlFlowNode* loop, const std::string& container) const {
        if (loop == nullptr) {
            return py::none();
        }
        const sdfg::tiles::Tile* t = analysis_.tile(*loop, container);
        if (t == nullptr) {
            return py::none();
        }

        py::dict d;
        d["container"] = t->container();
        d["reads"] = t->reads();
        d["writes"] = t->writes();
        d["cooperative"] = t->cooperative();
        d["space"] = tile_space_name(t->required_space());
        d["num_axes"] = t->axes().size();

        // Loop-scoped storage tier: where staging AT this loop actually lands (grid axes above are
        // fixed ancestors), or None if staging here is illegal. This is the actionable space for a
        // local-storage decision, unlike the global `space` above.
        if (auto* sloop = dynamic_cast<sdfg::structured_control_flow::StructuredLoop*>(loop)) {
            auto placement = t->placement(*sloop, manager_).required_space(t->writes());
            if (placement.has_value()) {
                d["placement_space"] = tile_space_name(*placement);
            } else {
                d["placement_space"] = py::none();
            }
        } else {
            d["placement_space"] = py::none();
        }

        const auto& layout = t->source();
        py::list shape;
        for (const auto& s : layout.shape()) {
            shape.append(tile_eval_const(s));
        }
        d["shape"] = shape;
        d["size"] = tile_eval_const(layout.size());
        return d;
    }
};
