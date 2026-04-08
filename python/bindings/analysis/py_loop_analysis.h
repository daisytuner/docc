#pragma once

#include <list>
#include <optional>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sdfg/analysis/assumptions_analysis.h>
#include <sdfg/analysis/loop_analysis.h>
#include <sdfg/structured_control_flow/structured_loop.h>
#include <sstream>
#include <unordered_set>
#include <vector>

namespace py = pybind11;

/**
 * @brief Python wrapper for LoopAnalysis
 */
class PyLoopAnalysis {
private:
    sdfg::analysis::AnalysisManager& manager_;
    sdfg::analysis::LoopAnalysis& analysis_;

public:
    PyLoopAnalysis(sdfg::analysis::AnalysisManager& manager)
        : manager_(manager), analysis_(manager.get<sdfg::analysis::LoopAnalysis>()) {}

    sdfg::analysis::LoopAnalysis& analysis() { return analysis_; }

    /**
     * @brief Get all loops in the SDFG in DFS order
     */
    std::vector<sdfg::structured_control_flow::ControlFlowNode*> loops() const { return analysis_.loops(); }

    /**
     * @brief Get loop information for a specific loop
     */
    sdfg::analysis::LoopInfo loop_info(sdfg::structured_control_flow::ControlFlowNode* loop) const {
        return analysis_.loop_info(loop);
    }

    /**
     * @brief Find a loop by its induction variable name
     * @return The loop node or nullptr if not found
     */
    sdfg::structured_control_flow::ControlFlowNode* find_loop_by_indvar(const std::string& indvar) {
        return analysis_.find_loop_by_indvar(indvar);
    }

    /**
     * @brief Get the parent loop of a given loop
     * @return The parent loop node or nullptr if this is an outermost loop
     */
    sdfg::structured_control_flow::ControlFlowNode* parent_loop(sdfg::structured_control_flow::ControlFlowNode* loop
    ) const {
        return analysis_.parent_loop(loop);
    }

    /**
     * @brief Get all outermost loops (loops with no parent loop)
     */
    std::vector<sdfg::structured_control_flow::ControlFlowNode*> outermost_loops() const {
        return analysis_.outermost_loops();
    }

    /**
     * @brief Check if a loop is an outermost loop
     */
    bool is_outermost_loop(sdfg::structured_control_flow::ControlFlowNode* loop) const {
        return analysis_.is_outermost_loop(loop);
    }

    /**
     * @brief Get all outermost Map nodes
     */
    std::vector<sdfg::structured_control_flow::ControlFlowNode*> outermost_maps() const {
        return analysis_.outermost_maps();
    }

    /**
     * @brief Get the immediate child loops of a given loop
     */
    std::vector<sdfg::structured_control_flow::ControlFlowNode*> children(sdfg::structured_control_flow::ControlFlowNode*
                                                                              node) const {
        return analysis_.children(node);
    }

    /**
     * @brief Get all descendant loops of a given loop
     */
    std::unordered_set<sdfg::structured_control_flow::ControlFlowNode*>
    descendants(sdfg::structured_control_flow::ControlFlowNode* loop) const {
        return analysis_.descendants(loop);
    }

    /**
     * @brief Get all paths from the given loop to leaf loops in the loop tree
     */
    std::list<std::vector<sdfg::structured_control_flow::ControlFlowNode*>>
    loop_tree_paths(sdfg::structured_control_flow::ControlFlowNode* loop) const {
        return analysis_.loop_tree_paths(loop);
    }
};
