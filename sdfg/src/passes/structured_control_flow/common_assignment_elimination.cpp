#include "sdfg/passes/structured_control_flow/common_assignment_elimination.h"

#include "sdfg/structured_control_flow/structured_loop.h"

namespace sdfg {
namespace passes {

CommonAssignmentElimination::CommonAssignmentElimination()
    : Pass() {

      };

std::string CommonAssignmentElimination::name() { return "CommonAssignmentElimination"; };

bool CommonAssignmentElimination::
    run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    bool applied = false;

    // Traverse structured SDFG
    std::list<structured_control_flow::ControlFlowNode*> queue = {&builder.subject().root()};
    while (!queue.empty()) {
        auto current = queue.front();
        queue.pop_front();

        // Add children to queue
        if (auto sequence_stmt = dyn_cast<structured_control_flow::Sequence*>(current)) {
            for (size_t i = 0; i < sequence_stmt->size(); i++) {
                auto child = sequence_stmt->at(i);
                if (auto if_else_stmt = dyn_cast<structured_control_flow::IfElse*>(&child.first)) {
                    if (if_else_stmt->size() < 2) {
                        continue;
                    }
                    auto& first_branch = if_else_stmt->at(0).first;
                    if (first_branch.size() == 0) {
                        continue;
                    }
                    auto& last_transition = first_branch.at(first_branch.size() - 1).second;
                    if (last_transition.assignments().size() == 0) {
                        continue;
                    }

                    // Check waw dependencies
                    for (auto& entry : last_transition.assignments()) {
                        auto& first_assign = entry.first;
                        auto& first_assignment = entry.second;
                        if (child.second.assignments().find(first_assign) != child.second.assignments().end()) {
                            continue;
                        }

                        // Check if all branches end with same assignment
                        bool all_branches_same = true;
                        for (size_t j = 1; j < if_else_stmt->size(); j++) {
                            auto& branch = if_else_stmt->at(j).first;
                            if (branch.size() == 0) {
                                all_branches_same = false;
                                break;
                            }

                            auto& transition = branch.at(branch.size() - 1).second;
                            if (transition.assignments().find(first_assign) == transition.assignments().end()) {
                                all_branches_same = false;
                                break;
                            }
                            if (!symbolic::eq(first_assignment, transition.assignments().at(first_assign))) {
                                all_branches_same = false;
                                break;
                            }
                        }

                        if (!all_branches_same) {
                            continue;
                        }

                        child.second.assignments().insert({first_assign, first_assignment});
                        applied = true;
                    }
                }
            }

            for (size_t i = 0; i < sequence_stmt->size(); i++) {
                queue.push_back(&sequence_stmt->at(i).first);
            }
        } else if (auto if_else_stmt = dyn_cast<structured_control_flow::IfElse*>(current)) {
            for (size_t i = 0; i < if_else_stmt->size(); i++) {
                queue.push_back(&if_else_stmt->at(i).first);
            }
        } else if (auto loop_stmt = dyn_cast<structured_control_flow::While*>(current)) {
            queue.push_back(&loop_stmt->root());
        } else if (auto sloop_stmt = dyn_cast<structured_control_flow::StructuredLoop*>(current)) {
            queue.push_back(&sloop_stmt->root());
        }
    }

    return applied;
};

} // namespace passes
} // namespace sdfg
