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
                auto& child = sequence_stmt->at(i);
                if (auto if_else_stmt = dyn_cast<structured_control_flow::IfElse*>(&child)) {
                    if (if_else_stmt->size() < 2) {
                        continue;
                    }
                    auto& first_branch = if_else_stmt->at(0).first;
                    if (first_branch.size() == 0) {
                        continue;
                    }
                    auto* last_assigns = dyn_cast<AssignmentBlock*>(&first_branch.at(first_branch.size() - 1));
                    if (!last_assigns || last_assigns->empty()) {
                        continue;
                    }

                    Assignments hoisted_assignments;

                    // Check waw dependencies
                    for (auto& entry : last_assigns->assignments()) {
                        auto& first_assign = entry.first;
                        auto& first_assignment = entry.second;

                        // Check if all branches end with same assignment
                        bool all_branches_same = true;
                        for (size_t j = 1; j < if_else_stmt->size(); j++) {
                            auto& branch = if_else_stmt->at(j).first;
                            if (branch.size() == 0) {
                                all_branches_same = false;
                                break;
                            }

                            auto* assign_block = dyn_cast<AssignmentBlock*>(&branch.at(branch.size() - 1));
                            if (!assign_block) {
                                all_branches_same = false;
                                break;
                            }
                            auto& other_branch_assigns = assign_block->assignments();
                            if (other_branch_assigns.find(first_assign) == other_branch_assigns.end()) {
                                all_branches_same = false;
                                break;
                            }
                            if (!symbolic::eq(first_assignment, other_branch_assigns.at(first_assign))) {
                                all_branches_same = false;
                                break;
                            }
                        }

                        if (!all_branches_same) {
                            continue;
                        }

                        hoisted_assignments.insert({first_assign, first_assignment});
                    }

                    if (!hoisted_assignments.empty()) {
                        AssignmentBlock* existing = nullptr;
                        if (i + 1 < sequence_stmt->size()) {
                            // currently, this pass is called in a loop to hoist assingments out multiple levels.
                            // And it does not remove the source, so this can easily lead to infinite loops if it does
                            // not achieve stable results. So needs to not make changes on its own results
                            auto& next_stmt = sequence_stmt->at(i + 1);
                            if (auto assign_block = dyn_cast<AssignmentBlock*>(&next_stmt)) {
                                existing = assign_block;
                                for (const auto& target : hoisted_assignments | std::views::keys) {
                                    for (auto& val : assign_block->assignments() | std::views::values) {
                                        if (has_symbol(*val, *target)) {
                                            // existing block uses what we would hoist, needs to go in a separate
                                            // assignments before
                                            existing = nullptr;
                                            break;
                                        }
                                    }
                                    if (!existing) {
                                        break;
                                    }
                                }

                                if (existing) {
                                    for (auto& hoisted : hoisted_assignments) {
                                        bool added = existing->add_if_not_overwritten(hoisted.first, hoisted.second);
                                        if (added) {
                                            applied = true;
                                        }
                                    }
                                }
                            }
                        }
                        if (!existing) {
                            builder.add_assignments_at(
                                *sequence_stmt, i + 1, hoisted_assignments, sequence_stmt->debug_info()
                            );
                            applied = true;
                        }
                    }
                }
            }

            for (size_t i = 0; i < sequence_stmt->size(); i++) {
                queue.push_back(&sequence_stmt->at(i));
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
