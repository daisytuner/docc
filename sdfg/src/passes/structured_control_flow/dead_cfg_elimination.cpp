#include "sdfg/passes/structured_control_flow/dead_cfg_elimination.h"

#include "sdfg/structured_control_flow/structured_loop.h"

namespace sdfg {
namespace passes {

bool DeadCFGElimination::is_dead(const structured_control_flow::ControlFlowNode& node) {
    if (auto block_stmt = dynamic_cast<const structured_control_flow::Block*>(&node)) {
        return (block_stmt->dataflow().nodes().size() == 0);
    } else if (auto sequence_stmt = dynamic_cast<const structured_control_flow::Sequence*>(&node)) {
        return (sequence_stmt->size() == 0);
    } else if (auto if_else_stmt = dynamic_cast<const structured_control_flow::IfElse*>(&node)) {
        return (if_else_stmt->size() == 0);
    } else if (auto while_stmt = dynamic_cast<const structured_control_flow::While*>(&node)) {
        return is_dead(while_stmt->root());
    } else if (auto sloop = dynamic_cast<const structured_control_flow::StructuredLoop*>(&node)) {
        if (sloop->root().size() != 0) {
            return false;
        }
        // TODO: Check use of indvar later
        return permissive_;
    }

    return false;
};

DeadCFGElimination::DeadCFGElimination()
    : Pass(), permissive_(false) {

      };

DeadCFGElimination::DeadCFGElimination(bool permissive)
    : Pass(), permissive_(permissive) {

      };

std::string DeadCFGElimination::name() { return "DeadCFGElimination"; };

bool DeadCFGElimination::run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    bool applied = false;

    auto& sdfg = builder.subject();

    auto& root = sdfg.root();
    if (root.size() == 0) {
        return false;
    }

    std::list<structured_control_flow::ControlFlowNode*> queue = {&sdfg.root()};
    while (!queue.empty()) {
        auto curr = queue.front();
        queue.pop_front();

        if (auto sequence_stmt = dynamic_cast<structured_control_flow::Sequence*>(curr)) {
            // Simplify
            size_t i = 0;
            while (i < sequence_stmt->size()) {
                auto child = sequence_stmt->at(i);
                symbolic::SymbolSet dead_lhs;
                for (auto& entry : child.second.assignments()) {
                    if (symbolic::eq(entry.first, entry.second)) {
                        dead_lhs.insert(entry.first);
                    }
                }
                for (auto& lhs : dead_lhs) {
                    child.second.assignments().erase(lhs);
                    applied = true;
                }

                // Return node found, everything after is dead
                if (auto return_node = dynamic_cast<structured_control_flow::Return*>(&child.first)) {
                    if (child.second.assignments().size() > 0) {
                        // Clear assignments
                        child.second.assignments().clear();
                        applied = true;
                    }
                    for (size_t j = i + 1; j < sequence_stmt->size();) {
                        builder.remove_child(*sequence_stmt, i + 1);
                        applied = true;
                    }
                    break;
                }

                // Non-empty transitions are not safe to remove
                if (!child.second.empty()) {
                    i++;
                    continue;
                }

                // Dead
                if (is_dead(child.first)) {
                    builder.remove_child(*sequence_stmt, i);
                    applied = true;
                    continue;
                }

                // Trivial branch
                if (auto if_else_stmt = dynamic_cast<structured_control_flow::IfElse*>(&child.first)) {
                    auto branch = if_else_stmt->at(0);
                    if (symbolic::is_true(branch.second)) {
                        builder.move_children(branch.first, *sequence_stmt, i + 1);
                        builder.remove_child(*sequence_stmt, i);
                        applied = true;
                        continue;
                    }
                }

                i++;
            }

            // Add to queue
            for (size_t j = 0; j < sequence_stmt->size(); j++) {
                queue.push_back(&sequence_stmt->at(j).first);
            }
        } else if (auto if_else_stmt = dynamic_cast<structured_control_flow::IfElse*>(curr)) {
            // False branches are safe to remove
            size_t i = 0;
            while (i < if_else_stmt->size()) {
                auto child = if_else_stmt->at(i);
                if (symbolic::is_false(child.second)) {
                    builder.remove_case(*if_else_stmt, i);
                    applied = true;
                    continue;
                }

                i++;
            }

            // Trailing dead branches are safe to remove
            if (if_else_stmt->size() > 0) {
                if (is_dead(if_else_stmt->at(if_else_stmt->size() - 1).first)) {
                    builder.remove_case(*if_else_stmt, if_else_stmt->size() - 1);
                    applied = true;
                }
            }

            // If-else to simple if conversion
            if (if_else_stmt->size() == 2) {
                auto if_condition = if_else_stmt->at(0).second;
                auto else_condition = if_else_stmt->at(1).second;
                if (symbolic::eq(if_condition->logical_not(), else_condition)) {
                    if (is_dead(if_else_stmt->at(1).first)) {
                        builder.remove_case(*if_else_stmt, 1);
                        applied = true;
                    } else if (is_dead(if_else_stmt->at(0).first)) {
                        builder.remove_case(*if_else_stmt, 0);
                        applied = true;
                    }
                }
            }

            // Add to queue
            for (size_t j = 0; j < if_else_stmt->size(); j++) {
                queue.push_back(&if_else_stmt->at(j).first);
            }
        } else if (auto loop_stmt = dynamic_cast<structured_control_flow::While*>(curr)) {
            auto& root = loop_stmt->root();
            queue.push_back(&root);
        } else if (auto sloop_stmt = dynamic_cast<structured_control_flow::StructuredLoop*>(curr)) {
            auto& root = sloop_stmt->root();
            queue.push_back(&root);
        } else if (auto map_stmt = dynamic_cast<structured_control_flow::Map*>(curr)) {
            auto& root = map_stmt->root();
            queue.push_back(&root);
        }
    }

    return applied;
};

} // namespace passes
} // namespace sdfg
