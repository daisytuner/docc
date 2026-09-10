#include "sdfg/passes/structured_control_flow/dead_cfg_elimination.h"

#include <list>
#include <unordered_set>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/memlet.h"
#include "sdfg/element.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/control_flow_node.h"
#include "sdfg/structured_control_flow/if_else.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/structured_control_flow/while.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/visitor/for_each.h"

namespace sdfg {
namespace passes {

bool DeadCFGElimination::is_dead(const structured_control_flow::ControlFlowNode& node) {
    if (auto block_stmt = dyn_cast<const structured_control_flow::Block*>(&node)) {
        return (block_stmt->dataflow().nodes().size() == 0);
    } else if (auto* assign_stmt = dyn_cast<AssignmentBlock*>(&node)) {
        return (assign_stmt->empty());
    } else if (auto sequence_stmt = dyn_cast<const structured_control_flow::Sequence*>(&node)) {
        return (sequence_stmt->size() == 0);
    } else if (auto if_else_stmt = dyn_cast<const structured_control_flow::IfElse*>(&node)) {
        return (if_else_stmt->size() == 0);
    } else if (auto while_stmt = dyn_cast<const structured_control_flow::While*>(&node)) {
        return is_dead(while_stmt->root());
    } else if (auto sloop = dyn_cast<const structured_control_flow::StructuredLoop*>(&node)) {
        if (sloop->root().size() != 0) {
            return false;
        }
        // TODO: Check use of indvar later
        return permissive_;
    }

    return false;
};

bool DeadCFGElimination::is_trivial(structured_control_flow::Map* loop) {
    // Check if stride is 1
    if (!loop->is_contiguous()) {
        return false;
    }
    // Check if bound - init == 1
    auto trip_count = loop->num_iterations();
    if (trip_count.is_null()) {
        return false;
    }
    return symbolic::eq(trip_count, symbolic::one());
}

void DeadCFGElimination::
    update_loop_indvar_accesses(builder::StructuredSDFGBuilder& builder, structured_control_flow::Map* loop) {
    symbolic::Symbol indvar = loop->indvar();
    const auto& indvar_type = builder.subject().type(indvar->get_name());
    visitor::for_each_block(*loop, [&builder, &indvar, &indvar_type](structured_control_flow::Block& block) -> void {
        auto access_nodes = block.dataflow().data_nodes();
        for (auto* access_node : access_nodes) {
            // Skip constant nodes
            if (is_a(access_node->type_id(), ElementType::ConstantNode)) {
                continue;
            }
            // Skip access nodes on containers other than the indvar
            if (access_node->data() != indvar->get_name()) {
                continue;
            }

            auto& new_constant_node = builder.add_constant(block, "0", indvar_type, access_node->debug_info());
            std::unordered_set<data_flow::Memlet*> old_memlets;
            for (auto& memlet : block.dataflow().out_edges(*access_node)) {
                builder.add_memlet(
                    block,
                    new_constant_node,
                    memlet.src_conn(),
                    memlet.dst(),
                    memlet.dst_conn(),
                    memlet.subset(),
                    memlet.base_type(),
                    memlet.debug_info()
                );
                old_memlets.insert(&memlet);
            }
            for (auto* old_memlet : old_memlets) {
                builder.remove_memlet(block, *old_memlet);
            }
            builder.remove_node(block, *access_node);
        }
    });
}

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

        if (auto* assign_block = dyn_cast<structured_control_flow::AssignmentBlock*>(curr)) {
            symbolic::SymbolSet dead_lhs;
            for (auto& entry : assign_block->assignments()) {
                if (symbolic::eq(entry.first, entry.second)) {
                    dead_lhs.insert(entry.first);
                }
            }
            for (auto& lhs : dead_lhs) {
                assign_block->assignments().erase(lhs);
                applied = true;
            }
        } else if (auto sequence_stmt = dyn_cast<structured_control_flow::Sequence*>(curr)) {
            // Simplify
            size_t i = 0;
            while (i < sequence_stmt->size()) {
                auto& child = sequence_stmt->at(i);

                // Return node found, everything after is dead
                if (auto return_node = dyn_cast<structured_control_flow::Return*>(&child)) {
                    for (size_t j = i + 1; j < sequence_stmt->size();) {
                        builder.remove_child(*sequence_stmt, i + 1);
                        applied = true;
                    }
                    break;
                }

                // Dead
                if (is_dead(child)) {
                    builder.remove_child(*sequence_stmt, i);
                    applied = true;
                    continue;
                }

                // Trivial branch
                if (auto if_else_stmt = dyn_cast<structured_control_flow::IfElse*>(&child)) {
                    auto branch = if_else_stmt->at(0);
                    if (symbolic::is_true(branch.second)) {
                        builder.move_children(branch.first, *sequence_stmt, i + 1);
                        builder.remove_child(*sequence_stmt, i);
                        applied = true;
                        continue;
                    }
                }

                // Trivial structured loop (bound - init == 1 and stride == 1)
                if (auto sloop = dyn_cast<structured_control_flow::Map*>(&child)) {
                    if (is_trivial(sloop)) {
                        auto indvar = sloop->indvar();
                        auto init = sloop->init();
                        sloop->root().replace(indvar, init);
                        this->update_loop_indvar_accesses(builder, sloop);

                        // Move children from loop body to parent sequence
                        builder.move_children(sloop->root(), *sequence_stmt, i + 1);

                        // Remove the loop
                        builder.remove_child(*sequence_stmt, i);
                        applied = true;
                        continue;
                    }
                }

                i++;
            }

            // Add to queue
            for (size_t j = 0; j < sequence_stmt->size(); j++) {
                queue.push_back(&sequence_stmt->at(j));
            }
        } else if (auto if_else_stmt = dyn_cast<structured_control_flow::IfElse*>(curr)) {
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
        } else if (auto loop_stmt = dyn_cast<structured_control_flow::While*>(curr)) {
            auto& root = loop_stmt->root();
            queue.push_back(&root);
        } else if (auto sloop_stmt = dyn_cast<structured_control_flow::StructuredLoop*>(curr)) {
            auto& root = sloop_stmt->root();
            queue.push_back(&root);
        } else if (auto map_stmt = dyn_cast<structured_control_flow::Map*>(curr)) {
            auto& root = map_stmt->root();
            queue.push_back(&root);
        }
    }

    return applied;
};

} // namespace passes
} // namespace sdfg
