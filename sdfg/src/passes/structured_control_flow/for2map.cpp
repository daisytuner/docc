#include "sdfg/passes/structured_control_flow/for2map.h"

#include "sdfg/analysis/loop_analysis.h"
#include "sdfg/analysis/scope_analysis.h"
#include "sdfg/analysis/users.h"
#include "sdfg/passes/pipeline.h"

namespace sdfg {
namespace passes {

std::string For2MapPass::name() { return "For2Map"; }

bool For2MapPass::can_be_applied(
    builder::StructuredSDFGBuilder& builder,
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::For& for_stmt
) {
    if (!for_stmt.is_monotonic()) {
        return false;
    }

    // Criterion: Loop must not have side-effecting body
    std::list<const structured_control_flow::ControlFlowNode*> queue = {&for_stmt.root()};
    while (!queue.empty()) {
        auto current = queue.front();
        queue.pop_front();

        if (auto block = dynamic_cast<const structured_control_flow::Block*>(current)) {
            for (auto& node : block->dataflow().nodes()) {
                if (auto library_node = dynamic_cast<const data_flow::LibraryNode*>(&node)) {
                    if (library_node->side_effect()) {
                        return false;
                    }
                }
            }
        } else if (auto seq = dynamic_cast<const structured_control_flow::Sequence*>(current)) {
            for (size_t i = 0; i < seq->size(); i++) {
                auto& child = seq->at(i).first;
                queue.push_back(&child);
            }
        } else if (auto ifelse = dynamic_cast<const structured_control_flow::IfElse*>(current)) {
            for (size_t i = 0; i < ifelse->size(); i++) {
                auto& branch = ifelse->at(i).first;
                queue.push_back(&branch);
            }
        } else if (auto loop = dynamic_cast<const structured_control_flow::StructuredLoop*>(current)) {
            queue.push_back(&loop->root());
        } else if (auto while_stmt = dynamic_cast<const structured_control_flow::While*>(current)) {
            queue.push_back(&while_stmt->root());
        } else if (auto for_stmt = dynamic_cast<const structured_control_flow::Break*>(current)) {
            // Do nothing
        } else if (auto for_stmt = dynamic_cast<const structured_control_flow::Continue*>(current)) {
            // Do nothing
        } else if (auto for_stmt = dynamic_cast<const structured_control_flow::Return*>(current)) {
            return false;
        } else {
            throw InvalidSDFGException("Unknown control flow node type in For2Map pass.");
        }
    }

    // Criterion: loop must be data-parallel w.r.t containers
    auto dependencies = data_dependency_analysis_->dependencies(for_stmt);

    // a. No true dependencies (RAW) between iterations
    for (auto& dep : dependencies) {
        if (dep.second.type == analysis::LoopCarriedDependency::LOOP_CARRIED_DEPENDENCY_READ_WRITE) {
            return false;
        }
    }

    // b. False dependencies (WAW) are limited to loop-local variables
    auto& users = analysis_manager.get<analysis::Users>();
    analysis::UsersView body_users(users, for_stmt.root());
    auto locals = users.locals(for_stmt.root());
    for (auto& dep : dependencies) {
        auto& container = dep.first;
        auto& type = builder.subject().type(container);

        // Must be loop-local variable
        if (locals.find(container) == locals.end()) {
            // Special case: Constant scalar assignments
            if (type.type_id() == types::TypeID::Scalar) {
                auto writes = body_users.writes(container);
                auto reads = body_users.reads(container);
                if (writes.size() == 1 && reads.empty()) {
                    auto write = writes.front();
                    if (auto write_transition =
                            dynamic_cast<const structured_control_flow::Transition*>(write->element())) {
                        auto lhs = symbolic::symbol(container);
                        auto rhs = write_transition->assignments().at(lhs);
                        if (SymEngine::is_a<SymEngine::Integer>(*rhs)) {
                            continue;
                        }
                    }
                }
            }

            return false;
        }

        // Check for pointers that they point to loop-local storage
        if (type.type_id() != types::TypeID::Pointer) {
            continue;
        }
        if (type.storage_type().allocation() == types::StorageType::AllocationType::Managed) {
            continue;
        }

        // or alias of loop-local storage
        if (users.moves(container).size() != 1) {
            return false;
        }
        auto move = users.moves(container).front();
        auto move_node = static_cast<const data_flow::AccessNode*>(move->element());
        auto& move_graph = move_node->get_parent();
        auto& move_edge = *move_graph.in_edges(*move_node).begin();
        auto& move_src = static_cast<const data_flow::AccessNode&>(move_edge.src());
        if (locals.find(move_src.data()) == locals.end()) {
            return false;
        }
        auto& move_type = builder.subject().type(move_src.data());
        if (move_type.storage_type().allocation() == types::StorageType::AllocationType::Unmanaged) {
            return false;
        }
    }

    // c. indvar not used after for
    if (locals.find(for_stmt.indvar()->get_name()) != locals.end()) {
        return false;
    }

    return true;
}

bool For2MapPass::run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    this->data_dependency_analysis_ = std::make_unique<analysis::DataDependencyAnalysis>(builder.subject(), true);
    this->data_dependency_analysis_->run(analysis_manager);

    auto& loop_analysis = analysis_manager.get<analysis::LoopAnalysis>();
    auto& loop_tree = loop_analysis.loop_tree();

    // Traverse loops in bottom-up fashion (reverse loop)
    std::list<structured_control_flow::For*> for_queue;
    for (auto& entry : loop_tree) {
        if (auto for_stmt = dynamic_cast<structured_control_flow::For*>(entry.first)) {
            for_queue.push_front(for_stmt);
        }
    }

    // Mark for loops that can be converted
    std::list<structured_control_flow::For*> map_queue;
    for (auto& for_loop : for_queue) {
        if (this->can_be_applied(builder, analysis_manager, *for_loop)) {
            map_queue.push_back(for_loop);
        }
    }

    // Convert marked for loops
    bool applied = false;
    auto& scope_analysis = analysis_manager.get<analysis::ScopeAnalysis>();
    for (auto& for_stmt : map_queue) {
        auto parent = static_cast<structured_control_flow::Sequence*>(scope_analysis.parent_scope(for_stmt));
        builder.convert_for(*parent, *for_stmt);
        applied = true;
    }

    return applied;
}

} // namespace passes
} // namespace sdfg
