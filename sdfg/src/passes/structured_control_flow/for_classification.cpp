#include "sdfg/passes/structured_control_flow/for_classification.h"

#include <set>
#include <vector>

#include "sdfg/analysis/loop_analysis.h"
#include "sdfg/analysis/loop_carried_dependency_analysis.h"
#include "sdfg/analysis/users.h"
#include "sdfg/passes/pipeline.h"

namespace sdfg {
namespace passes {

std::string ForClassificationPass::name() { return "ForClassification"; }

ForClassificationPass::Classification ForClassificationPass::classify(
    builder::StructuredSDFGBuilder& builder,
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::For& for_stmt,
    std::vector<structured_control_flow::ReductionInfo>& reductions
) {
    reductions.clear();

    if (!for_stmt.is_monotonic()) {
        return Classification::None;
    }

    auto& loop_analysis = analysis_manager.get<analysis::LoopAnalysis>();

    if (loop_analysis.loop_info(&for_stmt).has_side_effects) {
        return Classification::None;
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
                        return Classification::None;
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
            return Classification::None;
        } else {
            throw InvalidSDFGException("Unknown control flow node type in ForClassification pass.");
        }
    }

    // Criterion: loop must be data-parallel w.r.t containers
    auto& lcd = analysis_manager.get<analysis::LoopCarriedDependencyAnalysis>();
    auto& dependencies = lcd.dependencies(for_stmt);

    // Recognized reductions: loop-carried read-write dependencies that are
    // reorderable accumulations. These are NOT hazards for a Reduce loop.
    const auto& recognized_reductions = lcd.reductions(for_stmt);
    std::set<std::string> reduction_containers;
    for (auto& reduction : recognized_reductions) {
        reduction_containers.insert(reduction.container);
    }
    bool is_reduction = lcd.is_reduction_only(for_stmt);

    // a. The only true dependencies (RAW / undefined) allowed between iterations
    //    are recognized reductions. Any other hazard rules out both Map and
    //    Reduce.
    if (lcd.has_loop_carried_hazard(for_stmt) && !is_reduction) {
        return Classification::None;
    }

    // b. False dependencies (WAW) are limited to loop-local variables. Reduction
    //    accumulators are exempt: their cross-iteration writes are handled by
    //    the reduction itself.
    auto& users = analysis_manager.get<analysis::Users>();
    analysis::UsersView body_users(users, for_stmt.root());
    auto locals = users.locals(for_stmt.root());
    for (auto& dep : dependencies) {
        auto& container = dep.first;
        if (reduction_containers.count(container) != 0) {
            continue;
        }
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

            return Classification::None;
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
            return Classification::None;
        }
        auto move = users.moves(container).front();
        auto move_node = static_cast<const data_flow::AccessNode*>(move->element());
        auto& move_graph = move_node->get_parent();
        auto& move_edge = *move_graph.in_edges(*move_node).begin();
        auto& move_src = static_cast<const data_flow::AccessNode&>(move_edge.src());
        if (locals.find(move_src.data()) == locals.end()) {
            return Classification::None;
        }
        auto& move_type = builder.subject().type(move_src.data());
        if (move_type.storage_type().allocation() == types::StorageType::AllocationType::Unmanaged) {
            return Classification::None;
        }
    }

    // c. indvar not used after for
    if (locals.find(for_stmt.indvar()->get_name()) != locals.end()) {
        return Classification::None;
    }

    if (is_reduction) {
        reductions.assign(recognized_reductions.begin(), recognized_reductions.end());
        return Classification::Reduce;
    }
    return Classification::Map;
}

bool ForClassificationPass::run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    auto& loop_analysis = analysis_manager.get<analysis::LoopAnalysis>();

    // Traverse loops in bottom-up fashion (reverse loop)
    std::list<structured_control_flow::For*> for_queue;
    for (auto& loop : loop_analysis.loops_in_pre_order()) {
        if (auto for_stmt = dyn_cast<structured_control_flow::For*>(loop)) {
            for_queue.push_front(for_stmt);
        }
    }

    // Mark for loops that can be converted, recording the target classification
    // (and the reductions for Reduce loops) up front while the analyses are valid.
    std::list<structured_control_flow::For*> map_queue;
    std::list<std::pair<structured_control_flow::For*, std::vector<structured_control_flow::ReductionInfo>>>
        reduce_queue;
    for (auto& for_loop : for_queue) {
        std::vector<structured_control_flow::ReductionInfo> reductions;
        switch (this->classify(builder, analysis_manager, *for_loop, reductions)) {
            case Classification::Map:
                map_queue.push_back(for_loop);
                break;
            case Classification::Reduce:
                reduce_queue.emplace_back(for_loop, std::move(reductions));
                break;
            case Classification::None:
                break;
        }
    }

    // Convert marked for loops
    bool applied = false;
    for (auto& for_stmt : map_queue) {
        auto parent = static_cast<structured_control_flow::Sequence*>(for_stmt->get_parent());
        builder.convert_for(*parent, *for_stmt);
        applied = true;
    }
    for (auto& entry : reduce_queue) {
        auto* for_stmt = entry.first;
        auto parent = static_cast<structured_control_flow::Sequence*>(for_stmt->get_parent());
        builder.convert_for_to_reduce(*parent, *for_stmt, entry.second);
        applied = true;
    }

    return applied;
}

} // namespace passes
} // namespace sdfg
