#include "sdfg/passes/debug_info_propagation.h"

namespace sdfg {
namespace passes {

DebugInfoPropagation::DebugInfoPropagation() : Pass() {}

void DebugInfoPropagation::propagate(structured_control_flow::ControlFlowNode* current) {
    auto current_debug_info = current->debug_info();

    if (auto block = dyn_cast<structured_control_flow::Block*>(current)) {
        auto& graph = block->dataflow();
        for (auto& node : graph.nodes()) {
            current_debug_info = DebugInfo::merge(current_debug_info, node.debug_info());
        }
        for (auto& edge : graph.edges()) {
            current_debug_info = DebugInfo::merge(current_debug_info, edge.debug_info());
        }
    } else if (auto assignment_block = dyn_cast<structured_control_flow::AssignmentBlock*>(current)) {
        current_debug_info = DebugInfo::merge(current_debug_info, assignment_block->debug_info());
    } else if (auto sequence_stmt = dyn_cast<structured_control_flow::Sequence*>(current)) {
        for (size_t i = 0; i < sequence_stmt->size(); i++) {
            this->propagate(&sequence_stmt->at(i));
            current_debug_info = DebugInfo::merge(current_debug_info, sequence_stmt->at(i).debug_info());
        }
    } else if (auto if_else_stmt = dyn_cast<structured_control_flow::IfElse*>(current)) {
        for (size_t i = 0; i < if_else_stmt->size(); i++) {
            this->propagate(&if_else_stmt->at(i).first);
            current_debug_info = DebugInfo::merge(current_debug_info, if_else_stmt->at(i).first.debug_info());
        }
    } else if (auto while_stmt = dyn_cast<structured_control_flow::While*>(current)) {
        this->propagate(&while_stmt->root());
        current_debug_info = DebugInfo::merge(current_debug_info, while_stmt->root().debug_info());
    } else if (auto loop_stmt = dyn_cast<structured_control_flow::StructuredLoop*>(current)) {
        this->propagate(&loop_stmt->root());
        current_debug_info = DebugInfo::merge(current_debug_info, loop_stmt->root().debug_info());
    } else if (auto break_stmt = dyn_cast<structured_control_flow::Break*>(current)) {
        current_debug_info = DebugInfo::merge(current_debug_info, break_stmt->debug_info());
    } else if (auto continue_stmt = dyn_cast<structured_control_flow::Continue*>(current)) {
        current_debug_info = DebugInfo::merge(current_debug_info, continue_stmt->debug_info());
    } else if (auto return_stmt = dyn_cast<structured_control_flow::Return*>(current)) {
        current_debug_info = DebugInfo::merge(current_debug_info, return_stmt->debug_info());
    } else {
        throw InvalidSDFGException("Unsupported control flow node type");
    }

    current->set_debug_info(current_debug_info);
}

bool DebugInfoPropagation::run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    this->propagate(&builder.subject().root());
    return true;
}

std::string DebugInfoPropagation::name() { return "DebugInfoPropagation"; }

} // namespace passes
} // namespace sdfg
