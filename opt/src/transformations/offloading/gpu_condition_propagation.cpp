#include "sdfg/transformations/offloading/gpu_condition_propagation.h"
#include <vector>
#include "sdfg/analysis/scope_analysis.h"
#include "sdfg/analysis/users.h"
#include "sdfg/data_flow/library_nodes/barrier_local_node.h"
#include "sdfg/element.h"
#include "sdfg/structured_control_flow/control_flow_node.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/structured_sdfg.h"
#include "sdfg/targets/gpu/gpu_schedule_type.h"
#include "sdfg/visitor/structured_sdfg_visitor.h"

namespace sdfg {
namespace transformations {

GPUConditionPropagation::GPUConditionPropagation(structured_control_flow::Map& map_) : map_(map_) {};


bool GPUConditionPropagation::
    can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    // Criterion: Must be a GPU map (CUDA or ROCm)
    if (!gpu::is_gpu_schedule(map_.schedule_type())) {
        return false;
    }

    // Criterion: Loop must contain thread barriers
    BarrierFinder barrier_finder(builder, analysis_manager);
    if (!barrier_finder.visit(&map_)) {
        return false;
    }

    return true;
}

void GPUConditionPropagation::apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    //  1. iterate over all nodes in the map body
    //  2. for each node, check if it contains a barrier
    //  3. if, propagate the map condition to the barrier
    //  4. else mark the node as relevant for condition propagation

    auto new_sched_type = map_.schedule_type();
    new_sched_type.set_property("nested_sync", "true");
    builder.update_schedule_type(map_, new_sched_type);

    std::vector<structured_control_flow::ControlFlowNode*> nodes_to_visit;
    nodes_to_visit.push_back(&map_.root());
    BarrierFinder barrier_finder(builder, analysis_manager);

    auto& scope_analysis = analysis_manager.get<analysis::ScopeAnalysis>();
    auto& users = analysis_manager.get<analysis::Users>();

    while (!nodes_to_visit.empty()) {
        auto current_node = nodes_to_visit.back();
        nodes_to_visit.pop_back();
        auto parent_scope = scope_analysis.parent_scope(current_node);
        auto parent_sequence = dynamic_cast<structured_control_flow::Sequence*>(parent_scope);
        analysis::UsersView current_users(users, *current_node);
        auto uses = current_users.uses(map_.indvar()->get_name());
        if (uses.empty()) {
            // Node does not use the map indvar, skip
            continue;
        }

        if (auto block_node = dynamic_cast<structured_control_flow::Block*>(current_node)) {
            if (!barrier_finder.visit(block_node)) {
                auto& if_else = builder.add_if_else_before(*parent_sequence, *block_node, {}, DebugInfo());
                auto& branch = builder.add_case(if_else, map_.condition());
                builder.move_child(*parent_sequence, parent_sequence->index(*block_node), branch);
            }
        } else if (auto seq_node = dynamic_cast<structured_control_flow::Sequence*>(current_node)) {
            if (barrier_finder.visit(seq_node)) {
                for (int i = 0; i < seq_node->size(); i++) {
                    nodes_to_visit.push_back(&seq_node->at(i).first);
                }
            } else {
                auto& if_else = builder.add_if_else_before(*parent_sequence, seq_node->at(0).first, {}, DebugInfo());
                auto& branch = builder.add_case(if_else, map_.condition());
                for (int i = 0; i < seq_node->size(); i++) {
                    builder.move_child(*seq_node, seq_node->index(seq_node->at(0).first), branch);
                }
            }
        } else if (auto ifelse_node = dynamic_cast<structured_control_flow::IfElse*>(current_node)) {
            if (barrier_finder.visit(ifelse_node)) {
                for (size_t i = 0; i < ifelse_node->size(); i++) {
                    nodes_to_visit.push_back(&ifelse_node->at(i).first);
                }
            } else {
                auto& if_else = builder.add_if_else_before(*parent_sequence, *ifelse_node, {}, DebugInfo());
                auto& branch = builder.add_case(if_else, map_.condition());
                builder.move_child(*parent_sequence, parent_sequence->index(*ifelse_node), branch);
            }
        } else if (auto for_node = dynamic_cast<structured_control_flow::For*>(current_node)) {
            if (barrier_finder.visit(for_node)) {
                nodes_to_visit.push_back(&for_node->root());
            } else {
                auto& if_else = builder.add_if_else_before(*parent_sequence, *for_node, {}, DebugInfo());
                auto& branch = builder.add_case(if_else, map_.condition());
                builder.move_child(*parent_sequence, parent_sequence->index(*for_node), branch);
            }
        } else if (auto while_node = dynamic_cast<structured_control_flow::While*>(current_node)) {
            if (barrier_finder.visit(while_node)) {
                nodes_to_visit.push_back(&while_node->root());
            } else {
                auto& if_else = builder.add_if_else_before(*parent_sequence, *while_node, {}, DebugInfo());
                auto& branch = builder.add_case(if_else, map_.condition());
                builder.move_child(*parent_sequence, parent_sequence->index(*while_node), branch);
            }
        } else if (auto map_node = dynamic_cast<structured_control_flow::Map*>(current_node)) {
            if (barrier_finder.visit(map_node)) {
                nodes_to_visit.push_back(&map_node->root());
            } else {
                auto& if_else = builder.add_if_else_before(*parent_sequence, *map_node, {}, DebugInfo());
                auto& branch = builder.add_case(if_else, map_.condition());
                builder.move_child(*parent_sequence, parent_sequence->index(*map_node), branch);
            }
        }
    }
    analysis_manager.invalidate_all();
}

std::string GPUConditionPropagation::name() const { return "GPUConditionPropagation"; };

void GPUConditionPropagation::to_json(nlohmann::json& j) const {
    j["transformation_type"] = this->name();

    j["subgraph"] = {{"0", {{"element_id", this->map_.element_id()}, {"type", "map"}}}};

    // Legacy field for backward compatibility
    j["map_element_id"] = this->map_.element_id();
}

GPUConditionPropagation GPUConditionPropagation::
    from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& j) {
    size_t map_id;
    if (j.contains("subgraph")) {
        const auto& node_desc = j.at("subgraph").at("0");
        map_id = node_desc.at("element_id").get<size_t>();
    } else {
        map_id = j.at("map_element_id").get<size_t>();
    }

    auto element = builder.find_element_by_id(map_id);
    if (!element) {
        throw InvalidTransformationDescriptionException("Element with ID " + std::to_string(map_id) + " not found.");
    }
    auto map = dynamic_cast<structured_control_flow::Map*>(element);

    return GPUConditionPropagation(*map);
}

BarrierFinder::BarrierFinder(builder::StructuredSDFGBuilder& builder, sdfg::analysis::AnalysisManager& analysis_manager)
    : visitor::StructuredSDFGVisitor(builder, analysis_manager) {}

bool BarrierFinder::accept(structured_control_flow::Block& node) {
    for (auto& library_node : node.dataflow().nodes()) {
        if (auto barrier_node = dynamic_cast<data_flow::BarrierLocalNode*>(&library_node)) {
            return true;
        }
    }
    return false;
}

bool BarrierFinder::visit(structured_control_flow::ControlFlowNode* node) {
    if (auto block_stmt = dynamic_cast<structured_control_flow::Block*>(node)) {
        return this->accept(*block_stmt);
    } else if (auto sequence_stmt = dynamic_cast<structured_control_flow::Sequence*>(node)) {
        return this->visit_internal(*sequence_stmt);
    } else if (auto if_else_stmt = dynamic_cast<structured_control_flow::IfElse*>(node)) {
        for (int i = 0; i < if_else_stmt->size(); i++) {
            if (this->visit_internal(if_else_stmt->at(i).first)) {
                return true;
            }
        }
    } else if (auto for_stmt = dynamic_cast<structured_control_flow::For*>(node)) {
        return this->visit_internal(for_stmt->root());
    } else if (auto map_stmt = dynamic_cast<structured_control_flow::Map*>(node)) {
        return this->visit_internal(map_stmt->root());
    } else if (auto while_stmt = dynamic_cast<structured_control_flow::While*>(node)) {
        return this->visit_internal(while_stmt->root());
    } else if (auto continue_stmt = dynamic_cast<structured_control_flow::Continue*>(node)) {
        return this->accept(*continue_stmt);
    } else if (auto break_stmt = dynamic_cast<structured_control_flow::Break*>(node)) {
        return this->accept(*break_stmt);
    } else if (auto return_stmt = dynamic_cast<structured_control_flow::Return*>(node)) {
        return this->accept(*return_stmt);
    }

    return false;
}

} // namespace transformations
} // namespace sdfg
