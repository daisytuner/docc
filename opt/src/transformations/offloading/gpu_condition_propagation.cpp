#include "sdfg/transformations/offloading/gpu_condition_propagation.h"
#include <set>
#include <symengine/integer.h>
#include <vector>
#include "sdfg/analysis/assumptions_analysis.h"
#include "sdfg/analysis/loop_analysis.h"
#include "sdfg/analysis/users.h"
#include "sdfg/data_flow/library_nodes/barrier_local_node.h"
#include "sdfg/element.h"
#include "sdfg/structured_control_flow/control_flow_node.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/structured_sdfg.h"
#include "sdfg/symbolic/extreme_values.h"
#include "sdfg/targets/gpu/gpu_map_utils.h"
#include "sdfg/targets/gpu/gpu_schedule_type.h"
#include "sdfg/visitor/structured_sdfg_visitor.h"

namespace sdfg {
namespace transformations {

// A GPU offload map is "full" when its trip count is an exact multiple of the
// parallel size (block/grid dimension). Then the coverage loop assigns every
// thread only in-range indices — no thread overshoots — so the per-thread
// boundary predicate (map_.condition()) is always true and need not be
// propagated onto hoisted, barrier-free content. Ragged maps keep the guard.
// Sound: only fires on compile-time-constant trip and parallel size.
static bool map_is_full(structured_control_flow::Map& map, analysis::AnalysisManager& analysis_manager) {
    const auto& sched = map.schedule_type();
    if (sched.properties().find("parallel_size") == sched.properties().end()) {
        return false;
    }
    auto psize = gpu::ScheduleType_GPU_Offload::parallel_size(sched);
    if (psize.is_null() || !SymEngine::is_a<SymEngine::Integer>(*psize)) {
        return false;
    }
    auto p = SymEngine::rcp_static_cast<const SymEngine::Integer>(psize)->as_int();
    if (p <= 0) {
        return false;
    }
    auto trip = map.num_iterations();
    if (trip.is_null()) {
        return false;
    }
    long long t = -1;
    if (SymEngine::is_a<SymEngine::Integer>(*trip)) {
        t = SymEngine::rcp_static_cast<const SymEngine::Integer>(trip)->as_int();
    } else {
        auto& aa = analysis_manager.get<analysis::AssumptionsAnalysis>();
        const auto& assums = aa.get(map.root(), true);
        const auto& params = aa.parameters();
        auto mx = symbolic::maximum(trip, params, assums, true);
        auto mn = symbolic::minimum(trip, params, assums, true);
        if (mx.is_null() || mn.is_null() || !symbolic::eq(mx, mn) || !SymEngine::is_a<SymEngine::Integer>(*mx)) {
            return false;
        }
        t = SymEngine::rcp_static_cast<const SymEngine::Integer>(mx)->as_int();
    }
    return t >= 0 && (t % p) == 0;
}

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

    // A full map has no ragged threads: every thread's index satisfies the map
    // condition, so the per-thread predicate is redundant and need not be
    // propagated onto hoisted content. Barriers are still hit by all threads.
    if (map_is_full(map_, analysis_manager)) {
        analysis_manager.invalidate_all();
        return;
    }

    std::vector<structured_control_flow::ControlFlowNode*> nodes_to_visit;
    nodes_to_visit.push_back(&map_.root());
    BarrierFinder barrier_finder(builder, analysis_manager);

    auto& users = analysis_manager.get<analysis::Users>();

    while (!nodes_to_visit.empty()) {
        auto current_node = nodes_to_visit.back();
        nodes_to_visit.pop_back();
        auto parent_scope = current_node->get_parent();
        auto parent_sequence = dyn_cast<structured_control_flow::Sequence*>(parent_scope);
        analysis::UsersView current_users(users, *current_node);
        auto uses = current_users.uses(map_.indvar()->get_name());
        if (uses.empty()) {
            // Node does not use the map indvar, skip
            continue;
        }

        if (auto block_node = dyn_cast<structured_control_flow::Block*>(current_node)) {
            if (!barrier_finder.visit(block_node)) {
                auto& if_else = builder.add_if_else_before(*parent_sequence, *block_node);
                auto& branch = builder.add_case(if_else, map_.condition());
                builder.move_child(*parent_sequence, parent_sequence->index(*block_node), branch);
            }
        } else if (auto seq_node = dyn_cast<structured_control_flow::Sequence*>(current_node)) {
            if (barrier_finder.visit(seq_node)) {
                for (int i = 0; i < seq_node->size(); i++) {
                    nodes_to_visit.push_back(&seq_node->at(i));
                }
            } else {
                auto& if_else = builder.add_if_else_before(*parent_sequence, seq_node->at(0));
                auto& branch = builder.add_case(if_else, map_.condition());
                for (int i = 0; i < seq_node->size(); i++) {
                    builder.move_child(*seq_node, seq_node->index(seq_node->at(0)), branch);
                }
            }
        } else if (auto ifelse_node = dyn_cast<structured_control_flow::IfElse*>(current_node)) {
            if (barrier_finder.visit(ifelse_node)) {
                for (size_t i = 0; i < ifelse_node->size(); i++) {
                    nodes_to_visit.push_back(&ifelse_node->at(i).first);
                }
            } else {
                auto& if_else = builder.add_if_else_before(*parent_sequence, *ifelse_node);
                auto& branch = builder.add_case(if_else, map_.condition());
                builder.move_child(*parent_sequence, parent_sequence->index(*ifelse_node), branch);
            }
        } else if (auto for_node = dyn_cast<structured_control_flow::For*>(current_node)) {
            if (barrier_finder.visit(for_node)) {
                nodes_to_visit.push_back(&for_node->root());
            } else {
                auto& if_else = builder.add_if_else_before(*parent_sequence, *for_node);
                auto& branch = builder.add_case(if_else, map_.condition());
                builder.move_child(*parent_sequence, parent_sequence->index(*for_node), branch);
            }
        } else if (auto while_node = dyn_cast<structured_control_flow::While*>(current_node)) {
            if (barrier_finder.visit(while_node)) {
                nodes_to_visit.push_back(&while_node->root());
            } else {
                auto& if_else = builder.add_if_else_before(*parent_sequence, *while_node);
                auto& branch = builder.add_case(if_else, map_.condition());
                builder.move_child(*parent_sequence, parent_sequence->index(*while_node), branch);
            }
        } else if (auto map_node = dyn_cast<structured_control_flow::Map*>(current_node)) {
            if (barrier_finder.visit(map_node)) {
                nodes_to_visit.push_back(&map_node->root());
            } else {
                auto& if_else = builder.add_if_else_before(*parent_sequence, *map_node);
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
    j["parameters"] = nlohmann::json::object();

    serializer::JSONSerializer ser_flat(false);
    j["subgraph"] = nlohmann::json::object();
    j["subgraph"]["0"] = nlohmann::json::object();
    ser_flat.serialize_node(j["subgraph"]["0"], map_);
}

GPUConditionPropagation GPUConditionPropagation::
    from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& j) {
    size_t map_id;
    const auto& node_desc = j.at("subgraph").at("0");
    map_id = node_desc.at("element_id").get<size_t>();

    auto element = builder.find_element_by_id(map_id);
    if (!element) {
        throw InvalidTransformationDescriptionException("Element with ID " + std::to_string(map_id) + " not found.");
    }
    auto map = dyn_cast<structured_control_flow::Map*>(element);

    return GPUConditionPropagation(*map);
}

// Element ids of GPU-scheduled Maps in the subtree rooted at `root` (root included), collected up
// front because applying GPUConditionPropagation invalidates the analysis.
static std::vector<size_t>
gpu_map_ids(structured_control_flow::StructuredLoop& root, analysis::AnalysisManager& analysis_manager) {
    auto& loop_analysis = analysis_manager.get<analysis::LoopAnalysis>();
    std::vector<size_t> ids;
    std::set<size_t> seen;
    auto consider = [&](structured_control_flow::ControlFlowNode* node) {
        auto* map = dyn_cast<structured_control_flow::Map*>(node);
        if (map != nullptr && gpu::is_gpu_schedule(map->schedule_type()) && seen.insert(map->element_id()).second) {
            ids.push_back(map->element_id());
        }
    };
    consider(&root);
    for (auto& path : loop_analysis.loop_tree_paths(&root)) {
        for (auto* node : path) {
            consider(node);
        }
    }
    return ids;
}

GPUConditionPropagationScope::GPUConditionPropagationScope(structured_control_flow::StructuredLoop& root)
    : root_(root) {}

bool GPUConditionPropagationScope::
    can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    for (size_t id : gpu_map_ids(root_, analysis_manager)) {
        auto* map = dyn_cast<structured_control_flow::Map*>(builder.find_element_by_id(id));
        if (map == nullptr) {
            continue;
        }
        GPUConditionPropagation transformation(*map);
        if (transformation.can_be_applied(builder, analysis_manager)) {
            return true;
        }
    }
    return false;
}

void GPUConditionPropagationScope::
    apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    for (size_t id : gpu_map_ids(root_, analysis_manager)) {
        auto* map = dyn_cast<structured_control_flow::Map*>(builder.find_element_by_id(id));
        if (map == nullptr) {
            continue;
        }
        GPUConditionPropagation transformation(*map);
        if (transformation.can_be_applied(builder, analysis_manager)) {
            transformation.apply(builder, analysis_manager);
        }
    }
    analysis_manager.invalidate_all();
}

std::string GPUConditionPropagationScope::name() const { return "GPUConditionPropagationScope"; };

void GPUConditionPropagationScope::to_json(nlohmann::json& j) const {
    j["transformation_type"] = this->name();
    j["parameters"] = nlohmann::json::object();

    serializer::JSONSerializer ser_flat(false);
    j["subgraph"] = nlohmann::json::object();
    j["subgraph"]["0"] = nlohmann::json::object();
    ser_flat.serialize_node(j["subgraph"]["0"], root_);
}

GPUConditionPropagationScope GPUConditionPropagationScope::
    from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& j) {
    const auto& node_desc = j.at("subgraph").at("0");
    size_t root_id = node_desc.at("element_id").get<size_t>();

    auto element = builder.find_element_by_id(root_id);
    if (!element) {
        throw InvalidTransformationDescriptionException("Element with ID " + std::to_string(root_id) + " not found.");
    }
    auto root = dyn_cast<structured_control_flow::StructuredLoop*>(element);
    if (!root) {
        throw InvalidTransformationDescriptionException(
            "Element with ID " + std::to_string(root_id) + " is not a structured loop."
        );
    }

    return GPUConditionPropagationScope(*root);
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
    if (auto block_stmt = dyn_cast<structured_control_flow::Block*>(node)) {
        return this->accept(*block_stmt);
    } else if (auto sequence_stmt = dyn_cast<structured_control_flow::Sequence*>(node)) {
        return this->visit_internal(*sequence_stmt);
    } else if (auto if_else_stmt = dyn_cast<structured_control_flow::IfElse*>(node)) {
        for (int i = 0; i < if_else_stmt->size(); i++) {
            if (this->visit_internal(if_else_stmt->at(i).first)) {
                return true;
            }
        }
    } else if (auto for_stmt = dyn_cast<structured_control_flow::For*>(node)) {
        return this->visit_internal(for_stmt->root());
    } else if (auto map_stmt = dyn_cast<structured_control_flow::Map*>(node)) {
        return this->visit_internal(map_stmt->root());
    } else if (auto while_stmt = dyn_cast<structured_control_flow::While*>(node)) {
        return this->visit_internal(while_stmt->root());
    } else if (auto continue_stmt = dyn_cast<structured_control_flow::Continue*>(node)) {
        return this->accept(*continue_stmt);
    } else if (auto break_stmt = dyn_cast<structured_control_flow::Break*>(node)) {
        return this->accept(*break_stmt);
    } else if (auto return_stmt = dyn_cast<structured_control_flow::Return*>(node)) {
        return this->accept(*return_stmt);
    }

    return false;
}

} // namespace transformations
} // namespace sdfg
