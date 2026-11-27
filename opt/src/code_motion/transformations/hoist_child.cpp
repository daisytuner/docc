#include "sdfg/code_motion/transformations/hoist_child.h"

#include "sdfg/analysis/assumptions_analysis.h"
#include "sdfg/analysis/users.h"
#include "sdfg/code_motion/conflicts.h"
#include "sdfg/serializer/json_serializer.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/symbolic/extreme_values.h"

namespace sdfg {
namespace code_motion {
namespace transformations {

HoistChild::HoistChild(
    structured_control_flow::StructuredLoop& loop, structured_control_flow::ControlFlowNode& child, bool after
)
    : loop_(loop), child_(child), after_(after) {
}

std::string HoistChild::name() const {
    return "HoistChild";
}

bool HoistChild::can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    auto& body = loop_.root();
    if (child_.get_parent() != &body || !dynamic_cast<structured_control_flow::Sequence*>(loop_.get_parent())) {
        return false;
    }
    auto& assumptions_analysis = analysis_manager.get<analysis::AssumptionsAnalysis>();

    // The loop runs at least once, so the child runs as before.
    auto iterations = loop_.num_iterations();
    if (iterations.is_null()) {
        return false;
    }
    auto least = symbolic::minimum(iterations, {}, assumptions_analysis.get(loop_, true), false);
    if (least.is_null() || !SymEngine::is_a<SymEngine::Integer>(*least) ||
        SymEngine::down_cast<const SymEngine::Integer&>(*least).as_int() < 1) {
        return false;
    }

    for (size_t k = 0; k < body.size(); ++k) {
        if (!movable(builder.subject(), body.at(k))) {
            return false;
        }
    }

    // Every iteration runs the child alike, and running it again changes nothing.
    analysis::UsersView child_users(analysis_manager.get<analysis::Users>(), child_);
    if (!child_users.uses(loop_.indvar()->get_name()).empty()) {
        return false;
    }
    auto child = accesses_of(builder.subject(), analysis_manager, child_);
    if (reads_what_it_writes(child)) {
        return false;
    }

    // No other child of the body writes what it accesses, in any pair of iterations. Those after it may read what it
    // writes, as each iteration still finds it written; those before it read it before it was written in the first
    // iteration, and moved after the loop, all would.
    auto scope = assumptions_analysis.get(body, true);
    if (scope.contains(loop_.indvar())) {
        scope.at(loop_.indvar()).constant(false);
    }
    size_t index = body.index(child_);
    for (size_t k = 0; k < body.size(); ++k) {
        auto& sibling = body.at(k);
        if (k == index) {
            continue;
        }
        Dependences kinds{.flow = after_ || k < index, .anti = true, .output = true};
        if (may_conflict(child, accesses_of(builder.subject(), analysis_manager, sibling), scope, kinds)) {
            return false;
        }
    }
    return true;
}

void HoistChild::apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    auto& body = loop_.root();
    auto& parent = static_cast<structured_control_flow::Sequence&>(*loop_.get_parent());
    size_t position = parent.index(loop_) + (after_ ? 1 : 0);
    builder.move_child(body, body.index(child_), parent, position);
    // The child left the loop for the sequence around it.
    analysis_manager.invalidate_all(parent);
}

void HoistChild::to_json(nlohmann::json& j) const {
    j["transformation_type"] = this->name();
    j["parameters"] = {{"after", after_}};
    serializer::JSONSerializer serializer(false);
    j["subgraph"] = nlohmann::json::object();
    j["subgraph"]["0"] = nlohmann::json::object();
    serializer.serialize_node(j["subgraph"]["0"], loop_);
    j["subgraph"]["1"] = nlohmann::json::object();
    serializer.serialize_node(j["subgraph"]["1"], child_);
}

HoistChild HoistChild::from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& j) {
    auto loop_id = j["subgraph"]["0"]["element_id"].get<size_t>();
    auto child_id = j["subgraph"]["1"]["element_id"].get<size_t>();
    auto* loop = dynamic_cast<structured_control_flow::StructuredLoop*>(builder.find_element_by_id(loop_id));
    auto* child = dynamic_cast<structured_control_flow::ControlFlowNode*>(builder.find_element_by_id(child_id));
    if (loop == nullptr || child == nullptr) {
        throw ::sdfg::transformations::InvalidTransformationDescriptionException(
            "Elements with IDs " + std::to_string(loop_id) + " and " + std::to_string(child_id) +
            " are no loop and control-flow node."
        );
    }
    return HoistChild(*loop, *child, j["parameters"]["after"].get<bool>());
}

} // namespace transformations
} // namespace code_motion
} // namespace sdfg
