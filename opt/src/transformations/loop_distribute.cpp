#include "sdfg/transformations/loop_distribute.h"

#include "sdfg/analysis/data_dependency_analysis.h"
#include "sdfg/analysis/scope_analysis.h"
#include "sdfg/deepcopy/structured_sdfg_deep_copy.h"

namespace sdfg {
namespace transformations {

LoopDistribute::LoopDistribute(structured_control_flow::StructuredLoop& loop)
    : loop_(loop) {

      };

std::string LoopDistribute::name() const { return "LoopDistribute"; };

bool LoopDistribute::can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    // We distribute each child at a time
    auto& body = this->loop_.root();
    if (body.size() < 2) {
        return false;
    }
    auto& child = body.at(0).first;
    // Criterion: Only dataflow
    if (!body.at(0).second.assignments().empty()) {
        return false;
    }

    // Criterion: Child does not write to loop-local containers
    /**
     * Positive example:
     * A = ...;
     * ...
     * for (i = 0; i < 10; i++) {
     *  // child 0
     *  A[i] = 0;
     *  // child 1..n
     *  ... (uses A[i])
     * }
     *
     * Negative example:
     * for (i = 0; i < 10; i++) {
     *  // child 0
     *  double a = 0;
     *  // child 1..n
     *  ... (uses a)
     * }
     */

    // Criterion: If dependency exists, then it is
    // a) a child-local WAW, we can ignore it
    // b) is not used by child at all

    // collect dependencies
    analysis::DataDependencyAnalysis analysis(builder.subject(), true);
    analysis.run(analysis_manager);
    auto& dependencies = analysis.dependencies(this->loop_);

    // collect child locals and child users
    auto& users = analysis_manager.get<analysis::Users>();
    analysis::UsersView child_users(users, child);
    auto child_locals = users.locals(child);
    if (!child_users.views().empty() || !child_users.moves().empty()) {
        return false;
    }

    // Check dependencies
    for (auto& dep : dependencies) {
        auto& container = dep.first;
        if (child_users.uses(container).empty()) {
            continue;
        }
        if (dep.second.type == analysis::LoopCarriedDependency::LOOP_CARRIED_DEPENDENCY_WRITE_WRITE) {
            if (child_locals.find(container) != child_locals.end()) {
                continue;
            }
        }

        return false;
    }

    return true;
};

void LoopDistribute::apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    auto& sdfg = builder.subject();

    auto indvar = this->loop_.indvar();
    auto condition = this->loop_.condition();
    auto update = this->loop_.update();
    auto init = this->loop_.init();

    auto& analysis = analysis_manager.get<analysis::ScopeAnalysis>();
    auto parent = static_cast<structured_control_flow::Sequence*>(analysis.parent_scope(&this->loop_));
    structured_control_flow::ScheduleType schedule_type = structured_control_flow::ScheduleType_Sequential::create();
    if (auto map_stmt = dynamic_cast<structured_control_flow::Map*>(&this->loop_)) {
        schedule_type = map_stmt->schedule_type();
    }
    auto& new_map = builder.add_map_before(
        *parent, this->loop_, indvar, condition, init, update, schedule_type, {}, this->loop_.debug_info()
    );
    builder.move_child(this->loop_.root(), 0, new_map.root());

    // Replace indvar in new loop
    std::string new_indvar = builder.find_new_name(indvar->get_name());
    builder.add_container(new_indvar, sdfg.type(indvar->get_name()));
    new_map.replace(indvar, symbolic::symbol(new_indvar));

    analysis_manager.invalidate_all();
};

void LoopDistribute::to_json(nlohmann::json& j) const {
    std::string loop_type;
    if (dynamic_cast<structured_control_flow::For*>(&loop_)) {
        loop_type = "for";
    } else if (dynamic_cast<structured_control_flow::Map*>(&loop_)) {
        loop_type = "map";
    } else {
        throw std::runtime_error("Unsupported loop type for serialization of loop: " + loop_.indvar()->get_name());
    }

    j["transformation_type"] = this->name();
    j["subgraph"] = {{"0", {{"element_id", this->loop_.element_id()}, {"type", loop_type}}}};
    j["transformation_type"] = this->name();
};

LoopDistribute LoopDistribute::from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& desc) {
    auto loop_id = desc["subgraph"]["0"]["element_id"].get<size_t>();
    auto element = builder.find_element_by_id(loop_id);
    if (element == nullptr) {
        throw InvalidTransformationDescriptionException("Element with ID " + std::to_string(loop_id) + " not found.");
    }
    auto loop = dynamic_cast<structured_control_flow::StructuredLoop*>(element);
    if (loop == nullptr) {
        throw InvalidTransformationDescriptionException(
            "Element with ID " + std::to_string(loop_id) + " is not a StructuredLoop."
        );
    }

    return LoopDistribute(*loop);
};

} // namespace transformations
} // namespace sdfg
