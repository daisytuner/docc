#include "sdfg/transformations/trivial_loop_elimination.h"

#include <stdexcept>
#include <unordered_set>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/memlet.h"
#include "sdfg/element.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/visitor/for_each.h"

namespace sdfg {
namespace transformations {

namespace {

/// Replaces data accesses to `indvar` by constant nodes holding `value`.
void replace_indvar_accesses_by_constant(
    builder::StructuredSDFGBuilder& builder,
    structured_control_flow::Sequence& body,
    const symbolic::Symbol& indvar,
    const std::string& value
) {
    const auto& indvar_type = builder.subject().type(indvar->get_name());
    visitor::for_each_block(body, [&](structured_control_flow::Block& block) -> void {
        for (auto* access_node : block.dataflow().data_nodes()) {
            if (is_a(access_node->type_id(), ElementType::ConstantNode)) {
                continue;
            }
            if (access_node->data() != indvar->get_name()) {
                continue;
            }

            auto& constant_node = builder.add_constant(block, value, indvar_type, access_node->debug_info());
            std::unordered_set<data_flow::Memlet*> old_memlets;
            for (auto& memlet : block.dataflow().out_edges(*access_node)) {
                builder.add_memlet(
                    block,
                    constant_node,
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

/// Returns true if any data access to `indvar` remains in `body`.
bool has_indvar_accesses(structured_control_flow::Sequence& body, const symbolic::Symbol& indvar) {
    bool found = false;
    visitor::for_each_block(body, [&](structured_control_flow::Block& block) -> void {
        for (auto* access_node : block.dataflow().data_nodes()) {
            if (!is_a(access_node->type_id(), ElementType::ConstantNode) && access_node->data() == indvar->get_name()) {
                found = true;
            }
        }
    });
    return found;
}

} // namespace

TrivialLoopElimination::TrivialLoopElimination(structured_control_flow::Map& loop) : loop_(loop) {
}

std::string TrivialLoopElimination::name() const {
    return "TrivialLoopElimination";
}

bool TrivialLoopElimination::
    can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    // Non-sequential maps carry semantics beyond their iterations (e.g. a single-wave GPU map defines 64 lanes).
    if (loop_.schedule_type().value() != structured_control_flow::ScheduleType_Sequential::value()) {
        return false;
    }
    if (!loop_.is_contiguous()) {
        return false;
    }
    auto trip_count = loop_.num_iterations();
    if (trip_count.is_null() || !symbolic::eq(trip_count, symbolic::one())) {
        return false;
    }
    if (dyn_cast<structured_control_flow::Sequence*>(loop_.get_parent()) == nullptr) {
        return false;
    }
    return true;
}

void TrivialLoopElimination::apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    auto& parent = *dyn_cast<structured_control_flow::Sequence*>(loop_.get_parent());
    auto& body = loop_.root();
    auto indvar = loop_.indvar();
    auto init = loop_.init();

    // Symbolic uses (and accesses, if init is a symbol) of the indvar
    body.replace(indvar, init);

    // Remaining data accesses of the indvar container
    if (SymEngine::is_a<SymEngine::Integer>(*init)) {
        replace_indvar_accesses_by_constant(builder, body, indvar, init->__str__());
    } else if (has_indvar_accesses(body, indvar)) {
        builder.add_assignments_before(body, body.at(0), {{indvar, init}}, loop_.debug_info());
    }

    // Move body into parent and remove the loop
    size_t index = parent.index(loop_);
    builder.move_children(body, parent, index + 1);
    builder.remove_child(parent, index);

    analysis_manager.invalidate_all();
}

void TrivialLoopElimination::to_json(nlohmann::json& j) const {
    j["transformation_type"] = this->name();
    j["parameters"] = nlohmann::json::object();

    serializer::JSONSerializer ser_flat(false);
    j["subgraph"] = nlohmann::json::object();
    j["subgraph"]["0"] = nlohmann::json::object();
    ser_flat.serialize_node(j["subgraph"]["0"], loop_);
}

TrivialLoopElimination TrivialLoopElimination::from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& j) {
    auto loop_id = j["subgraph"]["0"]["element_id"].get<size_t>();

    auto element = builder.find_element_by_id(loop_id);
    if (element == nullptr) {
        throw InvalidTransformationDescriptionException(
            "TrivialLoopElimination: Element with ID " + std::to_string(loop_id) + " not found."
        );
    }

    auto loop = dyn_cast<structured_control_flow::Map*>(element);
    if (loop == nullptr) {
        throw InvalidTransformationDescriptionException(
            "TrivialLoopElimination: Element with ID " + std::to_string(loop_id) + " is not a map."
        );
    }

    return TrivialLoopElimination(*loop);
}

} // namespace transformations
} // namespace sdfg
