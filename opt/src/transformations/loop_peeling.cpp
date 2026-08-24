#include "sdfg/transformations/loop_peeling.h"

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/deepcopy/structured_sdfg_deep_copy.h"
#include "sdfg/structured_control_flow/for.h"
#include "sdfg/structured_control_flow/if_else.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/structured_control_flow/reduce.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/symbolic/symbolic.h"

#include <symengine/integer.h>

namespace sdfg {
namespace transformations {

LoopPeeling::LoopPeeling(structured_control_flow::StructuredLoop& loop, bool predicate)
    : loop_(loop), predicate_(predicate) {};

std::string LoopPeeling::name() const { return "LoopPeeling"; };

/// True if `expr` is a strictly positive integer constant.
static bool is_positive_int(const symbolic::Expression& expr) {
    return expr != SymEngine::null && SymEngine::is_a<SymEngine::Integer>(*expr) &&
           SymEngine::rcp_static_cast<const SymEngine::Integer>(expr)->as_int() > 0;
}

/// Applicable when the loop has a constant-trip overapproximation (so the nest
/// can be fully unrolled) but a non-constant exact trip count (so there is a
/// dynamic boundary to handle). Relies on the StructuredLoop trip-count helpers,
/// which handle `<=`, offsets, strides and tile-style `min(...)` bounds.
static bool has_predicable_boundary(structured_control_flow::StructuredLoop& loop) {
    if (!loop.is_monotonic()) {
        return false;
    }
    auto approx = loop.num_iterations_approx();
    if (!is_positive_int(approx)) {
        return false;
    }
    auto exact = loop.num_iterations();
    if (exact == SymEngine::null || is_positive_int(exact)) {
        return false;
    }
    return true;
}

/// Collect the perfectly nested chain of peelable loops starting at `loop`
/// (each level's body being exactly the next peelable loop).
static std::vector<structured_control_flow::StructuredLoop*> collect_nest(structured_control_flow::StructuredLoop& loop
) {
    std::vector<structured_control_flow::StructuredLoop*> nest{&loop};
    auto* current = &loop;
    while (true) {
        auto& body = current->root();
        if (body.size() != 1) {
            break;
        }
        auto* inner = dynamic_cast<structured_control_flow::StructuredLoop*>(&body.at(0));
        if (inner == nullptr || !has_predicable_boundary(*inner)) {
            break;
        }
        nest.push_back(inner);
        current = inner;
    }
    return nest;
}

/// Append a copy of `proto`'s loop header (same kind/schedule) into `parent`.
static structured_control_flow::StructuredLoop& append_loop(
    builder::StructuredSDFGBuilder& builder,
    structured_control_flow::Sequence& parent,
    structured_control_flow::StructuredLoop& proto,
    const symbolic::Symbol& indvar,
    const symbolic::Condition& condition,
    const symbolic::Expression& init,
    const symbolic::Expression& update
) {
    if (auto* map = dynamic_cast<structured_control_flow::Map*>(&proto)) {
        return builder.add_map(parent, indvar, condition, init, update, map->schedule_type(), proto.debug_info());
    }
    if (auto* reduce = dynamic_cast<structured_control_flow::Reduce*>(&proto)) {
        return builder.add_reduce(
            parent, indvar, condition, init, update, reduce->reductions(), reduce->schedule_type(), proto.debug_info()
        );
    }
    return builder.add_for(parent, indvar, condition, init, update, proto.debug_info());
}

bool LoopPeeling::can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    return has_predicable_boundary(loop_);
};

void LoopPeeling::apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    auto nest = collect_nest(loop_);
    auto* innermost = nest.back();
    auto zero = symbolic::integer(0);

    // Per-loop 0-based header + the shift `indvar -> indvar + init` that rewrites
    // the shifted body back to original induction values. Also accumulate:
    //  - combined_guard: dynamic bounds evaluated per iteration (predicate form);
    //  - combined_fits:  dynamic bounds evaluated at each loop's last iteration,
    //                    i.e. the whole tile is in bounds (hoisted form).
    struct LoopInfo {
        structured_control_flow::StructuredLoop* loop;
        symbolic::Symbol indvar;
        symbolic::Expression init;
        symbolic::Condition zero_condition;
        symbolic::Expression shifted;
    };
    std::vector<LoopInfo> infos;
    symbolic::Condition combined_guard = SymEngine::boolTrue;
    symbolic::Condition combined_fits = SymEngine::boolTrue;
    for (auto* l : nest) {
        auto indvar = l->indvar();
        auto init = l->init();
        auto stride = l->stride();
        auto trip = l->num_iterations_approx();
        auto shifted = symbolic::add(indvar, init);
        symbolic::Condition zero_condition = symbolic::Lt(indvar, symbolic::mul(trip, stride));
        combined_guard = symbolic::And(combined_guard, symbolic::subs(l->condition(), indvar, shifted));
        auto last = symbolic::add(init, symbolic::mul(symbolic::sub(trip, symbolic::integer(1)), stride));
        combined_fits = symbolic::And(combined_fits, symbolic::subs(l->condition(), indvar, last));
        infos.push_back({l, indvar, init, zero_condition, shifted});
    }

    auto* parent = static_cast<structured_control_flow::Sequence*>(loop_.get_parent());

    if (predicate_) {
        // 0-based nest; the whole boundary is re-checked once at the innermost body.
        auto& holder = builder.add_sequence_before(*parent, loop_, loop_.debug_info());
        structured_control_flow::Sequence* current = &holder;
        for (auto& info : infos) {
            auto& nl =
                append_loop(builder, *current, *info.loop, info.indvar, info.zero_condition, zero, info.loop->update());
            current = &nl.root();
        }
        auto& if_else = builder.add_if_else(*current, loop_.debug_info());
        auto& body = builder.add_case(if_else, combined_guard, loop_.debug_info());
        deepcopy::StructuredSDFGDeepCopy(builder, body, innermost->root()).insert();
        for (auto& info : infos) {
            if (!symbolic::eq(info.init, zero)) {
                body.replace(info.indvar, info.shifted);
            }
        }
    } else {
        // Hoisted: full clean tile in the "then" branch, original variable-trip
        // remainder in the "else". The "then" micro-kernel is unguarded (vectorizes).
        auto& if_else = builder.add_if_else_before(*parent, loop_, loop_.debug_info());

        auto& then_branch = builder.add_case(if_else, combined_fits, loop_.debug_info());
        structured_control_flow::Sequence* current = &then_branch;
        for (auto& info : infos) {
            auto& nl =
                append_loop(builder, *current, *info.loop, info.indvar, info.zero_condition, zero, info.loop->update());
            current = &nl.root();
        }
        deepcopy::StructuredSDFGDeepCopy(builder, *current, innermost->root()).insert();
        for (auto& info : infos) {
            if (!symbolic::eq(info.init, zero)) {
                current->replace(info.indvar, info.shifted);
            }
        }

        auto& else_branch = builder.add_case(if_else, symbolic::Not(combined_fits), loop_.debug_info());
        current = &else_branch;
        for (auto& info : infos) {
            auto& nl = append_loop(
                builder, *current, *info.loop, info.indvar, info.loop->condition(), info.init, info.loop->update()
            );
            current = &nl.root();
        }
        deepcopy::StructuredSDFGDeepCopy(builder, *current, innermost->root()).insert();
    }

    builder.remove_child(*parent, parent->index(loop_));

    analysis_manager.invalidate_all();
};

void LoopPeeling::to_json(nlohmann::json& j) const {
    j["transformation_type"] = this->name();
    j["parameters"] = nlohmann::json::object();
    j["parameters"]["predicate"] = predicate_;

    serializer::JSONSerializer ser_flat(false);
    j["subgraph"] = nlohmann::json::object();
    j["subgraph"]["0"] = nlohmann::json::object();
    ser_flat.serialize_node(j["subgraph"]["0"], loop_);
};

LoopPeeling LoopPeeling::from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& desc) {
    auto loop_id = desc["subgraph"]["0"]["element_id"].get<size_t>();
    auto element = builder.find_element_by_id(loop_id);
    if (element == nullptr) {
        throw InvalidTransformationDescriptionException("Element with ID " + std::to_string(loop_id) + " not found.");
    }
    auto* loop = dynamic_cast<structured_control_flow::StructuredLoop*>(element);
    if (loop == nullptr) {
        throw InvalidTransformationDescriptionException(
            "Element with ID " + std::to_string(loop_id) + " is not a structured loop."
        );
    }
    bool predicate = false;
    if (desc.contains("parameters") && desc["parameters"].contains("predicate")) {
        predicate = desc["parameters"]["predicate"].get<bool>();
    }
    return LoopPeeling(*loop, predicate);
};

} // namespace transformations
} // namespace sdfg
