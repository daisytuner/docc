#include "sdfg/transformations/loop_peeling.h"

#include "sdfg/analysis/scope_analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/deepcopy/structured_sdfg_deep_copy.h"
#include "sdfg/structured_control_flow/for.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/symbolic/symbolic.h"

#include <symengine/functions.h>
#include <symengine/logic.h>

namespace sdfg {
namespace transformations {

LoopPeeling::LoopPeeling(structured_control_flow::StructuredLoop& loop) : loop_(loop) {};

std::string LoopPeeling::name() const { return "LoopPeeling"; };

/// Extract upper bound from a condition of the form `indvar < bound`.
/// Returns SymEngine::null if not a simple strict-less-than on indvar.
static symbolic::Expression extract_upper_bound(const symbolic::Condition& cond, const symbolic::Symbol& indvar) {
    if (!SymEngine::is_a<SymEngine::StrictLessThan>(*cond)) {
        return SymEngine::null;
    }
    auto lt = SymEngine::rcp_static_cast<const SymEngine::StrictLessThan>(cond);
    auto lhs = lt->get_arg1();
    auto rhs = lt->get_arg2();
    if (symbolic::eq(lhs, indvar)) {
        return rhs;
    }
    return SymEngine::null;
}

/// Determine if `bound - init` simplifies to a positive integer constant.
/// Also handles the case where init is a max() expression: if bound - any_arg
/// gives a positive integer constant, the loop has a bounded trip count.
static bool is_constant_trip_bound(const symbolic::Expression& bound, const symbolic::Expression& init) {
    auto diff = symbolic::expand(symbolic::sub(bound, init));
    if (SymEngine::is_a<SymEngine::Integer>(*diff)) {
        auto val = SymEngine::rcp_static_cast<const SymEngine::Integer>(diff);
        return val->as_int() > 0;
    }
    // If init is max(a, b, ...), check if bound - arg is constant for any arg.
    // trip_count = bound - max(a, b, ...) = min(bound-a, bound-b, ...)
    // If any (bound - arg) is a positive constant, then the trip count is at most that constant.
    if (SymEngine::is_a<SymEngine::Max>(*init)) {
        auto max_op = SymEngine::rcp_static_cast<const SymEngine::Max>(init);
        auto args = max_op->get_args();
        for (auto& arg : args) {
            auto arg_diff = symbolic::expand(symbolic::sub(bound, arg));
            if (SymEngine::is_a<SymEngine::Integer>(*arg_diff)) {
                auto val = SymEngine::rcp_static_cast<const SymEngine::Integer>(arg_diff);
                if (val->as_int() > 0) {
                    return true;
                }
            }
        }
    }
    return false;
}

/// Get the constant trip count for a bound/init pair that passes is_constant_trip_bound.
/// Returns the positive integer trip count, handling max() in init.
static int64_t get_constant_trip_count(const symbolic::Expression& bound, const symbolic::Expression& init) {
    auto diff = symbolic::expand(symbolic::sub(bound, init));
    if (SymEngine::is_a<SymEngine::Integer>(*diff)) {
        return SymEngine::rcp_static_cast<const SymEngine::Integer>(diff)->as_int();
    }
    // For init = max(a, b, ...), find the smallest positive constant among (bound - arg)
    if (SymEngine::is_a<SymEngine::Max>(*init)) {
        auto max_op = SymEngine::rcp_static_cast<const SymEngine::Max>(init);
        auto args = max_op->get_args();
        int64_t best = INT64_MAX;
        for (auto& arg : args) {
            auto arg_diff = symbolic::expand(symbolic::sub(bound, arg));
            if (SymEngine::is_a<SymEngine::Integer>(*arg_diff)) {
                auto val = SymEngine::rcp_static_cast<const SymEngine::Integer>(arg_diff)->as_int();
                if (val > 0 && val < best) {
                    best = val;
                }
            }
        }
        return best;
    }
    // Should not reach here if is_constant_trip_bound returned true
    return 0;
}

/// Check if a loop has a compound condition with at least one canonical bound.
static bool loop_is_peelable(structured_control_flow::StructuredLoop& loop) {
    auto cond = loop.condition();
    if (!SymEngine::is_a<SymEngine::And>(*cond)) {
        return false;
    }
    auto and_cond = SymEngine::rcp_static_cast<const SymEngine::And>(cond);
    auto& conjuncts = and_cond->get_container();
    if (conjuncts.size() < 2) {
        return false;
    }

    auto indvar = loop.indvar();
    auto init = loop.init();
    bool has_canonical = false;
    bool has_dynamic = false;

    for (auto& conjunct : conjuncts) {
        auto bound = extract_upper_bound(conjunct, indvar);
        if (bound == SymEngine::null) {
            return false;
        }
        if (is_constant_trip_bound(bound, init)) {
            has_canonical = true;
        } else {
            has_dynamic = true;
        }
    }
    return has_canonical && has_dynamic;
}

/// For a peelable loop, extract the canonical bound (tightest constant-trip bound).
static symbolic::Expression find_canonical_bound(structured_control_flow::StructuredLoop& loop) {
    auto cond = loop.condition();
    auto and_cond = SymEngine::rcp_static_cast<const SymEngine::And>(cond);
    auto& conjuncts = and_cond->get_container();
    auto indvar = loop.indvar();
    auto init = loop.init();

    symbolic::Expression canonical = SymEngine::null;
    int64_t canonical_trip = INT64_MAX;
    for (auto& conjunct : conjuncts) {
        auto bound = extract_upper_bound(conjunct, indvar);
        if (bound == SymEngine::null) continue;
        if (!is_constant_trip_bound(bound, init)) continue;

        auto trip = get_constant_trip_count(bound, init);
        if (canonical == SymEngine::null || trip < canonical_trip) {
            canonical = bound;
            canonical_trip = trip;
        }
    }
    return canonical;
}

/// Build the peeling condition for a single loop: canonical_bound <= each dynamic bound.
static symbolic::Condition build_loop_peeling_condition(
    structured_control_flow::StructuredLoop& loop, const symbolic::Expression& canonical_bound
) {
    auto cond = loop.condition();
    auto and_cond = SymEngine::rcp_static_cast<const SymEngine::And>(cond);
    auto& conjuncts = and_cond->get_container();
    auto indvar = loop.indvar();
    auto init = loop.init();

    symbolic::Condition result = SymEngine::boolTrue;
    for (auto& conjunct : conjuncts) {
        auto bound = extract_upper_bound(conjunct, indvar);
        if (bound == SymEngine::null) continue;
        if (is_constant_trip_bound(bound, init) && symbolic::eq(bound, canonical_bound)) continue;
        // canonical_bound <= this dynamic/looser bound
        result = symbolic::And(result, symbolic::Le(canonical_bound, bound));
    }

    // If init is max(a, b, ...), we need to ensure the max resolves to the arg
    // that gives the constant trip count. Add conditions: chosen_arg >= other_args.
    if (SymEngine::is_a<SymEngine::Max>(*init)) {
        auto max_op = SymEngine::rcp_static_cast<const SymEngine::Max>(init);
        auto args = max_op->get_args();
        // Find the arg that gives the constant trip (smallest constant trip)
        symbolic::Expression chosen_arg = SymEngine::null;
        int64_t best_trip = INT64_MAX;
        for (auto& arg : args) {
            auto arg_diff = symbolic::expand(symbolic::sub(canonical_bound, arg));
            if (SymEngine::is_a<SymEngine::Integer>(*arg_diff)) {
                auto val = SymEngine::rcp_static_cast<const SymEngine::Integer>(arg_diff)->as_int();
                if (val > 0 && val < best_trip) {
                    best_trip = val;
                    chosen_arg = arg;
                }
            }
        }
        // Add conditions: chosen_arg >= each other arg (so max resolves to chosen_arg)
        if (chosen_arg != SymEngine::null) {
            for (auto& arg : args) {
                if (!symbolic::eq(arg, chosen_arg)) {
                    result = symbolic::And(result, symbolic::Le(arg, chosen_arg));
                }
            }
        }
    }

    return result;
}

/// Collect the perfectly nested chain of peelable loops starting from `loop`.
/// A chain continues as long as the loop body has exactly one child which is
/// another peelable structured loop.
static std::vector<structured_control_flow::StructuredLoop*> collect_peelable_nest(structured_control_flow::StructuredLoop&
                                                                                       loop) {
    std::vector<structured_control_flow::StructuredLoop*> nest;
    nest.push_back(&loop);

    auto* current = &loop;
    while (true) {
        auto& body = current->root();
        if (body.size() != 1) break;
        auto& child = body.at(0).first;
        auto* inner_loop = dynamic_cast<structured_control_flow::StructuredLoop*>(&child);
        if (!inner_loop) break;
        if (!loop_is_peelable(*inner_loop)) break;
        nest.push_back(inner_loop);
        current = inner_loop;
    }
    return nest;
}

bool LoopPeeling::can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    return loop_is_peelable(loop_);
};

void LoopPeeling::apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    auto& sdfg = builder.subject();

    // Collect perfectly nested chain of peelable loops
    auto nest = collect_peelable_nest(loop_);

    // For each loop in the nest, find its canonical bound and build peeling condition
    struct LoopPeelInfo {
        structured_control_flow::StructuredLoop* loop;
        symbolic::Expression canonical_bound;
        symbolic::Condition peeling_condition;
    };
    std::vector<LoopPeelInfo> peel_infos;

    symbolic::Condition combined_peeling_condition = SymEngine::boolTrue;
    for (auto* loop : nest) {
        auto canonical = find_canonical_bound(*loop);
        auto peel_cond = build_loop_peeling_condition(*loop, canonical);
        combined_peeling_condition = symbolic::And(combined_peeling_condition, peel_cond);
        peel_infos.push_back({loop, canonical, peel_cond});
    }

    // Get parent scope of the outermost loop
    auto& scope_analysis = analysis_manager.get<analysis::ScopeAnalysis>();
    auto parent = static_cast<structured_control_flow::Sequence*>(scope_analysis.parent_scope(&loop_));

    // Create IfElse before the outermost loop
    auto& if_else = builder.add_if_else_before(*parent, loop_, {}, loop_.debug_info());

    // === Then branch: all loops normalized to start at 0 with constant trip counts ===
    auto& then_branch = builder.add_case(if_else, combined_peeling_condition);

    // Build the nest of loops with clean 0-based bounds in the then branch
    structured_control_flow::Sequence* current_parent = &then_branch;
    // Track substitutions: original_indvar → indvar + init (for body fixup)
    std::vector<std::pair<symbolic::Symbol, symbolic::Expression>> substitutions;

    for (size_t i = 0; i < peel_infos.size(); i++) {
        auto& info = peel_infos[i];
        auto* loop = info.loop;
        auto indvar = loop->indvar();
        auto init = loop->init();

        // Compute constant trip count: canonical_bound - init (using helper for max() cases)
        auto trip_count = symbolic::integer(get_constant_trip_count(info.canonical_bound, init));

        // Resolve effective init for the then-branch.
        // If init = max(a, b, ...) and canonical_bound - b is the constant trip,
        // then in the then-branch (where peeling condition guarantees max = b), use b directly.
        symbolic::Expression effective_init = init;
        if (SymEngine::is_a<SymEngine::Max>(*init)) {
            auto max_op = SymEngine::rcp_static_cast<const SymEngine::Max>(init);
            auto args = max_op->get_args();
            int64_t best_trip = INT64_MAX;
            for (auto& arg : args) {
                auto arg_diff = symbolic::expand(symbolic::sub(info.canonical_bound, arg));
                if (SymEngine::is_a<SymEngine::Integer>(*arg_diff)) {
                    auto val = SymEngine::rcp_static_cast<const SymEngine::Integer>(arg_diff)->as_int();
                    if (val > 0 && val < best_trip) {
                        best_trip = val;
                        effective_init = arg;
                    }
                }
            }
        }

        // New loop: indvar goes from 0 to trip_count
        auto zero_condition = symbolic::Lt(indvar, trip_count);
        auto zero_init = symbolic::integer(0);

        structured_control_flow::StructuredLoop* new_loop = nullptr;
        if (auto map = dynamic_cast<structured_control_flow::Map*>(loop)) {
            new_loop = &builder.add_map(
                *current_parent,
                indvar,
                zero_condition,
                zero_init,
                loop->update(),
                map->schedule_type(),
                {},
                loop->debug_info()
            );
        } else {
            new_loop =
                &builder
                     .add_for(*current_parent, indvar, zero_condition, zero_init, loop->update(), {}, loop->debug_info());
        }

        // Record substitution: in the body, original indvar usage = new_indvar + effective_init
        substitutions.push_back({indvar, effective_init});
        current_parent = &new_loop->root();
    }

    // Deep copy the innermost loop's body into the new innermost loop
    auto* innermost = nest.back();
    deepcopy::StructuredSDFGDeepCopy main_copier(builder, *current_parent, innermost->root());
    main_copier.insert();

    // Apply shift substitutions in the copied body:
    // Replace indvar with (indvar + original_init) so body accesses use correct offsets.
    // Must apply outermost first (since inner body may reference outer indvars).
    for (auto& [indvar, init] : substitutions) {
        if (symbolic::eq(init, symbolic::zero())) continue;
        auto shifted_expr = symbolic::add(indvar, init);
        current_parent->replace(indvar, shifted_expr);
    }

    // === Else branch: original compound bounds (remainder) ===
    auto else_condition = symbolic::Not(combined_peeling_condition);
    auto& else_branch = builder.add_case(if_else, else_condition);

    // Build the nest of loops with original conditions in the else branch
    current_parent = &else_branch;
    for (size_t i = 0; i < peel_infos.size(); i++) {
        auto* loop = peel_infos[i].loop;

        structured_control_flow::StructuredLoop* new_loop = nullptr;
        if (auto map = dynamic_cast<structured_control_flow::Map*>(loop)) {
            new_loop = &builder.add_map(
                *current_parent,
                loop->indvar(),
                loop->condition(),
                loop->init(),
                loop->update(),
                map->schedule_type(),
                {},
                loop->debug_info()
            );
        } else {
            new_loop = &builder.add_for(
                *current_parent, loop->indvar(), loop->condition(), loop->init(), loop->update(), {}, loop->debug_info()
            );
        }
        current_parent = &new_loop->root();
    }

    // Deep copy the innermost loop's body into the remainder innermost loop
    deepcopy::StructuredSDFGDeepCopy remainder_copier(builder, *current_parent, innermost->root());
    remainder_copier.insert();

    // Remove the original loop
    builder.remove_child(*parent, parent->index(loop_));

    analysis_manager.invalidate_all();
};

void LoopPeeling::to_json(nlohmann::json& j) const {
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
};

LoopPeeling LoopPeeling::from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& desc) {
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

    return LoopPeeling(*loop);
};

} // namespace transformations
} // namespace sdfg
