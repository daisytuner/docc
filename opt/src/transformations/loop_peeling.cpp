#include "sdfg/transformations/loop_peeling.h"

#include "sdfg/analysis/assumptions_analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/deepcopy/structured_sdfg_deep_copy.h"
#include "sdfg/structured_control_flow/for.h"
#include "sdfg/structured_control_flow/if_else.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/structured_control_flow/reduce.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/symbolic/conjunctive_normal_form.h"
#include "sdfg/symbolic/extreme_values.h"
#include "sdfg/symbolic/symbolic.h"

#include <symengine/integer.h>
#include <symengine/logic.h>

namespace sdfg {
namespace transformations {

LoopPeeling::LoopPeeling(structured_control_flow::StructuredLoop& loop, bool predicate)
    : loop_(loop), predicate_(predicate) {};

std::string LoopPeeling::name() const { return "LoopPeeling"; };

/// Sound tri-state result of statically evaluating a boundary condition against
/// the SDFG's symbol assumptions.
enum class Provable { True, False, Unknown };

/// Prove a single relational literal True/False under @p assums, or Unknown when
/// undecidable. Only sound conclusions are returned. Gt/Ge are normalized by
/// SymEngine into swapped StrictLessThan/LessThan, so four classes suffice.
static Provable
prove_literal(const symbolic::Condition& lit, const symbolic::SymbolSet& params, const symbolic::Assumptions& assums) {
    if (symbolic::is_true(lit)) return Provable::True;
    if (symbolic::is_false(lit)) return Provable::False;
    if (!SymEngine::is_a_Relational(*lit)) return Provable::Unknown;

    auto rel = SymEngine::rcp_static_cast<const SymEngine::Relational>(lit);
    auto a = rel->get_arg1();
    auto b = rel->get_arg2();

    // Try both the tight (stride/evolution-aware) and loose bound sets: a
    // stride-tiled offset's provable range often needs the tight upper bound
    // (e.g. last-iteration value), while type-derived bounds are loose. Both are
    // sound, so proving under either suffices.
    auto lt = [&](const symbolic::Expression& x, const symbolic::Expression& y) {
        return symbolic::is_lt(x, y, params, assums, true) || symbolic::is_lt(x, y, params, assums, false);
    };
    auto le = [&](const symbolic::Expression& x, const symbolic::Expression& y) {
        return symbolic::is_le(x, y, params, assums, true) || symbolic::is_le(x, y, params, assums, false);
    };
    auto gt = [&](const symbolic::Expression& x, const symbolic::Expression& y) {
        return symbolic::is_gt(x, y, params, assums, true) || symbolic::is_gt(x, y, params, assums, false);
    };
    auto ge = [&](const symbolic::Expression& x, const symbolic::Expression& y) {
        return symbolic::is_ge(x, y, params, assums, true) || symbolic::is_ge(x, y, params, assums, false);
    };
    auto eq = [&](const symbolic::Expression& x, const symbolic::Expression& y) {
        return symbolic::is_eq(x, y, params, assums, true) || symbolic::is_eq(x, y, params, assums, false);
    };

    if (SymEngine::is_a<SymEngine::StrictLessThan>(*lit)) { // a < b
        if (lt(a, b)) return Provable::True;
        if (ge(a, b)) return Provable::False;
    } else if (SymEngine::is_a<SymEngine::LessThan>(*lit)) { // a <= b
        if (le(a, b)) return Provable::True;
        if (gt(a, b)) return Provable::False;
    } else if (SymEngine::is_a<SymEngine::Equality>(*lit)) { // a == b
        if (eq(a, b)) return Provable::True;
        if (lt(a, b) || gt(a, b)) return Provable::False;
    } else if (SymEngine::is_a<SymEngine::Unequality>(*lit)) { // a != b
        if (lt(a, b) || gt(a, b)) return Provable::True;
        if (eq(a, b)) return Provable::False;
    }
    return Provable::Unknown;
}

/// Prove a boundary condition (a conjunction of bounds) statically True/False, or
/// Unknown. True  => the guarded remainder is dead and can be dropped; False =>
/// the clean tile is dead and only the remainder is needed. Sound: only proven
/// conclusions are returned, so an Unknown keeps the full (correct) emission.
static Provable prove_condition(
    const symbolic::Condition& cond, const symbolic::SymbolSet& params, const symbolic::Assumptions& assums
) {
    auto simplified = symbolic::simplify(cond);
    if (symbolic::is_true(simplified)) return Provable::True;
    if (symbolic::is_false(simplified)) return Provable::False;

    symbolic::CNF cnf;
    try {
        cnf = symbolic::conjunctive_normal_form(cond);
    } catch (const symbolic::CNFException&) {
        return Provable::Unknown;
    }

    // The whole condition is the AND of clauses; each clause is an OR of literals.
    bool all_true = true;
    for (const auto& clause : cnf) {
        bool clause_true = false;
        bool clause_all_false = !clause.empty();
        for (const auto& lit : clause) {
            auto p = prove_literal(lit, params, assums);
            if (p == Provable::True) {
                clause_true = true;
                break;
            }
            if (p != Provable::False) {
                clause_all_false = false;
            }
        }
        if (clause_true) continue;
        // A clause whose every literal is provably false is unsatisfiable, so the
        // whole conjunction is provably false.
        if (clause_all_false) return Provable::False;
        all_true = false; // this clause is undecided
    }
    return all_true ? Provable::True : Provable::Unknown;
}


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

    // Build the 0-based loop nest into @p into and return its innermost body.
    auto build_zero_nest = [&](structured_control_flow::Sequence& into) -> structured_control_flow::Sequence& {
        structured_control_flow::Sequence* current = &into;
        for (auto& info : infos) {
            auto& nl =
                append_loop(builder, *current, *info.loop, info.indvar, info.zero_condition, zero, info.loop->update());
            current = &nl.root();
        }
        return *current;
    };
    // Rewrite a copied body's shifted induction variables back to originals.
    auto apply_shifts = [&](structured_control_flow::Sequence& body) {
        for (auto& info : infos) {
            if (!symbolic::eq(info.init, zero)) {
                body.replace(info.indvar, info.shifted);
            }
        }
    };
    // Clean, unguarded 0-based nest running the whole tile.
    auto emit_clean = [&](structured_control_flow::Sequence& into) {
        auto& body = build_zero_nest(into);
        deepcopy::StructuredSDFGDeepCopy(builder, body, innermost->root()).insert();
        apply_shifts(body);
    };
    // 0-based nest with the original per-iteration bounds as a body guard.
    auto emit_guarded = [&](structured_control_flow::Sequence& into) {
        auto& deep = build_zero_nest(into);
        auto& rem_if = builder.add_if_else(deep, loop_.debug_info());
        auto& body = builder.add_case(rem_if, combined_guard, loop_.debug_info());
        deepcopy::StructuredSDFGDeepCopy(builder, body, innermost->root()).insert();
        apply_shifts(body);
    };

    // Assumptions at the innermost body scope carry every enclosing loop's bounds
    // (e.g. tile-offset ranges), so a boundary that is dead given the surrounding
    // nest is provable here — unlike the coarser function-level assumptions.
    auto& assumptions_analysis = analysis_manager.get<analysis::AssumptionsAnalysis>();
    const auto& assums = assumptions_analysis.get(innermost->root(), true);
    const auto& params = assumptions_analysis.parameters();
    // The boundary is dead exactly when the last (worst-case, since the loops are
    // monotonic) iteration still fits: `combined_fits`. This governs both forms —
    // the hoisted remainder and the predicated per-iteration guard.
    auto fits = prove_condition(combined_fits, params, assums);

    if (predicate_) {
        auto& holder = builder.add_sequence_before(*parent, loop_, loop_.debug_info());
        if (fits == Provable::True) {
            // Every iteration is provably in bounds: the per-iteration guard is dead.
            emit_clean(holder);
        } else {
            emit_guarded(holder);
        }
    } else {
        if (fits == Provable::True) {
            // The whole tile always fits: the remainder branch is dead, emit only
            // the clean nest (no outer IfElse).
            auto& holder = builder.add_sequence_before(*parent, loop_, loop_.debug_info());
            emit_clean(holder);
        } else if (fits == Provable::False) {
            // The tile never fully fits: the clean branch is dead, emit only the
            // guarded remainder.
            auto& holder = builder.add_sequence_before(*parent, loop_, loop_.debug_info());
            emit_guarded(holder);
        } else {
            // Hoisted: full clean tile in the "then" branch, original variable-trip
            // remainder in the "else". The "then" micro-kernel is unguarded (vectorizes).
            auto& if_else = builder.add_if_else_before(*parent, loop_, loop_.debug_info());
            auto& then_branch = builder.add_case(if_else, combined_fits, loop_.debug_info());
            emit_clean(then_branch);
            auto& else_branch = builder.add_case(if_else, symbolic::Not(combined_fits), loop_.debug_info());
            emit_guarded(else_branch);
        }
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
