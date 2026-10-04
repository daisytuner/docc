#include "sdfg/analysis/assumptions_analysis.h"

#include <unordered_set>
#include <utility>
#include <vector>

#include "sdfg/analysis/analysis.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/library_node.h"
#include "sdfg/data_flow/memlet.h"
#include "sdfg/structured_control_flow/reduce.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/symbolic/assumptions.h"
#include "sdfg/symbolic/conjunctive_normal_form.h"
#include "sdfg/symbolic/polynomials.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/type.h"

namespace sdfg {
namespace analysis {

namespace {

// Add per-symbol bounds derived from a single atomic relation
// (`LessThan` / `StrictLessThan` / `Equality`) into `branch_assumptions`.
// All other clause shapes (logical connectives, `Unequality`, custom
// booleans...) are silently skipped: the caller is responsible for
// pre-normalizing via CNF and only invoking this on the literals of
// single-literal conjuncts.
//
// Only symbols already known to `outer_assumptions` get bounds (scope
// hygiene — don't introduce alien symbols into the branch state).
void extract_bound_from_literal(
    const symbolic::Condition& cond,
    const symbolic::Assumptions& outer_assumptions,
    symbolic::Assumptions& branch_assumptions
) {
    // Normalize to `delta OP K` where `delta = lhs - rhs`. SymEngine collapses
    // `Gt` and `Ge` into swapped `StrictLessThan` / `LessThan`, so we only
    // need to recognise the three canonical relational classes (same
    // approach as opt/src/transformations/loop_condition_normalize.cpp and
    // sdfglib-auto/.../feature_extractor.cpp::visit_condition).
    symbolic::Expression delta;
    symbolic::Expression K_le = SymEngine::null; // threshold for `delta <= K_le`
    symbolic::Expression K_ge = SymEngine::null; // threshold for `delta >= K_ge`

    if (SymEngine::is_a<SymEngine::LessThan>(*cond)) {
        // a <= b -> delta <= 0.
        auto rel = SymEngine::rcp_static_cast<const SymEngine::LessThan>(cond);
        delta = symbolic::sub(rel->get_arg1(), rel->get_arg2());
        K_le = symbolic::zero();
    } else if (SymEngine::is_a<SymEngine::StrictLessThan>(*cond)) {
        // a < b on the integer domain -> delta <= -1.
        auto rel = SymEngine::rcp_static_cast<const SymEngine::StrictLessThan>(cond);
        delta = symbolic::sub(rel->get_arg1(), rel->get_arg2());
        K_le = symbolic::integer(-1);
    } else if (SymEngine::is_a<SymEngine::Equality>(*cond)) {
        // a == b -> delta both <= 0 and >= 0.
        auto rel = SymEngine::rcp_static_cast<const SymEngine::Equality>(cond);
        delta = symbolic::sub(rel->get_arg1(), rel->get_arg2());
        K_le = symbolic::zero();
        K_ge = symbolic::zero();
    } else {
        return;
    }

    // Collect indvars and validate other symbols.
    //
    // Rationale:
    //   * Target indvars are identified by `Assumption::map()` being non-null.
    //     We narrow only loop variables — narrowing a non-indvar local would
    //     require knowing it isn't reassigned inside the branch, which is out
    //     of scope here.
    //   * Every non-indvar symbol must be marked `constant()` by
    //     `AssumptionsAnalysis` (SDFG-level parameters and outer-loop indvars
    //     not active in this scope qualify; locally-written variables do NOT).
    //     If a non-constant symbol appears, the bound would be invalidated by
    //     writes deeper in the body.
    //   * Symbols not in `outer_assumptions` are treated as unknown — skip
    //     rather than emit an unverified bound.
    std::vector<symbolic::Symbol> indvar_targets;
    for (const auto& sym : symbolic::atoms(delta)) {
        auto it = outer_assumptions.find(sym);
        if (it == outer_assumptions.end()) {
            return;
        }
        bool is_indvar = !it->second.map().is_null();
        if (is_indvar) {
            indvar_targets.push_back(sym);
        } else if (!it->second.constant()) {
            return; // non-constant local — would be invalidated by body writes
        }
    }
    if (indvar_targets.empty()) {
        return;
    }

    if (indvar_targets.size() == 1) {
        // Single-indvar path: emit a per-symbol list bound (and possibly
        // refine the tight slot). Cheapest path; no BoundAnalysis cycle risk
        // because the bound expression mentions only constants and other
        // constant-marked symbols.
        symbolic::Symbol target = indvar_targets.front();
        auto neg_delta = symbolic::expand(symbolic::mul(symbolic::integer(-1), delta));

        auto try_emit = [&](const symbolic::Expression& K, bool is_lower) {
            if (K.is_null()) {
                return;
            }
            auto neg_K = symbolic::expand(symbolic::mul(symbolic::integer(-1), K));

            auto add_bound = [&](const symbolic::Expression& b, bool lower) {
                if (b.is_null()) {
                    return false;
                }
                for (const auto& a : symbolic::atoms(b)) {
                    if (symbolic::eq(a, target)) {
                        return false;
                    }
                }
                if (lower) {
                    branch_assumptions[target].add_lower_bound(b);
                } else {
                    branch_assumptions[target].add_upper_bound(b);
                }
                // Refine the tight bound slot when the branch literal strictly
                // narrows the outer-scope tight bound.
                auto outer_it = outer_assumptions.find(target);
                if (outer_it == outer_assumptions.end()) {
                    return true;
                }
                if (!SymEngine::is_a<SymEngine::Integer>(*b)) {
                    return true;
                }
                if (lower) {
                    auto cur = branch_assumptions[target].tight_lower_bound();
                    if (cur.is_null()) {
                        cur = outer_it->second.tight_lower_bound();
                    }
                    if (cur.is_null() || !SymEngine::is_a<SymEngine::Integer>(*cur)) {
                        return true;
                    }
                    auto refined = symbolic::max(cur, b);
                    if (!SymEngine::is_a<SymEngine::Integer>(*refined)) {
                        return true;
                    }
                    branch_assumptions[target].tight_lower_bound(refined);
                } else {
                    auto cur = branch_assumptions[target].tight_upper_bound();
                    if (cur.is_null()) {
                        cur = outer_it->second.tight_upper_bound();
                    }
                    if (cur.is_null() || !SymEngine::is_a<SymEngine::Integer>(*cur)) {
                        return true;
                    }
                    auto refined = symbolic::min(cur, b);
                    if (!SymEngine::is_a<SymEngine::Integer>(*refined)) {
                        return true;
                    }
                    branch_assumptions[target].tight_upper_bound(refined);
                }
                return true;
            };

            if (auto b = symbolic::solve_affine_bound(delta, target, K, is_lower); !b.is_null()) {
                add_bound(b, is_lower);
                return;
            }
            if (auto b = symbolic::solve_affine_bound(neg_delta, target, neg_K, !is_lower); !b.is_null()) {
                add_bound(b, !is_lower);
            }
        };

        try_emit(K_le, /*is_lower=*/false);
        try_emit(K_ge, /*is_lower=*/true);
        return;
    }

    // Multi-indvar path: register the literal as a *constraint* on every
    // involved indvar's `Assumption::constraints()` set. Canonical form is
    // `expr <= 0` (so equality `delta == 0` produces two constraints,
    // `delta` and `-delta`).
    //
    // We require `delta` to be affine in every indvar — i.e. each indvar
    // appears with an integer coefficient — so `BoundAnalysis` can later
    // project the constraint onto a single variable via affine inversion.
    // Constants and `constant()` symbols stay opaque in the residue.
    for (const auto& sym : indvar_targets) {
        auto decomp = symbolic::affine_decomposition(delta, sym);
        if (!decomp.success) {
            return;
        }
        if (!SymEngine::is_a<SymEngine::Integer>(*decomp.coeff)) {
            return;
        }
    }

    auto register_constraint = [&](const symbolic::Expression& c) {
        auto canonical = symbolic::expand(c);
        for (const auto& sym : indvar_targets) {
            branch_assumptions[sym].add_constraint(canonical);
        }
    };

    if (!K_le.is_null()) {
        // delta <= K_le  <=>  delta - K_le <= 0
        register_constraint(symbolic::sub(delta, K_le));
    }
    if (!K_ge.is_null()) {
        // delta >= K_ge  <=>  K_ge - delta <= 0
        register_constraint(symbolic::sub(K_ge, delta));
    }
}

// Add per-symbol bounds derived from an IfElse branch condition into
// `branch_assumptions`. Uses CNF normalization to handle arbitrary boolean
// structure (And / Or / Not / nested / De Morgan) uniformly: after CNF the
// condition is `clause_1 AND clause_2 AND ... AND clause_n` where each
// clause is a disjunction (OR) of literals. Only **single-literal** clauses
// are sound to translate into per-symbol bounds — a multi-literal
// disjunctive clause `(L1 OR L2)` does not constrain any one symbol.
//
// CNF conversion can fail on shapes outside its supported grammar
// (CNFException). On failure we silently emit no bounds; the analysis stays
// sound, just conservative.
void extract_assumptions_from_condition(
    const symbolic::Condition& cond,
    const symbolic::Assumptions& outer_assumptions,
    symbolic::Assumptions& branch_assumptions
) {
    symbolic::CNF cnf;
    try {
        cnf = symbolic::conjunctive_normal_form(cond);
    } catch (const symbolic::CNFException&) {
        return;
    }
    for (const auto& clause : cnf) {
        if (clause.size() != 1) {
            continue; // disjunctive — not soundly splittable
        }
        extract_bound_from_literal(clause.front(), outer_assumptions, branch_assumptions);
    }
}

// Ensure `branch_assumptions[sym]` exists for every symbol that
// `extract_assumptions_from_condition` will add a bound for, so the per-symbol
// `Assumption` object is created with the right symbol identity.
void ensure_assumption_entries(const symbolic::Condition& cond, symbolic::Assumptions& branch_assumptions) {
    for (const auto& sym : symbolic::atoms(cond)) {
        if (branch_assumptions.find(sym) == branch_assumptions.end()) {
            branch_assumptions.insert({sym, symbolic::Assumption(sym)});
        }
    }
}

// Containers that may be written anywhere in `node`; address-taken containers count as written.
void collect_written_containers(structured_control_flow::ControlFlowNode& node, std::unordered_set<std::string>& written) {
    if (auto* block = dyn_cast<structured_control_flow::Block*>(&node)) {
        auto& dataflow = block->dataflow();
        for (auto* access_node : dataflow.data_nodes()) {
            if (dataflow.in_degree(*access_node) > 0) {
                written.insert(access_node->data());
                continue;
            }
            for (auto& oedge : dataflow.out_edges(*access_node)) {
                if (oedge.type() == data_flow::MemletType::Reference ||
                    oedge.type() == data_flow::MemletType::Dereference_Dst) {
                    written.insert(access_node->data());
                    break;
                }
                if (auto* lib = dynamic_cast<data_flow::LibraryNode*>(&oedge.dst())) {
                    auto meta = lib->pointer_access_type(oedge);
                    if (meta && meta->may_contain_writes()) {
                        written.insert(access_node->data());
                        break;
                    }
                }
            }
        }
    } else if (auto* assignment_block = dyn_cast<structured_control_flow::AssignmentBlock*>(&node)) {
        for (auto& entry : assignment_block->assignments()) {
            written.insert(entry.first->get_name());
        }
    } else if (auto* sequence = dyn_cast<structured_control_flow::Sequence*>(&node)) {
        for (size_t i = 0; i < sequence->size(); i++) {
            collect_written_containers(sequence->at(i), written);
        }
    } else if (auto* if_else = dyn_cast<structured_control_flow::IfElse*>(&node)) {
        for (size_t i = 0; i < if_else->size(); i++) {
            collect_written_containers(if_else->at(i).first, written);
        }
    } else if (auto* while_stmt = dyn_cast<structured_control_flow::While*>(&node)) {
        collect_written_containers(while_stmt->root(), written);
    } else if (auto* loop = dyn_cast<structured_control_flow::StructuredLoop*>(&node)) {
        written.insert(loop->indvar()->get_name());
        if (auto* reduce = dyn_cast<structured_control_flow::Reduce*>(loop)) {
            for (const auto& entry : reduce->reductions()) {
                written.insert(entry.container);
            }
        }
        collect_written_containers(loop->root(), written);
    }
}

} // namespace

symbolic::SymbolSet AssumptionsAnalysis::per_symbol_refined_symbols(const symbolic::Condition& cond) {
    // Structural classification, independent of scope/indvar status: a symbol is
    // "per-symbol refined" iff it is the sole variable of some single-literal CNF
    // clause. Multi-variable literals (`i + j <= 15`) are coupled constraints and
    // are deliberately excluded so their ancestor operands stay opaque.
    symbolic::SymbolSet result;
    symbolic::CNF cnf;
    try {
        cnf = symbolic::conjunctive_normal_form(cond);
    } catch (const symbolic::CNFException&) {
        return result;
    }
    for (const auto& clause : cnf) {
        if (clause.size() != 1) {
            continue; // disjunctive — constrains no single symbol
        }
        const auto& lit = clause.front();

        symbolic::Expression delta = SymEngine::null;
        if (SymEngine::is_a<SymEngine::LessThan>(*lit)) {
            auto rel = SymEngine::rcp_static_cast<const SymEngine::LessThan>(lit);
            delta = symbolic::sub(rel->get_arg1(), rel->get_arg2());
        } else if (SymEngine::is_a<SymEngine::StrictLessThan>(*lit)) {
            auto rel = SymEngine::rcp_static_cast<const SymEngine::StrictLessThan>(lit);
            delta = symbolic::sub(rel->get_arg1(), rel->get_arg2());
        } else if (SymEngine::is_a<SymEngine::Equality>(*lit)) {
            auto rel = SymEngine::rcp_static_cast<const SymEngine::Equality>(lit);
            delta = symbolic::sub(rel->get_arg1(), rel->get_arg2());
        } else {
            continue;
        }

        const auto atoms = symbolic::atoms(delta);
        if (atoms.size() == 1) {
            result.insert(*atoms.begin());
        }
    }
    return result;
}

AssumptionsAnalysis::AssumptionsAnalysis(StructuredSDFG& sdfg)
    : Analysis(sdfg) {

      };

AssumptionsAnalysis::AssumptionsAnalysis(StructuredSDFG& sdfg, bool with_branch_conditions)
    : Analysis(sdfg), with_branch_conditions_(with_branch_conditions) {

      };

void AssumptionsAnalysis::run(analysis::AnalysisManager& analysis_manager) {
    this->own_assumptions_.clear();
    this->parent_scope_.clear();
    this->scope_of_.clear();
    this->assumptions_.clear();
    this->assumptions_with_trivial_.clear();
    this->constant_symbols_.clear();
    this->constant_symbols_with_trivial_.clear();

    this->parameters_.clear();

    // Determine parameters
    this->determine_parameters(analysis_manager);

    // Root scope is materialized eagerly from the SDFG-level assumptions
    auto& root = sdfg_.root();
    this->assumptions_.insert({&root, this->additional_assumptions_});
    auto& initial = this->assumptions_[&root];

    this->assumptions_with_trivial_.insert({&root, initial});
    auto& initial_with_trivial = this->assumptions_with_trivial_[&root];
    for (auto& entry : sdfg_.assumptions()) {
        if (initial_with_trivial.find(entry.first) == initial_with_trivial.end()) {
            initial_with_trivial.insert({entry.first, entry.second});
        } else {
            for (auto& lb : entry.second.lower_bounds()) {
                initial_with_trivial.at(entry.first).add_lower_bound(lb);
            }
            for (auto& ub : entry.second.upper_bounds()) {
                initial_with_trivial.at(entry.first).add_upper_bound(ub);
            }
        }
    }

    this->traverse(root, root);
};

void AssumptionsAnalysis::traverse(Node& current, Node& scope) {
    this->scope_of_[&current] = &scope;

    if (auto sequence_stmt = dyn_cast<structured_control_flow::Sequence*>(&current)) {
        for (size_t i = 0; i < sequence_stmt->size(); i++) {
            this->traverse(sequence_stmt->at(i), scope);
        }
    } else if (auto if_else_stmt = dyn_cast<structured_control_flow::IfElse*>(&current)) {
        if (!with_branch_conditions_) {
            // Cheap path: don't refine branch assumptions from the case
            // conditions. Recurse into each branch sequence inheriting the
            // outer scope's assumptions verbatim. CNF normalization plus
            // coupled-constraint extraction (the expensive part) is skipped
            // entirely.
            for (size_t i = 0; i < if_else_stmt->size(); i++) {
                auto& branch_seq = if_else_stmt->at(i).first;
                this->traverse(const_cast<structured_control_flow::Sequence&>(branch_seq), scope);
            }
            return;
        }
        // Branch refinement looks symbols up in the full outer scope.
        const auto& outer_assumptions = this->materialize(scope, /*include_trivial_bounds=*/false);
        for (size_t i = 0; i < if_else_stmt->size(); i++) {
            auto& branch_seq = const_cast<structured_control_flow::Sequence&>(if_else_stmt->at(i).first);
            const auto& condition = if_else_stmt->at(i).second;

            // Build per-branch assumption deltas from the case condition.
            // Conjunctive structure is split into independent constraints;
            // disjunctions / negations contribute nothing (see
            // extract_assumptions_from_condition).
            symbolic::Assumptions branch_assumptions;
            ensure_assumption_entries(condition, branch_assumptions);
            extract_assumptions_from_condition(condition, outer_assumptions, branch_assumptions);

            this->add_scope(branch_seq, std::move(branch_assumptions), scope);
            this->traverse(branch_seq, branch_seq);
        }
    } else if (auto while_stmt = dyn_cast<structured_control_flow::While*>(&current)) {
        this->traverse(while_stmt->root(), scope);
    } else if (auto loop_stmt = dyn_cast<structured_control_flow::StructuredLoop*>(&current)) {
        this->traverse_structured_loop(loop_stmt, scope);
    } else {
        // Other control flow nodes (e.g., Block) do not introduce assumptions or comprise scopes
    }
};

void AssumptionsAnalysis::traverse_structured_loop(structured_control_flow::StructuredLoop* loop, Node& scope) {
    // A structured loop induces assumption for the loop body
    auto& body = loop->root();
    symbolic::Assumptions body_assumptions;

    // Define all constant symbols
    auto indvar = loop->indvar();
    auto update = loop->update();
    auto init = loop->init();

    // By definition, all symbols in the loop condition are constant within the loop body
    symbolic::SymbolSet loop_syms = symbolic::atoms(loop->condition());
    for (auto& sym : loop_syms) {
        body_assumptions.insert({sym, symbolic::Assumption(sym)});
        body_assumptions[sym].constant(true);
    }

    // Define map of indvar
    body_assumptions[indvar].map(update);
    body_assumptions[indvar].constant(true);

    // Monotonic -> infer bounds on indvar
    symbolic::Integer stride = loop->stride();
    if (!stride.is_null() && loop->is_monotonic()) {
        body_assumptions[indvar].add_lower_bound(init);
        body_assumptions[indvar].tight_lower_bound(init);

        auto ub = loop->canonical_bound_upper();
        if (!ub.is_null()) {
            // Convert into inclusive bound
            symbolic::Expression ub_inclusive;
            if (SymEngine::is_a<SymEngine::Min>(*ub)) {
                auto min = SymEngine::rcp_static_cast<const SymEngine::Min>(ub);
                std::vector<symbolic::Expression> inclusive_args;
                for (size_t i = 0; i < min->get_args().size(); i++) {
                    auto arg = min->get_args()[i];
                    inclusive_args.push_back(symbolic::sub(arg, symbolic::one()));
                }
                ub_inclusive = inclusive_args.at(0);
                for (size_t i = 1; i < inclusive_args.size(); i++) {
                    ub_inclusive = symbolic::min(ub_inclusive, inclusive_args.at(i));
                }
            } else {
                ub_inclusive = symbolic::sub(ub, symbolic::one());
            }

            // ub is a general upper bound
            // Stride-tighten an inclusive bound to its largest in-range value `init + k*stride`.
            // Subtracting `init` cancels the arg's parent-relative part (e.g.
            // `(63 + tile0) - tile0 = 63`), so `idiv(63, stride)*stride` folds to a
            // clean constant offset (`tile0 + 60`) the inequality prover can use.
            auto stride_tighten = [&](const symbolic::Expression& incl) -> symbolic::Expression {
                if (symbolic::eq(stride, symbolic::one())) {
                    return incl;
                }
                auto range = symbolic::expand(symbolic::sub(incl, init));
                if (SymEngine::is_a<SymEngine::Integer>(*range)) {
                    auto steps = symbolic::div(range, stride);
                    return symbolic::add(init, symbolic::mul(steps, stride));
                }
                // C semantics: s*(x/s) == x - x%s, so `init + s*idiv(incl-init, s) == incl - imod(incl-init, s)`.
                // This form keeps `init` out of the dominant term: BoundAnalysis bounds imod to [0, s-1].
                return symbolic::sub(incl, symbolic::mod(range, stride));
            };

            // Tight upper bound: the last iteration value. idiv is monotone, so for a Min bound
            // `init + s*idiv(min(a,b) - init, s) == min(tighten(a), tighten(b))`; the flat form
            // keeps the Min out of idiv, where it would be non-monotone in `init` for BoundAnalysis.
            if (SymEngine::is_a<SymEngine::Min>(*ub_inclusive)) {
                symbolic::Expression tight_ub = SymEngine::null;
                for (const auto& arg : ub_inclusive->get_args()) {
                    auto t = stride_tighten(arg);
                    tight_ub = tight_ub.is_null() ? t : symbolic::min(tight_ub, t);
                }
                body_assumptions[indvar].tight_upper_bound(tight_ub);
            } else {
                body_assumptions[indvar].tight_upper_bound(stride_tighten(ub_inclusive));
            }

            // Register the coupled constraint `indvar - tight <= 0` when `tight`
            // couples the indvar with another loop variable (e.g. `tile1 <= tile0 +
            // 60`). Per-symbol bounding decorrelates such a bound (`tile0 + 60 ->
            // [60, N]`), losing the coupling; storing it as a constraint lets the
            // Add coupled-constraint projection cancel the shared parent symbol
            // (proving e.g. `3 + tile1 < 64 + tile0`). Only loop-condition bounds
            // reach here — IfElse conditions already register constraints.
            auto add_coupled_constraint = [&](const symbolic::Expression& tight) {
                bool couples_indvar = false;
                for (const auto& a : symbolic::atoms(tight)) {
                    if (!symbolic::eq(a, indvar) && !this->parameters_.contains(a)) {
                        couples_indvar = true;
                        break;
                    }
                }
                if (!couples_indvar) {
                    return;
                }
                auto constraint = symbolic::expand(symbolic::sub(indvar, tight));
                body_assumptions[indvar].add_constraint(constraint);
                for (const auto& a : symbolic::atoms(tight)) {
                    if (!symbolic::eq(a, indvar) && body_assumptions.find(a) != body_assumptions.end()) {
                        body_assumptions[a].add_constraint(constraint);
                    }
                }
            };
            if (SymEngine::is_a<SymEngine::Min>(*ub)) {
                auto min = SymEngine::rcp_static_cast<const SymEngine::Min>(ub);
                for (size_t i = 0; i < min->get_args().size(); i++) {
                    auto arg = min->get_args()[i];
                    auto arg_inclusive = symbolic::sub(arg, symbolic::one());
                    body_assumptions[indvar].add_upper_bound(arg_inclusive);
                    auto tight = stride_tighten(arg_inclusive);
                    body_assumptions[indvar].add_upper_bound(tight);
                    add_coupled_constraint(tight);
                }
            } else {
                body_assumptions[indvar].add_upper_bound(ub_inclusive);
                auto tight = stride_tighten(ub_inclusive);
                body_assumptions[indvar].add_upper_bound(tight);
                add_coupled_constraint(tight);
            }

            // Furthermore, we can infer lower bounds for each upper bound's symbol
            // For a loop to execute, we need ub > init, so ub >= init + 1
            auto min_ub_value = symbolic::add(init, symbolic::one());

            // Helper to infer lower bound for symbols in an expression
            // If expr = coeff * sym + offset and we need expr >= min_ub_value,
            // then sym >= (min_ub_value - offset) / coeff
            //
            // Only infer bounds when min_ub_value is purely constant (no non-parameter
            // symbols). When init is symbolic (e.g., a tile indvar), the derived bounds
            // reference that symbol (e.g., j_tile0 >= j_tile1-255) and create
            // unresolvable chains in BoundAnalysis that prevent delinearization.
            bool min_ub_value_is_clean = true;
            for (auto& atom : symbolic::atoms(min_ub_value)) {
                if (!this->parameters_.contains(atom)) {
                    min_ub_value_is_clean = false;
                    break;
                }
            }

            auto infer_symbol_lower_bound = [&](const symbolic::Expression& expr) {
                if (!min_ub_value_is_clean) {
                    return;
                }
                auto atoms = symbolic::atoms(expr);
                for (const auto& sym : atoms) {
                    auto bound = symbolic::solve_affine_bound(expr, sym, min_ub_value, true);
                    if (!bound.is_null()) {
                        body_assumptions[sym].add_lower_bound(bound);
                    }
                }
            };

            if (SymEngine::is_a<SymEngine::Min>(*ub)) {
                auto min = SymEngine::rcp_static_cast<const SymEngine::Min>(ub);
                for (size_t i = 0; i < min->get_args().size(); i++) {
                    auto arg = min->get_args()[i];
                    infer_symbol_lower_bound(arg);
                }
            } else {
                infer_symbol_lower_bound(ub);
            }
        }
    }

    this->add_scope(body, std::move(body_assumptions), scope);
    this->traverse(body, body);
}

namespace {

// Inner-scope assumptions take precedence; outer bounds/constraints accumulate.
void merge_outer(symbolic::Assumptions& into, const symbolic::Assumptions& outer) {
    for (auto& entry : outer) {
        auto it = into.find(entry.first);
        if (it == into.end()) {
            into.insert({entry.first, entry.second});
            continue;
        }
        auto& lower_assum = it->second;
        for (auto& ub : entry.second.upper_bounds()) {
            lower_assum.add_upper_bound(ub);
        }
        for (auto& lb : entry.second.lower_bounds()) {
            lower_assum.add_lower_bound(lb);
        }
        for (auto& c : entry.second.constraints()) {
            lower_assum.add_constraint(c);
        }
        if (lower_assum.tight_upper_bound().is_null()) {
            lower_assum.tight_upper_bound(entry.second.tight_upper_bound());
        }
        if (lower_assum.tight_lower_bound().is_null()) {
            lower_assum.tight_lower_bound(entry.second.tight_lower_bound());
        }
        if (lower_assum.map().is_null()) {
            lower_assum.map(entry.second.map());
        }
        if (!lower_assum.constant()) {
            lower_assum.constant(entry.second.constant());
        }
    }
}

} // namespace

void AssumptionsAnalysis::add_scope(Node& node, symbolic::Assumptions own, Node& parent) {
    this->own_assumptions_[&node] = std::move(own);
    this->parent_scope_[&node] = &parent;
}

const symbolic::Assumptions& AssumptionsAnalysis::materialize(Node& scope, bool include_trivial_bounds) {
    auto& cache = include_trivial_bounds ? this->assumptions_with_trivial_ : this->assumptions_;
    auto it = cache.find(&scope);
    if (it != cache.end()) {
        return it->second;
    }
    // Node-based map: `outer` stays valid while the recursion inserts other scopes.
    const auto& outer = this->materialize(*this->parent_scope_.at(&scope), include_trivial_bounds);
    symbolic::Assumptions merged = this->own_assumptions_.at(&scope);
    merge_outer(merged, outer);
    return cache.emplace(&scope, std::move(merged)).first->second;
}

// Merging ORs `constant()`, so a scope's constants are its parent's plus its own constant entries.
const symbolic::SymbolSet& AssumptionsAnalysis::materialize_constants(Node& scope, bool include_trivial_bounds) {
    auto& cache = include_trivial_bounds ? this->constant_symbols_with_trivial_ : this->constant_symbols_;
    auto it = cache.find(&scope);
    if (it != cache.end()) {
        return it->second;
    }
    symbolic::SymbolSet constants;
    auto parent = this->parent_scope_.find(&scope);
    if (parent == this->parent_scope_.end()) {
        for (const auto& [sym, assum] : this->materialize(scope, include_trivial_bounds)) {
            if (assum.constant()) {
                constants.insert(sym);
            }
        }
    } else {
        constants = this->materialize_constants(*parent->second, include_trivial_bounds);
        for (const auto& [sym, assum] : this->own_assumptions_.at(&scope)) {
            if (assum.constant()) {
                constants.insert(sym);
            }
        }
    }
    return cache.emplace(&scope, std::move(constants)).first->second;
}

void AssumptionsAnalysis::determine_parameters(analysis::AnalysisManager& analysis_manager) {
    // Only scalar arguments: proving pointers are never moved needs alias reasoning.
    std::unordered_set<std::string> written;
    collect_written_containers(this->sdfg_.root(), written);
    for (auto& container : this->sdfg_.arguments()) {
        if (this->sdfg_.type(container).type_id() != types::TypeID::Scalar) {
            continue;
        }
        if (!written.contains(container)) {
            this->parameters_.insert(symbolic::symbol(container));
        }
    }
}

const symbolic::Assumptions& AssumptionsAnalysis::
    get(structured_control_flow::ControlFlowNode& node, bool include_trivial_bounds) {
    return this->materialize(*this->scope_of_.at(&node), include_trivial_bounds);
}

const symbolic::SymbolSet& AssumptionsAnalysis::
    constant_symbols(structured_control_flow::ControlFlowNode& node, bool include_trivial_bounds) {
    return this->materialize_constants(*this->scope_of_.at(&node), include_trivial_bounds);
}

const symbolic::SymbolSet& AssumptionsAnalysis::parameters() {
    return this->parameters_;
}

bool AssumptionsAnalysis::is_parameter(const symbolic::Symbol& container) {
    return this->parameters_.contains(container);
}

bool AssumptionsAnalysis::is_parameter(const std::string& container) {
    return this->is_parameter(symbolic::symbol(container));
}

} // namespace analysis
} // namespace sdfg
