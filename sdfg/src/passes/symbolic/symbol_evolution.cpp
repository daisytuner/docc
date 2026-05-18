#include "sdfg/passes/symbolic/symbol_evolution.h"

#include "sdfg/analysis/dominance_analysis.h"
#include "sdfg/analysis/users.h"
#include "sdfg/symbolic/polynomials.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {
namespace passes {

namespace {

// -----------------------------------------------------------------------------
// Semantic equality on symbolic expressions
// -----------------------------------------------------------------------------
//
// `symbolic::eq` is purely structural and frequently misses obvious identities
// (e.g. `2*(0-1)` vs `-2`). For verification conditions in the recurrence
// solver we need a notion of equality that survives algebraic rewrites.
bool sym_equiv(const symbolic::Expression& a, const symbolic::Expression& b) {
    if (symbolic::eq(a, b)) {
        return true;
    }
    auto diff = symbolic::simplify(symbolic::expand(symbolic::sub(a, b)));
    return SymEngine::is_a<SymEngine::Integer>(*diff) &&
           SymEngine::down_cast<const SymEngine::Integer&>(*diff).is_zero();
}

// -----------------------------------------------------------------------------
// Loop canonicality
// -----------------------------------------------------------------------------
bool is_canonical(structured_control_flow::StructuredLoop& loop) {
    if (loop.indvar() == SymEngine::null) return false;
    if (loop.init() == SymEngine::null) return false;
    if (loop.update() == SymEngine::null) return false;
    if (loop.condition() == SymEngine::null) return false;
    auto stride = loop.stride();
    if (stride == SymEngine::null) return false;
    // Stride must be non-zero (otherwise no progress and division by zero).
    if (SymEngine::is_a<SymEngine::Integer>(*stride) &&
        SymEngine::down_cast<const SymEngine::Integer&>(*stride).is_zero()) {
        return false;
    }
    return true;
}

// -----------------------------------------------------------------------------
// Affine recurrence decomposition
// -----------------------------------------------------------------------------
//
// We model the loop body's update as
//
//     sym_{n+1} = a * sym_n + b * indvar_n + c
//
// where `a`, `b`, `c` are required to be loop-invariant — i.e. they may not
// depend on `sym`, `indvar`, or any other symbol that is rewritten inside the
// same loop body (the `moving` set).
struct AffineRecurrence {
    bool ok = false;
    symbolic::Expression a;
    symbolic::Expression b;
    symbolic::Expression c;
};

symbolic::Expression coeff_or_zero(const symbolic::AffineCoeffs& coeffs, const symbolic::Symbol& s) {
    auto it = coeffs.find(s);
    if (it == coeffs.end()) return symbolic::zero();
    return it->second;
}

bool depends_on_moving(const symbolic::Expression& expr, const std::unordered_set<std::string>& moving) {
    for (auto& atom : symbolic::atoms(expr)) {
        if (moving.find(atom->get_name()) != moving.end()) {
            return true;
        }
    }
    return false;
}

AffineRecurrence decompose_recurrence(
    const symbolic::Expression& sym_update,
    const symbolic::Symbol& sym,
    const symbolic::Symbol& indvar,
    const std::unordered_set<std::string>& moving
) {
    AffineRecurrence rec;

    // Decompose `sym_update` as an affine combination of `sym` and `indvar`.
    symbolic::SymbolVec gens = {sym, indvar};
    auto poly = symbolic::polynomial(sym_update, gens);
    if (poly == SymEngine::null) return rec;
    auto coeffs = symbolic::affine_coefficients(poly);
    if (coeffs.empty()) return rec;
    auto a = coeff_or_zero(coeffs, sym);
    auto b = coeff_or_zero(coeffs, indvar);
    auto c = coeff_or_zero(coeffs, symbolic::symbol("__daisy_constant__"));

    // Coefficients must not reference any other symbol that is being rewritten
    // in the same loop body; otherwise the closed form would be ill-defined.
    if (depends_on_moving(a, moving)) return rec;
    if (depends_on_moving(b, moving)) return rec;
    if (depends_on_moving(c, moving)) return rec;

    rec.ok = true;
    rec.a = a;
    rec.b = b;
    rec.c = c;
    return rec;
}

// -----------------------------------------------------------------------------
// Closed-form solver
// -----------------------------------------------------------------------------
//
// Returns SymEngine::null if no closed form is available.
//
// Two solvable cases:
//   * Accumulator (a == 1, b == 0): sym_{n+1} = sym_n + c
//       closed(i) = sym_init + c * (i - indvar_init) / stride
//   * Indvar function (a == 0): sym_{n+1} = b*indvar_n + c
//       closed(i) = b * (i - stride) + c, gated by closed(indvar_init) == sym_init
//
// Pure-constant updates (a==0, b==0, sym_update == c) are subsumed by the
// indvar-function case; the verification condition collapses to c == sym_init.
symbolic::Expression solve_closed_form(
    const symbolic::Symbol& indvar,
    const symbolic::Expression& indvar_init,
    const symbolic::Expression& stride,
    const symbolic::Expression& sym_init,
    const AffineRecurrence& rec
) {
    if (sym_equiv(rec.a, symbolic::one()) && sym_equiv(rec.b, symbolic::zero())) {
        // Accumulator case: closed(i) = sym_init + c * iter
        // where iter = (i - indvar_init) / stride.
        //
        // We special-case strides of +/-1 to avoid emitting `idiv(...)`,
        // whose evaluation semantics for negative divisors are not what we
        // want (SymEngine's `idiv(-5, -1) == 0`, not 5).
        symbolic::Expression iter;
        if (sym_equiv(stride, symbolic::one())) {
            iter = symbolic::sub(indvar, indvar_init);
        } else if (sym_equiv(stride, symbolic::integer(-1))) {
            iter = symbolic::sub(indvar_init, indvar);
        } else {
            iter = symbolic::div(symbolic::sub(indvar, indvar_init), stride);
        }
        return symbolic::add(sym_init, symbolic::mul(rec.c, iter));
    }

    if (sym_equiv(rec.a, symbolic::zero())) {
        // Function-of-indvar case: closed(i) = b*(i - stride) + c
        // (At iteration k>=1 we have sym = b*indvar_{k-1} + c = b*(i - stride) + c.)
        auto closed = symbolic::add(symbolic::mul(rec.b, symbolic::sub(indvar, stride)), rec.c);
        // Verification: closed(indvar_init) must equal sym_init
        // so iteration 0 also matches.
        auto closed_at_init = symbolic::subs(closed, indvar, indvar_init);
        if (!sym_equiv(closed_at_init, sym_init)) {
            return SymEngine::null;
        }
        return closed;
    }

    // a != 0 and a != 1: would be a geometric or higher-order recurrence;
    // not handled.
    return SymEngine::null;
}

// -----------------------------------------------------------------------------
// Candidate descriptor + cheap filtering
// -----------------------------------------------------------------------------
struct Candidate {
    std::string name;
    analysis::User* update_use = nullptr;
    structured_control_flow::Transition* update_transition = nullptr;
    symbolic::Expression update_expr;

    analysis::User* init_use = nullptr;
    structured_control_flow::Transition* init_transition = nullptr;
    symbolic::Expression init_expr;
};

// Returns true if all cheap, structural prerequisites are satisfied and fills
// in the descriptor accordingly. Does not perform dominance / use-after checks.
bool collect_candidate_cheap(
    builder::StructuredSDFGBuilder& builder,
    analysis::Users& users,
    analysis::UsersView& body_users,
    structured_control_flow::StructuredLoop& loop,
    structured_control_flow::Transition& exit_transition,
    const std::string& name,
    Candidate& out
) {
    // Type filter: integer scalars only.
    auto& type = builder.subject().type(name);
    if (!dynamic_cast<const types::Scalar*>(&type)) return false;
    if (!types::is_integer(type.primitive_type())) return false;

    auto sym = symbolic::symbol(name);

    // Don't fight a user-set exit assignment.
    if (exit_transition.assignments().find(sym) != exit_transition.assignments().end()) {
        return false;
    }

    // Single write inside the loop body.
    auto body_writes = body_users.writes(name);
    if (body_writes.size() != 1) return false;

    // The write must be a Transition directly inside loop.root() (i.e. always
    // executed each iteration, never inside a conditional branch).
    auto* update_use = body_writes.front();
    auto* element = update_use->element();
    auto* update_transition = dynamic_cast<structured_control_flow::Transition*>(element);
    if (update_transition == nullptr) return false;
    if (&update_transition->parent() != &loop.root()) return false;

    auto upd_it = update_transition->assignments().find(sym);
    if (upd_it == update_transition->assignments().end()) return false;

    // Initial value: there must be exactly one definition reaching the loop
    // header. We approximate this with "exactly two writes globally, one of
    // which is the body write". (Conservative: rejects multi-path inits.)
    auto& all_writes = users.writes(name);
    if (all_writes.size() != 2) return false;

    analysis::User* init_use = nullptr;
    for (auto* w : all_writes) {
        if (w != update_use) {
            init_use = w;
            break;
        }
    }
    if (init_use == nullptr) return false;

    auto* init_transition = dynamic_cast<structured_control_flow::Transition*>(init_use->element());
    if (init_transition == nullptr) return false;

    auto init_it = init_transition->assignments().find(sym);
    if (init_it == init_transition->assignments().end()) return false;

    out.name = name;
    out.update_use = update_use;
    out.update_transition = update_transition;
    out.update_expr = upd_it->second;
    out.init_use = init_use;
    out.init_transition = init_transition;
    out.init_expr = init_it->second;
    return true;
}

// Expensive checks: dominance and "no read of `sym` after the body update".
bool passes_expensive_checks(
    analysis::UsersView& body_users, analysis::DominanceAnalysis& dominance, const Candidate& cand
) {
    if (!dominance.dominates(*cand.init_use, *cand.update_use)) return false;

    auto uses_after = body_users.all_uses_after(*cand.update_use);
    for (auto* use : uses_after) {
        if (use->container() == cand.name) return false;
    }
    return true;
}

// -----------------------------------------------------------------------------
// Apply rewrite
// -----------------------------------------------------------------------------
//
// 1. Insert `sym = closed_form(indvar)` at the top of the loop body.
// 2. Remove `sym` from the original body update transition.
// 3. Insert `sym = sym_update` into the loop's exit transition so that the
//    final post-loop value equals closed_form(post_loop_indvar) (the original
//    sym_update applied to closed_form(last_iter) yields the right value).
void apply_rewrite(
    builder::StructuredSDFGBuilder& builder,
    structured_control_flow::StructuredLoop& loop,
    structured_control_flow::Transition& exit_transition,
    const Candidate& cand,
    const symbolic::Expression& closed_form
) {
    auto sym = symbolic::symbol(cand.name);
    auto& old_first_block = loop.root().at(0).first;
    builder.add_block_before(loop.root(), old_first_block, {{sym, closed_form}}, old_first_block.debug_info());
    cand.update_transition->assignments().erase(sym);
    exit_transition.assignments().insert({sym, cand.update_expr});
}

// -----------------------------------------------------------------------------
// Post-order loop collection
// -----------------------------------------------------------------------------
//
// We walk every Sequence in the SDFG and emit the StructuredLoop children
// after we have descended into them, so innermost loops are processed first.
// The parent Sequence is recorded so we can recover the per-iteration exit
// transition by index (it shifts only when we modify the parent sequence,
// which happens neither for inner-loop rewrites — they touch loop bodies —
// nor for the loop currently being processed).
struct LoopEntry {
    structured_control_flow::StructuredLoop* loop;
    structured_control_flow::Sequence* parent_seq;
};

void collect_loops_postorder(structured_control_flow::ControlFlowNode& node, std::vector<LoopEntry>& out) {
    if (auto seq = dynamic_cast<structured_control_flow::Sequence*>(&node)) {
        for (size_t i = 0; i < seq->size(); ++i) {
            auto& child = seq->at(i).first;
            collect_loops_postorder(child, out);
            if (auto sloop = dynamic_cast<structured_control_flow::StructuredLoop*>(&child)) {
                out.push_back({sloop, seq});
            }
        }
    } else if (auto if_else = dynamic_cast<structured_control_flow::IfElse*>(&node)) {
        for (size_t i = 0; i < if_else->size(); ++i) {
            collect_loops_postorder(if_else->at(i).first, out);
        }
    } else if (auto while_stmt = dynamic_cast<structured_control_flow::While*>(&node)) {
        collect_loops_postorder(while_stmt->root(), out);
    } else if (auto sloop = dynamic_cast<structured_control_flow::StructuredLoop*>(&node)) {
        collect_loops_postorder(sloop->root(), out);
    }
}

} // namespace

// -----------------------------------------------------------------------------
// SymbolEvolution::eliminate_symbols
// -----------------------------------------------------------------------------
//
// One pass over the loop's body. Returns true on the first successful rewrite
// so the caller can refresh analyses and retry.
bool SymbolEvolution::eliminate_symbols(
    builder::StructuredSDFGBuilder& builder,
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::StructuredLoop& loop,
    structured_control_flow::Transition& exit_transition
) {
    if (loop.root().size() == 0) return false;
    if (!is_canonical(loop)) return false;

    auto indvar = loop.indvar();
    auto indvar_init = loop.init();
    auto stride = loop.stride();

    auto& users = analysis_manager.get<analysis::Users>();
    auto& dominance = analysis_manager.get<analysis::DominanceAnalysis>();
    analysis::UsersView body_users(users, loop.root());

    // Build the candidate name set first (one cheap scan).
    std::unordered_set<std::string> moving;
    for (auto* w : body_users.writes()) {
        auto& type = builder.subject().type(w->container());
        if (!dynamic_cast<const types::Scalar*>(&type)) continue;
        if (!types::is_integer(type.primitive_type())) continue;
        moving.insert(w->container());
    }

    for (const auto& name : moving) {
        Candidate cand;
        if (!collect_candidate_cheap(builder, users, body_users, loop, exit_transition, name, cand)) {
            continue;
        }

        // Cheap algebraic check next: solver before expensive analyses.
        auto rec = decompose_recurrence(cand.update_expr, symbolic::symbol(name), indvar, moving);
        if (!rec.ok) continue;

        auto closed = solve_closed_form(indvar, indvar_init, stride, cand.init_expr, rec);
        if (closed == SymEngine::null) continue;

        // Only now pay for dominance + use-after-update.
        if (!passes_expensive_checks(body_users, dominance, cand)) continue;

        apply_rewrite(builder, loop, exit_transition, cand, closed);
        return true;
    }

    return false;
}

SymbolEvolution::SymbolEvolution() : Pass() {}

std::string SymbolEvolution::name() { return "SymbolEvolution"; }

bool SymbolEvolution::run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    bool any_applied = false;

    std::vector<LoopEntry> loops;
    collect_loops_postorder(builder.subject().root(), loops);

    for (auto& entry : loops) {
        // Per-loop fixpoint: each successful rewrite mutates the IR and
        // invalidates cached analyses, so we refetch and retry until quiet.
        while (true) {
            // Recover the exit transition by current index (parent sequence
            // is not modified by inner-loop rewrites, but indices are stable
            // only when we don't touch the parent ourselves).
            auto idx = entry.parent_seq->index(*entry.loop);
            if (idx < 0) break;
            auto& exit_transition = entry.parent_seq->at(static_cast<size_t>(idx)).second;

            bool applied = eliminate_symbols(builder, analysis_manager, *entry.loop, exit_transition);
            if (!applied) break;

            any_applied = true;
            analysis_manager.invalidate_all();
        }
    }

    return any_applied;
}

} // namespace passes
} // namespace sdfg
