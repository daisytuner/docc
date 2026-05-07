#include "sdfg/analysis/loop_carried_dependency_analysis.h"

#include <cassert>
#include <string>
#include <vector>

#include <isl/ctx.h>
#include <isl/options.h>
#include <isl/set.h>

#include "sdfg/analysis/analysis.h"
#include "sdfg/analysis/assumptions_analysis.h"
#include "sdfg/analysis/data_dependency_analysis.h"
#include "sdfg/analysis/loop_analysis.h"
#include "sdfg/analysis/memory_layout_analysis.h"
#include "sdfg/analysis/scope_analysis.h"
#include "sdfg/analysis/users.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/memlet.h"
#include "sdfg/structured_sdfg.h"
#include "sdfg/symbolic/maps.h"
#include "sdfg/types/scalar.h"

namespace sdfg {
namespace analysis {

LoopCarriedDependencyAnalysis::LoopCarriedDependencyAnalysis(StructuredSDFG& sdfg)
    : Analysis(sdfg), node_(sdfg.root()) {}

LoopCarriedDependencyAnalysis::LoopCarriedDependencyAnalysis(StructuredSDFG& sdfg, structured_control_flow::Sequence& node)
    : Analysis(sdfg), node_(node) {}

DataDependencyAnalysis& LoopCarriedDependencyAnalysis::detailed_dda() {
    if (!detailed_dda_) {
        detailed_dda_ = std::make_unique<DataDependencyAnalysis>(this->sdfg_, this->node_);
        detailed_dda_->set_detailed(true);
    }
    return *detailed_dda_;
}

void LoopCarriedDependencyAnalysis::analyze_loop(
    analysis::AnalysisManager& /*analysis_manager*/, structured_control_flow::StructuredLoop& /*loop*/
) {
    // Per-loop work is done inline in `run()`.
}

namespace {

bool is_undefined_user(User& user) {
    // Mirror DDA::is_undefined_user — undefined users have a null vertex.
    return user.element() == nullptr;
}

// Collect a user's subsets, preferring delinearized (multi-dim) subsets from
// MemoryLayoutAnalysis when available. ISL cannot reason precisely about
// linearized expressions like `i*N + j`; delinearization recovers `[i, j]` so
// `dependence_deltas` can compute exact distance vectors.
//
// The returned vector is index-aligned with the user's underlying memlets:
// for each memlet, either MLA's delinearized subset (when MLA succeeded) or
// the memlet's original subset (linearized fallback).
std::vector<data_flow::Subset> collect_subsets(User& user, MemoryLayoutAnalysis& mla) {
    std::vector<data_flow::Subset> result;
    auto* access_node = dynamic_cast<data_flow::AccessNode*>(user.element());
    if (access_node == nullptr) {
        for (auto& s : user.subsets()) result.push_back(s);
        return result;
    }
    auto& graph = access_node->get_parent();
    if (user.use() == Use::READ || user.use() == Use::VIEW) {
        for (auto& edge : graph.out_edges(*access_node)) {
            if (auto* acc = mla.access(edge)) {
                result.push_back(acc->subset);
            } else {
                result.push_back(edge.subset());
            }
        }
    } else if (user.use() == Use::WRITE || user.use() == Use::MOVE) {
        for (auto& edge : graph.in_edges(*access_node)) {
            if (auto* acc = mla.access(edge)) {
                result.push_back(acc->subset);
            } else {
                result.push_back(edge.subset());
            }
        }
    }
    return result;
}

// Compute the union of iteration-distance delta sets between two users wrt
// `loop`'s indvar. Inner-loop indvars are marked non-constant so that the ISL
// formulation existentially quantifies them per-side (giving "different inner
// iteration" the freedom it needs).
//
// Returns {empty=true} when no loop-carried dependence exists between the two
// users. Returns {empty=false, deltas_str=""} for the scalar shortcut and for
// undefined users (dependence exists, distance unrepresentable).
symbolic::maps::DependenceDeltas pair_deltas(
    StructuredSDFG& sdfg,
    User& previous,
    User& current,
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::StructuredLoop& loop
) {
    symbolic::maps::DependenceDeltas empty_result{true, "", {}};

    if (previous.container() != current.container()) {
        return empty_result;
    }

    // Scalar shortcut: any pair of accesses to a scalar is loop-carried but the
    // distance vector is unrepresentable.
    auto& type = sdfg.type(previous.container());
    if (dynamic_cast<const types::Scalar*>(&type)) {
        return symbolic::maps::DependenceDeltas{false, "", {}};
    }

    if (is_undefined_user(previous) || is_undefined_user(current)) {
        return empty_result;
    }

    // Use MLA-delinearized subsets when available so ISL can reason precisely
    // about multi-dimensional pointer accesses.
    auto& mla = analysis_manager.get<analysis::MemoryLayoutAnalysis>();
    auto previous_subsets = collect_subsets(previous, mla);
    auto current_subsets = collect_subsets(current, mla);

    auto& assumptions_analysis = analysis_manager.get<analysis::AssumptionsAnalysis>();
    auto previous_scope = Users::scope(&previous);
    auto previous_assumptions = assumptions_analysis.get(*previous_scope, true);
    auto current_scope = Users::scope(&current);
    auto current_assumptions = assumptions_analysis.get(*current_scope, true);

    // Mark loop's indvar and all nested loop indvars as non-constant from this
    // loop's perspective.
    auto& loop_analysis = analysis_manager.get<analysis::LoopAnalysis>();
    if (previous_assumptions.find(loop.indvar()) != previous_assumptions.end()) {
        previous_assumptions.at(loop.indvar()).constant(false);
    }
    if (current_assumptions.find(loop.indvar()) != current_assumptions.end()) {
        current_assumptions.at(loop.indvar()).constant(false);
    }
    for (auto& inner_loop : loop_analysis.descendants(&loop)) {
        if (auto structured_loop = dynamic_cast<const structured_control_flow::StructuredLoop*>(inner_loop)) {
            auto indvar = structured_loop->indvar();
            if (previous_assumptions.find(indvar) != previous_assumptions.end()) {
                previous_assumptions.at(indvar).constant(false);
            }
            if (current_assumptions.find(indvar) != current_assumptions.end()) {
                current_assumptions.at(indvar).constant(false);
            }
        }
    }

    // Collect deltas across all subset pairs and union them.
    isl_ctx* union_ctx = nullptr;
    isl_set* accumulated = nullptr;
    std::vector<std::string> result_dimensions;

    for (auto& previous_subset : previous_subsets) {
        for (auto& current_subset : current_subsets) {
            auto deltas = symbolic::maps::dependence_deltas(
                previous_subset, current_subset, loop.indvar(), previous_assumptions, current_assumptions
            );
            if (deltas.empty) {
                continue;
            }
            if (deltas.deltas_str.empty()) {
                if (accumulated) {
                    isl_set_free(accumulated);
                    isl_ctx_free(union_ctx);
                }
                return symbolic::maps::DependenceDeltas{false, "", {}};
            }
            if (!union_ctx) {
                union_ctx = isl_ctx_alloc();
                isl_options_set_on_error(union_ctx, ISL_ON_ERROR_CONTINUE);
                accumulated = isl_set_read_from_str(union_ctx, deltas.deltas_str.c_str());
                result_dimensions = deltas.dimensions;
            } else {
                isl_set* new_set = isl_set_read_from_str(union_ctx, deltas.deltas_str.c_str());
                if (new_set && accumulated) {
                    accumulated = isl_set_union(accumulated, new_set);
                } else if (new_set) {
                    isl_set_free(new_set);
                }
            }
        }
    }

    if (!accumulated) {
        if (union_ctx) {
            isl_ctx_free(union_ctx);
        }
        return empty_result;
    }

    char* str = isl_set_to_str(accumulated);
    if (!str) {
        isl_set_free(accumulated);
        isl_ctx_free(union_ctx);
        return symbolic::maps::DependenceDeltas{false, "", {}};
    }
    std::string union_str(str);
    free(str);

    isl_set_free(accumulated);
    isl_ctx_free(union_ctx);

    return symbolic::maps::DependenceDeltas{false, union_str, result_dimensions};
}

void merge_deltas(LoopCarriedDependencyInfo& info, const symbolic::maps::DependenceDeltas& add) {
    if (add.empty) return;
    if (info.deltas.empty) {
        info.deltas = add;
        return;
    }
    if (add.deltas_str.empty() || info.deltas.deltas_str.empty()) {
        info.deltas.empty = false;
        return;
    }
    isl_ctx* ctx = isl_ctx_alloc();
    isl_options_set_on_error(ctx, ISL_ON_ERROR_CONTINUE);
    isl_set* s1 = isl_set_read_from_str(ctx, info.deltas.deltas_str.c_str());
    isl_set* s2 = isl_set_read_from_str(ctx, add.deltas_str.c_str());
    if (s1 && s2) {
        isl_set* u = isl_set_union(s1, s2);
        char* str = u ? isl_set_to_str(u) : nullptr;
        if (str) {
            info.deltas.deltas_str = std::string(str);
            free(str);
        } else {
            info.deltas.deltas_str = "";
            info.deltas.dimensions.clear();
        }
        if (u) isl_set_free(u);
    } else {
        if (s1) isl_set_free(s1);
        if (s2) isl_set_free(s2);
        info.deltas.deltas_str = "";
        info.deltas.dimensions.clear();
    }
    isl_ctx_free(ctx);
}

bool user_in_subtree(
    User& user, const structured_control_flow::ControlFlowNode& subtree, analysis::ScopeAnalysis& scope_analysis
) {
    auto* scope = Users::scope(&user);
    while (scope != nullptr) {
        if (scope == &subtree) {
            return true;
        }
        scope = scope_analysis.parent_scope(scope);
    }
    return false;
}

} // namespace

void LoopCarriedDependencyAnalysis::run(analysis::AnalysisManager& analysis_manager) {
    dependencies_.clear();
    pairs_.clear();

    // Drive entirely from DDA's reaching-definitions scaffold:
    //   - DDA computes per-loop boundary snapshots (upward-exposed reads,
    //     escaping definitions) — its primary job.
    //   - LCDA enumerates the cross-iteration pair space and computes delta
    //     sets via `pair_deltas` (using `symbolic::maps::dependence_deltas`).
    //
    // For a structured loop L with indvar i_L:
    //   pairs(L) = { (W,R, RAW, Δ_L(W,R)) : W ∈ esc(L), R ∈ ue(L),
    //                                       cont(W) = cont(R), Δ ≠ ∅ }
    //            ∪ { (W₁,W₂, WAW, Δ_L(W₁,W₂)) : W₁,W₂ ∈ esc(L),
    //                                           cont(W₁) = cont(W₂), Δ ≠ ∅ }
    auto& dda = detailed_dda();
    dda.run(analysis_manager);
    auto& loop_analysis = analysis_manager.get<analysis::LoopAnalysis>();
    auto& scope_analysis = analysis_manager.get<analysis::ScopeAnalysis>();

    for (auto* loop_node : loop_analysis.loops()) {
        auto* loop = dynamic_cast<structured_control_flow::StructuredLoop*>(loop_node);
        if (loop == nullptr) {
            continue;
        }

        // Restrict to loops within the analysis scope (`node_`).
        bool in_scope = false;
        structured_control_flow::ControlFlowNode* cur = loop;
        while (cur != nullptr) {
            if (cur == &node_) {
                in_scope = true;
                break;
            }
            cur = scope_analysis.parent_scope(cur);
        }
        if (!in_scope) {
            continue;
        }

        // Non-monotonic / unanalyzable loops: don't register; consumers must
        // treat absence as "no info" and fall back to safe defaults.
        if (!loop->is_monotonic()) {
            continue;
        }
        if (!dda.has_loop_boundary(*loop)) {
            continue;
        }

        auto& ue_reads = dda.upward_exposed_reads(*loop);
        auto& esc_defs = dda.escaping_definitions(*loop);

        auto& deps = dependencies_[loop];
        auto& pair_list = pairs_[loop];

        // RAW: escaping_writes × upward_exposed_reads
        for (auto& write_entry : esc_defs) {
            auto* write = write_entry.first;
            for (auto* read : ue_reads) {
                if (write->container() != read->container()) {
                    continue;
                }
                auto deltas = pair_deltas(this->sdfg_, *write, *read, analysis_manager, *loop);
                if (deltas.empty) continue;
                pair_list.push_back(LoopCarriedDependencyPair{write, read, LOOP_CARRIED_DEPENDENCY_READ_WRITE, deltas});
                auto it = deps.find(read->container());
                if (it == deps.end()) {
                    deps[read->container()] = LoopCarriedDependencyInfo{LOOP_CARRIED_DEPENDENCY_READ_WRITE, deltas};
                } else {
                    it->second.type = LOOP_CARRIED_DEPENDENCY_READ_WRITE;
                    merge_deltas(it->second, deltas);
                }
            }
        }

        // WAW: escaping_writes × escaping_writes (ordered pairs incl. self)
        for (auto& w1_entry : esc_defs) {
            auto* w1 = w1_entry.first;
            for (auto& w2_entry : esc_defs) {
                auto* w2 = w2_entry.first;
                if (w1->container() != w2->container()) {
                    continue;
                }
                auto deltas = pair_deltas(this->sdfg_, *w1, *w2, analysis_manager, *loop);
                if (deltas.empty) continue;
                pair_list.push_back(LoopCarriedDependencyPair{w1, w2, LOOP_CARRIED_DEPENDENCY_WRITE_WRITE, deltas});
                if (deps.find(w1->container()) == deps.end()) {
                    deps[w1->container()] = LoopCarriedDependencyInfo{LOOP_CARRIED_DEPENDENCY_WRITE_WRITE, deltas};
                }
            }
        }
    }
}

bool LoopCarriedDependencyAnalysis::available(structured_control_flow::StructuredLoop& loop) const {
    return pairs_.find(&loop) != pairs_.end();
}

const std::unordered_map<std::string, LoopCarriedDependencyInfo>& LoopCarriedDependencyAnalysis::
    dependencies(structured_control_flow::StructuredLoop& loop) const {
    auto it = dependencies_.find(&loop);
    assert(it != dependencies_.end() && "LoopCarriedDependencyAnalysis: loop not analyzed");
    return it->second;
}

const std::vector<LoopCarriedDependencyPair>& LoopCarriedDependencyAnalysis::pairs(structured_control_flow::StructuredLoop&
                                                                                       loop) const {
    auto it = pairs_.find(&loop);
    assert(it != pairs_.end() && "LoopCarriedDependencyAnalysis: loop not analyzed");
    return it->second;
}

std::vector<const LoopCarriedDependencyPair*> LoopCarriedDependencyAnalysis::pairs_between(
    structured_control_flow::StructuredLoop& loop,
    const structured_control_flow::ControlFlowNode& subtree_a,
    const structured_control_flow::ControlFlowNode& subtree_b,
    analysis::AnalysisManager& analysis_manager
) const {
    std::vector<const LoopCarriedDependencyPair*> result;
    auto it = pairs_.find(&loop);
    if (it == pairs_.end()) {
        return result;
    }
    auto& scope_analysis = analysis_manager.get<analysis::ScopeAnalysis>();

    for (auto& pair : it->second) {
        bool wa = user_in_subtree(*pair.writer, subtree_a, scope_analysis);
        bool wb = user_in_subtree(*pair.writer, subtree_b, scope_analysis);
        bool ra = user_in_subtree(*pair.reader, subtree_a, scope_analysis);
        bool rb = user_in_subtree(*pair.reader, subtree_b, scope_analysis);

        if ((wa && rb) || (wb && ra)) {
            result.push_back(&pair);
        }
    }
    return result;
}

bool LoopCarriedDependencyAnalysis::has_loop_carried(structured_control_flow::StructuredLoop& loop) const {
    auto it = pairs_.find(&loop);
    if (it == pairs_.end()) return false;
    return !it->second.empty();
}

bool LoopCarriedDependencyAnalysis::has_loop_carried_raw(structured_control_flow::StructuredLoop& loop) const {
    auto it = pairs_.find(&loop);
    if (it == pairs_.end()) return false;
    for (auto& p : it->second) {
        if (p.type == LOOP_CARRIED_DEPENDENCY_READ_WRITE) return true;
    }
    return false;
}

} // namespace analysis
} // namespace sdfg
