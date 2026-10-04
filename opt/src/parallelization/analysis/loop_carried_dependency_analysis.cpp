#include "sdfg/parallelization/analysis/loop_carried_dependency_analysis.h"

#include <algorithm>
#include <cassert>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include <isl/ctx.h>
#include <isl/options.h>
#include <isl/set.h>

#include "sdfg/analysis/analysis.h"
#include "sdfg/analysis/assumptions_analysis.h"
#include "sdfg/analysis/data_dependency_analysis.h"
#include "sdfg/analysis/loop_analysis.h"
#include "sdfg/analysis/memory_layout_analysis.h"
#include "sdfg/analysis/users.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/library_nodes/math/cmath/cmath_node.h"
#include "sdfg/data_flow/memlet.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/if_else.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/structured_sdfg.h"
#include "sdfg/symbolic/maps.h"
#include "sdfg/symbolic/polyhedral.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/scalar.h"

namespace sdfg {
namespace parallelization {

LoopCarriedDependencyAnalysis::LoopCarriedDependencyAnalysis(StructuredSDFG& sdfg) : Analysis(sdfg) {
}

void LoopCarriedDependencyAnalysis::run(analysis::AnalysisManager& analysis_manager) {
    analysis_manager_ = &analysis_manager;
    results_.clear();
    nest_ddas_.clear();
    detailed_assumptions_.reset();
}

analysis::AssumptionsAnalysis& LoopCarriedDependencyAnalysis::detailed_assumptions() {
    if (!detailed_assumptions_) {
        detailed_assumptions_ =
            std::make_unique<analysis::AssumptionsAnalysis>(this->sdfg_, /*with_branch_conditions=*/true);
        detailed_assumptions_->run(*analysis_manager_);
    }
    return *detailed_assumptions_;
}

analysis::DataDependencyAnalysis& LoopCarriedDependencyAnalysis::nest_dda(structured_control_flow::StructuredLoop& loop) {
    // Boundary snapshots of a loop do not depend on the code around it, so one run over the outermost
    // loop of the nest serves every loop inside it.
    structured_control_flow::StructuredLoop* outermost = &loop;
    for (auto* cur = loop.get_parent(); cur != nullptr; cur = cur->get_parent()) {
        if (auto* ancestor = dynamic_cast<structured_control_flow::StructuredLoop*>(cur)) {
            outermost = ancestor;
        }
    }
    auto& dda = nest_ddas_[outermost];
    if (!dda) {
        passes::CompileStatistics::enter_analysis_if_enabled("DetailedDDA");
        dda = std::make_unique<analysis::DataDependencyAnalysis>(this->sdfg_, *outermost);
        dda->set_detailed(true);
        dda->run(*analysis_manager_);
        passes::CompileStatistics::exit_analysis_if_enabled();
    }
    return *dda;
}

const LoopCarriedDependencyAnalysis::LoopResult& LoopCarriedDependencyAnalysis::
    result(structured_control_flow::StructuredLoop& loop) {
    assert(analysis_manager_ != nullptr && "LoopCarriedDependencyAnalysis: run() must be called before queries");
    auto it = results_.find(&loop);
    if (it != results_.end()) {
        return it->second;
    }
    auto& entry = results_[&loop];
    compute(loop, entry);
    return entry;
}

namespace {

bool is_undefined_user(analysis::User& user) {
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
std::vector<data_flow::Subset> collect_subsets(analysis::User& user, analysis::MemoryLayoutAnalysis& mla) {
    std::vector<data_flow::Subset> result;
    auto* access_node = dynamic_cast<data_flow::AccessNode*>(user.element());
    if (access_node == nullptr) {
        for (auto& s : user.subsets()) {
            result.push_back(s);
        }
        return result;
    }
    auto& graph = access_node->get_parent();
    if (user.use() == analysis::Use::READ || user.use() == analysis::Use::VIEW) {
        for (auto& edge : graph.out_edges(*access_node)) {
            if (auto* acc = mla.access(edge)) {
                result.push_back(acc->subset);
            } else {
                result.push_back(edge.subset());
            }
        }
    } else if (user.use() == analysis::Use::WRITE || user.use() == analysis::Use::MOVE) {
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
// Assumptions of a user scope with the loop's (and nested loops') indvars
// marked non-constant, plus the bounds bundle derived from them. Shared by all
// pairs of one loop so BoundAnalysis/delinearize memoization amortizes.
struct ScopeBounds {
    explicit ScopeBounds(symbolic::Assumptions a) : assums(std::move(a)), bounds(assums) {
    }
    symbolic::Assumptions assums;
    symbolic::AssumptionsBounds bounds;
};

using ScopeBoundsCache =
    std::unordered_map<const structured_control_flow::ControlFlowNode*, std::unique_ptr<ScopeBounds>>;

symbolic::AssumptionsBounds& scope_bounds(
    ScopeBoundsCache& cache,
    structured_control_flow::ControlFlowNode& scope,
    analysis::AnalysisManager& analysis_manager,
    analysis::AssumptionsAnalysis& assumptions_analysis,
    structured_control_flow::StructuredLoop& loop
) {
    auto it = cache.find(&scope);
    if (it != cache.end()) {
        return it->second->bounds;
    }
    auto assumptions = assumptions_analysis.get(scope, true);

    // Mark loop's indvar and all nested loop indvars as non-constant from this
    // loop's perspective.
    auto& loop_analysis = analysis_manager.get<analysis::LoopAnalysis>();
    if (assumptions.find(loop.indvar()) != assumptions.end()) {
        assumptions.at(loop.indvar()).constant(false);
    }
    for (auto& inner_loop : loop_analysis.descendants(&loop)) {
        if (auto structured_loop = dynamic_cast<const structured_control_flow::StructuredLoop*>(inner_loop)) {
            auto indvar = structured_loop->indvar();
            if (assumptions.find(indvar) != assumptions.end()) {
                assumptions.at(indvar).constant(false);
            }
        }
    }
    auto& entry = cache[&scope];
    entry = std::make_unique<ScopeBounds>(std::move(assumptions));
    return entry->bounds;
}

symbolic::maps::DependenceDeltas pair_deltas(
    StructuredSDFG& sdfg,
    analysis::User& previous,
    analysis::User& current,
    analysis::AnalysisManager& analysis_manager,
    analysis::AssumptionsAnalysis& assumptions_analysis,
    structured_control_flow::StructuredLoop& loop,
    ScopeBoundsCache& bounds_cache
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

    auto& previous_bounds =
        scope_bounds(bounds_cache, *analysis::Users::scope(&previous), analysis_manager, assumptions_analysis, loop);
    auto& current_bounds =
        scope_bounds(bounds_cache, *analysis::Users::scope(&current), analysis_manager, assumptions_analysis, loop);

    // Collect deltas across all subset pairs and union them.
    isl_ctx* union_ctx = nullptr;
    isl_set* accumulated = nullptr;
    std::vector<std::string> result_dimensions;

    for (auto& previous_subset : previous_subsets) {
        for (auto& current_subset : current_subsets) {
            auto deltas = symbolic::maps::
                dependence_deltas(previous_subset, current_subset, loop.indvar(), previous_bounds, current_bounds);
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
    if (add.empty) {
        return;
    }
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
        if (u) {
            isl_set_free(u);
        }
    } else {
        if (s1) {
            isl_set_free(s1);
        }
        if (s2) {
            isl_set_free(s2);
        }
        info.deltas.deltas_str = "";
        info.deltas.dimensions.clear();
    }
    isl_ctx_free(ctx);
}

bool user_in_subtree(analysis::User& user, const structured_control_flow::ControlFlowNode& subtree) {
    if (user.element() == nullptr) {
        return false;
    }
    auto* scope = analysis::Users::scope(&user);
    while (scope != nullptr) {
        if (scope == &subtree) {
            return true;
        }
        scope = scope->get_parent();
    }
    return false;
}

// Map an associative/commutative combine node (scalar tasklet or CMath library
// node) to its reduction operator. Returns nullopt for any node that is not a
// recognized reorderable reduction operator.
std::optional<structured_control_flow::ReductionOperation> combine_operator(const data_flow::DataFlowNode& node) {
    using structured_control_flow::ReductionOperation;
    if (auto* tasklet = dynamic_cast<const data_flow::Tasklet*>(&node)) {
        switch (tasklet->code()) {
            case data_flow::TaskletCode::fp_add:
            case data_flow::TaskletCode::int_add:
                return ReductionOperation::Add;
            case data_flow::TaskletCode::fp_fma:
                // Fused multiply-add `_out = _in0 * _in1 + _in2` is an additive
                // reduction over its addend operand.
                return ReductionOperation::Add;
            case data_flow::TaskletCode::fp_mul:
            case data_flow::TaskletCode::int_mul:
                return ReductionOperation::Mul;
            case data_flow::TaskletCode::int_smin:
            case data_flow::TaskletCode::int_umin:
                return ReductionOperation::Min;
            case data_flow::TaskletCode::int_smax:
            case data_flow::TaskletCode::int_umax:
                return ReductionOperation::Max;
            default:
                return std::nullopt;
        }
    }
    if (auto* cmath = dynamic_cast<const math::cmath::CMathNode*>(&node)) {
        switch (cmath->function()) {
            case math::cmath::CMathFunction::fmax:
                return ReductionOperation::Max;
            case math::cmath::CMathFunction::fmin:
                return ReductionOperation::Min;
            case math::cmath::CMathFunction::fma:
                // `fma(x, y, z) = x * y + z`: additive reduction over the addend
                // operand `z`
                return ReductionOperation::Add;
            default:
                return std::nullopt;
        }
    }
    return std::nullopt;
}

std::optional<std::string> fma_addend_connector(const data_flow::DataFlowNode& node) {
    if (auto* tasklet = dynamic_cast<const data_flow::Tasklet*>(&node)) {
        if (tasklet->code() == data_flow::TaskletCode::fp_fma) {
            return tasklet->inputs().at(2);
        }
    }
    if (auto* cmath = dynamic_cast<const math::cmath::CMathNode*>(&node)) {
        if (cmath->function() == math::cmath::CMathFunction::fma) {
            return cmath->inputs().at(2);
        }
    }
    return std::nullopt;
}

// Collect the dataflow Blocks directly belonging to `node`'s subtree, descending
// through Sequences and IfElse branches but NOT into nested loops (those have
// their own loop-carried analysis). Used to scan a loop body for the combine.
void collect_body_blocks(
    const structured_control_flow::ControlFlowNode& node, std::vector<const structured_control_flow::Block*>& out
) {
    if (auto* block = dynamic_cast<const structured_control_flow::Block*>(&node)) {
        out.push_back(block);
    } else if (auto* seq = dynamic_cast<const structured_control_flow::Sequence*>(&node)) {
        for (size_t i = 0; i < seq->size(); i++) {
            collect_body_blocks(seq->at(i), out);
        }
    } else if (auto* ifelse = dynamic_cast<const structured_control_flow::IfElse*>(&node)) {
        for (size_t i = 0; i < ifelse->size(); i++) {
            collect_body_blocks(ifelse->at(i).first, out);
        }
    }
    // Stop at nested StructuredLoop / While: their bodies belong to other loops.
}

// True if the accumulator subset is invariant across the reduction loop: the
// induction variable does not appear in any index expression. This is a sound
// (conservative) guard — an accumulator whose address moves with the induction
// variable is not a single-location reduction. The actual *equality* of the
// written and read-back addresses is proven separately and precisely by
// `symbolic::polyhedral::equal_on_domain`.
bool address_invariant_in_indvar(const data_flow::Subset& subset, const symbolic::Symbol& indvar) {
    for (auto& dim : subset) {
        for (auto& atom : symbolic::atoms(dim)) {
            if (atom->get_name() == indvar->get_name()) {
                return false;
            }
        }
    }
    return true;
}

} // namespace

void LoopCarriedDependencyAnalysis::compute(structured_control_flow::StructuredLoop& loop, LoopResult& result) {
    // Non-monotonic / unanalyzable loops stay unavailable; consumers must treat
    // that as "no info" and fall back to safe defaults.
    if (!loop.is_monotonic()) {
        return;
    }
    auto& dda = nest_dda(loop);
    if (!dda.has_loop_boundary(loop)) {
        return;
    }
    result.available = true;

    auto& analysis_manager = *analysis_manager_;
    auto& assumptions_analysis = detailed_assumptions();
    auto& ue_reads = dda.upward_exposed_reads(loop);
    auto& esc_defs = dda.escaping_definitions(loop);
    auto& deps = result.dependencies;
    auto& pair_list = result.pairs;
    ScopeBoundsCache bounds_cache;

    // RAW: escaping_writes × upward_exposed_reads
    for (auto& write_entry : esc_defs) {
        auto* write = write_entry.first;
        for (auto* read : ue_reads) {
            if (write->container() != read->container()) {
                continue;
            }
            auto deltas =
                pair_deltas(this->sdfg_, *write, *read, analysis_manager, assumptions_analysis, loop, bounds_cache);
            if (deltas.empty) {
                continue;
            }
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
            auto deltas =
                pair_deltas(this->sdfg_, *w1, *w2, analysis_manager, assumptions_analysis, loop, bounds_cache);
            if (deltas.empty) {
                continue;
            }
            pair_list.push_back(LoopCarriedDependencyPair{w1, w2, LOOP_CARRIED_DEPENDENCY_WRITE_WRITE, deltas});
            if (deps.find(w1->container()) == deps.end()) {
                deps[w1->container()] = LoopCarriedDependencyInfo{LOOP_CARRIED_DEPENDENCY_WRITE_WRITE, deltas};
            }
        }
    }

    detect_reductions(loop, result);
}

void LoopCarriedDependencyAnalysis::detect_reductions(structured_control_flow::StructuredLoop& loop, LoopResult& result) {
    auto& deps = result.dependencies;
    if (deps.empty()) {
        return;
    }

    std::vector<const structured_control_flow::Block*> blocks;
    collect_body_blocks(loop.root(), blocks);

    auto indvar = loop.indvar();

    // Assumptions for the loop body, used by the polyhedral equality test. They
    // must be taken from the loop body (`loop.root()`), not the loop node, so
    // that the induction variable's own bounds (e.g. 0 <= i < N) are present in
    // the domain. The induction variable must additionally be treated as
    // evolving (a domain dimension) rather than a constant parameter so that
    // shifted accesses such as A[i] vs A[i-1] are correctly rejected.
    symbolic::Assumptions assums = detailed_assumptions().get(loop.root(), true);
    auto indvar_it = assums.find(indvar);
    if (indvar_it != assums.end()) {
        indvar_it->second.constant(false);
    }

    // Candidate accumulator -> operator. A container is rejected (and stays out
    // of the result) if any of its writes in the body is not a clean combine.
    std::map<std::string, structured_control_flow::ReductionOperation> found;
    std::set<std::string> rejected;

    for (auto* block : blocks) {
        auto& graph = block->dataflow();
        for (auto& node : graph.nodes()) {
            auto* write_node = dynamic_cast<const data_flow::AccessNode*>(&node);
            if (write_node == nullptr) {
                continue;
            }

            std::vector<const data_flow::Memlet*> in_edges;
            for (auto& edge : graph.in_edges(*write_node)) {
                in_edges.push_back(&edge);
            }
            if (in_edges.empty()) {
                continue; // not written in this block
            }

            const std::string& container = write_node->data();

            // Only loop-carried read-write containers can be reductions.
            auto ci = deps.find(container);
            if (ci == deps.end() || ci->second.type != LOOP_CARRIED_DEPENDENCY_READ_WRITE) {
                continue;
            }

            // A reduction accumulator is produced by exactly one combine node.
            if (in_edges.size() != 1) {
                rejected.insert(container);
                continue;
            }
            auto& write_edge = *in_edges.front();
            auto& combine = write_edge.src();

            auto op = combine_operator(combine);
            if (!op.has_value()) {
                rejected.insert(container);
                continue;
            }

            // The combine must read the same accumulator back (acc = acc OP x).
            // For a fused multiply-add the accumulator must be the addend
            // operand; if it feeds a multiplicand the update is acc = acc * b + c,
            // which is not a reorderable reduction and is rejected.
            auto fma_addend = fma_addend_connector(combine);
            const data_flow::Memlet* read_edge = nullptr;
            bool acc_on_non_addend = false;
            for (auto& edge : graph.in_edges(combine)) {
                auto* read_node = dynamic_cast<const data_flow::AccessNode*>(&edge.src());
                if (read_node == nullptr || read_node->data() != container) {
                    continue;
                }
                if (fma_addend.has_value() && edge.dst_conn() != fma_addend.value()) {
                    // Accumulator feeds a multiplicand of the fma.
                    acc_on_non_addend = true;
                    continue;
                }
                read_edge = &edge;
                if (!fma_addend.has_value()) {
                    break;
                }
            }
            if (read_edge == nullptr || acc_on_non_addend) {
                rejected.insert(container);
                continue;
            }

            // The accumulator must address one fixed, loop-invariant location
            // and the written cell must equal the read-back cell. `equal_on_domain`
            // proves the latter precisely (parametric/linearized affine forms),
            // while the invariance guard rejects accumulators that move with the
            // induction variable.
            if (!address_invariant_in_indvar(write_edge.subset(), indvar) ||
                !symbolic::polyhedral::equal_on_domain(write_edge.subset(), read_edge->subset(), indvar, assums)) {
                rejected.insert(container);
                continue;
            }

            auto existing = found.find(container);
            if (existing != found.end() && existing->second != op.value()) {
                rejected.insert(container);
                continue;
            }
            found[container] = op.value();
        }
    }

    for (auto& entry : found) {
        if (rejected.count(entry.first) != 0) {
            continue;
        }
        result.reductions.push_back(structured_control_flow::ReductionInfo{entry.second, entry.first});
    }
}

bool LoopCarriedDependencyAnalysis::available(structured_control_flow::StructuredLoop& loop) {
    return result(loop).available;
}

const std::unordered_map<std::string, LoopCarriedDependencyInfo>& LoopCarriedDependencyAnalysis::
    dependencies(structured_control_flow::StructuredLoop& loop) {
    auto& r = result(loop);
    assert(r.available && "LoopCarriedDependencyAnalysis: loop not analyzable");
    return r.dependencies;
}

const std::vector<LoopCarriedDependencyPair>& LoopCarriedDependencyAnalysis::
    pairs(structured_control_flow::StructuredLoop& loop) {
    auto& r = result(loop);
    assert(r.available && "LoopCarriedDependencyAnalysis: loop not analyzable");
    return r.pairs;
}

std::vector<const LoopCarriedDependencyPair*> LoopCarriedDependencyAnalysis::pairs_between(
    structured_control_flow::StructuredLoop& loop,
    const structured_control_flow::ControlFlowNode& subtree_a,
    const structured_control_flow::ControlFlowNode& subtree_b
) {
    std::vector<const LoopCarriedDependencyPair*> between;
    for (auto& pair : result(loop).pairs) {
        bool wa = user_in_subtree(*pair.writer, subtree_a);
        bool wb = user_in_subtree(*pair.writer, subtree_b);
        bool ra = user_in_subtree(*pair.reader, subtree_a);
        bool rb = user_in_subtree(*pair.reader, subtree_b);

        if ((wa && rb) || (wb && ra)) {
            between.push_back(&pair);
        }
    }
    return between;
}

bool LoopCarriedDependencyAnalysis::has_loop_carried(structured_control_flow::StructuredLoop& loop) {
    return !result(loop).pairs.empty();
}

bool LoopCarriedDependencyAnalysis::has_loop_carried_raw(structured_control_flow::StructuredLoop& loop) {
    for (auto& p : result(loop).pairs) {
        if (p.type == LOOP_CARRIED_DEPENDENCY_READ_WRITE) {
            return true;
        }
    }
    return false;
}

bool LoopCarriedDependencyAnalysis::has_loop_carried_hazard(structured_control_flow::StructuredLoop& loop) {
    for (auto& p : result(loop).pairs) {
        if (p.type != LOOP_CARRIED_DEPENDENCY_WRITE_WRITE) {
            return true;
        }
    }
    return false;
}

const std::vector<structured_control_flow::ReductionInfo>& LoopCarriedDependencyAnalysis::
    reductions(structured_control_flow::StructuredLoop& loop) {
    auto& r = result(loop);
    assert(r.available && "LoopCarriedDependencyAnalysis: loop not analyzable");
    return r.reductions;
}

bool LoopCarriedDependencyAnalysis::has_reductions(structured_control_flow::StructuredLoop& loop) {
    return !result(loop).reductions.empty();
}

bool LoopCarriedDependencyAnalysis::is_reduction_only(structured_control_flow::StructuredLoop& loop) {
    auto& r = result(loop);
    if (r.reductions.empty()) {
        return false;
    }
    std::set<std::string> reduction_containers;
    for (auto& reduction : r.reductions) {
        reduction_containers.insert(reduction.container);
    }
    for (auto& pair : r.pairs) {
        if (pair.type == LOOP_CARRIED_DEPENDENCY_WRITE_WRITE) {
            continue;
        }
        if (reduction_containers.count(pair.writer->container()) == 0) {
            return false;
        }
    }
    return true;
}

} // namespace parallelization
} // namespace sdfg
