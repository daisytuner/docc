#include "sdfg/passes/symbolic/type_minimization.h"

#include <unordered_set>

#include "sdfg/analysis/analysis.h"
#include "sdfg/analysis/assumptions_analysis.h"
#include "sdfg/symbolic/extreme_values.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {
namespace passes {

TypeMinimization::TypeMinimization(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager)
    : visitor::NonStoppingStructuredSDFGVisitor(builder, analysis_manager) {};

bool TypeMinimization::is_safe_trunc(symbolic::Expression expr, const symbolic::Assumptions& assumptions) {
    size_t output_bitwidth = 32;
    int64_t output_min_value_signed = 0;
    int64_t output_max_value_signed = (1ULL << (output_bitwidth - 1)) - 1;

    auto mini = symbolic::minimum(expr, {}, assumptions, true);
    if (mini.is_null()) {
        return false;
    }
    auto lb_criterion = symbolic::Ge(mini, symbolic::integer(output_min_value_signed));
    if (!symbolic::is_true(lb_criterion)) {
        return false;
    }

    auto maxi = symbolic::maximum(expr, {}, assumptions, false);
    if (maxi.is_null()) {
        return false;
    }
    auto ub_criterion = symbolic::Le(maxi, symbolic::integer(output_max_value_signed));
    if (!symbolic::is_true(ub_criterion)) {
        return false;
    }

    return true;
}

bool TypeMinimization::accept(structured_control_flow::Block& block) {
    bool applied = false;
    auto& dfg = block.dataflow();
    auto& assumptions_analysis = this->analysis_manager_.get<analysis::AssumptionsAnalysis>();
    auto& block_assumptions = assumptions_analysis.get(block, true);

    symbolic::ExpressionMap replacements;
    for (auto& edge : dfg.edges()) {
        auto& subset = edge.subset();
        for (auto& dim : subset) {
            auto truncs = symbolic::find<SymEngine::FunctionSymbol>(dim);
            for (auto& trunc : truncs) {
                auto trunc_func = SymEngine::rcp_static_cast<const SymEngine::FunctionSymbol>(trunc);
                if (trunc_func->get_name() != "trunc_i32") {
                    continue;
                }
                if (replacements.find(trunc_func) != replacements.end()) {
                    continue;
                }
                auto arg = trunc_func->get_args()[0];
                if (!this->is_safe_trunc(arg, block_assumptions)) {
                    continue;
                }

                replacements[trunc_func] = arg;
            }
        }
    }
    if (replacements.empty()) {
        return false;
    }

    for (auto& edge : dfg.edges()) {
        auto subset = edge.subset();
        for (auto& dim : subset) {
            for (auto& [old_expr, new_expr] : replacements) {
                if (new_expr.is_null()) {
                    continue;
                }
                dim = symbolic::subs(dim, old_expr, new_expr);
                applied = true;
            }
        }
        edge.set_subset(subset);
    }

    return applied;
}

bool TypeMinimization::accept(structured_control_flow::For& loop) {
    bool applied = false;
    auto& assumptions_analysis = this->analysis_manager_.get<analysis::AssumptionsAnalysis>();
    auto& block_assumptions = assumptions_analysis.get(loop, true);

    symbolic::ExpressionMap replacements;
    auto truncs = symbolic::find<SymEngine::FunctionSymbol>(loop.condition());
    for (auto& trunc : truncs) {
        auto trunc_func = SymEngine::rcp_static_cast<const SymEngine::FunctionSymbol>(trunc);
        if (trunc_func->get_name() != "trunc_i32") {
            continue;
        }
        if (replacements.find(trunc_func) != replacements.end()) {
            continue;
        }
        auto arg = trunc_func->get_args()[0];
        if (!this->is_safe_trunc(arg, block_assumptions)) {
            continue;
        }

        replacements[trunc_func] = arg;
    }
    if (replacements.empty()) {
        return false;
    }

    auto condition = loop.condition();
    for (auto& [old_expr, new_expr] : replacements) {
        if (new_expr.is_null()) {
            continue;
        }
        condition = symbolic::subs(condition, old_expr, new_expr);
        applied = true;
    }
    builder_.update_loop(loop, loop.indvar(), condition, loop.init(), loop.update());

    return applied;
}

} // namespace passes
} // namespace sdfg
