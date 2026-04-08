#include "sdfg/passes/dataflow/memlet_simplification.h"

#include <algorithm>
#include <optional>
#include <vector>

#include "sdfg/data_flow/data_flow_graph.h"
#include "sdfg/data_flow/memlet.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/visitor/structured_sdfg_visitor.h"

namespace sdfg {
namespace passes {

namespace {

/**
 * @brief Represents a single term in a mixed-radix decomposition
 *
 * A term has the form: stride * ((base / divisor) % modulus)
 * Special cases:
 *   - Outermost: stride * (base / divisor), modulus = infinity (represented as 0)
 *   - Innermost: (base % modulus), stride = 1, divisor = 1
 */
struct MixedRadixTerm {
    symbolic::Expression base; // The base index variable
    int64_t stride; // Multiplier (product of dims after this one)
    int64_t divisor; // What to divide base by
    int64_t modulus; // What to mod by (0 = no modulus, i.e., outermost)
};

/**
 * @brief Checks if an expression is the idiv function and extracts its arguments
 */
std::optional<std::pair<symbolic::Expression, int64_t>> parse_idiv(const symbolic::Expression& expr) {
    if (!SymEngine::is_a<SymEngine::FunctionSymbol>(*expr)) {
        return std::nullopt;
    }
    auto func = SymEngine::rcp_static_cast<const SymEngine::FunctionSymbol>(expr);
    if (func->get_name() != "idiv") {
        return std::nullopt;
    }
    auto args = func->get_args();
    if (args.size() != 2) {
        return std::nullopt;
    }
    if (!SymEngine::is_a<SymEngine::Integer>(*args[1])) {
        return std::nullopt;
    }
    auto divisor = SymEngine::rcp_static_cast<const SymEngine::Integer>(args[1])->as_int();
    return std::make_pair(args[0], divisor);
}

/**
 * @brief Checks if an expression is the imod function and extracts its arguments
 */
std::optional<std::pair<symbolic::Expression, int64_t>> parse_imod(const symbolic::Expression& expr) {
    if (!SymEngine::is_a<SymEngine::FunctionSymbol>(*expr)) {
        return std::nullopt;
    }
    auto func = SymEngine::rcp_static_cast<const SymEngine::FunctionSymbol>(expr);
    if (func->get_name() != "imod") {
        return std::nullopt;
    }
    auto args = func->get_args();
    if (args.size() != 2) {
        return std::nullopt;
    }
    if (!SymEngine::is_a<SymEngine::Integer>(*args[1])) {
        return std::nullopt;
    }
    auto modulus = SymEngine::rcp_static_cast<const SymEngine::Integer>(args[1])->as_int();
    return std::make_pair(args[0], modulus);
}

/**
 * @brief Parses a single term of the mixed-radix expression
 *
 * Handles:
 *   - stride * imod(idiv(base, divisor), modulus)  (middle terms)
 *   - stride * idiv(base, divisor)                  (outermost term, no mod)
 *   - imod(base, modulus)                           (innermost term, stride=1, divisor=1)
 *   - imod(idiv(base, divisor), modulus)            (stride=1 middle term)
 */
std::optional<MixedRadixTerm> parse_term(const symbolic::Expression& term, const symbolic::Symbol& expected_base) {
    MixedRadixTerm result;
    result.stride = 1;
    result.divisor = 1;
    result.modulus = 0;

    symbolic::Expression inner = term;

    // Check if term is stride * something
    if (SymEngine::is_a<SymEngine::Mul>(*term)) {
        auto mul = SymEngine::rcp_static_cast<const SymEngine::Mul>(term);
        auto args = mul->get_args();

        // Look for an integer multiplier (stride)
        symbolic::Expression non_int_part = symbolic::one();
        bool found_stride = false;

        for (const auto& arg : args) {
            if (SymEngine::is_a<SymEngine::Integer>(*arg) && !found_stride) {
                result.stride = SymEngine::rcp_static_cast<const SymEngine::Integer>(arg)->as_int();
                found_stride = true;
            } else {
                non_int_part = SymEngine::mul(non_int_part, arg);
            }
        }

        if (found_stride) {
            inner = non_int_part;
        }
    }

    // Now inner should be one of:
    //   - imod(idiv(base, divisor), modulus)
    //   - idiv(base, divisor)  (outermost)
    //   - imod(base, modulus)  (innermost)
    //   - just base            (trivial case, stride=divisor=1, no mod)

    // Try: imod(something, modulus)
    if (auto mod_result = parse_imod(inner)) {
        result.modulus = mod_result->second;
        inner = mod_result->first;
    }

    // Try: idiv(base, divisor)
    if (auto div_result = parse_idiv(inner)) {
        result.divisor = div_result->second;
        inner = div_result->first;
    }

    // Now inner should be the base symbol
    if (!SymEngine::is_a<SymEngine::Symbol>(*inner)) {
        return std::nullopt;
    }

    auto sym = SymEngine::rcp_static_cast<const SymEngine::Symbol>(inner);
    if (!symbolic::eq(sym, expected_base)) {
        return std::nullopt;
    }

    result.base = sym;
    return result;
}

} // anonymous namespace

MemletSimplification::
    MemletSimplification(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager)
    : visitor::NonStoppingStructuredSDFGVisitor(builder, analysis_manager) {}

std::optional<symbolic::Expression> MemletSimplification::
    try_simplify_mixed_radix(const symbolic::Expression& expr, const symbolic::Symbol& expected_base) {
    // Handle trivial case: expression is just the base symbol
    if (SymEngine::is_a<SymEngine::Symbol>(*expr)) {
        if (symbolic::eq(expr, expected_base)) {
            return expr;
        }
        return std::nullopt;
    }

    // Must be an Add expression
    if (!SymEngine::is_a<SymEngine::Add>(*expr)) {
        return std::nullopt;
    }

    auto add = SymEngine::rcp_static_cast<const SymEngine::Add>(expr);
    auto args = add->get_args();

    if (args.size() < 2) {
        return std::nullopt;
    }

    // Parse all terms
    std::vector<MixedRadixTerm> terms;
    for (const auto& arg : args) {
        auto parsed = parse_term(arg, expected_base);
        if (!parsed) {
            return std::nullopt;
        }
        terms.push_back(*parsed);
    }

    // Sort by divisor descending (outermost first, innermost last)
    std::sort(terms.begin(), terms.end(), [](const MixedRadixTerm& a, const MixedRadixTerm& b) {
        return a.divisor > b.divisor;
    });

    // Verify chain property:
    // 1. stride[k] == divisor[k] for all k
    // 2. For each k (except innermost): divisor[k] == modulus[k+1] * divisor[k+1]
    // 3. Innermost divisor must be 1 (no division needed for last dimension)

    for (size_t k = 0; k < terms.size(); ++k) {
        // Check stride == divisor
        if (terms[k].stride != terms[k].divisor) {
            return std::nullopt;
        }

        if (k < terms.size() - 1) {
            // Outermost term (k=0) can have no modulus, others must have it
            if (terms[k].modulus == 0 && k != 0) {
                return std::nullopt;
            }
            // Verify: divisor[k] == modulus[k+1] * divisor[k+1]
            if (terms[k].divisor != terms[k + 1].modulus * terms[k + 1].divisor) {
                return std::nullopt;
            }
        }
    }

    // Innermost term must have divisor == 1
    if (terms.back().divisor != 1) {
        return std::nullopt;
    }

    // Innermost term must have modulus (since it's base % modulus)
    if (terms.back().modulus == 0) {
        return std::nullopt;
    }

    // All checks passed - the expression equals the base index
    return expected_base;
}

bool MemletSimplification::accept(structured_control_flow::Map& map) {
    // Only process Maps in loop normal form (init=0, stride=1)
    if (!map.is_loop_normal_form()) {
        return false;
    }

    auto indvar = map.indvar();
    bool applied = false;

    // Walk immediate blocks in the map's body
    auto& body = map.root();
    std::list<structured_control_flow::ControlFlowNode*> queue = {&body};
    while (!queue.empty()) {
        auto* current = queue.front();
        queue.pop_front();


        if (auto* block = dynamic_cast<structured_control_flow::Block*>(current)) {
            // Process all memlets in this block
            auto& dfg = block->dataflow();
            for (auto& memlet : dfg.edges()) {
                // Only process flat pointer memlets (single-element subset)
                if (memlet.subset().size() != 1) {
                    continue;
                }

                auto& subset_expr = memlet.subset()[0];

                // Try to simplify the index expression
                auto simplified = try_simplify_mixed_radix(subset_expr, indvar);
                if (simplified && !symbolic::eq(*simplified, subset_expr)) {
                    memlet.set_subset({*simplified});
                    applied = true;
                }
            }
        } else if (auto* seq = dynamic_cast<structured_control_flow::Sequence*>(current)) {
            // Add children of sequences to the queue
            for (size_t i = 0; i < seq->size(); ++i) {
                queue.push_back(&seq->at(i).first);
            }
        } else if (auto* if_stmt = dynamic_cast<structured_control_flow::IfElse*>(current)) {
            // Add branches of if statements to the queue
            for (size_t i = 0; i < if_stmt->size(); ++i) {
                queue.push_back(&if_stmt->at(i).first);
            }
        }
    }

    return applied;
}

} // namespace passes
} // namespace sdfg
