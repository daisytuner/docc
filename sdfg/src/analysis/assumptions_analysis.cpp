#include "sdfg/analysis/assumptions_analysis.h"

#include <utility>
#include <vector>

#include "sdfg/analysis/analysis.h"
#include "sdfg/analysis/scope_analysis.h"
#include "sdfg/analysis/users.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/memlet.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/symbolic/assumptions.h"
#include "sdfg/symbolic/polynomials.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/type.h"

namespace sdfg {
namespace analysis {

AssumptionsAnalysis::AssumptionsAnalysis(StructuredSDFG& sdfg)
    : Analysis(sdfg) {

      };

void AssumptionsAnalysis::run(analysis::AnalysisManager& analysis_manager) {
    this->assumptions_.clear();
    this->assumptions_with_trivial_.clear();
    this->ref_assumptions_.clear();
    this->ref_assumptions_with_trivial_.clear();

    this->parameters_.clear();
    this->users_analysis_ = &analysis_manager.get<Users>();

    // Determine parameters
    this->determine_parameters(analysis_manager);

    // Initialize root assumptions with SDFG-level assumptions
    this->assumptions_.insert({&sdfg_.root(), this->additional_assumptions_});
    auto& initial = this->assumptions_[&sdfg_.root()];

    this->assumptions_with_trivial_.insert({&sdfg_.root(), initial});
    auto& initial_with_trivial = this->assumptions_with_trivial_[&sdfg_.root()];
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

    // Traverse and propagate
    this->traverse(sdfg_.root(), initial, initial_with_trivial);
};

void AssumptionsAnalysis::traverse(
    structured_control_flow::ControlFlowNode& current,
    const symbolic::Assumptions& outer_assumptions,
    const symbolic::Assumptions& outer_assumptions_with_trivial
) {
    this->propagate_ref(current, outer_assumptions, outer_assumptions_with_trivial);

    if (auto sequence_stmt = dynamic_cast<structured_control_flow::Sequence*>(&current)) {
        for (size_t i = 0; i < sequence_stmt->size(); i++) {
            this->traverse(sequence_stmt->at(i).first, outer_assumptions, outer_assumptions_with_trivial);
        }
    } else if (auto if_else_stmt = dynamic_cast<structured_control_flow::IfElse*>(&current)) {
        for (size_t i = 0; i < if_else_stmt->size(); i++) {
            this->traverse(if_else_stmt->at(i).first, outer_assumptions, outer_assumptions_with_trivial);
        }
    } else if (auto while_stmt = dynamic_cast<structured_control_flow::While*>(&current)) {
        this->traverse(while_stmt->root(), outer_assumptions, outer_assumptions_with_trivial);
    } else if (auto loop_stmt = dynamic_cast<structured_control_flow::StructuredLoop*>(&current)) {
        this->traverse_structured_loop(loop_stmt, outer_assumptions, outer_assumptions_with_trivial);
    } else {
        // Other control flow nodes (e.g., Block) do not introduce assumptions or comprise scopes
    }
};

void AssumptionsAnalysis::traverse_structured_loop(
    structured_control_flow::StructuredLoop* loop,
    const symbolic::Assumptions& outer_assumptions,
    const symbolic::Assumptions& outer_assumptions_with_trivial
) {
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
            // Compute tight upper bound based on stride
            if (symbolic::eq(stride, symbolic::one())) {
                // Stride == 1: tight upper bound is simply ub - 1
                body_assumptions[indvar].tight_upper_bound(ub_inclusive);
            } else if (!stride.is_null()) {
                // Non-unit stride: tight upper bound = init + idiv(ub_inclusive - init, stride) * stride
                // This is the largest value of init + k*stride that is <= ub_inclusive
                auto range = symbolic::sub(ub_inclusive, init);
                auto num_steps = symbolic::div(range, stride);
                auto tight_ub = symbolic::add(init, symbolic::mul(num_steps, stride));
                body_assumptions[indvar].tight_upper_bound(tight_ub);
            }

            // If combined bound, each arg is also an upper bound
            if (SymEngine::is_a<SymEngine::Min>(*ub)) {
                auto min = SymEngine::rcp_static_cast<const SymEngine::Min>(ub);
                for (size_t i = 0; i < min->get_args().size(); i++) {
                    auto arg = min->get_args()[i];
                    auto arg_inclusive = symbolic::sub(arg, symbolic::one());
                    body_assumptions[indvar].add_upper_bound(arg_inclusive);
                }
            } else {
                body_assumptions[indvar].add_upper_bound(ub_inclusive);
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
                if (!min_ub_value_is_clean) return;
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

    this->propagate(body, body_assumptions, outer_assumptions, outer_assumptions_with_trivial);
    this->traverse(body, this->assumptions_[&body], this->assumptions_with_trivial_[&body]);
}

void AssumptionsAnalysis::propagate(
    structured_control_flow::ControlFlowNode& node,
    const symbolic::Assumptions& node_assumptions,
    const symbolic::Assumptions& outer_assumptions,
    const symbolic::Assumptions& outer_assumptions_with_trivial
) {
    // Propagate assumptions
    this->assumptions_.insert({&node, node_assumptions});
    auto& propagated_assumptions = this->assumptions_[&node];
    for (auto& entry : outer_assumptions) {
        if (propagated_assumptions.find(entry.first) == propagated_assumptions.end()) {
            // New assumption
            propagated_assumptions.insert({entry.first, entry.second});
            continue;
        }

        // Merge assumptions from lower scopes
        auto& lower_assum = propagated_assumptions[entry.first];

        // Add to set of bounds
        for (auto ub : entry.second.upper_bounds()) {
            lower_assum.add_upper_bound(ub);
        }
        for (auto lb : entry.second.lower_bounds()) {
            lower_assum.add_lower_bound(lb);
        }

        // Set tight bounds
        if (lower_assum.tight_upper_bound().is_null()) {
            lower_assum.tight_upper_bound(entry.second.tight_upper_bound());
        }
        if (lower_assum.tight_lower_bound().is_null()) {
            lower_assum.tight_lower_bound(entry.second.tight_lower_bound());
        }

        // Set map
        if (lower_assum.map().is_null()) {
            lower_assum.map(entry.second.map());
        }

        // Set constant
        if (!lower_assum.constant()) {
            lower_assum.constant(entry.second.constant());
        }
    }

    this->assumptions_with_trivial_.insert({&node, node_assumptions});
    auto& assumptions_with_trivial = this->assumptions_with_trivial_[&node];
    for (auto& entry : outer_assumptions_with_trivial) {
        if (assumptions_with_trivial.find(entry.first) == assumptions_with_trivial.end()) {
            // New assumption
            assumptions_with_trivial.insert({entry.first, entry.second});
            continue;
        }
        // Merge assumptions from lower scopes
        auto& lower_assum = assumptions_with_trivial[entry.first];

        // Add to set of bounds
        for (auto ub : entry.second.upper_bounds()) {
            lower_assum.add_upper_bound(ub);
        }
        for (auto lb : entry.second.lower_bounds()) {
            lower_assum.add_lower_bound(lb);
        }

        // Set tight bounds
        if (lower_assum.tight_upper_bound().is_null()) {
            lower_assum.tight_upper_bound(entry.second.tight_upper_bound());
        }
        if (lower_assum.tight_lower_bound().is_null()) {
            lower_assum.tight_lower_bound(entry.second.tight_lower_bound());
        }

        // Set map
        if (lower_assum.map().is_null()) {
            lower_assum.map(entry.second.map());
        }

        // Set constant
        if (!lower_assum.constant()) {
            lower_assum.constant(entry.second.constant());
        }
    }
}

void AssumptionsAnalysis::propagate_ref(
    structured_control_flow::ControlFlowNode& node,
    const symbolic::Assumptions& outer_assumptions,
    const symbolic::Assumptions& outer_assumptions_with_trivial
) {
    this->ref_assumptions_.insert({&node, &outer_assumptions});
    this->ref_assumptions_with_trivial_.insert({&node, &outer_assumptions_with_trivial});
}

void AssumptionsAnalysis::determine_parameters(analysis::AnalysisManager& analysis_manager) {
    for (auto& container : this->sdfg_.arguments()) {
        bool readonly = true;
        Use not_allowed;
        switch (this->sdfg_.type(container).type_id()) {
            case types::TypeID::Scalar:
                not_allowed = Use::WRITE;
                break;
            case types::TypeID::Pointer:
                not_allowed = Use::MOVE;
                break;
            case types::TypeID::Array:
            case types::TypeID::Structure:
            case types::TypeID::Reference:
            case types::TypeID::Function:
            case types::TypeID::Tensor:
                continue;
        }
        for (auto user : this->users_analysis_->uses(container)) {
            if (user->use() == not_allowed) {
                readonly = false;
                break;
            }
        }
        if (readonly) {
            this->parameters_.insert(symbolic::symbol(container));
        }
    }
}

const symbolic::Assumptions& AssumptionsAnalysis::
    get(structured_control_flow::ControlFlowNode& node, bool include_trivial_bounds) {
    if (include_trivial_bounds) {
        return *this->ref_assumptions_with_trivial_[&node];
    } else {
        return *this->ref_assumptions_[&node];
    }
}

const symbolic::SymbolSet& AssumptionsAnalysis::parameters() { return this->parameters_; }

bool AssumptionsAnalysis::is_parameter(const symbolic::Symbol& container) {
    return this->parameters_.contains(container);
}

bool AssumptionsAnalysis::is_parameter(const std::string& container) {
    return this->is_parameter(symbolic::symbol(container));
}

} // namespace analysis
} // namespace sdfg
