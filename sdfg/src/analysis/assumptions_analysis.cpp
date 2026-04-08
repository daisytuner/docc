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
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/type.h"

namespace sdfg {
namespace analysis {

symbolic::Expression AssumptionsAnalysis::cnf_to_upper_bound(const symbolic::CNF& cnf, const symbolic::Symbol indvar) {
    std::vector<symbolic::Expression> candidates;

    for (const auto& clause : cnf) {
        for (const auto& literal : clause) {
            // Comparison: indvar < expr
            if (SymEngine::is_a<SymEngine::StrictLessThan>(*literal)) {
                auto lt = SymEngine::rcp_static_cast<const SymEngine::StrictLessThan>(literal);
                if (symbolic::eq(lt->get_arg1(), indvar) && !symbolic::uses(lt->get_arg2(), indvar)) {
                    auto ub = symbolic::sub(lt->get_arg2(), symbolic::one());
                    candidates.push_back(ub);
                }
            }
            // Comparison: indvar <= expr
            else if (SymEngine::is_a<SymEngine::LessThan>(*literal)) {
                auto le = SymEngine::rcp_static_cast<const SymEngine::LessThan>(literal);
                if (symbolic::eq(le->get_arg1(), indvar) && !symbolic::uses(le->get_arg2(), indvar)) {
                    candidates.push_back(le->get_arg2());
                }
            }
            // Comparison: indvar == expr
            else if (SymEngine::is_a<SymEngine::Equality>(*literal)) {
                auto eq = SymEngine::rcp_static_cast<const SymEngine::Equality>(literal);
                if (symbolic::eq(eq->get_arg1(), indvar) && !symbolic::uses(eq->get_arg2(), indvar)) {
                    candidates.push_back(eq->get_arg2());
                }
            }
        }
    }

    if (candidates.empty()) {
        return SymEngine::null;
    }

    // Return the smallest upper bound across all candidate constraints
    symbolic::Expression result = candidates[0];
    for (size_t i = 1; i < candidates.size(); ++i) {
        result = symbolic::min(result, candidates[i]);
    }

    return result;
}

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

    // Collect other constant symbols based on uses
    UsersView users_view(*this->users_analysis_, body);
    std::unordered_set<std::string> visited;
    for (auto& read : users_view.reads()) {
        if (!sdfg_.exists(read->container())) {
            continue;
        }
        if (visited.find(read->container()) != visited.end()) {
            continue;
        }
        visited.insert(read->container());

        auto& type = this->sdfg_.type(read->container());
        if (!type.is_symbol() || type.type_id() != types::TypeID::Scalar) {
            continue;
        }

        if (users_view.reads(read->container()) != users_view.uses(read->container())) {
            continue;
        }

        if (body_assumptions.find(symbolic::symbol(read->container())) == body_assumptions.end()) {
            symbolic::Symbol sym = symbolic::symbol(read->container());
            body_assumptions.insert({sym, symbolic::Assumption(sym)});
            body_assumptions[sym].constant(true);
        }
    }

    // Define map of indvar
    body_assumptions[indvar].map(update);
    body_assumptions[indvar].constant(true);

    // Determine non-tight lower and upper bounds from inverse index access
    std::vector<symbolic::Expression> lbs, ubs;
    for (auto user : this->users_analysis_->reads(indvar->get_name())) {
        if (auto* memlet = dynamic_cast<data_flow::Memlet*>(user->element())) {
            const types::IType* memlet_type = &memlet->base_type();
            for (long long i = memlet->subset().size() - 1; i >= 0; i--) {
                symbolic::Expression num_elements = SymEngine::null;
                if (auto* pointer_type = dynamic_cast<const types::Pointer*>(memlet_type)) {
                    memlet_type = &pointer_type->pointee_type();
                } else if (auto* array_type = dynamic_cast<const types::Array*>(memlet_type)) {
                    memlet_type = &array_type->element_type();
                    num_elements = array_type->num_elements();
                } else {
                    break;
                }
                if (!symbolic::uses(memlet->subset().at(i), indvar)) {
                    continue;
                }
                symbolic::Expression inverse = symbolic::inverse(memlet->subset().at(i), indvar);
                if (inverse.is_null()) {
                    continue;
                }
                lbs.push_back(symbolic::subs(inverse, indvar, symbolic::zero()));
                if (num_elements.is_null()) {
                    std::string container;
                    if (memlet->src_conn() == "void") {
                        container = dynamic_cast<data_flow::AccessNode&>(memlet->src()).data();
                    } else {
                        container = dynamic_cast<data_flow::AccessNode&>(memlet->dst()).data();
                    }
                    if (this->is_parameter(container)) {
                        ubs.push_back(symbolic::sub(
                            symbolic::subs(inverse, indvar, symbolic::dynamic_sizeof(symbolic::symbol(container))),
                            symbolic::one()
                        ));
                    }
                } else {
                    ubs.push_back(symbolic::sub(symbolic::subs(inverse, indvar, num_elements), symbolic::one()));
                }
            }
        }
    }
    for (auto lb : lbs) {
        body_assumptions[indvar].add_lower_bound(lb);
    }
    for (auto ub : ubs) {
        body_assumptions[indvar].add_upper_bound(ub);
    }

    // Prove that update is monotonic -> assume bounds
    if (!loop->is_monotonic()) {
        this->propagate(body, body_assumptions, outer_assumptions, outer_assumptions_with_trivial);
        this->traverse(body, this->assumptions_[&body], this->assumptions_with_trivial_[&body]);
        return;
    }

    // Assumption: init is tight lower bound for indvar
    body_assumptions[indvar].add_lower_bound(init);
    body_assumptions[indvar].tight_lower_bound(init);
    body_assumptions[indvar].lower_bound_deprecated(init);

    // Assumption: inverse index access lower bounds are lower bound for init
    if (SymEngine::is_a<SymEngine::Symbol>(*init) && !lbs.empty()) {
        auto sym = SymEngine::rcp_static_cast<const SymEngine::Symbol>(init);
        if (!body_assumptions.contains(sym)) {
            body_assumptions.insert({sym, symbolic::Assumption(sym)});
        }
        for (auto lb : lbs) {
            body_assumptions[sym].add_lower_bound(lb);
        }
    }

    symbolic::CNF cnf;
    try {
        cnf = symbolic::conjunctive_normal_form(loop->condition());
    } catch (const symbolic::CNFException& e) {
        this->propagate(body, body_assumptions, outer_assumptions, outer_assumptions_with_trivial);
        this->traverse(body, this->assumptions_[&body], this->assumptions_with_trivial_[&body]);
        return;
    }
    auto ub = cnf_to_upper_bound(cnf, indvar);
    if (ub.is_null()) {
        this->propagate(body, body_assumptions, outer_assumptions, outer_assumptions_with_trivial);
        this->traverse(body, this->assumptions_[&body], this->assumptions_with_trivial_[&body]);
        return;
    }
    // Assumption: upper bound ub is tight for indvar if
    body_assumptions[indvar].add_upper_bound(ub);
    body_assumptions[indvar].upper_bound_deprecated(ub);
    // TODO: handle non-contiguous tight upper bounds with modulo
    // Example: for (i = 0; i < n; i += 3) -> tight_upper_bound = (n - 1) - ((n - 1) % 3)
    if (loop->is_contiguous()) {
        body_assumptions[indvar].tight_upper_bound(ub);
    }

    // Assumption: inverse index access upper bounds are upper bound for ub
    if (SymEngine::is_a<SymEngine::Symbol>(*ub) && !ubs.empty()) {
        auto sym = SymEngine::rcp_static_cast<const SymEngine::Symbol>(ub);
        if (!body_assumptions.contains(sym)) {
            body_assumptions.insert({sym, symbolic::Assumption(sym)});
        }
        for (auto ub : ubs) {
            body_assumptions[sym].add_upper_bound(ub);
        }
    }

    // Assumption: any ub symbol is at least init
    for (auto& sym : symbolic::atoms(ub)) {
        body_assumptions[sym].add_lower_bound(symbolic::add(init, symbolic::one()));
        body_assumptions[sym].lower_bound_deprecated(symbolic::add(init, symbolic::one()));
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

        // Deprecated: combine with min
        auto lower_ub_deprecated = lower_assum.upper_bound_deprecated();
        auto lower_lb_deprecated = lower_assum.lower_bound_deprecated();
        auto new_ub_deprecated = symbolic::min(entry.second.upper_bound_deprecated(), lower_ub_deprecated);
        auto new_lb_deprecated = symbolic::max(entry.second.lower_bound_deprecated(), lower_lb_deprecated);
        lower_assum.upper_bound_deprecated(new_ub_deprecated);
        lower_assum.lower_bound_deprecated(new_lb_deprecated);

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

        // Deprecated: combine with min
        auto lower_ub_deprecated = lower_assum.upper_bound_deprecated();
        auto lower_lb_deprecated = lower_assum.lower_bound_deprecated();
        auto new_ub_deprecated = symbolic::min(entry.second.upper_bound_deprecated(), lower_ub_deprecated);
        auto new_lb_deprecated = symbolic::max(entry.second.lower_bound_deprecated(), lower_lb_deprecated);
        lower_assum.upper_bound_deprecated(new_ub_deprecated);
        lower_assum.lower_bound_deprecated(new_lb_deprecated);

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
