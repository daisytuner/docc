#include "sdfg/transformations/loop_interchange.h"

#include <isl/ctx.h>
#include <isl/map.h>
#include <isl/options.h>
#include <isl/set.h>

#include "sdfg/exceptions.h"
#include "sdfg/parallelization/analysis/loop_carried_dependency_analysis.h"
#include "sdfg/structured_control_flow/for.h"
#include "sdfg/structured_control_flow/reduce.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/symbolic/polynomials.h"
#include "sdfg/types/scalar.h"

namespace sdfg {
namespace transformations {

// The projection of `deltas` onto `dim_name` is provably non-negative; false when unknown.
static bool is_inner_distance_nonneg(const symbolic::maps::DependenceDeltas& deltas, const std::string& dim_name) {
    if (deltas.deltas_str.empty()) {
        return false;
    }
    auto it = std::find(deltas.dimensions.begin(), deltas.dimensions.end(), dim_name);
    if (it == deltas.dimensions.end()) {
        return false;
    }
    int dim = it - deltas.dimensions.begin();

    isl_ctx* ctx = isl_ctx_alloc();
    isl_options_set_on_error(ctx, ISL_ON_ERROR_CONTINUE);
    bool legal = false;
    isl_set* set = isl_set_read_from_str(ctx, deltas.deltas_str.c_str());
    if (set && isl_set_dim(set, isl_dim_set) == static_cast<isl_size>(deltas.dimensions.size())) {
        int n = isl_set_dim(set, isl_dim_set);
        set = isl_set_project_out(set, isl_dim_set, dim + 1, n - dim - 1);
        set = isl_set_project_out(set, isl_dim_set, 0, dim);
        isl_set* negative = isl_set_read_from_str(ctx, "{ [x] : x < 0 }");
        set = isl_set_intersect(set, negative);
        legal = set && isl_set_is_empty(set) == isl_bool_true;
    }
    isl_set_free(set);
    isl_ctx_free(ctx);
    return legal;
}

/// Extract the upper bound from a condition of the form `indvar [+ offset] < expr`,
/// or `And(indvar < expr1, indvar < expr2, ...)`.
/// Returns the equivalent RHS such that `indvar < result` (using min for conjunctions).
/// Returns SymEngine::null if the condition is not extractable.
static symbolic::Expression
extract_strict_upper_bound(const symbolic::Condition& condition, const symbolic::Symbol& indvar) {
    if (SymEngine::is_a<SymEngine::StrictLessThan>(*condition)) {
        auto lt = SymEngine::rcp_static_cast<const SymEngine::StrictLessThan>(condition);
        auto lhs = lt->get_arg1();
        auto rhs = lt->get_arg2();
        if (symbolic::eq(lhs, indvar)) {
            return rhs;
        }
        // Handle: Lt(indvar + offset, bound) → indvar < bound - offset
        if (symbolic::uses(lhs, indvar->get_name()) && !symbolic::uses(rhs, indvar->get_name())) {
            auto offset = symbolic::sub(lhs, indvar);
            if (!symbolic::uses(offset, indvar->get_name())) {
                return symbolic::sub(rhs, offset);
            }
        }
    }
    // Handle: And(cond1, cond2, ...) → min of extracted bounds
    if (SymEngine::is_a<SymEngine::And>(*condition)) {
        auto conj = SymEngine::rcp_static_cast<const SymEngine::And>(condition);
        symbolic::Expression result = SymEngine::null;
        for (auto& arg : conj->get_container()) {
            auto bound = extract_strict_upper_bound(SymEngine::rcp_dynamic_cast<const SymEngine::Boolean>(arg), indvar);
            if (bound == SymEngine::null) {
                return SymEngine::null;
            }
            if (result == SymEngine::null) {
                result = bound;
            } else {
                result = symbolic::min(result, bound);
            }
        }
        return result;
    }
    return SymEngine::null;
}

/// Decompose `expr` as `coefficient * sym + constant` where coefficient is a
/// positive integer.  Returns the (coefficient, constant) pair on success, or
/// (null, null) when the expression is not affine in `sym` or the coefficient
/// is not a positive integer.
struct AffineDecomp {
    symbolic::Expression coefficient = SymEngine::null;
    symbolic::Expression constant = SymEngine::null;
    explicit operator bool() const {
        return coefficient != SymEngine::null;
    }
};

static AffineDecomp check_affine(const symbolic::Expression& expr, const symbolic::Symbol& sym) {
    symbolic::SymbolVec syms = {sym};
    auto poly = symbolic::polynomial(expr, syms);
    if (poly == SymEngine::null) {
        return {};
    }
    auto coeffs = symbolic::affine_coefficients(poly);
    if (coeffs.empty()) {
        return {};
    }
    auto coeff = coeffs[sym];
    // Coefficient must be a positive integer
    if (!SymEngine::is_a<SymEngine::Integer>(*coeff)) {
        return {};
    }
    if (SymEngine::down_cast<const SymEngine::Integer&>(*coeff).as_int() <= 0) {
        return {};
    }
    return {coeff, coeffs[symbolic::symbol("__daisy_constant__")]};
}

LoopInterchange::LoopInterchange(
    structured_control_flow::StructuredLoop& outer_loop, structured_control_flow::StructuredLoop& inner_loop
)
    : outer_loop_(outer_loop), inner_loop_(inner_loop) {

      };

std::string LoopInterchange::name() const {
    return "LoopInterchange";
};

// Build the loop headers shared by footprint preview and apply without mutating the graph.
tiles::ReductionInterchangeProposal LoopInterchange::proposal() const {
    tiles::ReductionInterchangeProposal result{
        outer_loop_,
        inner_loop_,
        {inner_loop_.init(), inner_loop_.condition(), inner_loop_.update()},
        {outer_loop_.init(), outer_loop_.condition(), outer_loop_.update()}
    };
    bool dependent = symbolic::uses(inner_loop_.init(), outer_loop_.indvar()) ||
                     symbolic::uses(inner_loop_.condition(), outer_loop_.indvar());
    if (!dependent) {
        return result;
    }

    auto outer_indvar = outer_loop_.indvar();
    auto inner_indvar = inner_loop_.indvar();
    auto outer_bound = extract_strict_upper_bound(outer_loop_.condition(), outer_indvar);
    auto inner_bound = extract_strict_upper_bound(inner_loop_.condition(), inner_indvar);
    auto init_decomp = check_affine(inner_loop_.init(), outer_indvar);
    auto bound_decomp = inner_bound.is_null() ? AffineDecomp{} : check_affine(inner_bound, outer_indvar);
    if (outer_bound.is_null() || !init_decomp || !bound_decomp ||
        !symbolic::eq(init_decomp.coefficient, bound_decomp.coefficient)) {
        throw InvalidSDFGException("LoopInterchange: unsupported dependent-bound proposal");
    }
    result.new_outer.init = symbolic::subs(inner_loop_.init(), outer_indvar, outer_loop_.init());
    result.new_outer.condition =
        symbolic::Lt(inner_indvar, symbolic::subs(inner_bound, outer_indvar, symbolic::sub(outer_bound, symbolic::one())));
    auto coefficient = init_decomp.coefficient;
    auto lower = symbolic::sub(inner_indvar, bound_decomp.constant);
    auto upper = symbolic::sub(inner_indvar, init_decomp.constant);
    if (!symbolic::eq(coefficient, symbolic::one())) {
        lower = symbolic::floor_div(lower, coefficient);
        upper = symbolic::floor_div(upper, coefficient);
    }
    result.new_inner.init = symbolic::max(outer_loop_.init(), symbolic::add(lower, symbolic::one()));
    result.new_inner.condition =
        symbolic::Lt(outer_indvar, symbolic::min(outer_bound, symbolic::add(upper, symbolic::one())));
    return result;
}

bool LoopInterchange::can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    auto& outer_indvar = this->outer_loop_.indvar();

    // Check if inner bounds depend on outer loop
    auto inner_loop_init = this->inner_loop_.init();
    auto inner_loop_condition = this->inner_loop_.condition();
    auto inner_loop_update = this->inner_loop_.update();

    // Inner update must never depend on outer
    if (symbolic::uses(inner_loop_update, outer_indvar->get_name())) {
        return false;
    }

    bool inner_depends_on_outer = symbolic::uses(inner_loop_init, outer_indvar->get_name()) ||
                                  symbolic::uses(inner_loop_condition, outer_indvar->get_name());

    if (inner_depends_on_outer) {
        // Fourier-Motzkin elimination: only For-For
        if (dyn_cast<structured_control_flow::Map*>(&outer_loop_) ||
            dyn_cast<structured_control_flow::Map*>(&inner_loop_)) {
            return false;
        }
        // Outer loop must have unit step
        if (!symbolic::eq(outer_loop_.update(), symbolic::add(outer_loop_.indvar(), symbolic::integer(1)))) {
            return false;
        }
        // Inner loop must have a positive integer step
        auto inner_stride = symbolic::sub(inner_loop_.update(), inner_loop_.indvar());
        if (!SymEngine::is_a<SymEngine::Integer>(*inner_stride) ||
            SymEngine::down_cast<const SymEngine::Integer&>(*inner_stride).as_int() <= 0) {
            return false;
        }
        // Outer condition must be extractable as indvar < bound
        auto outer_bound = extract_strict_upper_bound(outer_loop_.condition(), outer_loop_.indvar());
        if (outer_bound == SymEngine::null) {
            return false;
        }
        // Inner init must be affine in outer indvar with positive integer coeff
        auto init_decomp = check_affine(inner_loop_init, outer_indvar);
        if (!init_decomp) {
            return false;
        }
        // Inner bound must be affine in outer indvar with positive integer coeff
        auto inner_bound = extract_strict_upper_bound(inner_loop_.condition(), inner_loop_.indvar());
        if (inner_bound == SymEngine::null) {
            return false;
        }
        auto bound_decomp = check_affine(inner_bound, outer_indvar);
        if (!bound_decomp) {
            return false;
        }
        // Both must have the same coefficient (ensures rectangular projection)
        if (!symbolic::eq(init_decomp.coefficient, bound_decomp.coefficient)) {
            return false;
        }
    }

    // Criterion: Outer loop must not have any outer blocks
    if (outer_loop_.root().size() != 1) {
        return false;
    }
    if (&outer_loop_.root().at(0) != &inner_loop_) {
        return false;
    }

    // Criterion: Any of both loops is a map
    if (dyn_cast<structured_control_flow::Map*>(&outer_loop_) ||
        dyn_cast<structured_control_flow::Map*>(&inner_loop_)) {
        return reduction_buffers_supported(analysis_manager);
    }

    auto& users_analysis = analysis_manager.get<analysis::Users>();
    analysis::UsersView body_users(users_analysis, inner_loop_.root());
    if (!body_users.views().empty() || !body_users.moves().empty()) {
        // Views and moves may have complex semantics that we don't handle yet
        return false;
    }

    // For-For: check legality using dependence delta sets
    auto& lcd = analysis_manager.get<parallelization::LoopCarriedDependencyAnalysis>();

    if (!lcd.available(outer_loop_) || !lcd.available(inner_loop_)) {
        return false;
    }

    std::string outer_indvar_name = outer_loop_.indvar()->get_name();
    std::string inner_indvar_name = inner_loop_.indvar()->get_name();

    // Pairs carried by the inner loop share the outer iteration, whose relative order interchange preserves.
    // Pairs carried by the outer loop (d_outer > 0) stay ordered iff d_inner >= 0.
    auto& outer_pairs = lcd.pairs(outer_loop_);

    std::unordered_set<std::string> flow_containers;
    for (auto& pair : outer_pairs) {
        if (pair.type == parallelization::LOOP_CARRIED_DEPENDENCY_READ_WRITE) {
            flow_containers.insert(pair.writer->container());
        }
    }
    // Scalar temporaries without carried flow that are dead after the nest can be privatized.
    auto locals = users_analysis.locals(outer_loop_);
    auto is_private_scalar = [&](const std::string& container) {
        return !flow_containers.count(container) && locals.count(container) &&
               dynamic_cast<const types::Scalar*>(&builder.subject().type(container)) != nullptr;
    };
    // Reductions are associative and commutative, so any iteration order is valid.
    auto is_reduction = [&](const std::string& container) {
        for (auto& reduction : lcd.reductions(outer_loop_)) {
            if (reduction.container == container) {
                return true;
            }
        }
        // The outer body is exactly the inner loop, so an inner Reduce's accumulator is reduced over the whole nest.
        if (auto* reduce = dyn_cast<structured_control_flow::Reduce*>(&inner_loop_)) {
            for (auto& reduction : reduce->reductions()) {
                if (reduction.container == container) {
                    return true;
                }
            }
        }
        return false;
    };

    for (auto& pair : outer_pairs) {
        const auto& container = pair.writer->container();
        if (container == outer_indvar_name || container == inner_indvar_name) {
            continue;
        }
        auto& deltas = pair.deltas;
        if (deltas.empty) {
            continue;
        }
        if (is_reduction(container) || is_private_scalar(container)) {
            continue;
        }
        if (!is_inner_distance_nonneg(deltas, inner_indvar_name)) {
            return false;
        }
    }

    return reduction_buffers_supported(analysis_manager);
};

// Check affected GPU owners against the proposed nesting before changing the live graph.
bool LoopInterchange::reduction_buffers_supported(analysis::AnalysisManager& analysis_manager) const {
    auto& buffers = analysis_manager.get<tiles::ReductionBufferAnalysis>();
    if (buffers.affected_reductions(outer_loop_).empty()) {
        return true;
    }
    try {
        return buffers.supports_interchange(proposal());
    } catch (const InvalidSDFGException&) {
        return false;
    }
}

void LoopInterchange::apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    if (!reduction_buffers_supported(analysis_manager)) {
        throw InvalidSDFGException("LoopInterchange: proposed GPU reduction buffers are not representable");
    }
    auto& outer_scope = static_cast<structured_control_flow::Sequence&>(*outer_loop_.get_parent());
    auto& inner_scope = outer_loop_.root();

    int index = outer_scope.index(this->outer_loop_);

    // Add new outer and inner loops
    structured_control_flow::StructuredLoop* new_outer_loop = nullptr;
    structured_control_flow::StructuredLoop* new_inner_loop = nullptr;

    auto* inner_map = dyn_cast<structured_control_flow::Map*>(&inner_loop_);
    auto* outer_map = dyn_cast<structured_control_flow::Map*>(&outer_loop_);
    auto* inner_reduce = dyn_cast<structured_control_flow::Reduce*>(&inner_loop_);
    auto* outer_reduce = dyn_cast<structured_control_flow::Reduce*>(&outer_loop_);

    const auto geometry = proposal();

    bool dependent = !inner_map && !outer_map &&
                     (symbolic::uses(inner_loop_.init(), outer_loop_.indvar()->get_name()) ||
                      symbolic::uses(inner_loop_.condition(), outer_loop_.indvar()->get_name()));

    if (dependent) {
        auto outer_indvar = outer_loop_.indvar();
        auto inner_indvar = inner_loop_.indvar();
        auto new_outer_init = geometry.new_outer.init;
        auto new_outer_cond = geometry.new_outer.condition;
        auto new_inner_init = geometry.new_inner.init;
        auto new_inner_cond = geometry.new_inner.condition;

        if (inner_reduce) {
            new_outer_loop = &builder.add_reduce_after(
                outer_scope,
                this->outer_loop_,
                inner_indvar,
                new_outer_cond,
                new_outer_init,
                this->inner_loop_.update(),
                inner_reduce->reductions(),
                this->inner_loop_.schedule_type(),
                this->inner_loop_.debug_info()
            );
        } else {
            new_outer_loop = &builder.add_for_after(
                outer_scope,
                this->outer_loop_,
                inner_indvar,
                new_outer_cond,
                new_outer_init,
                this->inner_loop_.update(),
                this->inner_loop_.debug_info()
            );
        }

        if (outer_reduce) {
            new_inner_loop = &builder.add_reduce_after(
                inner_scope,
                this->inner_loop_,
                outer_indvar,
                new_inner_cond,
                new_inner_init,
                this->outer_loop_.update(),
                outer_reduce->reductions(),
                this->outer_loop_.schedule_type(),
                this->outer_loop_.debug_info()
            );
        } else {
            new_inner_loop = &builder.add_for_after(
                inner_scope,
                this->inner_loop_,
                outer_indvar,
                new_inner_cond,
                new_inner_init,
                this->outer_loop_.update(),
                this->outer_loop_.debug_info()
            );
        }
    } else {
        // Standard case: just swap loop headers
        if (inner_map) {
            new_outer_loop = &builder.add_map_after(
                outer_scope,
                this->outer_loop_,
                inner_map->indvar(),
                inner_map->condition(),
                inner_map->init(),
                inner_map->update(),
                inner_map->schedule_type(),
                this->inner_loop_.debug_info()
            );
        } else if (inner_reduce) {
            new_outer_loop = &builder.add_reduce_after(
                outer_scope,
                this->outer_loop_,
                this->inner_loop_.indvar(),
                this->inner_loop_.condition(),
                this->inner_loop_.init(),
                this->inner_loop_.update(),
                inner_reduce->reductions(),
                this->inner_loop_.schedule_type(),
                this->inner_loop_.debug_info()
            );
        } else {
            new_outer_loop = &builder.add_for_after(
                outer_scope,
                this->outer_loop_,
                this->inner_loop_.indvar(),
                this->inner_loop_.condition(),
                this->inner_loop_.init(),
                this->inner_loop_.update(),
                this->inner_loop_.debug_info()
            );
        }

        if (outer_map) {
            new_inner_loop = &builder.add_map_after(
                inner_scope,
                this->inner_loop_,
                outer_map->indvar(),
                outer_map->condition(),
                outer_map->init(),
                outer_map->update(),
                outer_map->schedule_type(),
                this->outer_loop_.debug_info()
            );
        } else if (outer_reduce) {
            new_inner_loop = &builder.add_reduce_after(
                inner_scope,
                this->inner_loop_,
                this->outer_loop_.indvar(),
                this->outer_loop_.condition(),
                this->outer_loop_.init(),
                this->outer_loop_.update(),
                outer_reduce->reductions(),
                this->outer_loop_.schedule_type(),
                this->outer_loop_.debug_info()
            );
        } else {
            new_inner_loop = &builder.add_for_after(
                inner_scope,
                this->inner_loop_,
                this->outer_loop_.indvar(),
                this->outer_loop_.condition(),
                this->outer_loop_.init(),
                this->outer_loop_.update(),
                this->outer_loop_.debug_info()
            );
        }
    }

    // Insert inner loop body into new inner loop
    builder.move_children(this->inner_loop_.root(), new_inner_loop->root());

    // Insert outer loop body into new outer loop
    builder.move_children(this->outer_loop_.root(), new_outer_loop->root());

    // Remove old loops
    builder.remove_child(new_outer_loop->root(), 0);
    builder.remove_child(outer_scope, index);

    analysis_manager.invalidate_all();
    applied_ = true;
    new_outer_loop_ = new_outer_loop;
    new_inner_loop_ = new_inner_loop;
};

void LoopInterchange::to_json(nlohmann::json& j) const {
    j["transformation_type"] = this->name();
    j["parameters"] = nlohmann::json::object();

    serializer::JSONSerializer ser_flat(false);
    j["subgraph"] = nlohmann::json::object();
    j["subgraph"]["0"] = nlohmann::json::object();
    ser_flat.serialize_node(j["subgraph"]["0"], this->outer_loop_);

    j["subgraph"]["1"] = nlohmann::json::object();
    ser_flat.serialize_node(j["subgraph"]["1"], this->inner_loop_);
};

LoopInterchange LoopInterchange::from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& desc) {
    auto outer_loop_id = desc["subgraph"]["0"]["element_id"].get<size_t>();
    auto inner_loop_id = desc["subgraph"]["1"]["element_id"].get<size_t>();
    auto outer_element = builder.find_element_by_id(outer_loop_id);
    auto inner_element = builder.find_element_by_id(inner_loop_id);
    if (outer_element == nullptr) {
        throw InvalidSDFGException("Element with ID " + std::to_string(outer_loop_id) + " not found.");
    }
    if (inner_element == nullptr) {
        throw InvalidSDFGException("Element with ID " + std::to_string(inner_loop_id) + " not found.");
    }
    auto outer_loop = dyn_cast<structured_control_flow::StructuredLoop*>(outer_element);
    if (outer_loop == nullptr) {
        throw InvalidSDFGException("Element with ID " + std::to_string(outer_loop_id) + " is not a StructuredLoop.");
    }
    auto inner_loop = dyn_cast<structured_control_flow::StructuredLoop*>(inner_element);
    if (inner_loop == nullptr) {
        throw InvalidSDFGException("Element with ID " + std::to_string(inner_loop_id) + " is not a StructuredLoop.");
    }

    return LoopInterchange(*outer_loop, *inner_loop);
};

structured_control_flow::StructuredLoop* LoopInterchange::new_outer_loop() const {
    if (!applied_) {
        throw InvalidSDFGException("Transformation has not been applied yet.");
    }
    return new_outer_loop_;
};

structured_control_flow::StructuredLoop* LoopInterchange::new_inner_loop() const {
    if (!applied_) {
        throw InvalidSDFGException("Transformation has not been applied yet.");
    }
    return new_inner_loop_;
};

} // namespace transformations
} // namespace sdfg
