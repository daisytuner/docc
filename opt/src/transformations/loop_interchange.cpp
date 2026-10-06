#include "sdfg/transformations/loop_interchange.h"

#include <isl/ctx.h>
#include <isl/map.h>
#include <isl/options.h>
#include <isl/set.h>

#include <optional>

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

/// `coefficient * sym + constant` with a non-negative integer coefficient.
struct AffinePiece {
    int64_t coefficient;
    symbolic::Expression constant;
};

static bool contains_min_max(const symbolic::Expression& expr) {
    if (SymEngine::is_a<SymEngine::Min>(*expr) || SymEngine::is_a<SymEngine::Max>(*expr)) {
        return true;
    }
    for (auto& arg : expr->get_args()) {
        if (contains_min_max(arg)) {
            return true;
        }
    }
    return false;
}

// Flattens `expr` into affine pieces it is the min (or max) of, distributing sums and positive scalings.
static bool collect_pieces(const symbolic::Expression& expr, bool is_min, std::vector<symbolic::Expression>& out) {
    constexpr size_t max_pieces = 16;
    if ((is_min && SymEngine::is_a<SymEngine::Min>(*expr)) || (!is_min && SymEngine::is_a<SymEngine::Max>(*expr))) {
        for (auto& arg : expr->get_args()) {
            if (!collect_pieces(arg, is_min, out) || out.size() > max_pieces) {
                return false;
            }
        }
        return true;
    }
    if (SymEngine::is_a<SymEngine::Add>(*expr)) {
        std::vector<symbolic::Expression> sums = {symbolic::zero()};
        for (auto& term : expr->get_args()) {
            std::vector<symbolic::Expression> term_pieces;
            if (!collect_pieces(term, is_min, term_pieces) || sums.size() * term_pieces.size() > max_pieces) {
                return false;
            }
            std::vector<symbolic::Expression> next;
            for (auto& sum : sums) {
                for (auto& piece : term_pieces) {
                    next.push_back(symbolic::add(sum, piece));
                }
            }
            sums = std::move(next);
        }
        out.insert(out.end(), sums.begin(), sums.end());
        return true;
    }
    if (SymEngine::is_a<SymEngine::Mul>(*expr)) {
        auto& mul = SymEngine::down_cast<const SymEngine::Mul&>(*expr);
        auto factor = mul.get_coef();
        auto rest = SymEngine::div(expr, factor);
        if (SymEngine::is_a<SymEngine::Integer>(*factor) &&
            SymEngine::down_cast<const SymEngine::Integer&>(*factor).is_positive() && !symbolic::eq(rest, expr)) {
            std::vector<symbolic::Expression> rest_pieces;
            if (!collect_pieces(rest, is_min, rest_pieces)) {
                return false;
            }
            for (auto& piece : rest_pieces) {
                out.push_back(symbolic::mul(factor, piece));
            }
            return true;
        }
    }
    if (contains_min_max(expr)) {
        return false;
    }
    out.push_back(expr);
    return true;
}

static std::optional<AffinePiece> affine_piece(const symbolic::Expression& expr, const symbolic::Symbol& sym) {
    symbolic::SymbolVec syms = {sym};
    auto poly = symbolic::polynomial(expr, syms);
    if (poly == SymEngine::null) {
        return std::nullopt;
    }
    auto coeffs = symbolic::affine_coefficients(poly);
    if (coeffs.empty()) {
        return std::nullopt;
    }
    auto coeff = coeffs[sym];
    if (!SymEngine::is_a<SymEngine::Integer>(*coeff)) {
        return std::nullopt;
    }
    int64_t value = SymEngine::down_cast<const SymEngine::Integer&>(*coeff).as_int();
    if (value < 0) {
        return std::nullopt;
    }
    return AffinePiece{value, coeffs[symbolic::symbol("__daisy_constant__")]};
}

static std::optional<std::vector<AffinePiece>>
affine_pieces(const symbolic::Expression& expr, bool is_min, const symbolic::Symbol& sym) {
    std::vector<symbolic::Expression> pieces;
    if (!collect_pieces(expr, is_min, pieces)) {
        return std::nullopt;
    }
    std::vector<AffinePiece> result;
    for (auto& piece : pieces) {
        auto decomp = affine_piece(piece, sym);
        if (!decomp) {
            return std::nullopt;
        }
        result.push_back(*decomp);
    }
    return result;
}

static std::optional<int64_t> positive_stride(const structured_control_flow::StructuredLoop& loop) {
    auto stride = symbolic::sub(loop.update(), loop.indvar());
    if (!SymEngine::is_a<SymEngine::Integer>(*stride)) {
        return std::nullopt;
    }
    int64_t value = SymEngine::down_cast<const SymEngine::Integer&>(*stride).as_int();
    if (value <= 0) {
        return std::nullopt;
    }
    return value;
}

/// Fourier-Motzkin projection of the 2-D nest
///   for o = o_init; o < o_bound; o += S_o:  for j = max_l(c_l*o + d_l); j < min_k(c_k*o + e_k); j += S_j
/// with all c >= 0. Returns the new outer (j) and inner (o) headers, or nullopt if unsupported.
static std::optional<LoopSwap> dependent_interchange(
    structured_control_flow::StructuredLoop& outer_loop, structured_control_flow::StructuredLoop& inner_loop
) {
    auto o = outer_loop.indvar();
    auto j = inner_loop.indvar();
    auto outer_stride = positive_stride(outer_loop);
    auto inner_stride = positive_stride(inner_loop);
    auto o_bound = extract_strict_upper_bound(outer_loop.condition(), o);
    auto j_bound = extract_strict_upper_bound(inner_loop.condition(), j);
    if (!outer_stride || !inner_stride || o_bound.is_null() || j_bound.is_null() ||
        symbolic::uses(o_bound, j->get_name())) {
        return std::nullopt;
    }
    auto lowers = affine_pieces(inner_loop.init(), /*is_min=*/false, o);
    auto uppers = affine_pieces(j_bound, /*is_min=*/true, o);
    if (!lowers || !uppers) {
        return std::nullopt;
    }
    // j keeps its stride, so every outer iteration must start j on the same lattice.
    if (*inner_stride > 1 &&
        (lowers->size() != 1 || (lowers->at(0).coefficient * *outer_stride) % *inner_stride != 0)) {
        return std::nullopt;
    }

    LoopSwap result{
        outer_loop,
        inner_loop,
        {inner_loop.init(), inner_loop.condition(), inner_loop.update()},
        {outer_loop.init(), outer_loop.condition(), outer_loop.update()}
    };
    // All pieces are non-decreasing in o, so the j range is spanned by the first and last o.
    result.new_outer.init = symbolic::subs(inner_loop.init(), o, outer_loop.init());
    result.new_outer.condition = symbolic::Lt(j, symbolic::subs(j_bound, o, symbolic::sub(o_bound, symbolic::one())));

    // j < c*o + e  <=>  o >= floor((j - e) / c) + 1;  j >= c*o + d  <=>  o < floor((j - d) / c) + 1
    auto solve = [&](const AffinePiece& piece) {
        auto numerator = symbolic::sub(j, piece.constant);
        auto quotient = piece.coefficient == 1 ? numerator
                                               : symbolic::floor_div(numerator, symbolic::integer(piece.coefficient));
        return symbolic::add(quotient, symbolic::one());
    };
    symbolic::Expression o_lower = SymEngine::null;
    for (auto& piece : *uppers) {
        if (piece.coefficient > 0) {
            o_lower = o_lower.is_null() ? solve(piece) : symbolic::max(o_lower, solve(piece));
        }
    }
    symbolic::Expression o_upper = o_bound;
    for (auto& piece : *lowers) {
        if (piece.coefficient > 0) {
            o_upper = symbolic::min(o_upper, solve(piece));
        }
    }
    if (o_lower.is_null()) {
        result.new_inner.init = outer_loop.init();
    } else if (*outer_stride == 1) {
        result.new_inner.init = symbolic::max(outer_loop.init(), o_lower);
    } else {
        // Round up onto the outer lattice o_init + S_o * k.
        auto offset = symbolic::max(symbolic::zero(), symbolic::sub(o_lower, outer_loop.init()));
        result.new_inner.init = symbolic::add(
            outer_loop.init(),
            symbolic::mul(symbolic::integer(*outer_stride), symbolic::ceil_div(offset, symbolic::integer(*outer_stride)))
        );
    }
    result.new_inner.condition = symbolic::Lt(o, o_upper);
    return result;
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
LoopSwap LoopInterchange::proposal() const {
    LoopSwap result{
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

    auto projected = dependent_interchange(outer_loop_, inner_loop_);
    if (!projected) {
        throw InvalidSDFGException("LoopInterchange: unsupported dependent-bound proposal");
    }
    return *projected;
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
        if (!dependent_interchange(outer_loop_, inner_loop_)) {
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
