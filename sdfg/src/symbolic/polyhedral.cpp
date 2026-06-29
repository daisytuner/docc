#include "sdfg/symbolic/polyhedral.h"

#include <isl/constraint.h>
#include <isl/local_space.h>
#include <isl/options.h>
#include <isl/space.h>

#include "sdfg/symbolic/delinearization.h"
#include "sdfg/symbolic/utils.h"

namespace sdfg {
namespace symbolic {
namespace polyhedral {

IslCtx::IslCtx() : ctx_(isl_ctx_alloc()) {
    if (ctx_ != nullptr) {
        isl_options_set_on_error(ctx_, ISL_ON_ERROR_CONTINUE);
    }
}

IslCtx::~IslCtx() {
    if (ctx_ != nullptr) {
        isl_ctx_free(ctx_);
    }
}

IslCtx::IslCtx(IslCtx&& other) noexcept : ctx_(other.ctx_) { other.ctx_ = nullptr; }

IslCtx& IslCtx::operator=(IslCtx&& other) noexcept {
    if (this != &other) {
        if (ctx_ != nullptr) {
            isl_ctx_free(ctx_);
        }
        ctx_ = other.ctx_;
        other.ctx_ = nullptr;
    }
    return *this;
}

bool equal_on_domain(const MultiExpression& f, const MultiExpression& g, const Symbol indvar, const Assumptions& assums) {
    (void) indvar;

    if (f.size() != g.size()) {
        return false;
    }
    // Scalar accesses (no index expressions) trivially refer to the same cell.
    if (f.empty()) {
        return true;
    }

    // ISL's string interface only accepts affine expressions: a parameter times
    // a loop variable (e.g. `N*i` from a linearized access) is rejected even
    // though it is parametrically affine. Delinearize single-dimension accesses
    // first so every recovered dimension is affine in the loop variables only.
    MultiExpression f_delin = f;
    if (f.size() == 1) {
        auto result = symbolic::delinearize(f.at(0), assums);
        if (result.success) {
            f_delin = result.indices;
        }
    }
    MultiExpression g_delin = g;
    if (g.size() == 1) {
        auto result = symbolic::delinearize(g.at(0), assums);
        if (result.success) {
            g_delin = result.indices;
        }
    }
    if (f_delin.size() != g_delin.size() || f_delin.empty()) {
        return false;
    }

    // Build the element-wise difference f - g over the *combined* domain. The
    // accesses are equal everywhere iff this difference map is identically zero,
    // which keeps a single shared domain and avoids any two-map space alignment.
    MultiExpression diff;
    diff.reserve(f_delin.size());
    for (size_t i = 0; i < f_delin.size(); ++i) {
        diff.push_back(symbolic::sub(f_delin.at(i), g_delin.at(i)));
    }

    std::string diff_map_str = expression_to_map_str(diff, assums);

    IslCtx ctx;
    if (!ctx) {
        return false;
    }

    IslMap diff_map(isl_map_read_from_str(ctx.get(), diff_map_str.c_str()));
    if (!diff_map) {
        return false;
    }

    IslSet range(isl_map_range(diff_map.release()));
    if (!range) {
        return false;
    }

    // Empty domain => no points to violate equality (vacuously equal).
    if (isl_set_is_empty(range.get()) == isl_bool_true) {
        return true;
    }

    // Constrain every output coordinate of the difference to be zero, on the
    // *same* space as `range` (identical dim names and parameters), then test
    // range ⊆ {0, ..., 0}.
    IslSpace space(isl_set_get_space(range.get()));
    if (!space) {
        return false;
    }

    IslSet zero(isl_set_universe(isl_space_copy(space.get())));
    if (!zero) {
        return false;
    }

    int n_dims = isl_set_dim(range.get(), isl_dim_set);
    isl_set* zero_raw = zero.release();
    for (int i = 0; i < n_dims; ++i) {
        isl_constraint* c = isl_constraint_alloc_equality(isl_local_space_from_space(isl_space_copy(space.get())));
        if (c == nullptr) {
            isl_set_free(zero_raw);
            return false;
        }
        c = isl_constraint_set_coefficient_si(c, isl_dim_set, i, 1);
        zero_raw = isl_set_add_constraint(zero_raw, c);
        if (zero_raw == nullptr) {
            return false;
        }
    }
    zero.reset(zero_raw);

    return isl_set_is_subset(range.get(), zero.get()) == isl_bool_true;
}

} // namespace polyhedral
} // namespace symbolic
} // namespace sdfg
