#include "sdfg/symbolic/sets.h"

#include <isl/ctx.h>
#include <isl/options.h>
#include <isl/set.h>
#include <isl/space.h>

#include "sdfg/symbolic/delinearization.h"
#include "sdfg/symbolic/polyhedral.h"
#include "sdfg/symbolic/utils.h"

namespace sdfg {
namespace symbolic {

namespace {

// Core ISL subset/disjoint kernel: takes already-delinearized expressions plus
// the matching assumptions, builds the ISL sets and runs the query. Both
// `is_subset` and `is_disjoint` share the entire ISL preamble (ctx, alignment,
// param projection, range extraction); they only differ in the final ISL call.
enum class SetQuery { Subset, Disjoint };

bool run_isl_set_query(
    const MultiExpression& expr1_delinearized,
    const MultiExpression& expr2_delinearized,
    const Assumptions& assums1,
    const Assumptions& assums2,
    SetQuery query
) {
    std::string map_1_str = expression_to_map_str(expr1_delinearized, assums1);
    std::string map_2_str = expression_to_map_str(expr2_delinearized, assums2);

    polyhedral::IslCtx ctx;
    if (!ctx) {
        return false;
    }

    polyhedral::IslMap map_1(isl_map_read_from_str(ctx.get(), map_1_str.c_str()));
    polyhedral::IslMap map_2(isl_map_read_from_str(ctx.get(), map_2_str.c_str()));
    if (!map_1 || !map_2) {
        return false;
    }
    polyhedral::IslSpace params_map1(isl_space_params(isl_map_get_space(map_1.get())));
    polyhedral::IslSpace params_map2(isl_space_params(isl_map_get_space(map_2.get())));

    // Align parameters carefully:
    polyhedral::IslSpace
        unified_params(isl_space_align_params(isl_space_copy(params_map1.get()), isl_space_copy(params_map2.get())));

    // Align maps to unified params:
    polyhedral::IslMap aligned_map_1(isl_map_align_params(map_1.release(), isl_space_copy(unified_params.get())));
    polyhedral::IslMap aligned_map_2(isl_map_align_params(map_2.release(), isl_space_copy(unified_params.get())));

    // Remove parameters explicitly (project them out)
    int n_param_1 = isl_map_dim(aligned_map_1.get(), isl_dim_param);
    aligned_map_1.reset(isl_map_project_out(aligned_map_1.release(), isl_dim_param, 0, n_param_1));
    int n_param_2 = isl_map_dim(aligned_map_2.get(), isl_dim_param);
    aligned_map_2.reset(isl_map_project_out(aligned_map_2.release(), isl_dim_param, 0, n_param_2));

    canonicalize_map_dims(aligned_map_1.get(), "in_", "out_");
    canonicalize_map_dims(aligned_map_2.get(), "in_", "out_");

    polyhedral::IslSet set_1(isl_map_range(aligned_map_1.release()));
    polyhedral::IslSet set_2(isl_map_range(aligned_map_2.release()));

    bool result = false;
    switch (query) {
        case SetQuery::Subset:
            result = isl_set_is_subset(set_1.get(), set_2.get()) == isl_bool_true;
            break;
        case SetQuery::Disjoint:
            result = isl_set_is_disjoint(set_1.get(), set_2.get()) == isl_bool_true;
            break;
    }

    return result;
}

} // namespace

bool is_subset(
    const MultiExpression& expr1, const MultiExpression& expr2, AssumptionsBounds& bounds1, AssumptionsBounds& bounds2
) {
    if (expr1.size() == 0 && expr2.size() == 0) {
        return true;
    }

    auto expr1_delinearized = expr1;
    if (expr1.size() == 1) {
        auto result = symbolic::delinearize(expr1.at(0), bounds1);
        if (result.success) {
            expr1_delinearized = result.indices;
        }
    }
    auto expr2_delinearized = expr2;
    if (expr2.size() == 1) {
        auto result = symbolic::delinearize(expr2.at(0), bounds2);
        if (result.success) {
            expr2_delinearized = result.indices;
        }
    }

    return run_isl_set_query(expr1_delinearized, expr2_delinearized, bounds1.assums(), bounds2.assums(), SetQuery::Subset);
}

bool is_subset(
    const MultiExpression& expr1, const MultiExpression& expr2, const Assumptions& assums1, const Assumptions& assums2
) {
    AssumptionsBounds b1(assums1);
    AssumptionsBounds b2(assums2);
    return is_subset(expr1, expr2, b1, b2);
}

bool is_disjoint(
    const MultiExpression& expr1, const MultiExpression& expr2, AssumptionsBounds& bounds1, AssumptionsBounds& bounds2
) {
    auto expr1_delinearized = expr1;
    if (expr1.size() == 1) {
        auto result = symbolic::delinearize(expr1.at(0), bounds1);
        if (result.success) {
            expr1_delinearized = result.indices;
        }
    }
    auto expr2_delinearized = expr2;
    if (expr2.size() == 1) {
        auto result = symbolic::delinearize(expr2.at(0), bounds2);
        if (result.success) {
            expr2_delinearized = result.indices;
        }
    }

    return run_isl_set_query(
        expr1_delinearized, expr2_delinearized, bounds1.assums(), bounds2.assums(), SetQuery::Disjoint
    );
}

bool is_disjoint(
    const MultiExpression& expr1, const MultiExpression& expr2, const Assumptions& assums1, const Assumptions& assums2
) {
    AssumptionsBounds b1(assums1);
    AssumptionsBounds b2(assums2);
    return is_disjoint(expr1, expr2, b1, b2);
}

} // namespace symbolic
} // namespace sdfg
