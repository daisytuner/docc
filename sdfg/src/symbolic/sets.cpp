#include "sdfg/symbolic/sets.h"

#include <isl/ctx.h>
#include <isl/options.h>
#include <isl/set.h>
#include <isl/space.h>

#include "sdfg/symbolic/delinearization.h"
#include "sdfg/symbolic/utils.h"

namespace sdfg {
namespace symbolic {

bool is_subset(
    const MultiExpression& expr1, const MultiExpression& expr2, const Assumptions& assums1, const Assumptions& assums2
) {
    if (expr1.size() == 0 && expr2.size() == 0) {
        return true;
    }

    auto expr1_delinearized = expr1;
    if (expr1.size() == 1) {
        auto result = symbolic::delinearize(expr1.at(0), assums1);
        if (result.success) {
            expr1_delinearized = result.indices;
        }
    }
    auto expr2_delinearized = expr2;
    if (expr2.size() == 1) {
        auto result = symbolic::delinearize(expr2.at(0), assums2);
        if (result.success) {
            expr2_delinearized = result.indices;
        }
    }

    std::string map_1_str = expression_to_map_str(expr1_delinearized, assums1);
    std::string map_2_str = expression_to_map_str(expr2_delinearized, assums2);

    isl_ctx* ctx = isl_ctx_alloc();
    isl_options_set_on_error(ctx, ISL_ON_ERROR_CONTINUE);

    isl_map* map_1 = isl_map_read_from_str(ctx, map_1_str.c_str());
    isl_map* map_2 = isl_map_read_from_str(ctx, map_2_str.c_str());
    if (!map_1 || !map_2) {
        if (map_1) {
            isl_map_free(map_1);
        }
        if (map_2) {
            isl_map_free(map_2);
        }
        isl_ctx_free(ctx);
        return false;
    }
    isl_space* params_map1 = isl_space_params(isl_map_get_space(map_1));
    isl_space* params_map2 = isl_space_params(isl_map_get_space(map_2));

    // Align parameters carefully:
    isl_space* unified_params = isl_space_align_params(isl_space_copy(params_map1), isl_space_copy(params_map2));

    // Align maps to unified params:
    isl_map* aligned_map_1 = isl_map_align_params(map_1, isl_space_copy(unified_params));
    isl_map* aligned_map_2 = isl_map_align_params(map_2, isl_space_copy(unified_params));

    // Remove parameters explicitly (project them out)
    aligned_map_1 = isl_map_project_out(aligned_map_1, isl_dim_param, 0, isl_map_dim(aligned_map_1, isl_dim_param));
    aligned_map_2 = isl_map_project_out(aligned_map_2, isl_dim_param, 0, isl_map_dim(aligned_map_2, isl_dim_param));

    canonicalize_map_dims(aligned_map_1, "in_", "out_");
    canonicalize_map_dims(aligned_map_2, "in_", "out_");

    isl_set* set_1 = isl_map_range(aligned_map_1);
    isl_set* set_2 = isl_map_range(aligned_map_2);

    bool subset = isl_set_is_subset(set_1, set_2) == isl_bool_true;

    isl_map_free(aligned_map_1);
    isl_map_free(aligned_map_2);
    isl_space_free(unified_params);
    isl_space_free(params_map1);
    isl_space_free(params_map2);
    isl_ctx_free(ctx);

    return subset;
}

bool is_disjoint(
    const MultiExpression& expr1, const MultiExpression& expr2, const Assumptions& assums1, const Assumptions& assums2
) {
    auto expr1_delinearized = expr1;
    if (expr1.size() == 1) {
        auto result = symbolic::delinearize(expr1.at(0), assums1);
        if (result.success) {
            expr1_delinearized = result.indices;
        }
    }
    auto expr2_delinearized = expr2;
    if (expr2.size() == 1) {
        auto result = symbolic::delinearize(expr2.at(0), assums2);
        if (result.success) {
            expr2_delinearized = result.indices;
        }
    }

    std::string map_1_str = expression_to_map_str(expr1_delinearized, assums1);
    std::string map_2_str = expression_to_map_str(expr2_delinearized, assums2);

    isl_ctx* ctx = isl_ctx_alloc();
    isl_options_set_on_error(ctx, ISL_ON_ERROR_CONTINUE);

    isl_map* map_1 = isl_map_read_from_str(ctx, map_1_str.c_str());
    isl_map* map_2 = isl_map_read_from_str(ctx, map_2_str.c_str());
    if (!map_1 || !map_2) {
        if (map_1) isl_map_free(map_1);
        if (map_2) isl_map_free(map_2);
        isl_ctx_free(ctx);
        return false;
    }

    isl_space* params_map1 = isl_space_params(isl_map_get_space(map_1));
    isl_space* params_map2 = isl_space_params(isl_map_get_space(map_2));

    // Align parameters carefully:
    isl_space* unified_params = isl_space_align_params(isl_space_copy(params_map1), isl_space_copy(params_map2));

    // Align maps to unified params:
    isl_map* aligned_map_1 = isl_map_align_params(map_1, isl_space_copy(unified_params));
    isl_map* aligned_map_2 = isl_map_align_params(map_2, isl_space_copy(unified_params));

    // Remove parameters explicitly (project them out)
    aligned_map_1 = isl_map_project_out(aligned_map_1, isl_dim_param, 0, isl_map_dim(aligned_map_1, isl_dim_param));
    aligned_map_2 = isl_map_project_out(aligned_map_2, isl_dim_param, 0, isl_map_dim(aligned_map_2, isl_dim_param));

    canonicalize_map_dims(aligned_map_1, "in_", "out_");
    canonicalize_map_dims(aligned_map_2, "in_", "out_");

    isl_set* set_1 = isl_map_range(aligned_map_1);
    isl_set* set_2 = isl_map_range(aligned_map_2);

    bool disjoint = isl_set_is_disjoint(set_1, set_2) == isl_bool_true;

    isl_map_free(aligned_map_1);
    isl_map_free(aligned_map_2);
    isl_space_free(unified_params);
    isl_space_free(params_map1);
    isl_space_free(params_map2);
    isl_ctx_free(ctx);

    return disjoint;
}

} // namespace symbolic
} // namespace sdfg
