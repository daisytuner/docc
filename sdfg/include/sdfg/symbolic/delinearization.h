#pragma once

#include <isl/map.h>

#include "sdfg/symbolic/assumptions.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {
namespace symbolic {

struct DelinearizeResult {
    MultiExpression indices;
    MultiExpression dimensions;
    bool success;
};

/**
 * @brief Delinearizes a linearized expression into indices and dimensions based on parameter assumptions
 * @param expr Linearized expression potentially containing linearized indices
 * @param assums Assumptions about parameters
 * @return Delinearized result containing indices, dimensions, and success flag
 *
 * Attempts to recover multi-dimensional structure from a linearized expression.
 * For example, if an expression represents a linearized 2D array access A[i*N + j],
 * this function tries to recover the 2D indices [i, j].
 *
 * The delinearization technique is based on the algorithm described in:
 * "Optimistic Delinearization of Parametrically Sized Arrays"
 * https://dl.acm.org/doi/10.1145/2751205.2751248
 */
DelinearizeResult delinearize(const Expression& expr, const Assumptions& assums);

} // namespace symbolic
} // namespace sdfg
