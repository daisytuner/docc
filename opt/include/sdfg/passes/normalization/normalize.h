#pragma once

#include "sdfg/structured_sdfg.h"

namespace sdfg {
namespace passes {
namespace normalization {

/**
 * Normalize a StructuredSDFG in place.
 *
 * Runs the loop-normalization pipeline. When `enable_fusion` is set, also performs
 * map fusion before and after loop normalization.
 *
 * This function performs:
 * 1. (Optional) Initial map fusion without init-into-reduction hoisting
 * 2. Loop distribution and stride minimization
 * 3. (Optional) Final map fusion with init-into-reduction hoisting
 */
void normalize(sdfg::StructuredSDFG& sdfg, bool enable_fusion = true);

} // namespace normalization
} // namespace passes
} // namespace sdfg
