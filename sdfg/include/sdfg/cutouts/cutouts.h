#pragma once

#include "sdfg/analysis/analysis.h"
#include "sdfg/analysis/arguments_analysis.h"
#include "sdfg/analysis/users.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/cutouts/cutout_serializer.h"
#include "sdfg/structured_sdfg.h"

namespace sdfg {
namespace util {

std::unique_ptr<StructuredSDFG> cutout(
    sdfg::StructuredSDFG& sdfg,
    sdfg::analysis::AnalysisManager& analysis_manager,
    structured_control_flow::ControlFlowNode& node
);

/**
 * Replace the control-flow node with element_id in builder.subject() by the children of
 * cutout_sdfg's root (the inverse of cutout).
 *
 * Arguments and externals of cutout_sdfg keep their names so they bind to the target's containers;
 * colliding transients are renamed and missing containers are added as transients, so the target's
 * signature is preserved. Merged nodes receive fresh element ids from the target; each outermost
 * merged loop records element_id as its source loop id.
 *
 * @throws std::runtime_error if no control-flow node with element_id exists or it has no parent scope.
 */
void apply_cutout(builder::StructuredSDFGBuilder& builder, size_t element_id, StructuredSDFG& cutout_sdfg);

} // namespace util
} // namespace sdfg
