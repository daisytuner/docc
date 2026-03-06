#pragma once

#include "sdfg/analysis/analysis.h"
#include "sdfg/analysis/arguments_analysis.h"
#include "sdfg/analysis/scope_analysis.h"
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

} // namespace util
} // namespace sdfg
