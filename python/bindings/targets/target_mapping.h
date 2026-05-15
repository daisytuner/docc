#pragma once
#include <string>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/plugins/plugins.h"
#include "sdfg/plugins/targets.h"

// TODO: move this into a library that can be reused by docc-llvm, do
namespace docc::plugins {

void apply_lib_node_target_mapping(
    sdfg::plugins::Context& docc_context,
    sdfg::builder::StructuredSDFGBuilder& builder,
    sdfg::analysis::AnalysisManager& analysis_manager,
    const target::TargetOptions& options
);

} // namespace docc::plugins
