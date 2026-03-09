#pragma once
#include <string>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"

// TODO: move this into a library that can be reused by docc-llvm, do
namespace docc::plugins {

struct TargetOptions {
    std::string target;
    std::string category;
    bool transfer_tuning;
};


void apply_lib_node_target_mapping(
    sdfg::builder::StructuredSDFGBuilder& builder,
    sdfg::analysis::AnalysisManager& analysis_manager,
    TargetOptions& options
);

} // namespace docc::plugins
