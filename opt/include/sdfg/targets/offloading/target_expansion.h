#pragma once

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_node.h"

namespace sdfg {
namespace offloading {

class TargetLibNodeExpander {
public:
    TargetLibNodeExpander(sdfg::data_flow::LibraryNode& library_node) {};
    bool expand(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager);
};
} // namespace offloading
} // namespace sdfg
