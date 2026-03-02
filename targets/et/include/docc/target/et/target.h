#pragma once
#include <sdfg/data_flow/data_flow_graph.h>

#include <sdfg/analysis/analysis.h>
#include <sdfg/builder/structured_sdfg_builder.h>
#include <sdfg/plugins/plugins.h>

namespace docc::target::et {

inline sdfg::data_flow::ImplementationType ImplementationType_ETSOC_WithTransfers{"ETSOC_WithTransfers"};
inline sdfg::data_flow::ImplementationType ImplementationType_ETSOC_WithoutTransfers{"ETSOC_WithoutTransfers"};

void register_plugin(sdfg::plugins::Context& context);

void et_scheduling_passes(
    sdfg::builder::StructuredSDFGBuilder& builder,
    sdfg::analysis::AnalysisManager& analysis_manager,
    const std::string& category
);

std::string et_get_host_additional_compile_args(
    const sdfg::StructuredSDFG&, const sdfg::codegen::CodeSnippetFactory& snippet_factory
);

std::string
et_get_host_additional_link_args(const sdfg::StructuredSDFG&, const sdfg::codegen::CodeSnippetFactory& snippet_factory);

} // namespace docc::target::et
