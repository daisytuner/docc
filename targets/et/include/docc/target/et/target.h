#pragma once
#include <sdfg/data_flow/data_flow_graph.h>

#include <sdfg/analysis/analysis.h>
#include <sdfg/builder/structured_sdfg_builder.h>
#include <sdfg/plugins/plugins.h>
#include <sdfg/plugins/targets.h>

namespace docc::target::et {

extern DoccTarget et_target;

inline sdfg::data_flow::ImplementationType ImplementationType_ETSOC_WithTransfers{"ETSOC_WithTransfers"};
inline sdfg::data_flow::ImplementationType ImplementationType_ETSOC_WithoutTransfers{"ETSOC_WithoutTransfers"};

inline const std::string ETSOC_KERNEL_FILE_EXT = "et.cpp";

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

struct EtBuildArgs {
    const std::filesystem::path& build_dir;
    const std::filesystem::path& plugin_rt_dir;
};

std::filesystem::path et_build_kernel(
    const sdfg::StructuredSDFG& sdfg,
    const sdfg::codegen::CodeSnippetFactory& snippet_factory,
    const std::filesystem::path& kernel_src,
    const EtBuildArgs& paths
);

} // namespace docc::target::et
