#pragma once

namespace docc::plugins {
struct TargetOptions;
}

namespace docc::compile {
class SrcFileCompilerBuilder;
}

namespace docc::target {

struct DoccTarget {
    std::string short_name;

    bool (*apply_additional_compile_options)(docc::compile::SrcFileCompilerBuilder& src_compile_builder);

    /**
     * Consider these hooks experimental. They allow completely custom modification of the SDFG.
     * In the future we may want to move more towards pass-management
     */
    bool (*apply_expand_time_mapping)(
        sdfg::builder::StructuredSDFGBuilder& builder,
        sdfg::analysis::AnalysisManager& analysis_manager,
        const plugins::TargetOptions& options
    );
    bool (*apply_sched_time_mapping)(
        sdfg::builder::StructuredSDFGBuilder& builder,
        sdfg::analysis::AnalysisManager& analysis_manager,
        const plugins::TargetOptions& options
    );
};

} // namespace docc::target
