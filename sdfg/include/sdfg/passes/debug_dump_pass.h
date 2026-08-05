#pragma once
#include "sdfg/passes/pass.h"

namespace sdfg::passes {

class DebugDumpPass : public sdfg::passes::Pass {
private:
    std::string name_;
    bool dump_json_;
    bool dump_dot_;
    size_t counter_ = 0;

public:
    DebugDumpPass(const std::string& name, bool dump_json = true, bool dump_dot = true);

    std::string name() override { return "DebugDumpPass"; }

    bool run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    static bool dump(const StructuredSDFG& sdfg, const std::string& type, bool dump_json = true, bool dump_dot = true);
    static bool dump(
        const StructuredSDFG& sdfg,
        const std::filesystem::path& dir,
        const std::string& type,
        bool dump_json = true,
        bool dump_dot = true
    );
};


} // namespace sdfg::passes
