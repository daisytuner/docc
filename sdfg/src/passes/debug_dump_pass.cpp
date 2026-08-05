#include "sdfg/passes/debug_dump_pass.h"

#include "sdfg/serializer/json_serializer.h"
#include "sdfg/visualizer/dot_visualizer.h"

namespace sdfg::passes {

DebugDumpPass::DebugDumpPass(const std::string& name, bool dump_json, bool dump_dot)
    : name_(name), dump_json_(dump_json), dump_dot_(dump_dot) {}

bool DebugDumpPass::run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    bool dumped = false;

    auto dir = builder.subject().metadata_if_exists("output_dir");
    if (dir) {
        auto count = counter_++;
        std::filesystem::path path = *dir;

        auto output_dir = path / ("debug_dump_" + name_);

        auto fname = "graph_" + std::to_string(count);

        dump(builder.subject(), output_dir, fname, dump_json_, dump_dot_);
    }


    return false; // pipeline will infinitely re-execute if true
}

bool DebugDumpPass::dump(const StructuredSDFG& sdfg, const std::string& type, bool dump_json, bool dump_dot) {
    auto* dir = sdfg.metadata_if_exists("output_dir");

    if (dir) {
        std::filesystem::path build_path(*dir);
        return dump(sdfg, build_path, type, dump_json, dump_dot);
    } else {
        return false;
    }
}

bool DebugDumpPass::dump(
    const StructuredSDFG& sdfg, const std::filesystem::path& dir, const std::string& type, bool dump_json, bool dump_dot
) {
    std::filesystem::path build_path(dir);
    if (!std::filesystem::exists(build_path)) {
        std::filesystem::create_directories(build_path);
    }

    // Add metadata to SDFG
    auto typeSuffix = type.empty() ? "" : ("." + type);
    auto suffixedName = sdfg.name() + typeSuffix;

    if (dump_json) {
        std::filesystem::path sdfg_file = build_path / (suffixedName + ".sdfg.json");

        // Dump json
        serializer::JSONSerializer::writeToFile(sdfg, sdfg_file);
    }

    if (dump_dot) {
        auto dot_file = build_path / (suffixedName + ".sdfg.dot");
        sdfg::visualizer::DotVisualizer::writeToFile(sdfg, &dot_file);
    }

    return dump_dot || dump_json;
}

} // namespace sdfg::passes
