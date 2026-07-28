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
        std::filesystem::create_directories(output_dir);

        auto fname = "graph_" + std::to_string(count);

        if (dump_dot_) {
            visualizer::DotVisualizer::writeToFile(builder.subject(), output_dir / (fname + ".sdfg.dot"));
            dumped = true;
        }
        if (dump_json_) {
            serializer::JSONSerializer::writeToFile(builder.subject(), output_dir / (fname + ".sdfg.json"));
            dumped = true;
        }
    }


    return false; // pipeline will infinitely re-execute if true
}

} // namespace sdfg::passes
