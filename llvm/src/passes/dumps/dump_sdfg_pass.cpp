#include "docc/passes/dumps/dump_sdfg_pass.h"

#include <sdfg/serializer/json_serializer.h>

#include "docc/analysis/sdfg_registry.h"
#include "sdfg/visualizer/dot_visualizer.h"

namespace docc {
namespace passes {

void DumpSDFGPass::dump(const sdfg::StructuredSDFG& sdfg, const std::string& stage, bool dump_visualization) {
    sdfg::serializer::JSONSerializer serializer;
    nlohmann::json j = serializer.serialize(sdfg);

    // Overwrite the SDFG file specified in the metadata
    std::filesystem::path sdfg_path = sdfg.metadata("sdfg_file");
    if (!stage.empty()) {
        sdfg_path.replace_filename(sdfg_path.stem().string() + "." + stage + sdfg_path.extension().string());
    }
    std::ofstream ofs(sdfg_path);
    if (!ofs.is_open()) {
        throw std::runtime_error("Failed to open file: " + sdfg_path.string());
    }
    ofs << j.dump(2);
    ofs.close();
    if (dump_visualization) {
        std::filesystem::path dot_path = sdfg_path.replace_extension(".dot");
        sdfg::visualizer::DotVisualizer::writeToFile(sdfg, &dot_path);
    }
}

llvm::PreservedAnalyses DumpSDFGPass::
    run(llvm::Module& Module, llvm::ModuleAnalysisManager& MAM, analysis::AnalysisManager& AM) {
    auto& registry = AM.get<analysis::SDFGRegistry>();
    if (!registry.has_module(Module)) {
        return llvm::PreservedAnalyses::all();
    }

    registry.for_each_sdfg_const(Module, [&](const sdfg::StructuredSDFG& sdfg) {
        dump(sdfg, stage_, dump_visualization_);
    });

    return llvm::PreservedAnalyses::all();
}

} // namespace passes
} // namespace docc
