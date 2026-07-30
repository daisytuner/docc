/**
 * post build analysis tool that consumes the index file and can enumarate SDFGs produces etc.
 */
#include <docc/analysis/analysis.h>
#include <docc/analysis/global_cfg_analysis.h>
#include <docc/analysis/sdfg_registry.h>
#include <docc/passes/dumps/global_cfg_dump_pass.h>
#include <llvm/Support/CommandLine.h>

#include <iostream>
#include <regex>

#include "docc/cmd_args.h"
#include "docc/docc.h"
#include "sdfg/visualizer/dot_visualizer.h"

llvm::cl::opt<std::string> TASK(llvm::cl::Positional, llvm::cl::desc("cfg | list"), llvm::cl::init(""));
llvm::cl::opt<std::string> FILTER("filter", llvm::cl::desc("Regex to filter modules"), llvm::cl::init(""));
llvm::cl::opt<std::string> SDFG_FILE("sdfg", llvm::cl::desc("SDFG file to analyze"), llvm::cl::init(""));

int main(int argc, char** argv) {
    llvm::cl::ParseCommandLineOptions(argc, argv);

    docc::register_sdfg_dispatchers();

    docc::analysis::AnalysisManager analysis_manager;

    if (TASK == "dot") {
        std::filesystem::path file = SDFG_FILE.getValue();
        std::cout << "Generating Dot Graph from " << file << std::endl;

        if (!std::filesystem::exists(file)) {
            std::cerr << "File " << file << " does not exist!" << std::endl;
            return 1;
        }

        if (file.extension() == ".json") {
            auto dir = file.parent_path();
            std::string basename = file.stem().string();

            std::ifstream stream(file);
            if (!stream.good()) {
                throw std::runtime_error("Failed to open json");
            }
            nlohmann::json json = nlohmann::json::parse(stream);
            sdfg::serializer::JSONSerializer js;
            auto sdfg = js.deserialize(json);
            sdfg::visualizer::DotVisualizer::writeToFile(*sdfg, dir / (basename + ".sdfg.dot"));
        } else {
            std::cerr << "File " << file << " is not a json file!" << std::endl;
            return 1;
        }
    } else if (TASK == "cfg") {
        std::cout << "Generating Global CFG..." << std::endl;

        auto& glob = analysis_manager.get<docc::analysis::GlobalCFGAnalysis>();

        std::string* path_opt = docc::DOCC_DUMP_GLBL_CFG.getValue().empty() ? nullptr
                                                                            : &docc::DOCC_DUMP_GLBL_CFG.getValue();

        docc::passes::GlobalCFGPrinterPass globPrinter(path_opt);

        auto& sdfgRegistry = analysis_manager.get<docc::analysis::SDFGRegistry>();

        std::regex filter(FILTER);

        llvm::LLVMContext ctx;

        for (const auto& modName : sdfgRegistry.get_known_modules()) {
            if (std::regex_match(modName, filter)) {
                std::cout << "Module " << modName << " matches filter!" << std::endl;
                auto mod = sdfgRegistry.get_module(modName, ctx);
                if (path_opt) {
                    globPrinter.dumpToDotFile(*path_opt, *mod, glob);
                } else {
                    globPrinter.dumpToConsole(*mod, glob);
                }
            }
        }

    } else {
        std::cout << "Listing all SDFGs:" << std::endl;

        auto& sdfgRegistry = analysis_manager.get<docc::analysis::SDFGRegistry>();

        for (auto& name : sdfgRegistry.get_known_modules()) {
            std::cout << "Module " << name << ":" << std::endl;
            for (auto& [nam, sdfgHolder] : sdfgRegistry.at(name)) {
                std::cout << "\t" << "SDFG: " << nam << std::endl;
            }
        }
    }

    return 0;
}
