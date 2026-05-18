#include "docc/analysis/sdfg_registry.h"

#include <llvm/ADT/StringRef.h>
#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/IR/CFG.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/ModuleSummaryIndex.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/raw_ostream.h>

#include "docc/utils.h"

#include <cassert>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <nlohmann/json_fwd.hpp>
#include <sdfg/serializer/json_serializer.h>
#include <stdexcept>

using json = nlohmann::json;

static llvm::ExitOnError ExitOnErr{"[docc_llvm_plugin] error: "};

llvm::cl::opt<std::string>
    IndexFile("docc-combined-index", llvm::cl::init(""), llvm::cl::desc("Thin-link .index.bc output file"));

static llvm::cl::opt<std::string> LibIndexFile(
    "docc-combined-lib-index",
    llvm::cl::init(""),
    llvm::cl::desc("Attributes for external function from shared objects")
);

#define DEBUG_TYPE "docc"

namespace docc {
namespace analysis {

bool is_null_hash(const llvm::ModuleHash& Hash) {
    for (auto Component : Hash)
        if (Component != 0) return false;
    return true;
}

void SDFGRegistry::load(const llvm::Module& Module) {
    auto ext_dir = SDFGRegistry::docc_extract_dir(Module);
    if (ext_dir.empty()) {
        return;
    }
    if (!std::filesystem::exists(ext_dir / "JSON")) {
        return;
    }

    std::ifstream infile(ext_dir / "JSON");
    if (!infile.is_open()) {
        throw std::runtime_error("Failed to open JSON file");
    }
    json index_json;
    infile >> index_json;
    infile.close();

    for (auto& entry : index_json["sdfgs"]) {
        std::string sdfg_name_from_index = entry["name"];
        std::filesystem::path sdfg_file_entry = entry["file"];
        std::filesystem::path sdfg_filepath;

        if (!sdfg_file_entry.is_absolute()) {
            sdfg_filepath = ext_dir / sdfg_file_entry;
        }

        // Load .scheduled.json file
        sdfg_filepath.replace_extension(".scheduled.json");

        std::ifstream sdfg_file(sdfg_filepath);
        if (!sdfg_file.is_open()) {
            throw std::runtime_error("Failed to open file: " + sdfg_filepath.string());
        }

        // Read JSON
        json j;
        sdfg_file >> j;
        sdfg_file.close();

        // Deserialize SDFG
        sdfg::serializer::JSONSerializer serializer;
        std::unique_ptr<sdfg::StructuredSDFG> sdfg = serializer.deserialize(j);
        auto& forMod = this->registry_[Module.getName().str()];
        auto& sdfg_name = sdfg->name();
        forMod.emplace(sdfg_name, std::make_unique<SDFGHolder>(sdfg));

        // Read attributes
        json attributes_json = entry["attributes"];
        analysis::Attributes attributes = analysis::Attributes::from_json(attributes_json);
        this->attributes_[sdfg_name] = attributes;
    }
    infile.close();
}

std::unique_ptr<llvm::MemoryBuffer> load_file(llvm::StringRef Path) {
    auto FileOrErr = llvm::MemoryBuffer::getFile(Path);
    if (std::error_code EC = FileOrErr.getError()) {
        std::string File = Path.str();
        llvm::errs() << "Error reading file: " << File.c_str() << "\n";
        exit(1);
    }
    return std::move(*FileOrErr);
}

bool SDFGRegistry::is_link_time(AnalysisManager& AM) { return !IndexFile.empty(); }

std::filesystem::path SDFGRegistry::docc_extract_dir(const llvm::Module& Module) {
    if (auto* S = llvm::dyn_cast_or_null<llvm::MDString>(Module.getModuleFlag("docc.extract.dir"))) {
        std::filesystem::path docc_dir = S->getString().str();
        std::filesystem::path module_dir = docc_dir / utils::hash_module_name(Module);
        return module_dir;
    }
    return {};
}

void SDFGRegistry::insert(const llvm::Module& Module, std::list<std::unique_ptr<sdfg::StructuredSDFG>> sdfgs) {
    assert(!this->registry_.contains(Module.getName().str()) && "Module already registered");

    std::unordered_map<std::string, std::unique_ptr<SDFGHolder>> holders;
    for (auto& sdfg : sdfgs) {
        holders.emplace(sdfg->name(), std::make_unique<SDFGHolder>(sdfg));
    }
    this->registry_.emplace(Module.getName().str(), std::move(holders));
}

std::unique_ptr<llvm::Module> SDFGRegistry::get_module(const std::string& module_name, llvm::LLVMContext& ctx) {
    std::unique_ptr<llvm::MemoryBuffer> module_file = load_file(module_name);
    std::unique_ptr<llvm::Module> Module = ExitOnErr(llvm::parseBitcodeFile(module_file->getMemBufferRef(), ctx));
    return Module;
}

void SDFGRegistry::run(AnalysisManager& AM) {
    assert(this->registry_.empty() && "Registry should be empty before running analysis");

    // At compile-time, start with empty registry
    if (!SDFGRegistry::is_link_time(AM)) {
        return;
    }

    // This file is written to disk during thin-link stage
    std::unique_ptr<llvm::MemoryBuffer> File = load_file(IndexFile);
    if (!File) {
        throw std::runtime_error("Failed to load index file: " + IndexFile);
    }

    // Populate registry
    auto combined_index = ExitOnErr(llvm::getModuleSummaryIndex(File->getMemBufferRef()));
    for (const auto& entry : combined_index->modulePaths()) {
        if (is_null_hash(entry.second)) {
            continue;
        }

        llvm::StringRef module_path_str = entry.first();
        std::filesystem::path module_path = module_path_str.str();
        if (!std::filesystem::exists(module_path)) {
            continue;
        }

        std::unique_ptr<llvm::Module> Module = get_module(module_path, Ctx_);

        assert(
            this->registry_.find(Module->getName().str()) == this->registry_.end() &&
            "Module already loaded in registry"
        );
        std::unordered_map<std::string, std::unique_ptr<SDFGHolder>> holders;
        this->registry_.emplace(Module->getName().str(), std::move(holders));
        this->load(*Module);
    }

    this->combined_index_ = std::move(combined_index);

    if (!LibIndexFile.empty()) {
        // Load library index file
        std::ifstream stream(LibIndexFile);
        if (!stream.good()) {
            throw std::runtime_error("Failed to load library index file: " + LibIndexFile);
        }
        nlohmann::json json = nlohmann::json::parse(stream);

        // Populate externals
        assert(json.contains("sdfgs"));
        assert(json["sdfgs"].is_array());
        for (size_t i = 0; i < json["sdfgs"].size(); i++) {
            auto& sdfg = json["sdfgs"][i];
            assert(sdfg.is_object());
            assert(sdfg.contains("attributes"));
            assert(sdfg["attributes"].is_object());
            assert(sdfg["attributes"].contains("arguments"));
            assert(sdfg["attributes"]["arguments"].is_array());
            assert(sdfg.contains("file"));
            assert(sdfg["file"].is_string());
            assert(sdfg.contains("name"));
            assert(sdfg["name"].is_string());
            this->external_attributes_
                .insert({sdfg["name"].get<std::string>(), Attributes::from_json(sdfg["attributes"])});
        }
    } else {
        LLVM_DEBUG(llvm::dbgs() << "No libIndex file provided\n");
    }
}

void SDFGRegistry::dump_sdfgs(const llvm::Module& Module) {
    std::filesystem::path build_path = SDFGRegistry::docc_extract_dir(Module);
    std::ofstream index_stream(build_path / "JSON", std::ofstream::out | std::ofstream::trunc);
    if (!index_stream.is_open()) {
        throw std::runtime_error("Failed to open SDFG index JSON");
    }
    json index_json;
    index_json["sdfgs"] = json::array();

    size_t i = 0;
    auto& sdfgs = this->at(Module);
    for (auto& [key, holder] : sdfgs) {
        auto [lock, sdfg] = holder->get_for_modify();
        std::string file_name = "sdfg_" + std::to_string(i++) + ".json";
        std::filesystem::path sdfg_path = build_path / file_name;

        // Add sdfg file to metadata
        sdfg->add_metadata("sdfg_file", sdfg_path.string());

        // Add arg capture path metadata
        std::filesystem::path arg_capture_path = build_path / "arg_captures";
        sdfg->add_metadata("arg_capture_path", arg_capture_path.string());

        // Dump SDFG to file
        sdfg::serializer::JSONSerializer serializer;
        json sdfg_json = serializer.serialize(*sdfg);
        std::ofstream ofs(sdfg_path);
        if (!ofs.is_open()) {
            throw std::runtime_error("Failed to open file: " + sdfg_path.string());
        }
        ofs << sdfg_json.dump(2);
        ofs.close();

        // Add to JSON file
        json sdfg_entry;
        sdfg_entry["file"] = file_name;
        sdfg_entry["name"] = sdfg->name();
        index_json["sdfgs"].push_back(sdfg_entry);
    }
    index_stream << index_json.dump(2);
    index_stream.close();
}

SDFGHolder::SDFGHolder(std::unique_ptr<sdfg::StructuredSDFG>& sdfg) : current_(std::move(sdfg)) {}

void SDFGRegistry::for_each_sdfg_modifiable(
    std::unordered_map<std::string, std::unique_ptr<SDFGHolder>>& map,
    const std::function<void(SDFGHolder&, sdfg::StructuredSDFG&)>& action
) {
    for (auto& [name, holderp] : map) {
        auto& holder = *holderp;
        std::unique_lock lock(holder.current_mutex_);

        // space to create copies of original SDFG if needed

        action(holder, *holder.current_);
    }
}

void SDFGRegistry::for_each_sdfg_const(
    std::unordered_map<std::string, std::unique_ptr<SDFGHolder>>& map,
    const std::function<void(SDFGHolder&, const sdfg::StructuredSDFG&)>& action
) {
    for (auto& [name, holderp] : map) {
        auto& holder = *holderp;
        std::shared_lock lock(holder.current_mutex_);

        action(holder, *holder.current_);
    }
}

std::tuple<std::unique_lock<std::shared_mutex>, sdfg::StructuredSDFG*> SDFGHolder::get_for_modify() {
    std::unique_lock<std::shared_mutex> lock(this->current_mutex_);

    // maybe create a copy of the original here, if it does not exist already. But also need to modify the for_each
    // calls in registry itself

    return {std::move(lock), &*this->current_};
}

std::tuple<std::shared_lock<std::shared_mutex>, const sdfg::StructuredSDFG*> SDFGHolder::get_for_read() {
    std::shared_lock<std::shared_mutex> lock(this->current_mutex_);

    return {std::move(lock), &*this->current_};
}

} // namespace analysis
} // namespace docc
