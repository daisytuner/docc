#pragma once

#include <shared_mutex>
#include <unordered_map>

#include <llvm/ADT/StringRef.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/ModuleSummaryIndex.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/raw_ostream.h>

#include <sdfg/structured_sdfg.h>

#include "docc/analysis/analysis.h"
#include "docc/analysis/attributes.h"

extern llvm::cl::opt<std::string> IndexFile;

namespace docc {
namespace analysis {

template<typename T>
inline T& indirect(const std::unique_ptr<T>& ptr) {
    return *ptr;
};

class GlobalCFGAnalysis;

class SDFGRegistry;

/**
 * This a wrapper around 1 SDFG (may include historic versions of the same SDFG in the future).
 * It is to ensure uncorrupted access to the SDFG without removing its reference from the index (as before, with unique
 * pointers only). If you cannot handle the lock-guard around the SDFG reference itself, reference the holder instead
 *
 * Allows multiple read-accesses in parallel, only modification requires an exclusive lock
 * Reads cannot guarantee you'll see a specific version, but you will see a consistent snapshot of its latest, complete
 * version
 */
class SDFGHolder {
    friend class SDFGRegistry;

private:
    std::unique_ptr<sdfg::StructuredSDFG> current_;
    std::shared_mutex current_mutex_;

public:
    explicit SDFGHolder(std::unique_ptr<sdfg::StructuredSDFG>& sdfg);

    std::tuple<std::unique_lock<std::shared_mutex>, sdfg::StructuredSDFG*> get_for_modify();

    std::tuple<std::shared_lock<std::shared_mutex>, const sdfg::StructuredSDFG*> get_for_read();
};

class SDFGRegistry : public Analysis {
    friend class GlobalCFGAnalysis;

private:
    llvm::LLVMContext Ctx_;

    std::unordered_map<std::string, std::unordered_map<std::string, std::unique_ptr<SDFGHolder>>> registry_;

    std::unique_ptr<llvm::ModuleSummaryIndex> combined_index_;

    std::unordered_map<std::string, Attributes> attributes_;

    std::unordered_map<std::string, Attributes> external_attributes_;

    void load(const llvm::Module& Module);

public:
    SDFGRegistry() = default;
    SDFGRegistry(const SDFGRegistry&) = delete;
    SDFGRegistry& operator=(const SDFGRegistry&) = delete;

    bool has_module(const llvm::Module& Module) const { return this->registry_.contains(Module.getName().str()); }

    std::vector<std::string> get_known_modules() {
        std::vector<std::string> modules;
        for (const auto& [mod_name, _] : this->registry_) {
            modules.push_back(mod_name);
        }
        return modules;
    }

    /**
     * Read the module from its file
     */
    std::unique_ptr<llvm::Module> get_module(const std::string& module_name, llvm::LLVMContext& ctx);

    std::unordered_map<std::string, std::unique_ptr<SDFGHolder>>& at(const std::string& ModuleName) {
        auto& perMod = this->registry_.at(ModuleName);

        return perMod;
    }

    std::unordered_map<std::string, std::unique_ptr<SDFGHolder>>& at(const llvm::StringRef& ModuleName) {
        return at(ModuleName.str());
    }

    bool has_function(const std::string& name) const { return this->attributes_.contains(name); }

    bool has_external_function(const std::string& name) const { return this->external_attributes_.contains(name); }

    std::unordered_map<std::string, std::unique_ptr<SDFGHolder>>& at(const llvm::Module& Module) {
        return at(Module.getName());
    }

    const Attributes& attributes(const std::string& name) const { return this->attributes_.at(name); }

    const Attributes& external_attributes(const std::string& name) const { return this->external_attributes_.at(name); }

    void insert(const llvm::Module& Module, std::list<std::unique_ptr<sdfg::StructuredSDFG>> sdfgs);

    void dump_sdfgs(const llvm::Module& Module);

    void run(AnalysisManager& WPAM);

    static bool is_link_time(AnalysisManager& WPAM);

    static std::filesystem::path docc_extract_dir(const llvm::Module& Module);

    static void for_each_sdfg_modifiable(
        std::unordered_map<std::string, std::unique_ptr<SDFGHolder>>& map,
        const std::function<void(SDFGHolder&, sdfg::StructuredSDFG&)>& action
    );

    void for_each_sdfg_modifiable(
        const llvm::Module& Module, const std::function<void(SDFGHolder&, sdfg::StructuredSDFG&)>& action
    ) {
        for_each_sdfg_modifiable(at(Module), action);
    }

    void for_each_sdfg_modifiable(const llvm::Module& Module, const std::function<void(sdfg::StructuredSDFG&)>& action) {
        for_each_sdfg_modifiable(at(Module), [&action](SDFGHolder& holder, sdfg::StructuredSDFG& sdfg) {
            action(sdfg);
        });
    }

    static void for_each_sdfg_const(
        std::unordered_map<std::string, std::unique_ptr<SDFGHolder>>& map,
        const std::function<void(SDFGHolder&, const sdfg::StructuredSDFG&)>& action
    );

    void for_each_sdfg_const(
        const llvm::Module& Module, const std::function<void(SDFGHolder&, const sdfg::StructuredSDFG&)>& action
    ) {
        for_each_sdfg_modifiable(at(Module), action);
    }

    void for_each_sdfg_const(const llvm::Module& Module, const std::function<void(const sdfg::StructuredSDFG&)>& action) {
        for_each_sdfg_const(at(Module), [&action](SDFGHolder& holder, const sdfg::StructuredSDFG& sdfg) {
            action(sdfg);
        });
    }
};

} // namespace analysis
} // namespace docc
