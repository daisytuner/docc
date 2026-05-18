#pragma once

#include <csignal>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <regex>
#include <string>

#include <llvm/Analysis/RegionInfo.h>
#include <llvm/BinaryFormat/Dwarf.h>
#include <llvm/Demangle/Demangle.h>
#include <llvm/IR/DebugInfoMetadata.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/IntrinsicInst.h>
#include <llvm/IR/Module.h>

#include <llvm/Support/Debug.h>

#include <sdfg/element.h>

#ifndef NDEBUG
#define LLVM_DEBUG_PRINTLN(msg)                    \
    do {                                           \
        llvm::errs() << "[DEBUG] " << msg << "\n"; \
    } while (0)
#define LLVM_DEBUG_PRINT(msg)              \
    do {                                   \
        llvm::errs() << "[DEBUG] " << msg; \
    } while (0)
#else
#define LLVM_DEBUG_PRINTLN(msg) \
    do {                        \
    } while (0)
#define LLVM_DEBUG_PRINT(msg) \
    do {                      \
    } while (0)
#endif

namespace docc {

enum TargetType { HOST, CUDA };

namespace utils {

inline std::string hash_function_name(const std::string& function_name) {
    std::hash<std::string> hash_fn;
    size_t hash_value = hash_fn(function_name);

    return std::to_string(hash_value);
};

inline std::string hash_module_name(const llvm::Module& Module) {
    std::string srcName = Module.getSourceFileName();
    std::string absPath = std::filesystem::absolute(srcName).string();

    std::hash<std::string> hash_fn;
    size_t hash_value = hash_fn(absPath);
    std::string stable_hash = std::to_string(hash_value);

    return stable_hash;
};

inline bool is_mangled_name(const std::string& name) {
    // Regex patterns for common mangling styles
    std::regex itaniumPattern("^_Z");

    return std::regex_search(name, itaniumPattern);
};

inline std::string get_demangled_name(const std::string& name) {
    if (is_mangled_name(name)) {
        char* demangled = llvm::itaniumDemangle(name.c_str());
        if (demangled) {
            std::string full_name(demangled);
            std::free(demangled);
            return full_name;
        }
    }
    return name;
};

inline std::string get_truncated_demangled_name(const std::string& name) {
    if (is_mangled_name(name)) {
        char* demangled = llvm::itaniumDemangle(name.c_str());
        if (demangled) {
            std::string full_name(demangled);
            std::free(demangled);
            size_t pos = full_name.find("(");
            assert(pos != std::string::npos);
            std::string sdfg_name = full_name.substr(0, pos);
            return sdfg_name;
        }
    }
    return name;
};

inline llvm::dwarf::SourceLanguage source_language(llvm::Module& Module) {
    if (auto* compile_unit = Module.getNamedMetadata("llvm.dbg.cu")) {
        if (auto* Node = compile_unit->getOperand(0)) {
            if (auto* CUNode = llvm::dyn_cast<llvm::DICompileUnit>(Node)) {
                unsigned lang = CUNode->getSourceLanguage();
                switch (lang) {
                    case llvm::dwarf::DW_LANG_C:
                    case llvm::dwarf::DW_LANG_C89:
                    case llvm::dwarf::DW_LANG_C99:
                    case llvm::dwarf::DW_LANG_C11: {
                        return llvm::dwarf::DW_LANG_C;
                    }
                    case llvm::dwarf::DW_LANG_C_plus_plus:
                    case llvm::dwarf::DW_LANG_C_plus_plus_03:
                    case llvm::dwarf::DW_LANG_C_plus_plus_11:
                    case llvm::dwarf::DW_LANG_C_plus_plus_14: {
                        return llvm::dwarf::DW_LANG_C_plus_plus;
                    }
                    default:
                        break;
                }
            }
        }
    }

    return llvm::dwarf::DW_LANG_C_plus_plus;
};

inline bool has_debug(const llvm::Module& Module) {
    for (const auto& F : Module) {
        if (F.getSubprogram() != nullptr) {
            return true;
        }
    }
    for (const auto& N : Module.named_metadata()) {
        if (N.getName().starts_with("llvm.dbg.")) {
            return true;
        }
    }
    return false;
};

inline sdfg::DebugInfo get_debug_info(const llvm::Instruction& inst) {
    // Retrieve the attached debug location (if any)
    llvm::DebugLoc dbg = inst.getDebugLoc();
    if (!dbg || dbg->getLine() == 0) {
        return sdfg::DebugInfo();
    }

    // LLVM 16+ stores the directory as part of the DIFile object.  Earlier versions
    // provided helper accessors on the location itself.  Use the DIFile object to
    // build an absolute path that works across all supported LLVM versions.
    const llvm::DILocation* loc = dbg.get();


    // Get the file name
    const llvm::DIFile* file = loc->getFile();
    if (!file) {
        return sdfg::DebugInfo();
    }
    llvm::StringRef directory = file->getDirectory();
    llvm::StringRef filename_ref = file->getFilename();
    std::filesystem::path filepath;
    if (!directory.empty()) {
        filepath = std::filesystem::path(directory.str()) / filename_ref.str();
    } else {
        filepath = std::filesystem::path(filename_ref.str());
    }

    // Make sure we operate on an absolute path so downstream components can rely
    // on a stable file identifier irrespective of the current working directory.
    std::string filename = std::filesystem::absolute(filepath).lexically_normal().string();

    // Get the line and column
    size_t line = loc->getLine();
    size_t col = loc->getColumn();

    // Get the subprogram
    std::string subprogram_name;
    const auto* LS = llvm::dyn_cast<llvm::DILocalScope>(loc->getScope());
    if (LS) {
        const llvm::DISubprogram* SP = LS->getSubprogram();
        if (SP) {
            if (!SP->getName().empty()) {
                // strip .extracted if it exists
                std::string name = SP->getLinkageName().str();
                if (name.empty()) {
                    name = SP->getName().str();
                }
                size_t pos = name.find(".extracted");
                if (pos != std::string::npos) {
                    name = name.substr(0, pos);
                }
                subprogram_name = get_demangled_name(name);
            }
        }
    }

    return sdfg::DebugInfo(filename, subprogram_name, line, col, line, col);
};

inline sdfg::DebugInfo get_debug_info(llvm::BasicBlock& block) {
    sdfg::DebugInfo dbg_info = docc::utils::get_debug_info(block.front());
    for (auto& inst : block) {
        auto inst_dbg_info = docc::utils::get_debug_info(inst);
        dbg_info = dbg_info.merge(inst_dbg_info, dbg_info);
    }
    return dbg_info;
};

inline sdfg::DebugInfo get_debug_info(llvm::Function& function) {
    sdfg::DebugInfo dbg_info = docc::utils::get_debug_info(function.getEntryBlock());
    for (auto& block : function) {
        auto block_dbg_info = docc::utils::get_debug_info(block);
        dbg_info = dbg_info.merge(block_dbg_info, dbg_info);
    }
    return dbg_info;
};

inline sdfg::DebugInfo get_debug_info(llvm::Region& region) {
    sdfg::DebugInfo dbg_info;
    for (auto block : region.blocks()) {
        auto block_dbg_info = docc::utils::get_debug_info(*block);
        dbg_info = dbg_info.merge(block_dbg_info, dbg_info);
    }
    return dbg_info;
};

// Grab a DebugLoc from the value itself (if Instruction), or from one of its users.
inline sdfg::DebugInfo bestEffortLoc(const llvm::Value& V) {
    if (auto* I = llvm::dyn_cast<llvm::Instruction>(&V)) return get_debug_info(*I);

    // Try users that are instructions with a location (common when V is an Argument or Constant).
    for (const llvm::User* U : V.users())
        if (auto* I = llvm::dyn_cast<llvm::Instruction>(U))
            if (auto DL = I->getDebugLoc()) return get_debug_info(*I);

    return sdfg::DebugInfo(); // nothing found
}

inline std::vector<std::string> get_recorded_driver_flags(const llvm::Module& M) {
    // Warning, this s broken. llvm.commandline may already contain additional lines from other modules due to thin-LTO
    // It makes no sense to merge them all, including multiple compiler-executables
    std::vector<std::string> Flags;

    if (auto* CMD = M.getNamedMetadata("llvm.commandline"))
        for (llvm::MDNode* Op : CMD->operands()) {
            auto* S = llvm::dyn_cast<llvm::MDString>(Op->getOperand(0));
            if (S) {
                // Split the recorded command line into individual flags
                std::string cmd = S->getString().str();
                std::stringstream ss(cmd);
                std::string part;
                while (std::getline(ss, part, ' ')) {
                    Flags.emplace_back(part);
                }
            }
        }
    return Flags;
}

template<typename T>
inline std::string toIRString(const T& value) {
    std::string result;
    llvm::raw_string_ostream os(result);
    value.print(os);
    return os.str();
}

} // namespace utils
} // namespace docc
