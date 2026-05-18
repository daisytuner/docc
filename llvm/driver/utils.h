#pragma once

#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <list>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#ifndef DOCC_LLVM_VERSION
#define DOCC_LLVM_VERSION "snapshot"
#endif

namespace docc {

enum DOCC_CI_LEVEL { DOCC_CI_LEVEL_NONE = 0, DOCC_CI_LEVEL_FULL, DOCC_CI_LEVEL_REGIONS, DOCC_CI_LEVEL_ARG_CAPTURE };

std::string getEnv(std::string const &key) {
    char *val = std::getenv(key.c_str());
    return val == NULL ? std::string("") : std::string(val);
};

void collect_args(std::vector<std::string> &collect_into, int argc, char *argv[]) {
    bool next_is_llvm = false;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-mllvm") {
            next_is_llvm = true;
            continue;
        }

        if (next_is_llvm) {
            collect_into.emplace_back("-mllvm=" + arg); // merge them to get rid of the difficult to parse double args
            next_is_llvm = false;
        } else {
            collect_into.emplace_back(arg);
        }
    }
}

std::vector<std::string> filter_non_docc_args(const std::vector<std::string> &args) {
    std::vector<std::string> filtered;
    for (const auto &item : args) {
        if (item.starts_with("-docc-")) continue;
        if (item.starts_with("-mllvm=-docc")) continue;
        if (item.starts_with("-Wl,-mllvm=-docc")) continue;
        filtered.emplace_back(item);
    }
    return filtered;
}

inline std::optional<std::filesystem::path> find_output_file(std::vector<std::string> &args) {
    for (auto it = args.begin(); it != args.end(); ++it) {
        auto &item = *it;
        if (item == "-o") {
            auto next = it + 1;
            if (next != args.end()) {
                return *next;
            }
        }
    }
    return {};
}

inline std::filesystem::path get_work_base_dir() {
    auto tmpBaseDirStr = getEnv("DOCC_TMP");
    std::filesystem::path tmpBaseDir;
    if (tmpBaseDirStr.empty()) {
        auto userName = getEnv("USER");
        tmpBaseDir = std::filesystem::path("/tmp") / userName / "DOCC";
    } else {
        tmpBaseDir = std::filesystem::path(tmpBaseDirStr);
    }
    return tmpBaseDir;
}

inline std::list<std::string> split_env(std::string env, char delim) {
    std::list<std::string> res;
    std::stringstream ss(std::move(env));
    std::string part;
    while (std::getline(ss, part, delim)) res.emplace_back(std::move(part));
    return res;
};

bool ends_with(const std::string &value, const std::string &ending) {
    return value.size() >= ending.size() && std::equal(ending.rbegin(), ending.rend(), value.rbegin());
}

template<typename RangeElem, typename Delim>
std::string str_join(const std::vector<RangeElem> &elems, const Delim &delim) {
    if (elems.empty()) return {};
    std::ostringstream oss;
    auto it = elems.begin();
    oss << *it++;
    for (; it != elems.end(); ++it) oss << delim << *it;
    return oss.str();
}

std::string find_file(const std::list<std::string> &search, const char *name) {
    for (const auto &path : search) {
        std::filesystem::path p(path);
        p /= name;
        if (std::filesystem::exists(p)) return p.string();
    }
    return {};
}

bool is_help(const std::vector<std::string> &args) {
    constexpr const char *trivial[] = {"--help", "-help"};
    for (auto t : trivial)
        if (std::find(args.begin(), args.end(), t) != args.end()) return true;
    return false;
}

bool is_version(const std::vector<std::string> &args) {
    constexpr const char *trivial[] = {"--version"};
    for (auto t : trivial)
        if (std::find(args.begin(), args.end(), t) != args.end()) return true;
    return false;
}

bool is_verbose(const std::vector<std::string> &args) {
    constexpr const char *trivial[] = {"-v", "--verbose"};
    for (auto t : trivial)
        if (std::find(args.begin(), args.end(), t) != args.end()) return true;
    return false;
}

/**
 * if given an opt like "-docc-link=" for a list of args containing "-docc-link=abc"
 * @return none, if not found
 */
inline std::optional<std::string> find_simple_opt(const std::vector<std::string> &args, const std::string &opt) {
    for (const auto &item : args) {
        if (item.starts_with(opt)) {
            return item.substr(opt.size());
        }
    }
    return {};
}

DOCC_CI_LEVEL ci_level() {
    std::string ci_env = getEnv("DOCC_CI");
    if (ci_env == "") {
        return DOCC_CI_LEVEL_NONE;
    } else if (ci_env == "regions") {
        return DOCC_CI_LEVEL_REGIONS;
    } else if (ci_env == "arg-capture") {
        return DOCC_CI_LEVEL_ARG_CAPTURE;
    } else {
        return DOCC_CI_LEVEL_FULL;
    }
}

int execvp_or_die(std::vector<std::string> &argv) {
    std::vector<char *> raw;
    raw.reserve(argv.size() + 1);
    for (auto &s : argv) raw.push_back(const_cast<char *>(s.c_str()));
    raw.push_back(nullptr);
    execvp(raw[0], raw.data());
    perror("execvp");
    return EXIT_FAILURE;
}

enum class DoccRootMode { None, CMake, Dist };

struct DoccPaths;

DoccPaths find_docc_paths();


static std::string plugin_filename = "libdocc_llvm_pass.so";

struct DoccPaths {
    DoccRootMode root_mode = DoccRootMode::None;
    std::filesystem::path docc_root_path;
    std::filesystem::path plugin_path;
    std::filesystem::path ld_path;

    [[nodiscard]] std::string docc_root_str() const {
        switch (root_mode) {
            case DoccRootMode::CMake:
                return "CMake:" + docc_root_path.string();
            case DoccRootMode::Dist:
                return "Dist:" + docc_root_path.string();
            default:
                return "";
        }
    }

    static std::filesystem::path cmake_plugin_path(const std::filesystem::path &root) {
        return root / plugin_filename;
    }

    static std::filesystem::path dist_plugin_path(const std::filesystem::path &root) {
        return root / "lib" / plugin_filename;
    }

    std::vector<std::filesystem::path> target_lib_paths() {
        switch (root_mode) {
            case DoccRootMode::CMake:
                return {docc_root_path / ".." / "rtl", docc_root_path / ".." / "arg-capture-io"};
            case DoccRootMode::Dist:
                return {docc_root_path / "lib"};
            default:
                return {};
        }
    }

    static DoccPaths from_root(const std::string_view &root) {
        auto idx = root.find(':');
        if (idx > 0) {
            auto type = root.substr(0, idx);
            std::filesystem::path path = root.substr(idx + 1);
            if (type == "CMake") {
                return {
                    .root_mode = DoccRootMode::CMake,
                    .docc_root_path = path,
                    .plugin_path = cmake_plugin_path(path),
                    .ld_path = path / "driver" / "docc-ld",
                };
            } else if (type == "Dist") {
                return {
                    .root_mode = DoccRootMode::CMake,
                    .docc_root_path = path,
                    .plugin_path = dist_plugin_path(path),
                    .ld_path = path / ".." / "bin" / "docc-ld",
                };
            } else {
                return find_docc_paths();
            }
        } else {
            return find_docc_paths();
        }
    }
};

inline std::filesystem::path self_path() {
    char path[1024 + 1];
    ssize_t length = readlink("/proc/self/exe", path, 1024);
    path[length] = '\0';
    return path;
}

inline DoccPaths find_docc_paths() {
    std::filesystem::path driver_path = self_path();
    std::filesystem::path docc_root_path;
    DoccRootMode root_mode = DoccRootMode::None;

    std::filesystem::path maybe_plugin_path;
    if (driver_path.has_root_path()) {
        docc_root_path = driver_path.parent_path().parent_path();
        root_mode = DoccRootMode::CMake;
        maybe_plugin_path = DoccPaths::cmake_plugin_path(docc_root_path);
    }
    if (!std::filesystem::exists(maybe_plugin_path)) {
        docc_root_path = driver_path.parent_path().parent_path() / "lib" / ("docc-llvm-" + std::string(DOCC_LLVM_VERSION));
        root_mode = DoccRootMode::Dist;
        maybe_plugin_path = DoccPaths::dist_plugin_path(docc_root_path);
    }
    if (maybe_plugin_path.empty() || !std::filesystem::exists(maybe_plugin_path)) {
        docc_root_path = "";
        root_mode = DoccRootMode::None;
        maybe_plugin_path = plugin_filename;
        std::cerr << "plugin: not found at usual places, expecting everything on env paths " << std::endl;
    }

    std::string ld_filename = "docc-ld";
    std::filesystem::path maybe_ld_path;
    if (driver_path.has_root_path()) {
        maybe_ld_path = driver_path.parent_path().parent_path() / ld_filename;
    }
    if (maybe_ld_path.empty() || !std::filesystem::exists(maybe_ld_path)) {
        maybe_ld_path = find_file(split_env(getEnv("PATH"), ':'), "docc-ld");
        ;
    }

    if (!std::filesystem::exists(maybe_ld_path)) {
        maybe_ld_path = "";
    }

    return {
        .root_mode = root_mode,
        .docc_root_path = docc_root_path,
        .plugin_path = maybe_plugin_path,
        .ld_path = maybe_ld_path,
    };
}

} // namespace docc
