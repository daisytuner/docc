#include <dlfcn.h>

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <list>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "utils.h"

static bool is_preprocess_only(const std::vector<std::string> &args) {
    return std::find(args.begin(), args.end(), "-E") != args.end() &&
           std::find(args.begin(), args.end(), "-dM") != args.end();
}

static bool is_driver_dump(const std::vector<std::string> &args) {
    return std::find(args.begin(), args.end(), "-###") != args.end();
}

static bool is_cmake_compiler_id(const std::vector<std::string> &args) {
    for (const auto &a : args)
        if (a.find("CMakeCCompilerId") != std::string::npos || a.find("CMakeCXXCompilerId") != std::string::npos)
            return true;
    return false;
}

static bool is_docc_noop(const std::vector<std::string> &args) {
    return std::any_of(args.begin(), args.end(), [](const std::string &arg) { return arg == "-docc-noop"; });
}

static std::string add_docc_work_dir(std::vector<std::string> &args, bool alsoLinking = true) {
    const std::string flag = "-docc-work-dir=";

    for (auto it = args.begin(); it != args.end(); ++it) {
        auto &item = *it;
        if (item.starts_with(flag)) {
            auto wdir = item.substr(flag.size());
            if (alsoLinking) {
                args.emplace_back("-Wl,-mllvm=" + item);
            }
            args.emplace_back("-docc-save-temps");
            *it = "-mllvm=" + item; // rewrite to mllvm form

            return wdir;
            // all set.
        }
    }

    auto baseDir = docc::get_work_base_dir();

    auto pid = std::to_string(getpid());
    auto oFile = docc::find_output_file(args);

    std::filesystem::path wDir;
    if (oFile) {
        wDir = baseDir / (oFile.value().filename().string() + "-" + pid);
    } else {
        wDir = baseDir / pid;
    }

    args.emplace_back("-mllvm=" + flag + "'" + wDir.string() + "'");
    if (alsoLinking) {
        args.emplace_back("-Wl,-mllvm=-docc-work-dir=" + wDir.string());
    }
    return wDir.string();
}

static void forward_docc_args(std::vector<std::string> &args, bool alsoLinking = true) {
    for (auto it = args.begin(); it != args.end(); ++it) {
        auto item = *it;
        if (item.starts_with("-docc-")) {
            auto llvmArg = "-mllvm=" + item;
            *it = llvmArg;
            if (alsoLinking) {
                it = args.insert(it + 1, "-Wl," + llvmArg);
            }
        }
    }
}

static bool is_link_step(const std::vector<std::string> &args) {
    // Compile‑only options that suppress linking.
    constexpr const char *compile_only_flags[] = {"-c", "-S", "-E", "-emit-llvm"};
    for (auto f : compile_only_flags)
        if (std::find(args.begin(), args.end(), f) != args.end()) return false;
    return true;
}

// ──────────────────────────────────────────────────────────────────────────────
// main
// ──────────────────────────────────────────────────────────────────────────────
int main(int argc, char *argv[]) {
    const bool cpp_mode = docc::ends_with(argv[0], "docc-cpp");
    const std::string cc = cpp_mode ? "clang++-19" : "clang-19";

    std::vector<std::string> cmd{cc};
    docc::collect_args(cmd, argc, argv);

    // ── Fast‑path bypasses ────────────────────────────────────────────────────
    if (docc::is_version(cmd)) {
        std::cout << "docc version: " << DOCC_LLVM_VERSION << std::endl;
        return EXIT_SUCCESS;
    }
    if (docc::is_help(cmd) || is_preprocess_only(cmd) || is_driver_dump(cmd) || is_cmake_compiler_id(cmd))
        return docc::execvp_or_die(cmd);

    if (is_docc_noop(cmd)) { // coompletely skip ANY of our processing, just forward to clang
        auto clangArgs = docc::filter_non_docc_args(cmd);
        return docc::execvp_or_die(clangArgs);
    }

    auto docc_paths = docc::find_docc_paths();
    if (docc_paths.ld_path.empty()) {
        std::cerr << "error: docc-ld not found with docc or on PATH!" << std::endl;
        return EXIT_FAILURE;
    }

    std::string plugin_path = docc_paths.plugin_path.string();
    // ── Inject plugin arguments ─────────────────────────────────────
    cmd.insert(cmd.begin() + 1, "-fplugin=" + plugin_path);
    cmd.insert(cmd.begin() + 1, "-fpass-plugin=" + plugin_path);

    bool containsLinking = is_link_step(cmd);

    // ── Inject extraction directory ───────────────────────────────
    auto docc_work_dir = add_docc_work_dir(cmd, containsLinking);
    forward_docc_args(cmd, containsLinking);

    // ── Inject command-line recording ───────────────────────────────
    cmd.insert(cmd.begin() + 1, "-frecord-command-line");

    // ── LTO setup (Global Optimization) ───────────────────────────────────────

    // Filter out lto flags from the command line
    cmd.erase(std::remove(cmd.begin(), cmd.end(), "-flto=auto"), cmd.end());
    cmd.erase(std::remove(cmd.begin(), cmd.end(), "-ffat-lto-objects"), cmd.end());


    cmd.emplace_back("-flto=thin");
    if (is_link_step(cmd)) {
        auto docc_root = docc_paths.docc_root_str();
        if (!docc_root.empty()) {
            if (containsLinking) {
                cmd.emplace_back("-Wl,-mllvm=-docc-root='" + docc_root + "'");
            }
            cmd.emplace_back("-mllvm=-docc-root='" + docc_root + "'");
        }
        cmd.insert(cmd.begin() + 1, "-fuse-ld=" + docc_paths.ld_path.string());
    }

    // ── Execute ───────────────────────────────────────────────────────────────
    return docc::execvp_or_die(cmd);
}
