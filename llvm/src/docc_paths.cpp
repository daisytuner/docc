#include "docc/docc_paths.h"
#include <llvm/Support/Debug.h>

#include "docc/cmd_args.h"

#include <filesystem>
#include <memory>


namespace docc::utils {

std::vector<std::filesystem::path> DoccPaths::target_inc_paths() const {
    switch (root_mode) {
        case DoccRootMode::CMake:
            return {
                docc_root_path / ".." / "docc" / "rtl" / "include",
                docc_root_path / ".." / "docc" / "arg-capture-io" / "include"
            };
        case DoccRootMode::Dist:
            return {docc_root_path / "include"};
        default:
            return {};
    }
}

DoccPaths DoccPaths::from_root(const std::string_view &root) {
    std::string_view stripped_root = root;
    if (stripped_root.size() >= 2) {
        if ((stripped_root.starts_with('"') && stripped_root.ends_with('"')) ||
            (stripped_root.starts_with('\'') && stripped_root.ends_with('\''))) {
            stripped_root = stripped_root.substr(1, stripped_root.size() - 2);
        }
    }

    auto idx = stripped_root.find(':');
    if (idx > 0) {
        auto type = stripped_root.substr(0, idx);
        std::filesystem::path path = stripped_root.substr(idx + 1);
        if (type == "CMake") {
            return {
                .root_mode = DoccRootMode::CMake,
                .docc_root_path = path,
            };
        } else if (type == "Dist") {
            return {
                .root_mode = DoccRootMode::Dist,
                .docc_root_path = path,
            };
        } else {
            LLVM_DEBUG_PRINTLN("DOCC Root undefined. Expecting resources on paths");
            return {
                .root_mode = DoccRootMode::None,
                .docc_root_path = "",
            };
        }
    } else {
        LLVM_DEBUG_PRINTLN("Invalid DOCC Root. Expecting resources on paths");
        return {
            .root_mode = DoccRootMode::None,
            .docc_root_path = "",
        };
    }
}

DoccPaths DoccPaths::get_instance() {
    auto root = DOCC_ROOT.getValue();
    static DoccPaths instance = DoccPaths::from_root(root);
    return instance;
}

std::string getEnv(std::string const &key) {
    char *val = std::getenv(key.c_str());
    return val == NULL ? std::string("") : std::string(val);
};

std::filesystem::path get_docc_work_dir() {
    auto argDir = DOCC_WORK_DIR.getValue();
    if (argDir.starts_with('"') || argDir.starts_with('\'')) {
        argDir = argDir.substr(1, argDir.size() - 2);
    }

    if (argDir.empty()) { // backup in case somebody calls us without going through the driver, does not need to be
                          // perfect!
        auto baseDir = get_work_base_dir();

        auto pid = std::to_string(getpid());
        auto wDir = baseDir / pid;
        return wDir;
    } else {
        return argDir;
    }
}

std::filesystem::path get_work_base_dir() {
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

} // namespace docc::utils
