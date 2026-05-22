#include "docc/docc_llvm_paths.h"
#include <llvm/Support/Debug.h>

#include "docc/cmd_args.h"

#include <filesystem>
#include <memory>


namespace docc::utils {

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
