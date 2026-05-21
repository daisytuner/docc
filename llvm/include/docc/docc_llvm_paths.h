#pragma once

#include <llvm/Support/raw_ostream.h>
#include <sdfg/helpers/helpers.h>

#include <filesystem>
#include <string>

#include "docc/cmd_args.h"
#include "docc/utils.h"

namespace docc::utils {

enum class DoccRootMode { None, CMake, Dist };

struct DoccPaths {
    DoccRootMode root_mode = DoccRootMode::None;
    std::filesystem::path docc_root_path;

    [[nodiscard]] std::vector<std::filesystem::path> target_inc_paths() const;

    static DoccPaths from_root(const std::string_view& root);

    static DoccPaths get_instance();
};

std::filesystem::path get_work_base_dir();

std::filesystem::path get_docc_work_dir();

} // namespace docc::utils
