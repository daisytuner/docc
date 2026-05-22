#pragma once

#include <llvm/Support/raw_ostream.h>
#include <sdfg/helpers/helpers.h>

#include <filesystem>
#include <string>

#include "docc/cmd_args.h"
#include "docc/utils.h"

namespace docc::utils {

std::filesystem::path get_work_base_dir();

std::filesystem::path get_docc_work_dir();

} // namespace docc::utils
