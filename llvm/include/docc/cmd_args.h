#pragma once

#include <llvm/Support/CommandLine.h>

namespace docc {

extern llvm::cl::opt<std::string> DOCC_ROOT;

extern llvm::cl::opt<std::string> DOCC_WORK_DIR;

extern llvm::cl::opt<std::string> DOCC_TUNE;

extern llvm::cl::opt<bool> DOCC_TRANSFERTUNE;

extern llvm::cl::opt<std::string> DOCC_TRANSFERTUNE_CATEGORY;

extern llvm::cl::opt<bool> DOCC_FORCE_SYNCHRONOUS_OFFLOADING;

extern llvm::cl::opt<bool> DOCC_SAVE_TEMPS;

extern bool DOCC_DUMP_GLBL_CFG_EN;
extern llvm::cl::opt<std::string> DOCC_DUMP_GLBL_CFG;

extern llvm::cl::list<std::string, bool> DOCC_COMP_OPTS;

extern llvm::cl::opt<std::string> DOCC_FUNC_BLACKLIST;

extern llvm::cl::opt<bool> DOCC_OFFLOAD_UNKNOWN_SIZES;

namespace args {

extern llvm::cl::opt<bool> DOCC_DOT_DUMP_SCHEDULED;

/**
 * Only needed in linker right now, only here so that it can pass arg-validation.
 */
extern llvm::cl::opt<std::string> DOCC_LINK_MODE;

extern llvm::cl::opt<bool> DOCC_NO_OFFLOADING_TRANSFER_OPT;

std::string collect_subcompile_override_flags();
} // namespace args

} // namespace docc
