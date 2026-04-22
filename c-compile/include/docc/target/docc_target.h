#pragma once
#include <string>

#include "sdfg/plugins/plugins.h"
#include "sdfg/plugins/targets.h"

namespace docc::target {


DoccTarget* get_target_handler(const std::string& target);

bool add_highway_build_support(compile::SrcFileCompilerBuilder& builder);

void register_builtin_targets(sdfg::plugins::Context& context);

} // namespace docc::target
