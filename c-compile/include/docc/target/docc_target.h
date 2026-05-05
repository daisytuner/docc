#pragma once
#include <string>

#include "sdfg/plugins/plugins.h"
#include "sdfg/plugins/targets.h"

namespace docc::target {


DoccTarget* get_target_handler(const std::string& target);

void register_builtin_targets(sdfg::plugins::Context& context);

} // namespace docc::target
