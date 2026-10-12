#pragma once

#include <sdfg/plugins/plugins.h>

namespace docc::target::axelera {

/// Registers the "axelera" target: its loop scheduler and the datapath dispatcher of the axscript codegen.
void register_plugin(sdfg::plugins::Context& context);

} // namespace docc::target::axelera
