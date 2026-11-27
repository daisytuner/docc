#pragma once

#include <sdfg/plugins/plugins.h>

#include "docc/plugin_registry.h"

namespace docc {

extern docc::PluginRegistry plugin_registry;

void register_sdfg_dispatchers();

/// The plugin context of the LLVM frontend, with the built-in targets and those of loaded plugins.
sdfg::plugins::Context& docc_context();

} // namespace docc
