#pragma once

#include "docc/plugin_registry.h"

namespace docc {

extern docc::PluginRegistry plugin_registry;

void register_sdfg_dispatchers();

} // namespace docc
