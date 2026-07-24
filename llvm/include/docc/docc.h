#pragma once

#include "docc/plugin_registry.h"

namespace docc {

extern docc::PluginRegistry plugin_registry;

std::shared_ptr<sdfg::plugins::Context> register_sdfg_dispatchers();

} // namespace docc
