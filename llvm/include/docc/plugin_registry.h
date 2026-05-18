#pragma once

#include <dlfcn.h>

#include <vector>

#include "sdfg/plugins/plugins.h"

namespace docc {

struct Plugin {
    sdfg::plugins::Plugin sdfg_plugin;
    void* handle;
};

struct PluginRegistry {
    std::vector<Plugin> plugins;

    ~PluginRegistry() {
        for (auto& plugin : this->plugins) {
            dlclose(plugin.handle);
        }
    }
};

} // namespace docc
