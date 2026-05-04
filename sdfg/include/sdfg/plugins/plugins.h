#pragma once

#include <list>
#include <memory>
#include <string>

#include "sdfg/codegen/dispatchers/map_dispatcher.h"
#include "sdfg/codegen/dispatchers/node_dispatcher_registry.h"
#include "sdfg/passes/scheduler/scheduler_registry.h"
#include "sdfg/plugins/targets.h"
#include "sdfg/serializer/json_serializer.h"
#include "sdfg/structured_sdfg.h"

namespace sdfg {
namespace plugins {

struct Context {
    // Serialization
    serializer::LibraryNodeSerializerRegistry& library_node_serializer_registry;

    // Dispatchers
    codegen::NodeDispatcherRegistry& node_dispatcher_registry;
    codegen::MapDispatcherRegistry& map_dispatcher_registry;
    codegen::LibraryNodeDispatcherRegistry& library_node_dispatcher_registry;

    // Schedulers
    passes::scheduler::SchedulerRegistry& scheduler_registry;

    std::unordered_map<std::string, docc::target::DoccTarget*> available_targets;

    static Context global_context() {
        return Context{
            serializer::LibraryNodeSerializerRegistry::instance(),
            codegen::NodeDispatcherRegistry::instance(),
            codegen::MapDispatcherRegistry::instance(),
            codegen::LibraryNodeDispatcherRegistry::instance(),
            passes::scheduler::SchedulerRegistry::instance()
        };
    }

    bool add_target(docc::target::DoccTarget* target) {
        auto res = available_targets.insert_or_assign(target->short_name, target);
        return res.second;
    }

    docc::target::DoccTarget* get_target_handler(const std::string& target) const {
        auto it = available_targets.find(target);
        if (it != available_targets.end()) {
            return it->second;
        }
        return nullptr;
    }
};

struct Plugin {
    const char* name;
    const char* version;
    const char* description;

    // Register callback
    void (*register_plugin_callback)(Context& context);

    // SDFG lookup
    std::list<std::unique_ptr<sdfg::StructuredSDFG>> (*sdfg_lookup)(std::string name);
};

} // namespace plugins
} // namespace sdfg
