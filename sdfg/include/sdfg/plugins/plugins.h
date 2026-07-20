#pragma once

#include <list>
#include <memory>
#include <string>

#include "sdfg/codegen/dispatchers/map_dispatcher.h"
#include "sdfg/codegen/dispatchers/node_dispatcher_registry.h"
#include "sdfg/codegen/dispatchers/reduce_dispatcher.h"
#include "sdfg/passes/scheduler/scheduler_registry.h"
#include "sdfg/plugins/targets.h"
#include "sdfg/serializer/json_serializer.h"
#include "sdfg/structured_sdfg.h"

namespace sdfg {
namespace plugins {

/**
 * TODO this class is broken. We need an interface that exposes the references on methods. So that there can be a future
 * implementation that actually owns the different registries as local state. And this legacy adapter can exist that
 * just redirects to problematic broken state. The methods exist, so we can migrate users over to the methods, before
 * stripping the implementation-specific parts into subclasses.
 *
 * Also, SchedulerRegistry currently lives not in this library, so poses another API hazard, as you could compile this
 * as part of the library it is in!
 */
struct Context {
    // Serialization
    /// @deprecated
    serializer::LibraryNodeSerializerRegistry& library_node_serializer_registry;

    serializer::LibraryNodeSerializerRegistry& get_library_node_serializer_registry() {
        return library_node_serializer_registry;
    }

    // Dispatchers
    codegen::NodeDispatcherRegistry& get_node_dispatcher_registry() { return node_dispatcher_registry; }
    codegen::MapDispatcherRegistry& get_map_dispatcher_registry() { return map_dispatcher_registry; }
    codegen::ReduceDispatcherRegistry& get_reduce_dispatcher_registry() { return reduce_dispatcher_registry; }
    codegen::LibraryNodeDispatcherRegistry& get_library_node_dispatcher_registry() {
        return library_node_dispatcher_registry;
    }

    /// @deprecated
    codegen::NodeDispatcherRegistry& node_dispatcher_registry;
    /// @deprecated
    codegen::MapDispatcherRegistry& map_dispatcher_registry;
    /// @deprecated
    codegen::ReduceDispatcherRegistry& reduce_dispatcher_registry;
    /// @deprecated
    codegen::LibraryNodeDispatcherRegistry& library_node_dispatcher_registry;

    // Schedulers
    passes::scheduler::SchedulerRegistry& scheduler_registry;
    passes::scheduler::SchedulerRegistry& get_scheduler_registry() { return scheduler_registry; }

protected:
    std::unordered_map<std::string, docc::target::DoccTarget*> available_targets;

public:
    /// @deprecated
    Context(
        serializer::LibraryNodeSerializerRegistry& library_node_serializer_registry,
        codegen::NodeDispatcherRegistry& node_dispatcher_registry,
        codegen::MapDispatcherRegistry& map_dispatcher_registry,
        codegen::ReduceDispatcherRegistry& reduce_dispatcher_registry,
        codegen::LibraryNodeDispatcherRegistry& library_node_dispatcher_registry,
        passes::scheduler::SchedulerRegistry& scheduler_registry
    )
        : library_node_serializer_registry(library_node_serializer_registry),
          node_dispatcher_registry(node_dispatcher_registry), map_dispatcher_registry(map_dispatcher_registry),
          reduce_dispatcher_registry(reduce_dispatcher_registry),
          library_node_dispatcher_registry(library_node_dispatcher_registry), scheduler_registry(scheduler_registry) {}

    static Context global_context() {
        return Context{
            serializer::LibraryNodeSerializerRegistry::instance(),
            codegen::NodeDispatcherRegistry::instance(),
            codegen::MapDispatcherRegistry::instance(),
            codegen::ReduceDispatcherRegistry::instance(),
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
