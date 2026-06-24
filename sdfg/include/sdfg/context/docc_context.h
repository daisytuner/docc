#pragma once
#include "sdfg/codegen/dispatchers/map_dispatcher.h"
#include "sdfg/codegen/dispatchers/node_dispatcher_registry.h"
#include "sdfg/passes/scheduler/scheduler_registry.h"
#include "sdfg/plugins/targets.h"
#include "sdfg/serializer/json_serializer.h"

namespace docc::context {

class DoccContext {
protected:
    std::unordered_map<std::string, docc::target::DoccTarget*> available_targets;

public:
    virtual ~DoccContext() = default;

    virtual sdfg::serializer::LibraryNodeSerializerRegistry& library_node_serializer_registry() = 0;
    virtual sdfg::codegen::NodeDispatcherRegistry& node_dispatcher_registry() = 0;
    virtual sdfg::codegen::MapDispatcherRegistry& map_dispatcher_registry() = 0;
    virtual sdfg::codegen::LibraryNodeDispatcherRegistry& library_node_dispatcher_registry() = 0;

    virtual sdfg::passes::scheduler::SchedulerRegistry& scheduler_registry() = 0;


    bool add_target(docc::target::DoccTarget* target);

    docc::target::DoccTarget* get_target_handler(const std::string& target) const;
};

class DefaultDoccContext : public DoccContext {
protected:
    std::unique_ptr<sdfg::serializer::LibraryNodeSerializerRegistry> library_node_serializer_registry;
    std::unique_ptr<sdfg::codegen::NodeDispatcherRegistry> node_dispatcher_registry;
    std::unique_ptr<sdfg::codegen::MapDispatcherRegistry> map_dispatcher_registry;
    std::unique_ptr<sdfg::codegen::LibraryNodeDispatcherRegistry> library_node_dispatcher_registry;
    std::unique_ptr<sdfg::passes::scheduler::SchedulerRegistry> scheduler_registry;

public:
    sdfg::serializer::LibraryNodeSerializerRegistry& library_node_serializer_registry() override {
        return *library_node_serializer_registry.get();
    }
    sdfg::codegen::NodeDispatcherRegistry& node_dispatcher_registry() override {
        return *node_dispatcher_registry.get();
    }
    sdfg::codegen::MapDispatcherRegistry& map_dispatcher_registry() override { return *map_dispatcher_registry.get(); }
    sdfg::codegen::LibraryNodeDispatcherRegistry& library_node_dispatcher_registry() override {
        return *library_node_dispatcher_registry.get();
    }
};

class LegacyRefContext : public DoccContext {
protected:
    // Serialization
    sdfg::serializer::LibraryNodeSerializerRegistry& library_node_serializer_registry;

    // Dispatchers
    sdfg::codegen::NodeDispatcherRegistry& node_dispatcher_registry;
    sdfg::codegen::MapDispatcherRegistry& map_dispatcher_registry;
    sdfg::codegen::LibraryNodeDispatcherRegistry& library_node_dispatcher_registry;

    // Schedulers
    passes::scheduler::SchedulerRegistry& scheduler_registry;

public:
    sdfg::serializer::LibraryNodeSerializerRegistry& library_node_serializer_registry() override {
        return this->library_node_serializer_registry;
    }
    sdfg::codegen::NodeDispatcherRegistry& node_dispatcher_registry() override { return node_dispatcher_registry; }
    sdfg::codegen::MapDispatcherRegistry& map_dispatcher_registry() override { return map_dispatcher_registry; }
    sdfg::codegen::LibraryNodeDispatcherRegistry& library_node_dispatcher_registry() override {
        return library_node_dispatcher_registry;
    }
};

} // namespace docc::context
