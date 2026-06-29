/**
 * @file node_dispatcher_registry.h
 * @brief Registry system for dispatchers and library node dispatchers
 *
 * This file defines registry classes for dispatchers, allowing dynamic registration
 * and lookup of code generation handlers for different node types and library nodes.
 *
 * ## Dispatcher Registries
 *
 * The registry system provides three main registries:
 * 1. **NodeDispatcherRegistry**: Maps control flow node types to dispatchers
 * 2. **MapDispatcherRegistry**: Maps map schedule types to specialized dispatchers
 * 3. **LibraryNodeDispatcherRegistry**: Maps library node codes to dispatchers
 *
 * Each registry follows the singleton pattern and provides thread-safe registration
 * and lookup of factory functions.
 *
 * ## Library Node Dispatchers
 *
 * Library node dispatchers generate code for specific library operations. Each library
 * node code (e.g., "GEMM", "Memcpy") can be registered with a custom dispatcher that
 * knows how to generate the appropriate library call or implementation.
 *
 * ## Example
 *
 * Registering a library node dispatcher:
 * @code
 * LibraryNodeDispatcherRegistry::instance().register_library_node_dispatcher(
 *     "GEMM",
 *     [](LanguageExtension& lang, const Function& func,
 *        const data_flow::DataFlowGraph& dfg, const data_flow::LibraryNode& node) {
 *         return std::make_unique<GemmDispatcher>(lang, func, dfg, node);
 *     }
 * );
 * @endcode
 *
 * @see node_dispatcher.h for the base dispatcher class
 * @see block_dispatcher.h for LibraryNodeDispatcher base class
 */

#pragma once

#include <mutex>
#include <typeindex>
#include <unordered_map>

#include "sdfg/analysis/analysis.h"
#include "sdfg/codegen/dispatchers/block_dispatcher.h"
#include "sdfg/codegen/dispatchers/node_dispatcher.h"
#include "sdfg/codegen/instrumentation/arg_capture_plan.h"
#include "sdfg/codegen/instrumentation/instrumentation_plan.h"
#include "sdfg/codegen/language_extension.h"
#include "sdfg/structured_control_flow/reduce.h"

namespace sdfg {
namespace codegen {

using NodeDispatcherFn = std::function<std::unique_ptr<
    NodeDispatcher>(LanguageExtension&, StructuredSDFG&, analysis::AnalysisManager&, structured_control_flow::ControlFlowNode&, InstrumentationPlan&, ArgCapturePlan&)>;

class NodeDispatcherRegistry {
private:
    mutable std::mutex mutex_;
    std::unordered_map<std::type_index, NodeDispatcherFn> factory_map_;

public:
    static NodeDispatcherRegistry& instance() {
        static NodeDispatcherRegistry registry;
        return registry;
    }

    void register_dispatcher(std::type_index type, NodeDispatcherFn fn) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (factory_map_.find(type) != factory_map_.end()) {
            return;
        }
        factory_map_[type] = std::move(fn);
    }

    NodeDispatcherFn get_dispatcher(std::type_index type) const {
        auto it = factory_map_.find(type);
        if (it != factory_map_.end()) {
            return it->second;
        }
        return nullptr;
    }

    size_t size() const { return factory_map_.size(); }
};

std::unique_ptr<NodeDispatcher> create_dispatcher(
    LanguageExtension& language_extension,
    StructuredSDFG& sdfg,
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::ControlFlowNode& node,
    InstrumentationPlan& instrumentation_plan,
    ArgCapturePlan& arg_capture_plan
);

using MapDispatcherFn = std::function<std::unique_ptr<
    NodeDispatcher>(LanguageExtension&, StructuredSDFG&, analysis::AnalysisManager&, structured_control_flow::Map&, InstrumentationPlan&, ArgCapturePlan&)>;

/**
 * @class MapDispatcherRegistry
 * @brief Registry for map-specific dispatchers based on schedule type
 *
 * Maps can have different schedule types (CPU_Sequential, GPU_Default, etc.)
 * that require different code generation strategies. This registry maps
 * schedule type strings to appropriate dispatcher factories.
 */
class MapDispatcherRegistry {
private:
    mutable std::mutex mutex_;
    std::unordered_map<std::string, MapDispatcherFn> factory_map_;

public:
    static MapDispatcherRegistry& instance() {
        static MapDispatcherRegistry registry;
        return registry;
    }

    void register_map_dispatcher(std::string schedule_type, MapDispatcherFn fn) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (factory_map_.find(schedule_type) != factory_map_.end()) {
            return;
        }
        factory_map_[schedule_type] = std::move(fn);
    }

    MapDispatcherFn get_map_dispatcher(std::string schedule_type) const {
        auto it = factory_map_.find(schedule_type);
        if (it != factory_map_.end()) {
            return it->second;
        }
        return nullptr;
    }

    size_t size() const { return factory_map_.size(); }
};

using ReduceDispatcherFn = std::function<std::unique_ptr<
    NodeDispatcher>(LanguageExtension&, StructuredSDFG&, analysis::AnalysisManager&, structured_control_flow::Reduce&, InstrumentationPlan&, ArgCapturePlan&)>;

/**
 * @class ReduceDispatcherRegistry
 * @brief Registry for reduce-specific dispatchers based on schedule type
 *
 * Mirrors @ref MapDispatcherRegistry. A @ref structured_control_flow::Reduce
 * carries a schedule type (sequential, OpenMP, ...) and this registry maps that
 * schedule type string to the dispatcher factory that knows how to lower the
 * reduction for that target.
 */
class ReduceDispatcherRegistry {
private:
    mutable std::mutex mutex_;
    std::unordered_map<std::string, ReduceDispatcherFn> factory_map_;

public:
    static ReduceDispatcherRegistry& instance() {
        static ReduceDispatcherRegistry registry;
        return registry;
    }

    void register_reduce_dispatcher(std::string schedule_type, ReduceDispatcherFn fn) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (factory_map_.find(schedule_type) != factory_map_.end()) {
            return;
        }
        factory_map_[schedule_type] = std::move(fn);
    }

    ReduceDispatcherFn get_reduce_dispatcher(std::string schedule_type) const {
        auto it = factory_map_.find(schedule_type);
        if (it != factory_map_.end()) {
            return it->second;
        }
        return nullptr;
    }

    size_t size() const { return factory_map_.size(); }
};

using LibraryNodeDispatcherFn = std::function<std::unique_ptr<
    LibraryNodeDispatcher>(LanguageExtension&, const Function&, const data_flow::DataFlowGraph&, const data_flow::LibraryNode&)>;

/**
 * @class LibraryNodeDispatcherRegistry
 * @brief Registry for library node dispatchers
 *
 * This registry maps library node codes (operation identifiers like "GEMM", "Memcpy")
 * to dispatcher factory functions. Each library node type can register a custom
 * dispatcher that generates the appropriate code for that operation.
 *
 * The registry is thread-safe and follows the singleton pattern. Dispatchers are
 * registered at program startup and looked up during code generation.
 *
 * ## Usage
 *
 * Register a dispatcher:
 * @code
 * LibraryNodeDispatcherRegistry::instance().register_library_node_dispatcher(
 *     "MyOperation",
 *     [](LanguageExtension& lang, const Function& func,
 *        const data_flow::DataFlowGraph& dfg, const data_flow::LibraryNode& node) {
 *         return std::make_unique<MyOperationDispatcher>(lang, func, dfg, node);
 *     }
 * );
 * @endcode
 *
 * Lookup a dispatcher:
 * @code
 * auto factory = LibraryNodeDispatcherRegistry::instance()
 *                    .get_library_node_dispatcher("MyOperation");
 * if (factory) {
 *     auto dispatcher = factory(lang, func, dfg, node);
 *     // Use dispatcher...
 * }
 * @endcode
 */
class LibraryNodeDispatcherRegistry {
private:
    mutable std::mutex mutex_;
    std::unordered_map<std::string, LibraryNodeDispatcherFn> factory_map_;

public:
    static LibraryNodeDispatcherRegistry& instance() {
        static LibraryNodeDispatcherRegistry registry;
        return registry;
    }

    void register_library_node_dispatcher(std::string library_node_code, LibraryNodeDispatcherFn fn) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (factory_map_.find(library_node_code) != factory_map_.end()) {
            return;
        }
        factory_map_[library_node_code] = std::move(fn);
    }

    LibraryNodeDispatcherFn get_library_node_dispatcher(std::string library_node_code) const {
        auto it = factory_map_.find(library_node_code);
        if (it != factory_map_.end()) {
            return it->second;
        }
        return nullptr;
    }

    size_t size() const { return factory_map_.size(); }
};

void register_default_dispatchers();

} // namespace codegen
} // namespace sdfg
