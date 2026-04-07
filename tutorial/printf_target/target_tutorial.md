# Tutorial: Creating a Custom Offload Target in sdfglib

This tutorial guides you through implementing a custom offload target in sdfglib, using a simple **Printf** debug target as a hands-on example. The Printf target replaces actual kernel and data transfer generation with printf statements, making it ideal for debugging and understanding the offloading infrastructure.

This tutorial assumes an offloading target with dedicated memory between host and device, requiring explicit data transfers. A target device with shared memory is also possible, but using the offloading infrastructure will introduce restrictions regarding data movement that may not be necessary.

The transformation explicitly targets maps, i.e., for loops with proven parallelism. If you want to offload other SDFG nodes, a fully custom transformation may be more appropriate.

## Overview

An offload target in sdfglib requires implementing five main components:

1. **Schedule Type** - Defines how loops are scheduled on the target
2. **Data Offloading Node** - Handles memory transfers and allocation
3. **Map Dispatcher** - Generates target-specific kernel code
4. **Transform** - Orchestrates the offloading transformation
5. **Plugin Registration** - Ensures that the dispatchers etc. are integrated

## Component 1: Schedule Type

The schedule type defines target-specific scheduling properties and metadata for a single map.

### Example: Printf Schedule Type

```cpp
// printf_target.h
#pragma once

#include <sdfg/codegen/instrumentation/target_type.h>
#include <sdfg/schedule_type.h>
#include <sdfg/symbolic/symbolic.h>

namespace sdfg {
namespace printf_target {

// Prefix for device-side variables
inline const std::string PRINTF_DEVICE_PREFIX = "__printf_device_";

/**
 * Printf schedule type - a debug target that generates printf statements
 * instead of actual device code.
 */
class ScheduleType_Printf {
public:
    // Required: unique string identifier for this target
    static const std::string value() { return "Printf"; }

    // Required: create a default schedule instance
    static ScheduleType create() {
        auto schedule_type = structured_control_flow::ScheduleType(value());
        return schedule_type;
    }

    // Optional: helper for generating target-specific code
    static void target_stream(codegen::PrettyPrinter& stream);
};

// Define the target type constant
inline codegen::TargetType TargetType_Printf{ScheduleType_Printf::value()};

} // namespace printf_target
} // namespace sdfg
```

### Key Points
- Store target-specific properties using `set_property()` and retrieve with `properties().at()`
- Implement `value()` to return a unique string identifier for your target
- Implement `create()` to construct a default schedule with sensible defaults
- Define a `TargetType` constant using the schedule type's value
- Add helper methods like `target_stream()` to add target-specific metadata to generated code

## Component 2: Data Offloading Node

Handles memory allocation, deallocation, and transfers between host and device.
Extending the base class `DataOffloadingNode` is recommended to enable reuse of common functionality and data transfer minimization logic.

### Example: Printf Data Offloading Node

```cpp
// printf_data_offloading_node.h
#pragma once

#include <sdfg/codegen/dispatchers/library_node_dispatcher.h>
#include <sdfg/serializer/library_node_serializer.h>
#include <sdfg/targets/offloading/data_offloading_node.h>

namespace sdfg {
namespace printf_target {

// Define library node type constant
inline const data_flow::LibraryNodeType LibraryNodeType_Printf_Offloading("PrintfOffloading");

class PrintfDataOffloadingNode : public offloading::DataOffloadingNode {
public:
    PrintfDataOffloadingNode(
        size_t element_id,
        const DebugInfo& debug_info,
        const graph::Vertex vertex,
        data_flow::DataFlowGraph& parent,
        symbolic::Expression size,
        offloading::DataTransferDirection transfer_direction,
        offloading::BufferLifecycle buffer_lifecycle
    );

    // Override base class methods
    void validate(const Function& function) const override;
    std::unique_ptr<data_flow::DataFlowNode> clone(
        size_t element_id,
        const graph::Vertex vertex,
        data_flow::DataFlowGraph& parent
    ) const override;
    symbolic::SymbolSet symbols() const override;
    void replace(
        const symbolic::Expression old_expression,
        const symbolic::Expression new_expression
    ) override;
    bool blocking() const override;
    bool redundant_with(const offloading::DataOffloadingNode& other) const override;
    bool equal_with(const offloading::DataOffloadingNode& other) const override;
};
```

### Constructor Implementation

Populates parameters based on the operation type.
Make sure that the input and output connectors match the attached memlets during creation.

```cpp
// printf_data_offloading_node.cpp
PrintfDataOffloadingNode::PrintfDataOffloadingNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    symbolic::Expression size,
    offloading::DataTransferDirection transfer_direction,
    offloading::BufferLifecycle buffer_lifecycle
)
    : offloading::DataOffloadingNode(
          element_id,
          debug_info,
          vertex,
          parent,
          LibraryNodeType_Printf_Offloading,
          {},  // inputs - populated below
          {},  // outputs - populated below
          transfer_direction,
          buffer_lifecycle,
          size
      ) {
    // Configure inputs/outputs based on operation type
    if (!offloading::is_NONE(transfer_direction)) {
        this->inputs_.push_back("_src");
        this->outputs_.push_back("_dst");
    } else if (offloading::is_ALLOC(buffer_lifecycle)) {
        this->outputs_.push_back("_ret");
    } else if (offloading::is_FREE(buffer_lifecycle)) {
        this->inputs_.push_back("_ptr");
        this->outputs_.push_back("_ptr");
    }
}
```

### Dispatcher for Code Generation

Ensures that the appropriate code is generated for each offloading operation.
Since DataOffloadingNodes are library nodes, implement a `LibraryNodeDispatcher`.

The Printf target generates informative printf statements instead of actual device operations:

```cpp
class PrintfDataOffloadingNodeDispatcher : public codegen::LibraryNodeDispatcher {
public:
    PrintfDataOffloadingNodeDispatcher(
        codegen::LanguageExtension& language_extension,
        const Function& function,
        const data_flow::DataFlowGraph& data_flow_graph,
        const data_flow::LibraryNode& node
    );

    void dispatch_code(
        codegen::PrettyPrinter& stream,
        codegen::PrettyPrinter& globals_stream,
        codegen::CodeSnippetFactory& library_snippet_factory
    ) override {
        auto& offloading_node = static_cast<const PrintfDataOffloadingNode&>(this->node_);

        // Generate printf for allocation
        if (offloading_node.is_alloc()) {
            stream << "printf(\"[PRINTF_TARGET] Allocating %zu bytes for %s\\n\", (size_t)("
                   << this->language_extension_.expression(offloading_node.size()) << "), \""
                   << offloading_node.output(0) << "\");" << std::endl;
            // Use malloc for simulation
            stream << offloading_node.output(0) << " = malloc("
                   << this->language_extension_.expression(offloading_node.size()) << ");" << std::endl;
        }

        // Generate printf for H2D transfer
        if (offloading_node.is_h2d()) {
            stream << "printf(\"[PRINTF_TARGET] Copying %zu bytes from host (%s) to device (%s)\\n\", (size_t)("
                   << this->language_extension_.expression(offloading_node.size()) << "), \""
                   << offloading_node.input(0) << "\", \"" << offloading_node.output(0) << "\");" << std::endl;
            // Simulate with memcpy
            stream << "memcpy(" << offloading_node.output(0) << ", " << offloading_node.input(0) << ", "
                   << this->language_extension_.expression(offloading_node.size()) << ");" << std::endl;
        }

        // Generate printf for D2H transfer
        if (offloading_node.is_d2h()) {
            stream << "printf(\"[PRINTF_TARGET] Copying %zu bytes from device (%s) to host (%s)\\n\", (size_t)("
                   << this->language_extension_.expression(offloading_node.size()) << "), \""
                   << offloading_node.input(0) << "\", \"" << offloading_node.output(0) << "\");" << std::endl;
            stream << "memcpy(" << offloading_node.output(0) << ", " << offloading_node.input(0) << ", "
                   << this->language_extension_.expression(offloading_node.size()) << ");" << std::endl;
        }

        // Generate printf for deallocation
        if (offloading_node.is_free()) {
            stream << "printf(\"[PRINTF_TARGET] Freeing %s\\n\", \""
                   << offloading_node.input(0) << "\");" << std::endl;
            stream << "free(" << offloading_node.input(0) << ");" << std::endl;
        }
    }
};
```

### Serialization Support

In order to save/load SDFGs with your offloading nodes, implement a `LibraryNodeSerializer`:

```cpp
class PrintfDataOffloadingNodeSerializer : public serializer::LibraryNodeSerializer {
public:
    nlohmann::json serialize(const data_flow::LibraryNode& library_node) override;
    data_flow::LibraryNode& deserialize(
        const nlohmann::json& j,
        builder::StructuredSDFGBuilder& builder,
        structured_control_flow::Block& parent
    ) override;
};
```

## Component 3: Map Dispatcher

Generates kernel code for parallel loops scheduled on the target.

### Example: Printf Map Dispatcher

```cpp
// printf_map_dispatcher.h
#pragma once

#include <sdfg/analysis/analysis_manager.h>
#include <sdfg/codegen/arg_capture_plan.h>
#include <sdfg/codegen/code_snippet_factory.h>
#include <sdfg/codegen/dispatchers/node_dispatcher.h>
#include <sdfg/codegen/instrumentation/instrumentation_plan.h>
#include <sdfg/codegen/language_extension.h>
#include <sdfg/codegen/pretty_printer.h>
#include <sdfg/structured_control_flow/map.h>
#include <sdfg/structured_sdfg.h>

namespace sdfg {
namespace printf_target {

class PrintfMapDispatcher : public codegen::NodeDispatcher {
private:
    structured_control_flow::Map& node_;

    // Helper: dispatch the loop body
    void dispatch_printf_body(
        codegen::PrettyPrinter& stream,
        codegen::CodeSnippetFactory& library_snippet_factory,
        codegen::PrettyPrinter& globals_stream
    );

public:
    PrintfMapDispatcher(
        codegen::LanguageExtension& language_extension,
        StructuredSDFG& sdfg,
        analysis::AnalysisManager& analysis_manager,
        structured_control_flow::Map& node,
        codegen::InstrumentationPlan& instrumentation_plan,
        codegen::ArgCapturePlan& arg_capture_plan
    );

    void dispatch_node(
        codegen::PrettyPrinter& main_stream,
        codegen::PrettyPrinter& globals_stream,
        codegen::CodeSnippetFactory& library_snippet_factory
    ) override;

    codegen::InstrumentationInfo instrumentation_info() const override;
};

} // namespace printf_target
} // namespace sdfg
```

### Key Implementation Steps

1. **Analyze the loop structure** - Get induction variable, bounds, and iteration count
2. **Gather arguments** - Identify which function arguments are used in the map
3. **Generate entry diagnostics** - Print map metadata on entry
4. **Generate the loop** - Create the for loop with iteration tracing
5. **Dispatch body** - Recursively generate code for the map body
6. **Generate exit diagnostics** - Print when leaving the map

### Implementation

The Printf dispatcher generates informative debug output showing map execution:

```cpp
// printf_map_dispatcher.cpp
void PrintfMapDispatcher::dispatch_node(
    codegen::PrettyPrinter& main_stream,
    codegen::PrettyPrinter& globals_stream,
    codegen::CodeSnippetFactory& library_snippet_factory
) {
    // 1. Analyze arguments and loop structure
    analysis::AnalysisManager analysis_manager(sdfg_);
    analysis::ArgumentsAnalysis& arguments_analysis = analysis_manager.get<analysis::ArgumentsAnalysis>();

    auto& used_arguments = arguments_analysis.arguments(analysis_manager, node_);
    auto indvar = node_.indvar();
    auto num_iterations = node_.num_iterations();

    // 2. Collect and sort argument names
    std::vector<std::string> argument_names;
    for (auto& argument : used_arguments) {
        argument_names.push_back(argument.first);
    }
    std::sort(argument_names.begin(), argument_names.end());

    // 3. Generate printf for map entry
    main_stream << "printf(\"[PRINTF_TARGET] Entering map (element_id=%zu)\\n\", (size_t)"
                << node_.element_id() << ");" << std::endl;
    main_stream << "printf(\"[PRINTF_TARGET]   Indvar: %s\\n\", \"" << indvar->get_name() << "\");" << std::endl;
    main_stream << "printf(\"[PRINTF_TARGET]   Iterations: %s\\n\", \""
                << this->language_extension_.expression(num_iterations) << "\");" << std::endl;

    // Print arguments
    main_stream << "printf(\"[PRINTF_TARGET]   Arguments: ";
    for (size_t i = 0; i < argument_names.size(); ++i) {
        main_stream << argument_names[i];
        if (i < argument_names.size() - 1) main_stream << ", ";
    }
    main_stream << "\\n\");" << std::endl;

    // 4. Generate loop with iteration tracing
    main_stream << "for (long " << indvar->get_name() << " = 0; " << indvar->get_name() << " < "
                << this->language_extension_.expression(num_iterations) << "; ++" << indvar->get_name() << ") {"
                << std::endl;
    main_stream.setIndent(main_stream.indent() + 4);

    // Print first few iterations (avoid spam)
    main_stream << "if (" << indvar->get_name() << " < 3 || " << indvar->get_name() << " == "
                << this->language_extension_.expression(num_iterations) << " - 1) {" << std::endl;
    main_stream.setIndent(main_stream.indent() + 4);
    main_stream << "printf(\"[PRINTF_TARGET]   Iteration %s = %ld\\n\", \"" << indvar->get_name()
                << "\", (long)" << indvar->get_name() << ");" << std::endl;
    main_stream.setIndent(main_stream.indent() - 4);
    main_stream << "}" << std::endl;

    // 5. Dispatch the actual body content
    dispatch_printf_body(main_stream, library_snippet_factory, globals_stream);

    main_stream.setIndent(main_stream.indent() - 4);
    main_stream << "}" << std::endl;

    // 6. Generate printf for map exit
    main_stream << "printf(\"[PRINTF_TARGET] Exiting map (element_id=%zu)\\n\", (size_t)"
                << node_.element_id() << ");" << std::endl;
}

void PrintfMapDispatcher::dispatch_printf_body(
    codegen::PrettyPrinter& stream,
    codegen::CodeSnippetFactory& library_snippet_factory,
    codegen::PrettyPrinter& globals_stream
) {
    codegen::SequenceDispatcher dispatcher(
        language_extension_, sdfg_, analysis_manager_, node_.root(),
        instrumentation_plan_, arg_capture_plan_
    );
    dispatcher.dispatch(stream, globals_stream, library_snippet_factory);
}

codegen::InstrumentationInfo PrintfMapDispatcher::instrumentation_info() const {
    return codegen::InstrumentationInfo(
        node_.element_id(),
        codegen::ElementType_Map,
        TargetType_Printf,
        analysis::LoopInfo{},
        {}
    );
}
```

## Component 4: Transform

Orchestrates the complete offloading transformation.

### Example: Printf Transform

```cpp
// printf_transform.h
#pragma once

#include <sdfg/transformations/offloading/offload_transform.h>
#include <sdfg/structured_control_flow/map.h>
#include "printf_target.h"
#include "printf_data_offloading_node.h"

namespace sdfg {
namespace printf_target {

class PrintfTransform : public transformations::OffloadTransform {
public:
    explicit PrintfTransform(
        structured_control_flow::Map& map,
        bool allow_dynamic_sizes = false
    ) : OffloadTransform(map, allow_dynamic_sizes) {}

    std::string name() const override { return "PrintfTransform"; }

protected:
    // Storage types (use generic host memory for Printf target)
    types::StorageType local_device_storage_type() override {
        return types::StorageType::generic_local();
    }

    types::StorageType global_device_storage_type(symbolic::Expression arg_size) override {
        return types::StorageType::generic_global(arg_size);
    }

    // Define the schedule type for transformed maps
    ScheduleType transformed_schedule_type() override {
        return ScheduleType_Printf::create();
    }

    // Prefix for device copies of arguments
    std::string copy_prefix() override { return PRINTF_DEVICE_PREFIX; }

    // Memory operations using PrintfDataOffloadingNode
    void allocate_device_arg(
        builder::StructuredSDFGBuilder& builder,
        structured_control_flow::Block& copy_block,
        const std::string& device_arg_name,
        const symbolic::Expression& size,
        const types::Type& type
    ) override;

    void deallocate_device_arg(
        builder::StructuredSDFGBuilder& builder,
        structured_control_flow::Block& copy_block,
        const std::string& device_arg_name,
        const types::Type& type
    ) override;

    void copy_to_device(
        builder::StructuredSDFGBuilder& builder,
        structured_control_flow::Block& copy_block,
        const std::string& host_arg_name,
        const std::string& device_arg_name,
        const symbolic::Expression& size,
        const types::Type& in_type,
        const types::Type& out_type
    ) override;

    void copy_from_device(
        builder::StructuredSDFGBuilder& builder,
        structured_control_flow::Block& copy_block,
        const std::string& device_arg_name,
        const std::string& host_arg_name,
        const symbolic::Expression& size,
        const types::Type& in_type,
        const types::Type& out_type
    ) override;
};

} // namespace printf_target
} // namespace sdfg
```

### Memory Operation Implementation

Each method creates the appropriate `PrintfDataOffloadingNode`:

```cpp
// printf_transform.cpp
void PrintfTransform::copy_to_device(
    builder::StructuredSDFGBuilder& builder,
    structured_control_flow::Block& copy_block,
    const std::string& host_arg_name,
    const std::string& device_arg_name,
    const symbolic::Expression& size,
    const types::Type& in_type,
    const types::Type& out_type
) {
    // Create access nodes for source and destination
    auto& access_node_host = builder.add_access(copy_block, host_arg_name);
    auto& access_node_device = builder.add_access(copy_block, device_arg_name);

    // Create the PrintfDataOffloadingNode for H2D transfer
    auto& memcpy_node = builder.add_library_node<PrintfDataOffloadingNode>(
        copy_block,
        this->map_.debug_info(),
        size,
        offloading::DataTransferDirection::H2D,
        offloading::BufferLifecycle::NO_CHANGE
    );

    // Connect with memlets
    builder.add_computational_memlet(
        copy_block, access_node_host, memcpy_node, "_src", {}, in_type
    );
    builder.add_computational_memlet(
        copy_block, memcpy_node, "_dst", access_node_device, {}, out_type
    );
}

void PrintfTransform::allocate_device_arg(
    builder::StructuredSDFGBuilder& builder,
    structured_control_flow::Block& copy_block,
    const std::string& device_arg_name,
    const symbolic::Expression& size,
    const types::Type& type
) {
    auto& access_node_device = builder.add_access(copy_block, device_arg_name);

    auto& alloc_node = builder.add_library_node<PrintfDataOffloadingNode>(
        copy_block,
        this->map_.debug_info(),
        size,
        offloading::DataTransferDirection::NONE,
        offloading::BufferLifecycle::ALLOC
    );

    builder.add_computational_memlet(
        copy_block, alloc_node, "_ret", access_node_device, {}, type
    );
}
```

## Component 5: Plugin Registration

The plugin system provides a clean way to register all target components with the sdfglib runtime.
The registration of serializers, map dispatchers, and library node dispatchers is required for proper integration.
It ensures that external extensions, like your plugin, can be used seamlessly with sdfglib's code generation and serialization infrastructure.

Each registry has a singleton instance that manages the registered components.
The components are registered using factory functions (lambdas) that create instances of the respective classes when needed.
To find the correct dispatchers and serializers, they are mapped using the schedule type, library node type, and (implementation type, in case of the library node dispatcher).

If your plugin is not correctly registered, sdfglib will raise errors when trying to generate code or serialize SDFGs that use your target.

### Location
- Header: `include/plugin.h` (in your tutorial/target directory)

### Example: Printf Plugin

```cpp
// plugin.h
#pragma once

#include <sdfg/codegen/dispatchers/node_dispatcher_registry.h>
#include <sdfg/serializer/json_serializer.h>

#include "printf_target.h"
#include "printf_data_offloading_node.h"
#include "printf_map_dispatcher.h"

namespace sdfg {
namespace printf_target {

/**
 * @brief Registers all printf target components with sdfglib
 *
 * This function must be called before using any printf target features.
 * It registers:
 * - Map dispatcher for Printf schedule type
 * - Library node dispatcher for PrintfOffloading nodes
 * - Serializer for PrintfDataOffloadingNode
 */
inline void register_printf_plugin() {
    // 1. Register Map Dispatcher
    // Associates the Printf schedule type with the PrintfMapDispatcher
    codegen::MapDispatcherRegistry::instance().register_map_dispatcher(
        ScheduleType_Printf::value(),
        [](codegen::LanguageExtension& language_extension,
           StructuredSDFG& sdfg,
           analysis::AnalysisManager& analysis_manager,
           structured_control_flow::Map& node,
           codegen::InstrumentationPlan& instrumentation_plan,
           codegen::ArgCapturePlan& arg_capture_plan) {
            return std::make_unique<PrintfMapDispatcher>(
                language_extension, sdfg, analysis_manager, node,
                instrumentation_plan, arg_capture_plan
            );
        }
    );

    // 2. Register Library Node Dispatcher
    // Associates PrintfOffloading library nodes with their code generator
    codegen::LibraryNodeDispatcherRegistry::instance().register_library_node_dispatcher(
        LibraryNodeType_Printf_Offloading.value() + "::" +
        data_flow::ImplementationType_NONE.value(),
        [](codegen::LanguageExtension& language_extension,
           const Function& function,
           const data_flow::DataFlowGraph& data_flow_graph,
           const data_flow::LibraryNode& node) {
            return std::make_unique<PrintfDataOffloadingNodeDispatcher>(
                language_extension, function, data_flow_graph, node
            );
        }
    );

    // 3. Register Serializer
    // Enables saving/loading SDFGs with PrintfDataOffloadingNodes
    serializer::LibraryNodeSerializerRegistry::instance()
        .register_library_node_serializer(
            LibraryNodeType_Printf_Offloading.value(),
            []() {
                return std::make_unique<PrintfDataOffloadingNodeSerializer>();
            }
        );
}

} // namespace printf_target
} // namespace sdfg
```

### Key Components

#### 1. Map Dispatcher Registration

```cpp
codegen::MapDispatcherRegistry::instance().register_map_dispatcher(
    ScheduleType_Printf::value(),  // "Printf" - schedule type identifier
    [](/* parameters */) {
        return std::make_unique<PrintfMapDispatcher>(...);
    }
);
```

**Purpose**: Associates your schedule type with the dispatcher that generates code for loops scheduled on your target.

#### 2. Library Node Dispatcher Registration

```cpp
codegen::LibraryNodeDispatcherRegistry::instance()
    .register_library_node_dispatcher(
        LibraryNodeType_Printf_Offloading.value() + "::" +
        data_flow::ImplementationType_NONE.value(),  // "PrintfOffloading::NONE"
        [](/* parameters */) {
            return std::make_unique<PrintfDataOffloadingNodeDispatcher>(...);
        }
    );
```

**Purpose**: Associates your library node type with the dispatcher that generates code for data offloading operations.

#### 3. Serializer Registration

```cpp
serializer::LibraryNodeSerializerRegistry::instance()
    .register_library_node_serializer(
        LibraryNodeType_Printf_Offloading.value(),  // "PrintfOffloading"
        []() {
            return std::make_unique<PrintfDataOffloadingNodeSerializer>();
        }
    );
```

**Purpose**: Enables saving/loading SDFGs with your offloading nodes to/from JSON.

### Using Your Plugin

To use your plugin, call the registration function early in your pipeline and tests.
Registering plugins multiple times will raise errors.

```cpp
#include "plugin.h"

int main() {
    // Register plugin before using any Printf features
    sdfg::printf_target::register_printf_plugin();

    // Now you can create SDFGs with Printf scheduling
    // ...
}
```

### Plugin Best Practices

1. **Single Registration Function**: Encapsulate all registrations in one function
2. **Inline Header**: Use `inline` to allow header-only registration
3. **Namespace Organization**: Keep plugin in your target's namespace
4. **Early Registration**: Call registration before creating any SDFGs

## Integration Steps

### 1. Create Plugin Header

Create `include/plugin.h` with registration function as shown above.

### 2. Call Registration

In your application or test setup:

```cpp
#include "plugin.h"

sdfg::printf_target::register_printf_plugin();
```

### 3. Verify Registration

Test that your components are properly registered:

```cpp
// Check map dispatcher
auto& map_registry = codegen::MapDispatcherRegistry::instance();
ASSERT_TRUE(map_registry.has_dispatcher(ScheduleType_Printf::value()));

// Check library node dispatcher
auto& lib_registry = codegen::LibraryNodeDispatcherRegistry::instance();
ASSERT_TRUE(lib_registry.has_dispatcher(
    LibraryNodeType_Printf_Offloading.value() + "::" +
    data_flow::ImplementationType_NONE.value()
));
```

### Code generation, Compilation, and Linking (WIP)

Currently, if your target needs to adapt the automatic compilation and linking process, you currently need to adapt the corresponding passes, e.g., in python/bindings/py_structured_sdfg.cpp, by hand.
We are working on refactoring the compilation process of generated code, to allow registration of includes, libraries, and flags, depending on used dispatchers.
We will update the tutorial according its progress.

## Testing

Create comprehensive tests for your target. See the `tests/` directory for examples:

```cpp
// printf_transform_test.cpp
#include <gtest/gtest.h>
#include "plugin.h"
#include "printf_transform.h"

class PrintfTransformTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Register plugin once
        static bool registered = false;
        if (!registered) {
            sdfg::printf_target::register_printf_plugin();
            registered = true;
        }
    }
};

TEST_F(PrintfTransformTest, BasicOffload) {
    // 1. Create SDFG with a parallel map
    // 2. Apply PrintfTransform
    // 3. Verify the transformed SDFG structure
    // 4. Generate code and check for printf statements
}
```

## Summary Checklist

- [ ] Implement `ScheduleType_Printf` class with `value()` and `create()`
- [ ] Define `TargetType_Printf` constant
- [ ] Define `LibraryNodeType_Printf_Offloading` constant
- [ ] Implement `PrintfDataOffloadingNode` class
- [ ] Implement `PrintfDataOffloadingNodeDispatcher` class
- [ ] Implement `PrintfDataOffloadingNodeSerializer` class
- [ ] Implement `PrintfMapDispatcher` class
- [ ] Implement `PrintfTransform` class
- [ ] Create `plugin.h` with `register_printf_plugin()` function
- [ ] Register all dispatchers and serializers in plugin
- [ ] Call plugin registration in application/tests
- [ ] Write comprehensive tests
- [ ] Update build system (CMakeLists.txt)

## Best Practices

1. **Reuse base classes**: Extend `DataOffloadingNode` and `OffloadTransform` rather than reimplementing
2. **Handle nested parallelism**: Check if you're the outermost map before generating kernel code (for real targets)
3. **Validate early**: Implement `validate()` methods to catch errors during SDFG construction
4. **Support serialization**: Ensure your nodes can be saved/loaded for testing and persistence
5. **Use symbolic expressions**: Support symbolic sizes and parameters throughout
6. **Instrument for profiling**: Implement `instrumentation_info()` for performance analysis

## Project Structure

The Printf target tutorial follows this directory structure:

```
tutorial/printf_target/
├── include/
│   ├── printf_target.h           # Schedule type and constants
│   ├── printf_data_offloading_node.h  # Data offloading node + dispatcher + serializer
│   ├── printf_map_dispatcher.h   # Map code generator
│   ├── printf_transform.h        # Transform orchestration
│   └── plugin.h                  # Plugin registration
├── src/
│   ├── printf_data_offloading_node.cpp
│   ├── printf_map_dispatcher.cpp
│   └── printf_transform.cpp
├── tests/
│   ├── CMakeLists.txt
│   ├── test_main.cpp
│   ├── printf_data_offloading_node_test.cpp
│   ├── printf_map_dispatcher_test.cpp
│   └── printf_transform_test.cpp
├── cmake/
│   └── sdfgtutorial_printf_targetConfig.cmake.in
├── CMakeLists.txt
└── target_tutorial.md            # This tutorial
```

## Additional Resources

- Base classes: `sdfg/targets/offloading/data_offloading_node.h`
- Transform base: `sdfg/transformations/offloading/offload_transform.h`
- Code generation: `sdfg/codegen/dispatchers/`
- For a real-world example, see the CUDA implementation in sdfglib

---

*This tutorial uses the Printf debug target as a learning example. The same patterns apply to real offload targets like CUDA, OpenCL, or custom accelerators - just replace the printf statements with actual device API calls.*
