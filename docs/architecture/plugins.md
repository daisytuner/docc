---
title: docc Plugins
sidebar_label: Plugins
sidebar_position: 4
description: How targets and extensions plug into docc, and how to load your own plugin
---

A **plugin** extends the docc core without changing it. Plugins add new library nodes, code generators, loop schedulers for a target, or replace whole functions with hand-written SDFGs. The built-in targets are plugins as well: OpenMP, CUDA, ROCm, and Tenstorrent register themselves through the same interface that external plugins use.

## What a plugin registers

The core keeps registries for everything that is target- or domain-specific. A plugin adds entries to them through a `sdfg::plugins::Context` (`sdfg/include/sdfg/plugins/plugins.h`):

| Registry                                    | Contains                                                                                     |
| ------------------------------------------- | -------------------------------------------------------------------------------------------- |
| `get_library_node_serializer_registry()`    | How to read and write a library node as JSON                                                  |
| `get_library_node_dispatcher_registry()`    | How to generate code for a library node, per target                                           |
| `get_map_dispatcher_registry()`             | How to generate code for a `Map` with a given schedule, for example an OpenMP loop or a GPU kernel |
| `get_reduce_dispatcher_registry()`          | The same for `Reduce` loops                                                                   |
| `get_node_dispatcher_registry()`            | Code generation for other control flow nodes                                                  |
| `get_scheduler_registry()`                  | [Loop schedulers](./core-concepts/passes.md#scheduling) of a target                           |
| `get_tile_target_registry()`                | Tile-level code generation for a target                                                       |
| `option_registry()`                         | [Pass options](./core-concepts/passes.md#pass-options) declared by the plugin                 |

The built-in plugins are registered at startup, for example `register_omp_plugin`, `register_cuda_plugin`, `register_rocm_plugin`, and `register_tenstorrent_plugin`, together with the einsum, parallelization, tiles, and vectorization plugins (`llvm/src/docc.cpp`).

## External plugins

An external plugin is a shared library that exports one C function, `register_docc_plugin`. It returns a description with two callbacks:

| Field                      | Description                                                                                          |
| -------------------------- | ---------------------------------------------------------------------------------------------------- |
| `name`, `version`, `description` | Shown when the plugin is loaded                                                                |
| `register_plugin_callback` | Called once with the `Context`; registers dispatchers, schedulers, and serializers                   |
| `sdfg_lookup`              | Called with the (mangled) name of each function; returns SDFGs that replace the function, or an empty list |

Both callbacks are required. `sdfg_lookup` lets a plugin replace a function with a hand-optimized implementation, for example a domain-specific kernel that docc could not derive itself.

### Example

The following plugin replaces `bool foo()` (mangled name `_Z3foov`) with an SDFG that returns `true`. It is the complete plugin from docc's integration tests (`llvm/integration/tests/plugins/simple/`):

```cpp title="plugin.cpp"
#include <sdfg/builder/structured_sdfg_builder.h>
#include <sdfg/plugins/plugins.h>

static std::list<std::unique_ptr<sdfg::StructuredSDFG>> create_foo() {
    sdfg::builder::StructuredSDFGBuilder builder("_Z3foov", sdfg::FunctionType_CPU);
    sdfg::types::Scalar boolean(sdfg::types::PrimitiveType::Bool);
    builder.set_return_type(boolean);
    builder.add_constant_return(builder.subject().root(), "true", boolean);

    std::list<std::unique_ptr<sdfg::StructuredSDFG>> result;
    result.push_back(builder.move());
    return result;
}

extern "C" sdfg::plugins::Plugin register_docc_plugin() {
    return {
        .name = "simple",
        .version = "0.0.1",
        .description = "A simple example plugin",
        .register_plugin_callback = [](sdfg::plugins::Context& context) {},
        .sdfg_lookup = [](std::string name) {
            if (name == "_Z3foov") return create_foo();
            return std::list<std::unique_ptr<sdfg::StructuredSDFG>>();
        },
    };
}
```

Build it against the docc libraries. They are part of the docc package (component `sdfglib-development`) and of a [source build](../c-cpp/custom-installation.md):

```cmake title="CMakeLists.txt"
cmake_minimum_required(VERSION 3.26)
project(simple-plugin LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)

find_package(sdfglib CONFIG REQUIRED)

add_library(SimplePlugin SHARED plugin.cpp)
target_link_libraries(SimplePlugin PRIVATE sdfg::sdfg sdfg::opt)
```

### Load a plugin

Pass a comma-separated list of plugin libraries to the C/C++ compiler with [`-docc-plugins`](../c-cpp/usage.md#control-the-optimization):

```bash
docc-cpp -O3 -docc-plugins=$PWD/libSimplePlugin.so main.cpp -o main
```

Use absolute paths, or make sure that the libraries are on `LD_LIBRARY_PATH`. If a plugin cannot be loaded or lacks one of the callbacks, compilation stops with an error.

## Add a target

A new hardware target is a plugin that registers a schedule type, a map dispatcher that generates the device code, data transfers, and a loop scheduler. [Adding Custom Offload Targets](./custom-offload-targets.md) walks through a complete example. The Tenstorrent backend in `targets/tenstorrent/` is a full real-world target built this way.
