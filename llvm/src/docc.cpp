#include "docc/docc.h"

#include <llvm/Support/CommandLine.h>

#include <cstdlib>
#include <dlfcn.h>
#include <mutex>

#include <sdfg/codegen/dispatchers/node_dispatcher_registry.h>
#include <sdfg/plugins/plugins.h>
#include <sdfg/targets/cuda/plugin.h>
#include <sdfg/targets/memory/plugin.h>
#include <sdfg/targets/omp/plugin.h>
#ifdef DOCC_BUILD_TARGET_TENSTORRENT
#include <docc/target/tenstorrent/plugin.h>
#endif

#include "docc/utils.h"
#include "sdfg/passes/scheduler/scheduler_registry.h"
#include "sdfg/targets/rocm/plugin.h"

namespace docc {

llvm::cl::opt<std::string> DOCC_Plugins(
    "docc-plugins",
    llvm::cl::desc("Loads DOCC plugins"),
    llvm::cl::init(""),
    llvm::cl::value_desc("a comma-separated list of plugins")
);

docc::PluginRegistry plugin_registry;

static std::once_flag dispatcher_registration_flag;

void register_sdfg_dispatchers() {
    std::call_once(dispatcher_registration_flag, []() {
        sdfg::codegen::register_default_dispatchers();
        sdfg::serializer::register_default_serializers();

        sdfg::omp::register_omp_plugin();
        sdfg::cuda::register_cuda_plugin();
        sdfg::rocm::register_rocm_plugin();
#ifdef DOCC_BUILD_TARGET_TENSTORRENT
        sdfg::tenstorrent::register_tenstorrent_plugin();
#endif
        sdfg::offloading::register_external_data_transfers_plugin();

        sdfg::plugins::Context context = sdfg::plugins::Context::global_context();

        if (!DOCC_Plugins.empty()) {
            std::stringstream ss(DOCC_Plugins);
            std::string plugin;
            while (std::getline(ss, plugin, ',')) {
                void* handle = dlopen(plugin.c_str(), RTLD_LAZY);
                if (!handle) {
                    LLVM_DEBUG_PRINTLN("[docc] Failed to load plugin: " << plugin << ", " << dlerror());
                    exit(EXIT_FAILURE);
                }
                auto _register_docc_plugin =
                    reinterpret_cast<sdfg::plugins::Plugin (*)(void)>(dlsym(handle, "register_docc_plugin"));
                if (!_register_docc_plugin) {
                    LLVM_DEBUG_PRINTLN("[docc] Failed to find register_docc_plugin in plugin: " << plugin);
                    exit(EXIT_FAILURE);
                }

                sdfg::plugins::Plugin p = _register_docc_plugin();
                if (!p.register_plugin_callback) {
                    LLVM_DEBUG_PRINTLN("[docc] Plugin " << plugin << " does not have a register callback");
                    exit(EXIT_FAILURE);
                }
                p.register_plugin_callback(context);
                if (!p.sdfg_lookup) {
                    LLVM_DEBUG_PRINTLN("[docc] Plugin " << plugin << " does not have an sdfg lookup function");
                    exit(EXIT_FAILURE);
                }
                plugin_registry.plugins.push_back({p, handle});
                LLVM_DEBUG_PRINTLN("[docc] Loaded plugin: " << p.name << " (" << p.version << ")");
            }
        }
    });
}

} // namespace docc
