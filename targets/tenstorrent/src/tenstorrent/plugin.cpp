#include "docc/target/tenstorrent/plugin.h"

#include "docc/compile/src_file_compiler_builder.h"
#include "docc/target/tenstorrent/math_node_implementation_override_pass.h"
#include "docc/target/tenstorrent/target.h"
#include "docc/target/tenstorrent/tenstorrent_scheduler.h"
#include "sdfg/passes/scheduler/scheduler_registry.h"
#include "sdfg/passes/targets/target_mapping_pass.h"
#include "sdfg/plugins/target_mapping.h"
#include "sdfg/plugins/targets.h"

namespace sdfg::tenstorrent {
bool tt_emit_full_metrics = false;
bool tt_force_close_devices_after_kernel = false;

docc::target::DoccTarget tenstorrent_target = {
    .short_name = "tenstorrent",
    .apply_additional_compile_options = [](docc::compile::SrcFileCompilerBuilder &builder) -> bool {
        auto tt_metal_path = std::getenv("TT_METAL_HOME");
        if (!tt_metal_path) {
            throw std::runtime_error("TT_METAL_HOME not set");
        }
        std::filesystem::path tt_path = std::filesystem::path(tt_metal_path) / "build";
        auto lib_path = tt_path / "lib";
        builder.add_library_path(lib_path);
        builder.add_link_option("-ltt_metal");
        builder.add_link_option("-Wl,-rpath,\"" + lib_path.string() + "\"");
        builder.add_include_path(tt_path / "include");
        builder.add_include_path(tt_path / "include" / "metalium-thirdparty");
        builder.add_compile_option("-DSPDLOG_FMT_EXTERNAL");

        docc::compile::SrcFileCompilerBuilder sub;
        sub.inherit(builder);
        sub.set_src_extension(TTKernelManagementCodegen::TT_SNIPPET_EXT);
        sub.codegen_only();
        builder.redirect_snippet(TTKernelManagementCodegen::TT_SNIPPET_EXT, std::move(sub));
        return true;
    },
    .apply_expand_time_mapping = nullptr,
    .apply_sched_time_mapping = [](sdfg::builder::StructuredSDFGBuilder &builder,
                                   sdfg::analysis::AnalysisManager &analysis_manager,
                                   const docc::target::TargetOptions &options) -> bool {
        std::vector<std::shared_ptr<plugins::TargetMapper>> mappers{std::make_shared<TTLibNodeMapper>()};
        passes::TargetMappingPass mappingPass(mappers);
        return mappingPass.run_pass(builder, analysis_manager);
    },
    .get_target_loop_schedulers = [](const docc::target::TargetOptions &options
                                  ) -> std::vector<std::shared_ptr<sdfg::passes::scheduler::LoopScheduler>> {
        std::vector<std::shared_ptr<sdfg::passes::scheduler::LoopScheduler>> schedulers;
        schedulers.push_back(std::make_shared<sdfg::passes::scheduler::TenstorrentScheduler>());
        return schedulers;
    }
};

void register_tenstorrent_plugin(bool emit_full_metrics, bool force_close_devices) {
    tt_emit_full_metrics = emit_full_metrics;
    tt_force_close_devices_after_kernel = force_close_devices;
    auto context = plugins::Context::global_context();
    docc::target::tenstorrent::register_plugin(context);
}

} // namespace sdfg::tenstorrent

namespace docc::target::tenstorrent {

using namespace sdfg::tenstorrent;
using namespace sdfg;

void register_plugin(sdfg::plugins::Context &context) {
    context.map_dispatcher_registry.register_map_dispatcher(
        ScheduleType_Tenstorrent_Device::value(),
        [](codegen::LanguageExtension &language_extension,
           StructuredSDFG &sdfg,
           analysis::AnalysisManager &analysis_manager,
           Map &node,
           codegen::InstrumentationPlan &instrumentation_plan,
           codegen::ArgCapturePlan &arg_capture_plan) {
            return std::make_unique<TenstorrentMapDispatcher>(
                language_extension, sdfg, analysis_manager, node, instrumentation_plan, arg_capture_plan
            );
        }
    );


    context.library_node_dispatcher_registry.register_library_node_dispatcher(
        LibraryNodeType_Tenstorrent_Offloading.value() + "::" + data_flow::ImplementationType_NONE.value(),
        [](codegen::LanguageExtension &language_extension,
           const Function &function,
           const data_flow::DataFlowGraph &data_flow_graph,
           const data_flow::LibraryNode &node) {
            return std::make_unique<TTDataOffloadingNodeDispatcher>(language_extension, function, data_flow_graph, node);
        }
    );

    context.library_node_serializer_registry
        .register_library_node_serializer(LibraryNodeType_Tenstorrent_Offloading.value(), []() {
            return std::make_unique<TTDataOffloadingNodeSerializer>();
        });

    context.library_node_dispatcher_registry.register_library_node_dispatcher(
        LibraryNodeType_Tenstorrent_CreateDevice.value() + "::" + data_flow::ImplementationType_NONE.value(),
        [](codegen::LanguageExtension &language_extension,
           const Function &function,
           const data_flow::DataFlowGraph &data_flow_graph,
           const data_flow::LibraryNode &node) {
            return std::make_unique<TTCreateDeviceDispatcher>(language_extension, function, data_flow_graph, node);
        }
    );

    context.library_node_serializer_registry
        .register_library_node_serializer(LibraryNodeType_Tenstorrent_CreateDevice.value(), []() {
            return std::make_unique<TTCreateDeviceSerializer>();
        });

    // blas dispatchers

    context.library_node_dispatcher_registry.register_library_node_dispatcher(
        math::blas::LibraryNodeType_GEMM.value() + "::" + ImplementationType_Tenstorrent_WithTransfers.value(),
        [](codegen::LanguageExtension &language_extension,
           const Function &function,
           const data_flow::DataFlowGraph &data_flow_graph,
           const data_flow::LibraryNode &node) {
            return std::make_unique<blas::GEMMNodeDispatcher_Tenstorrent>(
                language_extension, function, data_flow_graph, dynamic_cast<const math::blas::GEMMNode &>(node)
            );
        }
    );

    context.library_node_dispatcher_registry.register_library_node_dispatcher(
        math::blas::LibraryNodeType_DOT.value() + "::" + ImplementationType_Tenstorrent_WithTransfers.value(),
        [](codegen::LanguageExtension &language_extension,
           const Function &function,
           const data_flow::DataFlowGraph &data_flow_graph,
           const data_flow::LibraryNode &node) {
            return std::make_unique<blas::DotNodeDispatcher_Tenstorrent>(
                language_extension, function, data_flow_graph, dynamic_cast<const math::blas::DotNode &>(node)
            );
        }
    );

    context.scheduler_registry.register_loop_scheduler<
        passes::scheduler::TenstorrentScheduler>(passes::scheduler::TenstorrentScheduler::target());

    context.add_target(&tenstorrent_target);
}

} // namespace docc::target::tenstorrent
