#include "sdfg/targets/tenstorrent/plugin.h"
#include "sdfg/passes/scheduler/scheduler_registry.h"
#include "sdfg/targets/tenstorrent/tenstorrent_scheduler.h"

namespace sdfg::tenstorrent {

bool tt_emit_full_metrics = false;
bool tt_force_close_devices_after_kernel = false;

void register_tenstorrent_plugin(bool emit_full_metrics, bool force_close_devices) {
    tt_emit_full_metrics = emit_full_metrics;
    tt_force_close_devices_after_kernel = force_close_devices;

    codegen::MapDispatcherRegistry::instance().register_map_dispatcher(
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


    codegen::LibraryNodeDispatcherRegistry::instance().register_library_node_dispatcher(
        LibraryNodeType_Tenstorrent_Offloading.value() + "::" + data_flow::ImplementationType_NONE.value(),
        [](codegen::LanguageExtension &language_extension,
           const Function &function,
           const data_flow::DataFlowGraph &data_flow_graph,
           const data_flow::LibraryNode &node) {
            return std::make_unique<TTDataOffloadingNodeDispatcher>(language_extension, function, data_flow_graph, node);
        }
    );

    serializer::LibraryNodeSerializerRegistry::instance()
        .register_library_node_serializer(LibraryNodeType_Tenstorrent_Offloading.value(), []() {
            return std::make_unique<TTDataOffloadingNodeSerializer>();
        });

    codegen::LibraryNodeDispatcherRegistry::instance().register_library_node_dispatcher(
        LibraryNodeType_Tenstorrent_CreateDevice.value() + "::" + data_flow::ImplementationType_NONE.value(),
        [](codegen::LanguageExtension &language_extension,
           const Function &function,
           const data_flow::DataFlowGraph &data_flow_graph,
           const data_flow::LibraryNode &node) {
            return std::make_unique<TTCreateDeviceDispatcher>(language_extension, function, data_flow_graph, node);
        }
    );

    serializer::LibraryNodeSerializerRegistry::instance()
        .register_library_node_serializer(LibraryNodeType_Tenstorrent_CreateDevice.value(), []() {
            return std::make_unique<TTCreateDeviceSerializer>();
        });

    // blas dispatchers

    codegen::LibraryNodeDispatcherRegistry::instance().register_library_node_dispatcher(
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

    codegen::LibraryNodeDispatcherRegistry::instance().register_library_node_dispatcher(
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

    codegen::LibraryNodeDispatcherRegistry::instance().register_library_node_dispatcher(
        math::blas::LibraryNodeType_DOT.value() + "::" + ImplementationType_Tenstorrent_WithoutTransfers.value(),
        [](codegen::LanguageExtension &language_extension,
           const Function &function,
           const data_flow::DataFlowGraph &data_flow_graph,
           const data_flow::LibraryNode &node) {
            return std::make_unique<blas::DotNodeDispatcher_Tenstorrent>(
                language_extension, function, data_flow_graph, dynamic_cast<const math::blas::DotNode &>(node)
            );
        }
    );

    passes::scheduler::SchedulerRegistry::instance()
        .register_loop_scheduler<
            passes::scheduler::TenstorrentScheduler>(passes::scheduler::TenstorrentScheduler::target());
}

} // namespace sdfg::tenstorrent
