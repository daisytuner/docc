#include "sdfg/targets/cuda/plugin.h"


namespace sdfg::cuda {

void register_cuda_plugin(plugins::Context& context) {
    auto& libNodeDispatcherRegistry = context.library_node_dispatcher_registry;
    auto& mapDispatcherRegistry = context.map_dispatcher_registry;
    auto& libNodeSerRegistry = context.library_node_serializer_registry;

    mapDispatcherRegistry.register_map_dispatcher(
        ScheduleType_CUDA::value(),
        [](codegen::LanguageExtension& language_extension,
           StructuredSDFG& sdfg,
           analysis::AnalysisManager& analysis_manager,
           structured_control_flow::Map& node,
           codegen::InstrumentationPlan& instrumentation_plan,
           codegen::ArgCapturePlan& arg_capture_plan) {
            return std::make_unique<CUDAMapDispatcher>(
                language_extension, sdfg, analysis_manager, node, instrumentation_plan, arg_capture_plan
            );
        }
    );

    libNodeDispatcherRegistry.register_library_node_dispatcher(
        cuda::LibraryNodeType_CUDA_Offloading.value() + "::" + data_flow::ImplementationType_NONE.value(),
        [](codegen::LanguageExtension& language_extension,
           const Function& function,
           const data_flow::DataFlowGraph& data_flow_graph,
           const data_flow::LibraryNode& node) {
            return std::make_unique<
                cuda::CUDADataOffloadingNodeDispatcher>(language_extension, function, data_flow_graph, node);
        }
    );

    libNodeSerRegistry.register_library_node_serializer(cuda::LibraryNodeType_CUDA_Offloading.value(), []() {
        return std::make_unique<cuda::CUDADataOffloadingNodeSerializer>();
    });


    // Dot - CUBLAS with data transfers
    libNodeDispatcherRegistry.register_library_node_dispatcher(
        math::blas::LibraryNodeType_DOT.value() + "::" + cuda::ImplementationType_CUDAWithTransfers.value(),
        [](codegen::LanguageExtension& language_extension,
           const Function& function,
           const data_flow::DataFlowGraph& data_flow_graph,
           const data_flow::LibraryNode& node) {
            return std::make_unique<blas::DotNodeDispatcher_CUBLASWithTransfers>(
                language_extension, function, data_flow_graph, dynamic_cast<const math::blas::DotNode&>(node)
            );
        }
    );
    // Dot - CUBLAS without data transfers
    libNodeDispatcherRegistry.register_library_node_dispatcher(
        math::blas::LibraryNodeType_DOT.value() + "::" + cuda::ImplementationType_CUDAWithoutTransfers.value(),
        [](codegen::LanguageExtension& language_extension,
           const Function& function,
           const data_flow::DataFlowGraph& data_flow_graph,
           const data_flow::LibraryNode& node) {
            return std::make_unique<blas::DotNodeDispatcher_CUBLASWithoutTransfers>(
                language_extension, function, data_flow_graph, dynamic_cast<const math::blas::DotNode&>(node)
            );
        }
    );

    // GEMM - CUBLAS with data transfers
    libNodeDispatcherRegistry.register_library_node_dispatcher(
        math::blas::LibraryNodeType_GEMM.value() + "::" + cuda::ImplementationType_CUDAWithTransfers.value(),
        [](codegen::LanguageExtension& language_extension,
           const Function& function,
           const data_flow::DataFlowGraph& data_flow_graph,
           const data_flow::LibraryNode& node) {
            return std::make_unique<blas::GEMMNodeDispatcher_CUBLASWithTransfers>(
                language_extension, function, data_flow_graph, dynamic_cast<const math::blas::GEMMNode&>(node)
            );
        }
    );
    // GEMM - CUBLAS without data transfers
    libNodeDispatcherRegistry.register_library_node_dispatcher(
        math::blas::LibraryNodeType_GEMM.value() + "::" + cuda::ImplementationType_CUDAWithoutTransfers.value(),
        [](codegen::LanguageExtension& language_extension,
           const Function& function,
           const data_flow::DataFlowGraph& data_flow_graph,
           const data_flow::LibraryNode& node) {
            return std::make_unique<blas::GEMMNodeDispatcher_CUBLASWithoutTransfers>(
                language_extension, function, data_flow_graph, dynamic_cast<const math::blas::GEMMNode&>(node)
            );
        }
    );


    // Memset - CUDA with data transfers
    libNodeDispatcherRegistry.register_library_node_dispatcher(
        sdfg::stdlib::LibraryNodeType_Memset.value() + "::" + cuda::ImplementationType_CUDAWithTransfers.value(),
        [](codegen::LanguageExtension& language_extension,
           const Function& function,
           const data_flow::DataFlowGraph& data_flow_graph,
           const data_flow::LibraryNode& node) {
            return std::make_unique<cuda::stdlib::MemsetNodeDispatcher_CUDAWithTransfers>(
                language_extension, function, data_flow_graph, dynamic_cast<const sdfg::stdlib::MemsetNode&>(node)
            );
        }
    );
    // Memset - CUDA without data transfers
    libNodeDispatcherRegistry.register_library_node_dispatcher(
        sdfg::stdlib::LibraryNodeType_Memset.value() + "::" + cuda::ImplementationType_CUDAWithoutTransfers.value(),
        [](codegen::LanguageExtension& language_extension,
           const Function& function,
           const data_flow::DataFlowGraph& data_flow_graph,
           const data_flow::LibraryNode& node) {
            return std::make_unique<cuda::stdlib::MemsetNodeDispatcher_CUDAWithoutTransfers>(
                language_extension, function, data_flow_graph, dynamic_cast<const sdfg::stdlib::MemsetNode&>(node)
            );
        }
    );


    context.scheduler_registry
        .register_loop_scheduler<passes::scheduler::CUDAScheduler>(passes::scheduler::CUDAScheduler::target());
}

} // namespace sdfg::cuda
