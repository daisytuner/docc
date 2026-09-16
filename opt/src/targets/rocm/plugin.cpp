#include "sdfg/targets/rocm/plugin.h"

#include <memory>

#include "sdfg/passes/scheduler/rocm_offload_scheduler.h"
#include "sdfg/targets/gpu/gpu_offload_map_dispatcher.h"
#include "sdfg/targets/gpu/gpu_offload_reduce_dispatcher.h"
#include "sdfg/targets/gpu/gpu_tile_target.h"
#include "sdfg/targets/rocm/rocm.h"
#include "sdfg/targets/rocm/rocm_mma_dispatcher.h"
#include "sdfg/targets/rocm/rocm_offload_dispatcher_strategy.h"
#include "sdfg/targets/rocm/rocm_reduce_dispatcher.h"
#include "sdfg/targets/rocm/tiles/async_copy_node.h"
#include "sdfg/tiles/tile_target_registry.h"

namespace sdfg::rocm {

void register_rocm_plugin(plugins::Context& context) {
    auto& libNodeDispatcherRegistry = context.get_library_node_dispatcher_registry();
    auto& mapDispatcherRegistry = context.get_map_dispatcher_registry();
    auto& reduceDispatcherRegistry = context.get_reduce_dispatcher_registry();
    auto& libNodeSerRegistry = context.get_library_node_serializer_registry();

    // The tile algebra's view of ROCm: 64-wide wavefront + level/storage mapping,
    // shared under both the legacy and the offload schedule value.
    auto rocm_tile_target = std::make_shared<
        ::sdfg::tiles::GPUTileTarget>(ROCM_WARP_SIZE, ScheduleType_ROCM_Offload::value(), ImplementationType_ROCM);
    context.get_tile_target_registry().register_target(ScheduleType_ROCM::value(), rocm_tile_target);
    context.get_tile_target_registry().register_target(ScheduleType_ROCM_Offload::value(), rocm_tile_target);

    mapDispatcherRegistry.register_map_dispatcher(
        ScheduleType_ROCM::value(),
        [](codegen::LanguageExtension& language_extension,
           StructuredSDFG& sdfg,
           analysis::AnalysisManager& analysis_manager,
           structured_control_flow::Map& node,
           codegen::InstrumentationPlan& instrumentation_plan,
           codegen::ArgCapturePlan& arg_capture_plan) {
            return std::make_unique<ROCMMapDispatcher>(
                language_extension, sdfg, analysis_manager, node, instrumentation_plan, arg_capture_plan
            );
        }
    );

    reduceDispatcherRegistry.register_reduce_dispatcher(
        ScheduleType_ROCM::value(),
        [](codegen::LanguageExtension& language_extension,
           StructuredSDFG& sdfg,
           analysis::AnalysisManager& analysis_manager,
           structured_control_flow::Reduce& node,
           codegen::InstrumentationPlan& instrumentation_plan,
           codegen::ArgCapturePlan& arg_capture_plan) {
            return std::make_unique<ROCMReduceDispatcher>(
                language_extension, sdfg, analysis_manager, node, instrumentation_plan, arg_capture_plan
            );
        }
    );

    // Offload (target_level-based) dispatchers. Registration is first-registration-wins, so these
    // are currently shadowed by the deprecated dimension-based ROCm dispatchers registered above
    // and remain inactive until the deprecated registrations are removed.
    mapDispatcherRegistry.register_map_dispatcher(
        ScheduleType_ROCM_Offload::value(),
        [](codegen::LanguageExtension& language_extension,
           StructuredSDFG& sdfg,
           analysis::AnalysisManager& analysis_manager,
           structured_control_flow::Map& node,
           codegen::InstrumentationPlan& instrumentation_plan,
           codegen::ArgCapturePlan& arg_capture_plan) {
            return std::make_unique<gpu::GPUOffloadMapDispatcher>(
                language_extension,
                sdfg,
                analysis_manager,
                node,
                instrumentation_plan,
                arg_capture_plan,
                std::make_unique<rocm::ROCMOffloadDispatcherStrategy>(sdfg)
            );
        }
    );

    reduceDispatcherRegistry.register_reduce_dispatcher(
        ScheduleType_ROCM_Offload::value(),
        [](codegen::LanguageExtension& language_extension,
           StructuredSDFG& sdfg,
           analysis::AnalysisManager& analysis_manager,
           structured_control_flow::Reduce& node,
           codegen::InstrumentationPlan& instrumentation_plan,
           codegen::ArgCapturePlan& arg_capture_plan) {
            return std::make_unique<gpu::GPUOffloadReduceDispatcher>(
                language_extension,
                sdfg,
                analysis_manager,
                node,
                instrumentation_plan,
                arg_capture_plan,
                std::make_unique<rocm::ROCMOffloadDispatcherStrategy>(sdfg)
            );
        }
    );

    libNodeDispatcherRegistry.register_library_node_dispatcher(
        rocm::LibraryNodeType_ROCM_Offloading,
        data_flow::ImplementationType_NONE,
        [](codegen::LanguageExtension& language_extension,
           const Function& function,
           const data_flow::DataFlowGraph& data_flow_graph,
           const data_flow::LibraryNode& node) {
            return std::make_unique<
                rocm::ROCMDataOffloadingNodeDispatcher>(language_extension, function, data_flow_graph, node);
        }
    );

    libNodeSerRegistry.register_library_node_serializer(rocm::LibraryNodeType_ROCM_Offloading.value(), []() {
        return std::make_unique<rocm::ROCMDataOffloadingNodeSerializer>();
    });


    // Dot - ROCMBLAS with data transfers
    libNodeDispatcherRegistry.register_library_node_dispatcher(
        math::blas::LibraryNodeType_DOT,
        rocm::ImplementationType_ROCMWithTransfers,
        [](codegen::LanguageExtension& language_extension,
           const Function& function,
           const data_flow::DataFlowGraph& data_flow_graph,
           const data_flow::LibraryNode& node) {
            return std::make_unique<blas::DotNodeDispatcher_ROCMBLASWithTransfers>(
                language_extension, function, data_flow_graph, dynamic_cast<const math::blas::DotNode&>(node)
            );
        }
    );
    // Dot - ROCMBLAS without data transfers
    libNodeDispatcherRegistry.register_library_node_dispatcher(
        math::blas::LibraryNodeType_DOT,
        rocm::ImplementationType_ROCMWithoutTransfers,
        [](codegen::LanguageExtension& language_extension,
           const Function& function,
           const data_flow::DataFlowGraph& data_flow_graph,
           const data_flow::LibraryNode& node) {
            return std::make_unique<blas::DotNodeDispatcher_ROCMBLASWithoutTransfers>(
                language_extension, function, data_flow_graph, dynamic_cast<const math::blas::DotNode&>(node)
            );
        }
    );

    // GEMM - ROCMBLAS with data transfers
    libNodeDispatcherRegistry.register_library_node_dispatcher(
        math::blas::LibraryNodeType_GEMM,
        rocm::ImplementationType_ROCMWithTransfers,
        [](codegen::LanguageExtension& language_extension,
           const Function& function,
           const data_flow::DataFlowGraph& data_flow_graph,
           const data_flow::LibraryNode& node) {
            return std::make_unique<blas::GEMMNodeDispatcher_ROCMBLASWithTransfers>(
                language_extension, function, data_flow_graph, dynamic_cast<const math::blas::GEMMNode&>(node)
            );
        }
    );

    // GEMM - ROCM hand-tuned kernel (data already on GPU)
    libNodeDispatcherRegistry.register_library_node_dispatcher(
        math::blas::LibraryNodeType_GEMM,
        rocm::ImplementationType_ROCMWithoutTransfers,
        [](codegen::LanguageExtension& language_extension,
           const Function& function,
           const data_flow::DataFlowGraph& data_flow_graph,
           const data_flow::LibraryNode& node) -> std::unique_ptr<codegen::LibraryNodeDispatcher> {
            auto& library_node = dynamic_cast<const math::blas::GEMMNode&>(node);
            if (library_node.precision() != math::blas::BLAS_Precision::s) {
                return std::make_unique<blas::GEMMNodeDispatcher_ROCMBLASWithoutTransfers>(
                    language_extension, function, data_flow_graph, library_node
                );
            } else {
                return std::make_unique<
                    blas::GEMMNodeDispatcher_ROCMHandTuned>(language_extension, function, data_flow_graph, library_node);
            }
        }
    );

    // BatchedGEMM - ROCMBLAS with data transfers
    libNodeDispatcherRegistry.register_library_node_dispatcher(
        math::blas::LibraryNodeType_BatchedGEMM,
        rocm::ImplementationType_ROCMWithTransfers,
        [](codegen::LanguageExtension& language_extension,
           const Function& function,
           const data_flow::DataFlowGraph& data_flow_graph,
           const data_flow::LibraryNode& node) {
            return std::make_unique<blas::BatchedGEMMNodeDispatcher_ROCMBLASWithTransfers>(
                language_extension, function, data_flow_graph, dynamic_cast<const math::blas::BatchedGEMMNode&>(node)
            );
        }
    );
    // BatchedGEMM - ROCMBLAS without data transfers
    libNodeDispatcherRegistry.register_library_node_dispatcher(
        math::blas::LibraryNodeType_BatchedGEMM,
        rocm::ImplementationType_ROCMWithoutTransfers,
        [](codegen::LanguageExtension& language_extension,
           const Function& function,
           const data_flow::DataFlowGraph& data_flow_graph,
           const data_flow::LibraryNode& node) {
            return std::make_unique<blas::BatchedGEMMNodeDispatcher_ROCMBLASWithoutTransfers>(
                language_extension, function, data_flow_graph, dynamic_cast<const math::blas::BatchedGEMMNode&>(node)
            );
        }
    );


    // Memset - ROCM with data transfers
    libNodeDispatcherRegistry.register_library_node_dispatcher(
        sdfg::stdlib::LibraryNodeType_Memset,
        rocm::ImplementationType_ROCMWithTransfers,
        [](codegen::LanguageExtension& language_extension,
           const Function& function,
           const data_flow::DataFlowGraph& data_flow_graph,
           const data_flow::LibraryNode& node) {
            return std::make_unique<rocm::stdlib::MemsetNodeDispatcher_ROCMWithTransfers>(
                language_extension, function, data_flow_graph, dynamic_cast<const sdfg::stdlib::MemsetNode&>(node)
            );
        }
    );
    // Memset - ROCM without data transfers
    libNodeDispatcherRegistry.register_library_node_dispatcher(
        sdfg::stdlib::LibraryNodeType_Memset,
        rocm::ImplementationType_ROCMWithoutTransfers,
        [](codegen::LanguageExtension& language_extension,
           const Function& function,
           const data_flow::DataFlowGraph& data_flow_graph,
           const data_flow::LibraryNode& node) {
            return std::make_unique<rocm::stdlib::MemsetNodeDispatcher_ROCMWithoutTransfers>(
                language_extension, function, data_flow_graph, dynamic_cast<const sdfg::stdlib::MemsetNode&>(node)
            );
        }
    );


    // Memcpy - ROCM with data transfers
    libNodeDispatcherRegistry.register_library_node_dispatcher(
        sdfg::stdlib::LibraryNodeType_Memcpy,
        rocm::ImplementationType_ROCMWithTransfers,
        [](codegen::LanguageExtension& language_extension,
           const Function& function,
           const data_flow::DataFlowGraph& data_flow_graph,
           const data_flow::LibraryNode& node) {
            return std::make_unique<rocm::stdlib::MemcpyNodeDispatcher_ROCMWithTransfers>(
                language_extension, function, data_flow_graph, dynamic_cast<const sdfg::stdlib::MemcpyNode&>(node)
            );
        }
    );
    // Memcpy - ROCM without data transfers
    libNodeDispatcherRegistry.register_library_node_dispatcher(
        sdfg::stdlib::LibraryNodeType_Memcpy,
        rocm::ImplementationType_ROCMWithoutTransfers,
        [](codegen::LanguageExtension& language_extension,
           const Function& function,
           const data_flow::DataFlowGraph& data_flow_graph,
           const data_flow::LibraryNode& node) {
            return std::make_unique<rocm::stdlib::MemcpyNodeDispatcher_ROCMWithoutTransfers>(
                language_extension, function, data_flow_graph, dynamic_cast<const sdfg::stdlib::MemcpyNode&>(node)
            );
        }
    );

    libNodeDispatcherRegistry.register_library_node_dispatcher(
        math::tensor::LibraryNodeType_MatMul,
        gpu::rocm::ImplementationType_ROCM_MMA_GFX1201,
        [](codegen::LanguageExtension& language_extension,
           const Function& function,
           const data_flow::DataFlowGraph& data_flow_graph,
           const data_flow::LibraryNode& node) {
            return std::make_unique<gpu::rocm::RocmMmaMatmulDispatcher>(
                language_extension, function, data_flow_graph, dynamic_cast<const math::tensor::MatMulNode&>(node)
            );
        }
    );
    libNodeDispatcherRegistry.register_library_node_dispatcher(
        math::tensor::LibraryNodeType_MatMul,
        gpu::rocm::ImplementationType_ROCM_MMA_GFX90A,
        [](codegen::LanguageExtension& language_extension,
           const Function& function,
           const data_flow::DataFlowGraph& data_flow_graph,
           const data_flow::LibraryNode& node) {
            return std::make_unique<gpu::rocm::RocmMmaMatmulDispatcher>(
                language_extension, function, data_flow_graph, dynamic_cast<const math::tensor::MatMulNode&>(node)
            );
        }
    );
    // Async copy / pipeline primitives (software pipelining)
    libNodeDispatcherRegistry.register_library_node_dispatcher(
        ::sdfg::tiles::LibraryNodeType_CpAsyncCopy,
        ImplementationType_ROCM,
        [](codegen::LanguageExtension& language_extension,
           const Function& function,
           const data_flow::DataFlowGraph& data_flow_graph,
           const data_flow::LibraryNode& node) {
            return std::make_unique<rocm::tiles::CpAsyncCopyNodeDispatcher>(
                language_extension, function, data_flow_graph, dynamic_cast<const ::sdfg::tiles::CpAsyncCopyNode&>(node)
            );
        }
    );
    libNodeDispatcherRegistry.register_library_node_dispatcher(
        ::sdfg::tiles::LibraryNodeType_VectorCopy,
        ImplementationType_ROCM,
        [](codegen::LanguageExtension& language_extension,
           const Function& function,
           const data_flow::DataFlowGraph& data_flow_graph,
           const data_flow::LibraryNode& node) {
            return std::make_unique<rocm::tiles::VectorCopyNodeDispatcher>(
                language_extension, function, data_flow_graph, dynamic_cast<const ::sdfg::tiles::VectorCopyNode&>(node)
            );
        }
    );
    libNodeDispatcherRegistry.register_library_node_dispatcher(
        ::sdfg::tiles::LibraryNodeType_PipelineCommit,
        ImplementationType_ROCM,
        [](codegen::LanguageExtension& language_extension,
           const Function& function,
           const data_flow::DataFlowGraph& data_flow_graph,
           const data_flow::LibraryNode& node) {
            return std::make_unique<rocm::tiles::PipelineCommitNodeDispatcher>(
                language_extension,
                function,
                data_flow_graph,
                dynamic_cast<const ::sdfg::tiles::PipelineCommitNode&>(node)
            );
        }
    );
    libNodeDispatcherRegistry.register_library_node_dispatcher(
        ::sdfg::tiles::LibraryNodeType_PipelineWait,
        ImplementationType_ROCM,
        [](codegen::LanguageExtension& language_extension,
           const Function& function,
           const data_flow::DataFlowGraph& data_flow_graph,
           const data_flow::LibraryNode& node) {
            return std::make_unique<rocm::tiles::PipelineWaitNodeDispatcher>(
                language_extension, function, data_flow_graph, dynamic_cast<const ::sdfg::tiles::PipelineWaitNode&>(node)
            );
        }
    );


    context.get_scheduler_registry()
        .register_loop_scheduler<
            passes::scheduler::ROCMOffloadScheduler>(passes::scheduler::ROCMOffloadScheduler::target());
}

} // namespace sdfg::rocm
