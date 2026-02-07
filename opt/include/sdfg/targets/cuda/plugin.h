#pragma once

#include <sdfg/codegen/dispatchers/node_dispatcher_registry.h>
#include <sdfg/data_flow/library_nodes/math/blas/dot_node.h>
#include <sdfg/data_flow/library_nodes/math/blas/gemm_node.h>
#include <sdfg/serializer/json_serializer.h>

#include "sdfg/codegen/language_extension.h"
#include "sdfg/passes/scheduler/cuda_scheduler.h"
#include "sdfg/passes/scheduler/scheduler_registry.h"
#include "sdfg/targets/cuda/blas/dot.h"
#include "sdfg/targets/cuda/blas/gemm.h"
#include "sdfg/targets/cuda/cuda.h"
#include "sdfg/targets/cuda/cuda_data_offloading_node.h"
#include "sdfg/targets/cuda/cuda_map_dispatcher.h"

namespace sdfg {
namespace cuda {

inline void register_cuda_plugin() {
    codegen::MapDispatcherRegistry::instance().register_map_dispatcher(
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

    codegen::LibraryNodeDispatcherRegistry::instance().register_library_node_dispatcher(
        cuda::LibraryNodeType_CUDA_Offloading.value() + "::" + data_flow::ImplementationType_NONE.value(),
        [](codegen::LanguageExtension& language_extension,
           const Function& function,
           const data_flow::DataFlowGraph& data_flow_graph,
           const data_flow::LibraryNode& node) {
            return std::make_unique<
                cuda::CUDADataOffloadingNodeDispatcher>(language_extension, function, data_flow_graph, node);
        }
    );

    serializer::LibraryNodeSerializerRegistry::instance()
        .register_library_node_serializer(cuda::LibraryNodeType_CUDA_Offloading.value(), []() {
            return std::make_unique<cuda::CUDADataOffloadingNodeSerializer>();
        });


    // Dot - CUBLAS with data transfers
    codegen::LibraryNodeDispatcherRegistry::instance().register_library_node_dispatcher(
        math::blas::LibraryNodeType_DOT.value() + "::" + cuda::blas::ImplementationType_CUBLASWithTransfers.value(),
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
    codegen::LibraryNodeDispatcherRegistry::instance().register_library_node_dispatcher(
        math::blas::LibraryNodeType_DOT.value() + "::" + cuda::blas::ImplementationType_CUBLASWithoutTransfers.value(),
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
    codegen::LibraryNodeDispatcherRegistry::instance().register_library_node_dispatcher(
        math::blas::LibraryNodeType_GEMM.value() + "::" + cuda::blas::ImplementationType_CUBLASWithTransfers.value(),
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
    codegen::LibraryNodeDispatcherRegistry::instance().register_library_node_dispatcher(
        math::blas::LibraryNodeType_GEMM.value() + "::" + cuda::blas::ImplementationType_CUBLASWithoutTransfers.value(),
        [](codegen::LanguageExtension& language_extension,
           const Function& function,
           const data_flow::DataFlowGraph& data_flow_graph,
           const data_flow::LibraryNode& node) {
            return std::make_unique<blas::GEMMNodeDispatcher_CUBLASWithoutTransfers>(
                language_extension, function, data_flow_graph, dynamic_cast<const math::blas::GEMMNode&>(node)
            );
        }
    );


    passes::scheduler::SchedulerRegistry::instance()
        .register_loop_scheduler<passes::scheduler::CUDAScheduler>(passes::scheduler::CUDAScheduler::target());
}

} // namespace cuda
} // namespace sdfg
