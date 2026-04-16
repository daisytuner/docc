#pragma once

#include <sdfg/codegen/dispatchers/node_dispatcher_registry.h>
#include <sdfg/data_flow/library_nodes/math/blas/dot_node.h>
#include <sdfg/data_flow/library_nodes/math/blas/gemm_node.h>
#include <sdfg/data_flow/library_nodes/stdlib/memset.h>
#include <sdfg/serializer/json_serializer.h>

#include "sdfg/codegen/language_extension.h"
#include "sdfg/passes/scheduler/rocm_scheduler.h"
#include "sdfg/passes/scheduler/scheduler_registry.h"
#include "sdfg/targets/rocm/blas/dot.h"
#include "sdfg/targets/rocm/blas/gemm.h"
#include "sdfg/targets/rocm/blas/gemm_handtuned.h"
#include "sdfg/targets/rocm/rocm.h"
#include "sdfg/targets/rocm/rocm_data_offloading_node.h"
#include "sdfg/targets/rocm/rocm_map_dispatcher.h"
#include "sdfg/targets/rocm/stdlib/memset.h"

namespace sdfg {
namespace rocm {

inline void register_rocm_plugin() {
    codegen::MapDispatcherRegistry::instance().register_map_dispatcher(
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

    codegen::LibraryNodeDispatcherRegistry::instance().register_library_node_dispatcher(
        rocm::LibraryNodeType_ROCM_Offloading.value() + "::" + data_flow::ImplementationType_NONE.value(),
        [](codegen::LanguageExtension& language_extension,
           const Function& function,
           const data_flow::DataFlowGraph& data_flow_graph,
           const data_flow::LibraryNode& node) {
            return std::make_unique<
                rocm::ROCMDataOffloadingNodeDispatcher>(language_extension, function, data_flow_graph, node);
        }
    );

    serializer::LibraryNodeSerializerRegistry::instance()
        .register_library_node_serializer(rocm::LibraryNodeType_ROCM_Offloading.value(), []() {
            return std::make_unique<rocm::ROCMDataOffloadingNodeSerializer>();
        });


    // Dot - ROCMBLAS with data transfers
    codegen::LibraryNodeDispatcherRegistry::instance().register_library_node_dispatcher(
        math::blas::LibraryNodeType_DOT.value() + "::" + rocm::ImplementationType_ROCMWithTransfers.value(),
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
    codegen::LibraryNodeDispatcherRegistry::instance().register_library_node_dispatcher(
        math::blas::LibraryNodeType_DOT.value() + "::" + rocm::ImplementationType_ROCMWithoutTransfers.value(),
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
    codegen::LibraryNodeDispatcherRegistry::instance().register_library_node_dispatcher(
        math::blas::LibraryNodeType_GEMM.value() + "::" + rocm::ImplementationType_ROCMWithTransfers.value(),
        [](codegen::LanguageExtension& language_extension,
           const Function& function,
           const data_flow::DataFlowGraph& data_flow_graph,
           const data_flow::LibraryNode& node) {
            return std::make_unique<blas::GEMMNodeDispatcher_ROCMBLASWithTransfers>(
                language_extension, function, data_flow_graph, dynamic_cast<const math::blas::GEMMNode&>(node)
            );
        }
    );
    // GEMM - ROCMBLAS without data transfers
    // codegen::LibraryNodeDispatcherRegistry::instance().register_library_node_dispatcher(
    //     math::blas::LibraryNodeType_GEMM.value() + "::" + rocm::ImplementationType_ROCMWithoutTransfers.value(),
    //     [](codegen::LanguageExtension& language_extension,
    //        const Function& function,
    //        const data_flow::DataFlowGraph& data_flow_graph,
    //        const data_flow::LibraryNode& node) {
    //         return std::make_unique<blas::GEMMNodeDispatcher_ROCMBLASWithoutTransfers>(
    //             language_extension, function, data_flow_graph, dynamic_cast<const math::blas::GEMMNode&>(node)
    //         );
    //     }
    // );


    // GEMM - ROCM hand-tuned kernel (data already on GPU)
    codegen::LibraryNodeDispatcherRegistry::instance().register_library_node_dispatcher(
        math::blas::LibraryNodeType_GEMM.value() + "::" + rocm::ImplementationType_ROCMWithoutTransfers.value(),
        [](codegen::LanguageExtension& language_extension,
           const Function& function,
           const data_flow::DataFlowGraph& data_flow_graph,
           const data_flow::LibraryNode& node) {
            return std::make_unique<blas::GEMMNodeDispatcher_ROCMHandTuned>(
                language_extension, function, data_flow_graph, dynamic_cast<const math::blas::GEMMNode&>(node)
            );
        }
    );

    // Memset - ROCM with data transfers
    codegen::LibraryNodeDispatcherRegistry::instance().register_library_node_dispatcher(
        sdfg::stdlib::LibraryNodeType_Memset.value() + "::" + rocm::ImplementationType_ROCMWithTransfers.value(),
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
    codegen::LibraryNodeDispatcherRegistry::instance().register_library_node_dispatcher(
        sdfg::stdlib::LibraryNodeType_Memset.value() + "::" + rocm::ImplementationType_ROCMWithoutTransfers.value(),
        [](codegen::LanguageExtension& language_extension,
           const Function& function,
           const data_flow::DataFlowGraph& data_flow_graph,
           const data_flow::LibraryNode& node) {
            return std::make_unique<rocm::stdlib::MemsetNodeDispatcher_ROCMWithoutTransfers>(
                language_extension, function, data_flow_graph, dynamic_cast<const sdfg::stdlib::MemsetNode&>(node)
            );
        }
    );


    passes::scheduler::SchedulerRegistry::instance()
        .register_loop_scheduler<passes::scheduler::ROCMScheduler>(passes::scheduler::ROCMScheduler::target());
}

} // namespace rocm
} // namespace sdfg
