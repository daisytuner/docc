#pragma once

#include <sdfg/codegen/dispatchers/node_dispatcher_registry.h>
#include <sdfg/data_flow/library_nodes/math/blas/dot_node.h>
#include <sdfg/data_flow/library_nodes/math/blas/gemm_node.h>
#include <sdfg/serializer/json_serializer.h>

#include "sdfg/codegen/language_extension.h"
#include "sdfg/passes/scheduler/hip_scheduler.h"
#include "sdfg/passes/scheduler/scheduler_registry.h"
#include "sdfg/targets/hip/blas/dot.h"
#include "sdfg/targets/hip/blas/gemm.h"
#include "sdfg/targets/hip/hip.h"
#include "sdfg/targets/hip/hip_data_offloading_node.h"
#include "sdfg/targets/hip/hip_map_dispatcher.h"

namespace sdfg {
namespace hip {

inline void register_hip_plugin() {
    codegen::MapDispatcherRegistry::instance().register_map_dispatcher(
        ScheduleType_HIP::value(),
        [](codegen::LanguageExtension& language_extension,
           StructuredSDFG& sdfg,
           analysis::AnalysisManager& analysis_manager,
           structured_control_flow::Map& node,
           codegen::InstrumentationPlan& instrumentation_plan,
           codegen::ArgCapturePlan& arg_capture_plan) {
            return std::make_unique<HIPMapDispatcher>(
                language_extension, sdfg, analysis_manager, node, instrumentation_plan, arg_capture_plan
            );
        }
    );

    codegen::LibraryNodeDispatcherRegistry::instance().register_library_node_dispatcher(
        hip::LibraryNodeType_HIP_Offloading.value() + "::" + data_flow::ImplementationType_NONE.value(),
        [](codegen::LanguageExtension& language_extension,
           const Function& function,
           const data_flow::DataFlowGraph& data_flow_graph,
           const data_flow::LibraryNode& node) {
            return std::make_unique<
                hip::HIPDataOffloadingNodeDispatcher>(language_extension, function, data_flow_graph, node);
        }
    );

    serializer::LibraryNodeSerializerRegistry::instance()
        .register_library_node_serializer(hip::LibraryNodeType_HIP_Offloading.value(), []() {
            return std::make_unique<hip::HIPDataOffloadingNodeSerializer>();
        });


    // Dot - HIPBLAS with data transfers
    codegen::LibraryNodeDispatcherRegistry::instance().register_library_node_dispatcher(
        math::blas::LibraryNodeType_DOT.value() + "::" + hip::blas::ImplementationType_HIPBLASWithTransfers.value(),
        [](codegen::LanguageExtension& language_extension,
           const Function& function,
           const data_flow::DataFlowGraph& data_flow_graph,
           const data_flow::LibraryNode& node) {
            return std::make_unique<blas::DotNodeDispatcher_HIPBLASWithTransfers>(
                language_extension, function, data_flow_graph, dynamic_cast<const math::blas::DotNode&>(node)
            );
        }
    );
    // Dot - HIPBLAS without data transfers
    codegen::LibraryNodeDispatcherRegistry::instance().register_library_node_dispatcher(
        math::blas::LibraryNodeType_DOT.value() + "::" + hip::blas::ImplementationType_HIPBLASWithoutTransfers.value(),
        [](codegen::LanguageExtension& language_extension,
           const Function& function,
           const data_flow::DataFlowGraph& data_flow_graph,
           const data_flow::LibraryNode& node) {
            return std::make_unique<blas::DotNodeDispatcher_HIPBLASWithoutTransfers>(
                language_extension, function, data_flow_graph, dynamic_cast<const math::blas::DotNode&>(node)
            );
        }
    );

    // GEMM - HIPBLAS with data transfers
    codegen::LibraryNodeDispatcherRegistry::instance().register_library_node_dispatcher(
        math::blas::LibraryNodeType_GEMM.value() + "::" + hip::blas::ImplementationType_HIPBLASWithTransfers.value(),
        [](codegen::LanguageExtension& language_extension,
           const Function& function,
           const data_flow::DataFlowGraph& data_flow_graph,
           const data_flow::LibraryNode& node) {
            return std::make_unique<blas::GEMMNodeDispatcher_HIPBLASWithTransfers>(
                language_extension, function, data_flow_graph, dynamic_cast<const math::blas::GEMMNode&>(node)
            );
        }
    );
    // GEMM - HIPBLAS without data transfers
    codegen::LibraryNodeDispatcherRegistry::instance().register_library_node_dispatcher(
        math::blas::LibraryNodeType_GEMM.value() + "::" + hip::blas::ImplementationType_HIPBLASWithoutTransfers.value(),
        [](codegen::LanguageExtension& language_extension,
           const Function& function,
           const data_flow::DataFlowGraph& data_flow_graph,
           const data_flow::LibraryNode& node) {
            return std::make_unique<blas::GEMMNodeDispatcher_HIPBLASWithoutTransfers>(
                language_extension, function, data_flow_graph, dynamic_cast<const math::blas::GEMMNode&>(node)
            );
        }
    );


    passes::scheduler::SchedulerRegistry::instance()
        .register_loop_scheduler<passes::scheduler::HIPScheduler>(passes::scheduler::HIPScheduler::target());
}

} // namespace hip
} // namespace sdfg
