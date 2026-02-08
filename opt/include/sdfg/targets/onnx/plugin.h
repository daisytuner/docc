#pragma once

#include <sdfg/codegen/dispatchers/node_dispatcher_registry.h>
#include <sdfg/data_flow/library_nodes/math/math.h>

// ONNX target headers
#include "sdfg/targets/onnx/blas/dot_dispatcher.h"
#include "sdfg/targets/onnx/blas/gemm_dispatcher.h"
#include "sdfg/targets/onnx/onnx.h"
#include "sdfg/targets/onnx/tensor/broadcast_dispatcher.h"
#include "sdfg/targets/onnx/tensor/conv_dispatcher.h"
#include "sdfg/targets/onnx/tensor/elementwise_dispatcher.h"
#include "sdfg/targets/onnx/tensor/reduce_dispatcher.h"
#include "sdfg/targets/onnx/tensor/transpose_dispatcher.h"

namespace sdfg {
namespace onnx {

/**
 * @brief Register all ONNX tensor node dispatchers
 *
 * This function registers library node dispatchers for all tensor operations
 * that can be executed via ONNX Runtime. The dispatchers emit ONNX graph
 * representations to the library snippet factory and generate runtime calls
 * in the actual code stream.
 */
inline void register_onnx_plugin() {
    using namespace codegen;

    // GEMM
    LibraryNodeDispatcherRegistry::instance().register_library_node_dispatcher(
        math::blas::LibraryNodeType_GEMM.value() + "::" + ImplementationType_ONNX.value(),
        [](LanguageExtension& language_extension,
           const Function& function,
           const data_flow::DataFlowGraph& data_flow_graph,
           const data_flow::LibraryNode& node) {
            return std::make_unique<blas::GEMMNodeDispatcher_ONNX>(
                language_extension, function, data_flow_graph, dynamic_cast<const math::blas::GEMMNode&>(node)
            );
        }
    );

    // Dot
    LibraryNodeDispatcherRegistry::instance().register_library_node_dispatcher(
        math::blas::LibraryNodeType_DOT.value() + "::" + ImplementationType_ONNX.value(),
        [](LanguageExtension& language_extension,
           const Function& function,
           const data_flow::DataFlowGraph& data_flow_graph,
           const data_flow::LibraryNode& node) {
            return std::make_unique<blas::DotNodeDispatcher_ONNX>(
                language_extension, function, data_flow_graph, dynamic_cast<const math::blas::DotNode&>(node)
            );
        }
    );


    // =========================================================================
    // Elementwise Unary Operations
    // =========================================================================

    // Abs
    LibraryNodeDispatcherRegistry::instance().register_library_node_dispatcher(
        math::tensor::LibraryNodeType_Abs.value() + "::" + ImplementationType_ONNX.value(),
        [](LanguageExtension& language_extension,
           const Function& function,
           const data_flow::DataFlowGraph& data_flow_graph,
           const data_flow::LibraryNode& node) {
            return std::make_unique<tensor::ElementWiseUnaryNodeDispatcher_ONNX>(
                language_extension, function, data_flow_graph, dynamic_cast<const math::tensor::AbsNode&>(node)
            );
        }
    );

    // Sqrt
    LibraryNodeDispatcherRegistry::instance().register_library_node_dispatcher(
        math::tensor::LibraryNodeType_Sqrt.value() + "::" + ImplementationType_ONNX.value(),
        [](LanguageExtension& language_extension,
           const Function& function,
           const data_flow::DataFlowGraph& data_flow_graph,
           const data_flow::LibraryNode& node) {
            return std::make_unique<tensor::ElementWiseUnaryNodeDispatcher_ONNX>(
                language_extension, function, data_flow_graph, dynamic_cast<const math::tensor::SqrtNode&>(node)
            );
        }
    );

    // Exp
    LibraryNodeDispatcherRegistry::instance().register_library_node_dispatcher(
        math::tensor::LibraryNodeType_Exp.value() + "::" + ImplementationType_ONNX.value(),
        [](LanguageExtension& language_extension,
           const Function& function,
           const data_flow::DataFlowGraph& data_flow_graph,
           const data_flow::LibraryNode& node) {
            return std::make_unique<tensor::ElementWiseUnaryNodeDispatcher_ONNX>(
                language_extension, function, data_flow_graph, dynamic_cast<const math::tensor::ExpNode&>(node)
            );
        }
    );

    // Tanh
    LibraryNodeDispatcherRegistry::instance().register_library_node_dispatcher(
        math::tensor::LibraryNodeType_Tanh.value() + "::" + ImplementationType_ONNX.value(),
        [](LanguageExtension& language_extension,
           const Function& function,
           const data_flow::DataFlowGraph& data_flow_graph,
           const data_flow::LibraryNode& node) {
            return std::make_unique<tensor::ElementWiseUnaryNodeDispatcher_ONNX>(
                language_extension, function, data_flow_graph, dynamic_cast<const math::tensor::TanhNode&>(node)
            );
        }
    );

    // Sigmoid
    LibraryNodeDispatcherRegistry::instance().register_library_node_dispatcher(
        math::tensor::LibraryNodeType_Sigmoid.value() + "::" + ImplementationType_ONNX.value(),
        [](LanguageExtension& language_extension,
           const Function& function,
           const data_flow::DataFlowGraph& data_flow_graph,
           const data_flow::LibraryNode& node) {
            return std::make_unique<tensor::ElementWiseUnaryNodeDispatcher_ONNX>(
                language_extension, function, data_flow_graph, dynamic_cast<const math::tensor::SigmoidNode&>(node)
            );
        }
    );

    // ReLU
    LibraryNodeDispatcherRegistry::instance().register_library_node_dispatcher(
        math::tensor::LibraryNodeType_ReLU.value() + "::" + ImplementationType_ONNX.value(),
        [](LanguageExtension& language_extension,
           const Function& function,
           const data_flow::DataFlowGraph& data_flow_graph,
           const data_flow::LibraryNode& node) {
            return std::make_unique<tensor::ElementWiseUnaryNodeDispatcher_ONNX>(
                language_extension, function, data_flow_graph, dynamic_cast<const math::tensor::ReLUNode&>(node)
            );
        }
    );

    // LeakyReLU
    LibraryNodeDispatcherRegistry::instance().register_library_node_dispatcher(
        math::tensor::LibraryNodeType_LeakyReLU.value() + "::" + ImplementationType_ONNX.value(),
        [](LanguageExtension& language_extension,
           const Function& function,
           const data_flow::DataFlowGraph& data_flow_graph,
           const data_flow::LibraryNode& node) {
            return std::make_unique<tensor::ElementWiseUnaryNodeDispatcher_ONNX>(
                language_extension, function, data_flow_graph, dynamic_cast<const math::tensor::LeakyReLUNode&>(node)
            );
        }
    );

    // Elu
    LibraryNodeDispatcherRegistry::instance().register_library_node_dispatcher(
        math::tensor::LibraryNodeType_Elu.value() + "::" + ImplementationType_ONNX.value(),
        [](LanguageExtension& language_extension,
           const Function& function,
           const data_flow::DataFlowGraph& data_flow_graph,
           const data_flow::LibraryNode& node) {
            return std::make_unique<tensor::ElementWiseUnaryNodeDispatcher_ONNX>(
                language_extension, function, data_flow_graph, dynamic_cast<const math::tensor::EluNode&>(node)
            );
        }
    );

    // HardSigmoid
    LibraryNodeDispatcherRegistry::instance().register_library_node_dispatcher(
        math::tensor::LibraryNodeType_HardSigmoid.value() + "::" + ImplementationType_ONNX.value(),
        [](LanguageExtension& language_extension,
           const Function& function,
           const data_flow::DataFlowGraph& data_flow_graph,
           const data_flow::LibraryNode& node) {
            return std::make_unique<tensor::ElementWiseUnaryNodeDispatcher_ONNX>(
                language_extension, function, data_flow_graph, dynamic_cast<const math::tensor::HardSigmoidNode&>(node)
            );
        }
    );

    // Erf
    LibraryNodeDispatcherRegistry::instance().register_library_node_dispatcher(
        math::tensor::LibraryNodeType_Erf.value() + "::" + ImplementationType_ONNX.value(),
        [](LanguageExtension& language_extension,
           const Function& function,
           const data_flow::DataFlowGraph& data_flow_graph,
           const data_flow::LibraryNode& node) {
            return std::make_unique<tensor::ElementWiseUnaryNodeDispatcher_ONNX>(
                language_extension, function, data_flow_graph, dynamic_cast<const math::tensor::ErfNode&>(node)
            );
        }
    );

    // Cast
    LibraryNodeDispatcherRegistry::instance().register_library_node_dispatcher(
        math::tensor::LibraryNodeType_Cast.value() + "::" + ImplementationType_ONNX.value(),
        [](LanguageExtension& language_extension,
           const Function& function,
           const data_flow::DataFlowGraph& data_flow_graph,
           const data_flow::LibraryNode& node) {
            return std::make_unique<tensor::ElementWiseUnaryNodeDispatcher_ONNX>(
                language_extension, function, data_flow_graph, dynamic_cast<const math::tensor::CastNode&>(node)
            );
        }
    );

    // Fill (ConstantOfShape in ONNX)
    LibraryNodeDispatcherRegistry::instance().register_library_node_dispatcher(
        math::tensor::LibraryNodeType_Fill.value() + "::" + ImplementationType_ONNX.value(),
        [](LanguageExtension& language_extension,
           const Function& function,
           const data_flow::DataFlowGraph& data_flow_graph,
           const data_flow::LibraryNode& node) {
            return std::make_unique<tensor::ElementWiseUnaryNodeDispatcher_ONNX>(
                language_extension, function, data_flow_graph, dynamic_cast<const math::tensor::FillNode&>(node)
            );
        }
    );

    // =========================================================================
    // Elementwise Binary Operations
    // =========================================================================

    // Add
    LibraryNodeDispatcherRegistry::instance().register_library_node_dispatcher(
        math::tensor::LibraryNodeType_Add.value() + "::" + ImplementationType_ONNX.value(),
        [](LanguageExtension& language_extension,
           const Function& function,
           const data_flow::DataFlowGraph& data_flow_graph,
           const data_flow::LibraryNode& node) {
            return std::make_unique<tensor::ElementWiseBinaryNodeDispatcher_ONNX>(
                language_extension, function, data_flow_graph, dynamic_cast<const math::tensor::AddNode&>(node)
            );
        }
    );

    // Sub
    LibraryNodeDispatcherRegistry::instance().register_library_node_dispatcher(
        math::tensor::LibraryNodeType_Sub.value() + "::" + ImplementationType_ONNX.value(),
        [](LanguageExtension& language_extension,
           const Function& function,
           const data_flow::DataFlowGraph& data_flow_graph,
           const data_flow::LibraryNode& node) {
            return std::make_unique<tensor::ElementWiseBinaryNodeDispatcher_ONNX>(
                language_extension, function, data_flow_graph, dynamic_cast<const math::tensor::SubNode&>(node)
            );
        }
    );

    // Mul
    LibraryNodeDispatcherRegistry::instance().register_library_node_dispatcher(
        math::tensor::LibraryNodeType_Mul.value() + "::" + ImplementationType_ONNX.value(),
        [](LanguageExtension& language_extension,
           const Function& function,
           const data_flow::DataFlowGraph& data_flow_graph,
           const data_flow::LibraryNode& node) {
            return std::make_unique<tensor::ElementWiseBinaryNodeDispatcher_ONNX>(
                language_extension, function, data_flow_graph, dynamic_cast<const math::tensor::MulNode&>(node)
            );
        }
    );

    // Div
    LibraryNodeDispatcherRegistry::instance().register_library_node_dispatcher(
        math::tensor::LibraryNodeType_Div.value() + "::" + ImplementationType_ONNX.value(),
        [](LanguageExtension& language_extension,
           const Function& function,
           const data_flow::DataFlowGraph& data_flow_graph,
           const data_flow::LibraryNode& node) {
            return std::make_unique<tensor::ElementWiseBinaryNodeDispatcher_ONNX>(
                language_extension, function, data_flow_graph, dynamic_cast<const math::tensor::DivNode&>(node)
            );
        }
    );

    // Pow
    LibraryNodeDispatcherRegistry::instance().register_library_node_dispatcher(
        math::tensor::LibraryNodeType_Pow.value() + "::" + ImplementationType_ONNX.value(),
        [](LanguageExtension& language_extension,
           const Function& function,
           const data_flow::DataFlowGraph& data_flow_graph,
           const data_flow::LibraryNode& node) {
            return std::make_unique<tensor::ElementWiseBinaryNodeDispatcher_ONNX>(
                language_extension, function, data_flow_graph, dynamic_cast<const math::tensor::PowNode&>(node)
            );
        }
    );

    // Maximum
    LibraryNodeDispatcherRegistry::instance().register_library_node_dispatcher(
        math::tensor::LibraryNodeType_Maximum.value() + "::" + ImplementationType_ONNX.value(),
        [](LanguageExtension& language_extension,
           const Function& function,
           const data_flow::DataFlowGraph& data_flow_graph,
           const data_flow::LibraryNode& node) {
            return std::make_unique<tensor::ElementWiseBinaryNodeDispatcher_ONNX>(
                language_extension, function, data_flow_graph, dynamic_cast<const math::tensor::MaximumNode&>(node)
            );
        }
    );

    // Minimum
    LibraryNodeDispatcherRegistry::instance().register_library_node_dispatcher(
        math::tensor::LibraryNodeType_Minimum.value() + "::" + ImplementationType_ONNX.value(),
        [](LanguageExtension& language_extension,
           const Function& function,
           const data_flow::DataFlowGraph& data_flow_graph,
           const data_flow::LibraryNode& node) {
            return std::make_unique<tensor::ElementWiseBinaryNodeDispatcher_ONNX>(
                language_extension, function, data_flow_graph, dynamic_cast<const math::tensor::MinimumNode&>(node)
            );
        }
    );

    // =========================================================================
    // Reduction Operations
    // =========================================================================

    // Sum
    LibraryNodeDispatcherRegistry::instance().register_library_node_dispatcher(
        math::tensor::LibraryNodeType_Sum.value() + "::" + ImplementationType_ONNX.value(),
        [](LanguageExtension& language_extension,
           const Function& function,
           const data_flow::DataFlowGraph& data_flow_graph,
           const data_flow::LibraryNode& node) {
            return std::make_unique<tensor::ReduceNodeDispatcher_ONNX>(
                language_extension, function, data_flow_graph, dynamic_cast<const math::tensor::SumNode&>(node)
            );
        }
    );

    // Mean
    LibraryNodeDispatcherRegistry::instance().register_library_node_dispatcher(
        math::tensor::LibraryNodeType_Mean.value() + "::" + ImplementationType_ONNX.value(),
        [](LanguageExtension& language_extension,
           const Function& function,
           const data_flow::DataFlowGraph& data_flow_graph,
           const data_flow::LibraryNode& node) {
            return std::make_unique<tensor::ReduceNodeDispatcher_ONNX>(
                language_extension, function, data_flow_graph, dynamic_cast<const math::tensor::MeanNode&>(node)
            );
        }
    );

    // Max (reduction)
    LibraryNodeDispatcherRegistry::instance().register_library_node_dispatcher(
        math::tensor::LibraryNodeType_Max.value() + "::" + ImplementationType_ONNX.value(),
        [](LanguageExtension& language_extension,
           const Function& function,
           const data_flow::DataFlowGraph& data_flow_graph,
           const data_flow::LibraryNode& node) {
            return std::make_unique<tensor::ReduceNodeDispatcher_ONNX>(
                language_extension, function, data_flow_graph, dynamic_cast<const math::tensor::MaxNode&>(node)
            );
        }
    );

    // Min (reduction)
    LibraryNodeDispatcherRegistry::instance().register_library_node_dispatcher(
        math::tensor::LibraryNodeType_Min.value() + "::" + ImplementationType_ONNX.value(),
        [](LanguageExtension& language_extension,
           const Function& function,
           const data_flow::DataFlowGraph& data_flow_graph,
           const data_flow::LibraryNode& node) {
            return std::make_unique<tensor::ReduceNodeDispatcher_ONNX>(
                language_extension, function, data_flow_graph, dynamic_cast<const math::tensor::MinNode&>(node)
            );
        }
    );

    // Std
    LibraryNodeDispatcherRegistry::instance().register_library_node_dispatcher(
        math::tensor::LibraryNodeType_Std.value() + "::" + ImplementationType_ONNX.value(),
        [](LanguageExtension& language_extension,
           const Function& function,
           const data_flow::DataFlowGraph& data_flow_graph,
           const data_flow::LibraryNode& node) {
            return std::make_unique<tensor::ReduceNodeDispatcher_ONNX>(
                language_extension, function, data_flow_graph, dynamic_cast<const math::tensor::StdNode&>(node)
            );
        }
    );

    // Softmax
    LibraryNodeDispatcherRegistry::instance().register_library_node_dispatcher(
        math::tensor::LibraryNodeType_Softmax.value() + "::" + ImplementationType_ONNX.value(),
        [](LanguageExtension& language_extension,
           const Function& function,
           const data_flow::DataFlowGraph& data_flow_graph,
           const data_flow::LibraryNode& node) {
            return std::make_unique<tensor::ReduceNodeDispatcher_ONNX>(
                language_extension, function, data_flow_graph, dynamic_cast<const math::tensor::SoftmaxNode&>(node)
            );
        }
    );

    // =========================================================================
    // Other Tensor Operations
    // =========================================================================

    // Conv
    LibraryNodeDispatcherRegistry::instance().register_library_node_dispatcher(
        math::tensor::LibraryNodeType_Conv.value() + "::" + ImplementationType_ONNX.value(),
        [](LanguageExtension& language_extension,
           const Function& function,
           const data_flow::DataFlowGraph& data_flow_graph,
           const data_flow::LibraryNode& node) {
            return std::make_unique<tensor::ConvNodeDispatcher_ONNX>(
                language_extension, function, data_flow_graph, dynamic_cast<const math::tensor::ConvNode&>(node)
            );
        }
    );

    // Transpose
    LibraryNodeDispatcherRegistry::instance().register_library_node_dispatcher(
        math::tensor::LibraryNodeType_Transpose.value() + "::" + ImplementationType_ONNX.value(),
        [](LanguageExtension& language_extension,
           const Function& function,
           const data_flow::DataFlowGraph& data_flow_graph,
           const data_flow::LibraryNode& node) {
            return std::make_unique<tensor::TransposeNodeDispatcher_ONNX>(
                language_extension, function, data_flow_graph, dynamic_cast<const math::tensor::TransposeNode&>(node)
            );
        }
    );

    // Broadcast
    LibraryNodeDispatcherRegistry::instance().register_library_node_dispatcher(
        math::tensor::LibraryNodeType_Broadcast.value() + "::" + ImplementationType_ONNX.value(),
        [](LanguageExtension& language_extension,
           const Function& function,
           const data_flow::DataFlowGraph& data_flow_graph,
           const data_flow::LibraryNode& node) {
            return std::make_unique<tensor::BroadcastNodeDispatcher_ONNX>(
                language_extension, function, data_flow_graph, dynamic_cast<const math::tensor::BroadcastNode&>(node)
            );
        }
    );
}

} // namespace onnx
} // namespace sdfg
