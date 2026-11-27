#include "sdfg/targets/gpu/gpu_wave_barrier_node.h"

#include <nlohmann/json.hpp>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/exceptions.h"
#include "sdfg/targets/cuda/codegen/cuda_language_extension.h"
#include "sdfg/targets/rocm/codegen/rocm_language_extension.h"

namespace sdfg::gpu {

GpuWaveBarrierNode::GpuWaveBarrierNode(
    size_t element_id, const DebugInfo& debug_info, const graph::Vertex vertex, data_flow::DataFlowGraph& parent
)
    : LibraryNode(
          element_id,
          debug_info,
          vertex,
          parent,
          LibraryNodeType_GpuWaveBarrier,
          {},
          {},
          true,
          data_flow::ImplementationType_NONE
      ) {
}

void GpuWaveBarrierNode::validate(const Function& function) const {
    LibraryNode::validate(function);
}

symbolic::SymbolSet GpuWaveBarrierNode::symbols() const {
    return {};
}

void GpuWaveBarrierNode::replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) {
}

void GpuWaveBarrierNode::replace(const symbolic::ExpressionMapping& replacements) {
}

std::unique_ptr<data_flow::DataFlowNode> GpuWaveBarrierNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::unique_ptr<data_flow::DataFlowNode>(new GpuWaveBarrierNode(element_id, debug_info(), vertex, parent));
}

nlohmann::json GpuWaveBarrierNodeSerializer::serialize(const data_flow::LibraryNode& library_node) {
    if (library_node.code() != LibraryNodeType_GpuWaveBarrier) {
        throw InvalidSDFGException("GpuWaveBarrierNodeSerializer: invalid library node code");
    }
    nlohmann::json j;
    j["code"] = library_node.code().value();
    return j;
}

data_flow::LibraryNode& GpuWaveBarrierNodeSerializer::deserialize(
    const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
) {
    if (j.at("code").get<std::string>() != LibraryNodeType_GpuWaveBarrier.value()) {
        throw InvalidSDFGException("GpuWaveBarrierNodeSerializer: invalid library node code");
    }
    serializer::JSONSerializer serializer;
    DebugInfo debug_info = j.contains("debug_info") ? serializer.json_to_debug_info(j["debug_info"]) : DebugInfo();
    return builder.add_library_node<GpuWaveBarrierNode>(parent, debug_info);
}

GpuWaveBarrierDispatcher::GpuWaveBarrierDispatcher(
    codegen::LanguageExtension& language_extension,
    const Function& function,
    const data_flow::DataFlowGraph& data_flow_graph,
    const data_flow::LibraryNode& node
)
    : codegen::LibraryNodeDispatcher(language_extension, function, data_flow_graph, node) {
}

void GpuWaveBarrierDispatcher::dispatch(
    codegen::PrettyPrinter& stream,
    codegen::PrettyPrinter& globals_stream,
    codegen::CodeSnippetFactory& library_snippet_factory
) {
    if (dynamic_cast<::sdfg::cuda::CUDALanguageExtension*>(&this->language_extension_) != nullptr) {
        stream << "__syncwarp();" << std::endl;
    } else if (dynamic_cast<::sdfg::rocm::ROCMLanguageExtension*>(&this->language_extension_) != nullptr) {
        // A wavefront executes in lockstep; the fences order the lanes' memory accesses at wavefront scope.
        stream << "__builtin_amdgcn_fence(__ATOMIC_RELEASE, \"wavefront\");" << std::endl;
        stream << "__builtin_amdgcn_wave_barrier();" << std::endl;
        stream << "__builtin_amdgcn_fence(__ATOMIC_ACQUIRE, \"wavefront\");" << std::endl;
    } else {
        throw InvalidSDFGException("GpuWaveBarrierDispatcher: requires a CUDA or ROCm kernel language extension");
    }
}

} // namespace sdfg::gpu
