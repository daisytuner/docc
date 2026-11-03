#include "sdfg/targets/gpu/gpu_tile_arrival_node.h"

#include <nlohmann/json.hpp>
#include <sstream>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/exceptions.h"
#include "sdfg/targets/cuda/codegen/cuda_language_extension.h"
#include "sdfg/targets/rocm/codegen/rocm_language_extension.h"

namespace sdfg::gpu {

GpuTileArrivalNode::GpuTileArrivalNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const symbolic::Expression& index,
    const symbolic::Expression& expected,
    SynchronizationStrategy synchronization_strategy
)
    : LibraryNode(
          element_id,
          debug_info,
          vertex,
          parent,
          LibraryNodeType_GpuTileArrival,
          {"_last"},
          {"counter"},
          true,
          data_flow::ImplementationType_NONE
      ),
      index_(index), expected_(expected), synchronization_strategy_(synchronization_strategy) {
}

void GpuTileArrivalNode::validate(const Function& function) const {
    auto& graph = this->get_parent();
    if (graph.in_degree(*this) != 1) {
        throw InvalidSDFGException("GpuTileArrivalNode: Expected exactly 1 input (counter)");
    }
    if (graph.out_degree(*this) != 1) {
        throw InvalidSDFGException("GpuTileArrivalNode: Expected exactly 1 output (_last)");
    }
    if (SymEngine::is_a<SymEngine::Integer>(*expected_) &&
        !SymEngine::rcp_static_cast<const SymEngine::Integer>(expected_)->is_positive()) {
        throw InvalidSDFGException("GpuTileArrivalNode: expected arrivals must be positive");
    }
}

symbolic::SymbolSet GpuTileArrivalNode::symbols() const {
    auto syms = symbolic::atoms(index_);
    for (auto& s : symbolic::atoms(expected_)) {
        syms.insert(s);
    }
    return syms;
}

void GpuTileArrivalNode::replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) {
    index_ = symbolic::subs(index_, old_expression, new_expression);
    expected_ = symbolic::subs(expected_, old_expression, new_expression);
}

void GpuTileArrivalNode::replace(const symbolic::ExpressionMapping& replacements) {
    index_ = symbolic::subs(index_, replacements);
    expected_ = symbolic::subs(expected_, replacements);
}

std::unique_ptr<data_flow::DataFlowNode> GpuTileArrivalNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::unique_ptr<data_flow::DataFlowNode>(
        new GpuTileArrivalNode(element_id, debug_info(), vertex, parent, index_, expected_, synchronization_strategy_)
    );
}

std::string GpuTileArrivalNode::toStr() const {
    std::stringstream ss;
    ss << "GpuTileArrival(index: " << index_->__str__() << ", expected: " << expected_->__str__();
    if (synchronization_strategy_ == SynchronizationStrategy::CoherentStores) {
        ss << ", coherent stores";
    }
    ss << ")";
    return ss.str();
}

nlohmann::json GpuTileArrivalNodeSerializer::serialize(const data_flow::LibraryNode& library_node) {
    const auto& node = static_cast<const GpuTileArrivalNode&>(library_node);
    nlohmann::json j;
    j["code"] = node.code().value();
    j["index"] = node.index()->__str__();
    j["expected"] = node.expected()->__str__();
    j["coherent_stores"] = node.synchronization_strategy() == SynchronizationStrategy::CoherentStores;
    return j;
}

data_flow::LibraryNode& GpuTileArrivalNodeSerializer::deserialize(
    const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
) {
    serializer::JSONSerializer serializer;
    DebugInfo debug_info = serializer.json_to_debug_info(j["debug_info"]);
    const bool coherent = j.value("coherent_stores", false);
    return builder.add_library_node<GpuTileArrivalNode>(
        parent,
        debug_info,
        symbolic::parse(j.at("index").get<std::string>()),
        symbolic::parse(j.at("expected").get<std::string>()),
        coherent ? SynchronizationStrategy::CoherentStores : SynchronizationStrategy::DeviceFence
    );
}

GpuTileArrivalDispatcher::GpuTileArrivalDispatcher(
    codegen::LanguageExtension& language_extension,
    const Function& function,
    const data_flow::DataFlowGraph& data_flow_graph,
    const data_flow::LibraryNode& node
)
    : codegen::LibraryNodeDispatcher(language_extension, function, data_flow_graph, node) {
}

void GpuTileArrivalDispatcher::dispatch_code_with_edges(
    codegen::CodegenOutput& out,
    std::vector<codegen::DispatchInput>& inputs,
    std::vector<codegen::DispatchOutput>& outputs
) {
    if (!dynamic_cast<::sdfg::cuda::CUDALanguageExtension*>(&out.language_extension) &&
        !dynamic_cast<::sdfg::rocm::ROCMLanguageExtension*>(&out.language_extension)) {
        throw std::runtime_error("GpuTileArrivalDispatcher requires the CUDA or ROCm kernel language extension");
    }
    auto& node = static_cast<const GpuTileArrivalNode&>(node_);
    auto& lang = out.language_extension;

    flag_ = "__daisy_arrival_" + std::to_string(node.element_id());
    const std::string& flag = flag_;
    const std::string slot = "(" + inputs.at(GpuTileArrivalNode::COUNTER_INPUT_IDX).expr + ")[" +
                             lang.expression(node.index()) + "]";
    const std::string leader = "threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0";
    // On gfx9, device-coherent stores are complete (visible device-wide) once vmcnt drains, so the release
    // reduces to that wait; the device-scope fence would also write back (and later invalidate) the whole L2.
    const bool coherent = node.synchronization_strategy() == SynchronizationStrategy::CoherentStores &&
                          dynamic_cast<::sdfg::rocm::ROCMLanguageExtension*>(&lang) != nullptr;

    if (coherent) {
        out.stream << "#if defined(__GFX9__)" << std::endl;
        out.stream << "asm volatile(\"s_waitcnt vmcnt(0)\" ::: \"memory\");" << std::endl;
        out.stream << "#else" << std::endl;
        out.stream << "__threadfence();" << std::endl;
        out.stream << "#endif" << std::endl;
    } else {
        out.stream << "__threadfence();" << std::endl;
    }
    out.stream << "__syncthreads();" << std::endl;
    out.stream << "__shared__ int " << flag << ";" << std::endl;
    out.stream << "if (" << leader << ") {" << std::endl;
    out.stream.changeIndent(+4);
    out.stream << flag << " = atomicAdd(&" << slot << ", 1) == (" << lang.expression(node.expected()) << ") - 1;"
               << std::endl;
    out.stream << "if (" << flag << ") {" << std::endl;
    out.stream.changeIndent(+4);
    out.stream << slot << " = 0;" << std::endl;
    out.stream.changeIndent(-4);
    out.stream << "}" << std::endl;
    out.stream.changeIndent(-4);
    out.stream << "}" << std::endl;
    out.stream << "__syncthreads();" << std::endl;
    out.stream << "if (" << flag << ") {" << std::endl;
    out.stream.changeIndent(+4);
    if (coherent) {
        // The partials are read back through device-coherent loads, which bypass the non-coherent caches.
        out.stream << "#if !defined(__GFX9__)" << std::endl;
        out.stream << "__threadfence();" << std::endl;
        out.stream << "#endif" << std::endl;
    } else {
        out.stream << "__threadfence();" << std::endl;
    }
    out.stream.changeIndent(-4);
    out.stream << "}" << std::endl;

    auto& output = outputs.at(0);
    if (output.used) {
        register_output(output, flag_);
    }
}

} // namespace sdfg::gpu
