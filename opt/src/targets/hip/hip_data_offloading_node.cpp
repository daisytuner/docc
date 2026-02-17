#include "sdfg/targets/hip/hip_data_offloading_node.h"

#include <cstddef>
#include <memory>
#include <nlohmann/json_fwd.hpp>

#include "sdfg/analysis/loop_analysis.h"
#include "sdfg/codegen/code_snippet_factory.h"
#include "sdfg/codegen/dispatchers/block_dispatcher.h"
#include "sdfg/codegen/instrumentation/instrumentation_info.h"
#include "sdfg/codegen/language_extension.h"
#include "sdfg/codegen/utils.h"
#include "sdfg/data_flow/data_flow_graph.h"
#include "sdfg/data_flow/data_flow_node.h"
#include "sdfg/data_flow/library_node.h"
#include "sdfg/function.h"
#include "sdfg/graph/graph.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/targets/hip/hip.h"
#include "sdfg/targets/offloading/data_offloading_node.h"
#include "symengine/symengine_rcp.h"

namespace sdfg {
namespace hip {

HIPDataOffloadingNode::HIPDataOffloadingNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    symbolic::Expression size,
    symbolic::Expression device_id,
    offloading::DataTransferDirection transfer_direction,
    offloading::BufferLifecycle buffer_lifecycle
)
    : offloading::DataOffloadingNode(
          element_id,
          debug_info,
          vertex,
          parent,
          LibraryNodeType_HIP_Offloading,
          {},
          {},
          transfer_direction,
          buffer_lifecycle,
          size
      ),
      device_id_(device_id) {
    if (!is_NONE(transfer_direction)) {
        this->inputs_.push_back("_src");
        this->outputs_.push_back("_dst");
    } else if (is_ALLOC(buffer_lifecycle)) {
        this->outputs_.push_back("_ret");
    } else if (is_FREE(buffer_lifecycle)) {
        this->inputs_.push_back("_ptr");
        this->outputs_.push_back("_ptr");
    }
}

void HIPDataOffloadingNode::validate(const Function& function) const {
    // Prevent copy-in and free
    if (this->is_h2d() && this->is_free()) {
        throw InvalidSDFGException("HIPDataOffloadingNode: Combination copy-in and free is not allowed");
    }

    // Prevent copy-out and alloc
    if (this->is_d2h() && this->is_alloc()) {
        throw InvalidSDFGException("HIPDataOffloadingNode: Combination copy-out and alloc is not allowed");
    }
}

const symbolic::Expression HIPDataOffloadingNode::device_id() const { return this->device_id_; }

std::unique_ptr<data_flow::DataFlowNode> HIPDataOffloadingNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::make_unique<HIPDataOffloadingNode>(
        element_id,
        this->debug_info(),
        vertex,
        parent,
        this->size(),
        this->device_id(),
        this->transfer_direction(),
        this->buffer_lifecycle()
    );
}

symbolic::SymbolSet HIPDataOffloadingNode::symbols() const {
    if (this->device_id().is_null()) {
        return offloading::DataOffloadingNode::symbols();
    }
    auto symbols = offloading::DataOffloadingNode::symbols();
    auto device_id_atoms = symbolic::atoms(this->device_id());
    symbols.insert(device_id_atoms.begin(), device_id_atoms.end());
    return symbols;
}

void HIPDataOffloadingNode::replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) {
    offloading::DataOffloadingNode::replace(old_expression, new_expression);
    this->device_id_ = symbolic::subs(this->device_id_, old_expression, new_expression);
}

bool HIPDataOffloadingNode::blocking() const { return true; }

bool HIPDataOffloadingNode::redundant_with(const offloading::DataOffloadingNode& other) const {
    if (!offloading::DataOffloadingNode::redundant_with(other)) {
        return false;
    }

    auto& other_node = static_cast<const HIPDataOffloadingNode&>(other);
    if (!symbolic::null_safe_eq(this->device_id(), other_node.device_id())) {
        return false;
    }

    return true;
}

bool HIPDataOffloadingNode::equal_with(const offloading::DataOffloadingNode& other) const {
    if (!offloading::DataOffloadingNode::equal_with(other)) {
        return false;
    }

    auto& other_node = static_cast<const HIPDataOffloadingNode&>(other);
    if (!symbolic::null_safe_eq(this->device_id(), other_node.device_id())) {
        return false;
    }

    return true;
}

HIPDataOffloadingNodeDispatcher::HIPDataOffloadingNodeDispatcher(
    codegen::LanguageExtension& language_extension,
    const Function& function,
    const data_flow::DataFlowGraph& data_flow_graph,
    const data_flow::LibraryNode& node
)
    : codegen::LibraryNodeDispatcher(language_extension, function, data_flow_graph, node) {}

void HIPDataOffloadingNodeDispatcher::dispatch_code(
    codegen::PrettyPrinter& stream,
    codegen::PrettyPrinter& globals_stream,
    codegen::CodeSnippetFactory& library_snippet_factory
) {
    auto& offloading_node = static_cast<const HIPDataOffloadingNode&>(this->node_);

    stream << "hipError_t err;" << std::endl;

    if (offloading_node.is_alloc()) {
        stream << "err = hipMalloc(&" << offloading_node.output(0) << ", "
               << this->language_extension_.expression(offloading_node.size()) << ");" << std::endl;
        hip_error_checking(stream, this->language_extension_, "err");
    }

    if (offloading_node.is_h2d()) {
        stream << "err = hipMemcpy(" << offloading_node.output(0) << ", " << offloading_node.input(0) << ", "
               << this->language_extension_.expression(offloading_node.size()) << ", hipMemcpyHostToDevice);"
               << std::endl;
        hip_error_checking(stream, this->language_extension_, "err");
    } else if (offloading_node.is_d2h()) {
        stream << "err = hipMemcpy(" << offloading_node.output(0) << ", " << offloading_node.input(0) << ", "
               << this->language_extension_.expression(offloading_node.size()) << ", hipMemcpyDeviceToHost);"
               << std::endl;
        hip_error_checking(stream, this->language_extension_, "err");
    }

    if (offloading_node.is_free()) {
        stream << "err = hipFree(" << offloading_node.input(0) << ");" << std::endl;
        hip_error_checking(stream, this->language_extension_, "err");
    }
}

codegen::InstrumentationInfo HIPDataOffloadingNodeDispatcher::instrumentation_info() const {
    auto& hip_node = static_cast<const HIPDataOffloadingNode&>(node_);
    if (hip_node.is_d2h()) {
        return codegen::InstrumentationInfo(
            node_.element_id(),
            codegen::ElementType_D2HTransfer,
            TargetType_HIP,
            analysis::LoopInfo{},
            {{"pcie_bytes", language_extension_.expression(hip_node.size())}}
        );
    } else if (hip_node.is_h2d()) {
        return codegen::InstrumentationInfo(
            node_.element_id(),
            codegen::ElementType_H2DTransfer,
            TargetType_HIP,
            analysis::LoopInfo{},
            {{"pcie_bytes", language_extension_.expression(hip_node.size())}}
        );
    } else {
        return codegen::LibraryNodeDispatcher::instrumentation_info();
    }
}

nlohmann::json HIPDataOffloadingNodeSerializer::serialize(const sdfg::data_flow::LibraryNode& library_node) {
    const auto& node = static_cast<const HIPDataOffloadingNode&>(library_node);
    nlohmann::json j;

    // Library node
    j["type"] = "library_node";
    j["element_id"] = library_node.element_id();

    // Debug info
    auto& debug_info = library_node.debug_info();
    j["has"] = debug_info.has();
    j["filename"] = debug_info.filename();
    j["start_line"] = debug_info.start_line();
    j["start_column"] = debug_info.start_column();
    j["end_line"] = debug_info.end_line();
    j["end_column"] = debug_info.end_column();

    // Library node properties
    j["code"] = std::string(library_node.code().value());

    // Offloading node properties
    sdfg::serializer::JSONSerializer serializer;
    if (node.size().is_null()) {
        j["size"] = nlohmann::json::value_t::null;
    } else {
        j["size"] = serializer.expression(node.size());
    }
    j["device_id"] = serializer.expression(node.device_id());
    j["transfer_direction"] = static_cast<int8_t>(node.transfer_direction());
    j["buffer_lifecycle"] = static_cast<int8_t>(node.buffer_lifecycle());

    return j;
}

data_flow::LibraryNode& HIPDataOffloadingNodeSerializer::deserialize(
    const nlohmann::json& j, sdfg::builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
) {
    auto code = j["code"].get<std::string>();
    if (code != LibraryNodeType_HIP_Offloading.value()) {
        throw std::runtime_error("Invalid library node code");
    }

    sdfg::serializer::JSONSerializer serializer;
    DebugInfo debug_info = serializer.json_to_debug_info(j["debug_info"]);

    symbolic::Expression size;
    if (!j.contains("size") || j.at("size").is_null()) {
        size = SymEngine::null;
    } else {
        size = symbolic::parse(j.at("size"));
    }
    SymEngine::Expression device_id(j.at("device_id"));
    auto transfer_direction = static_cast<offloading::DataTransferDirection>(j["transfer_direction"].get<int8_t>());
    auto buffer_lifecycle = static_cast<offloading::BufferLifecycle>(j["buffer_lifecycle"].get<int8_t>());

    return builder.add_library_node<
        HIPDataOffloadingNode>(parent, debug_info, size, device_id, transfer_direction, buffer_lifecycle);
}

} // namespace hip
} // namespace sdfg
