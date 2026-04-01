#include "printf_data_offloading_node.h"

#include <cstddef>
#include <memory>

#include <sdfg/codegen/code_snippet_factory.h>
#include <sdfg/codegen/dispatchers/block_dispatcher.h>
#include <sdfg/codegen/instrumentation/instrumentation_info.h>
#include <sdfg/codegen/language_extension.h>
#include <sdfg/data_flow/data_flow_graph.h>
#include <sdfg/data_flow/data_flow_node.h>
#include <sdfg/data_flow/library_node.h>
#include <sdfg/function.h>
#include <sdfg/graph/graph.h>
#include <sdfg/symbolic/symbolic.h>
#include <sdfg/targets/offloading/data_offloading_node.h>
#include "printf_target.h"

namespace sdfg {
namespace printf_target {

PrintfDataOffloadingNode::PrintfDataOffloadingNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    symbolic::Expression size,
    offloading::DataTransferDirection transfer_direction,
    offloading::BufferLifecycle buffer_lifecycle
)
    : offloading::DataOffloadingNode(
          element_id,
          debug_info,
          vertex,
          parent,
          LibraryNodeType_Printf_Offloading,
          {},
          {},
          transfer_direction,
          buffer_lifecycle,
          size
      ) {
    // Configure inputs/outputs based on operation type
    if (!offloading::is_NONE(transfer_direction)) {
        this->inputs_.push_back("_src");
        this->outputs_.push_back("_dst");
    } else if (offloading::is_ALLOC(buffer_lifecycle)) {
        this->outputs_.push_back("_ret");
    } else if (offloading::is_FREE(buffer_lifecycle)) {
        this->inputs_.push_back("_ptr");
        this->outputs_.push_back("_ptr");
    }
}

void PrintfDataOffloadingNode::validate(const Function& function) const {
    // Prevent copy-in and free
    if (this->is_h2d() && this->is_free()) {
        throw InvalidSDFGException("PrintfDataOffloadingNode: Combination copy-in and free is not allowed");
    }

    // Prevent copy-out and alloc
    if (this->is_d2h() && this->is_alloc()) {
        throw InvalidSDFGException("PrintfDataOffloadingNode: Combination copy-out and alloc is not allowed");
    }
}

std::unique_ptr<data_flow::DataFlowNode> PrintfDataOffloadingNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::make_unique<PrintfDataOffloadingNode>(
        element_id,
        this->debug_info(),
        vertex,
        parent,
        this->size(),
        this->transfer_direction(),
        this->buffer_lifecycle()
    );
}

symbolic::SymbolSet PrintfDataOffloadingNode::symbols() const { return offloading::DataOffloadingNode::symbols(); }

void PrintfDataOffloadingNode::
    replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) {
    offloading::DataOffloadingNode::replace(old_expression, new_expression);
}

bool PrintfDataOffloadingNode::blocking() const {
    // Printf operations are non-blocking
    return false;
}

bool PrintfDataOffloadingNode::redundant_with(const offloading::DataOffloadingNode& other) const {
    return offloading::DataOffloadingNode::redundant_with(other);
}

bool PrintfDataOffloadingNode::equal_with(const offloading::DataOffloadingNode& other) const {
    return offloading::DataOffloadingNode::equal_with(other);
}

bool PrintfDataOffloadingNode::is_same_target(const DataOffloadingNode& other) const {
    return dynamic_cast<const PrintfDataOffloadingNode*>(&other) != nullptr;
}

// ============================================================================
// Dispatcher Implementation
// ============================================================================

PrintfDataOffloadingNodeDispatcher::PrintfDataOffloadingNodeDispatcher(
    codegen::LanguageExtension& language_extension,
    const Function& function,
    const data_flow::DataFlowGraph& data_flow_graph,
    const data_flow::LibraryNode& node
)
    : codegen::LibraryNodeDispatcher(language_extension, function, data_flow_graph, node) {}

void PrintfDataOffloadingNodeDispatcher::dispatch_code(
    codegen::PrettyPrinter& stream,
    codegen::PrettyPrinter& globals_stream,
    codegen::CodeSnippetFactory& library_snippet_factory
) {
    auto& offloading_node = static_cast<const PrintfDataOffloadingNode&>(this->node_);

    // Generate printf statements instead of actual memory operations
    if (offloading_node.is_alloc()) {
        stream << "printf(\"[PRINTF_TARGET] Allocating %zu bytes for %s\\n\", (size_t)("
               << this->language_extension_.expression(offloading_node.size()) << "), \"" << offloading_node.output(0)
               << "\");" << std::endl;
        // Still need to set the pointer to something (use malloc for simulation)
        stream << offloading_node.output(0) << " = malloc("
               << this->language_extension_.expression(offloading_node.size()) << ");" << std::endl;
    }

    if (offloading_node.is_h2d()) {
        stream << "printf(\"[PRINTF_TARGET] Copying %zu bytes from host (%s) to device (%s)\\n\", (size_t)("
               << this->language_extension_.expression(offloading_node.size()) << "), \"" << offloading_node.input(0)
               << "\", \"" << offloading_node.output(0) << "\");" << std::endl;
        // Simulate the copy with memcpy
        stream << "memcpy(" << offloading_node.output(0) << ", " << offloading_node.input(0) << ", "
               << this->language_extension_.expression(offloading_node.size()) << ");" << std::endl;
    } else if (offloading_node.is_d2h()) {
        stream << "printf(\"[PRINTF_TARGET] Copying %zu bytes from device (%s) to host (%s)\\n\", (size_t)("
               << this->language_extension_.expression(offloading_node.size()) << "), \"" << offloading_node.input(0)
               << "\", \"" << offloading_node.output(0) << "\");" << std::endl;
        // Simulate the copy with memcpy
        stream << "memcpy(" << offloading_node.output(0) << ", " << offloading_node.input(0) << ", "
               << this->language_extension_.expression(offloading_node.size()) << ");" << std::endl;
    }

    if (offloading_node.is_free()) {
        stream << "printf(\"[PRINTF_TARGET] Freeing %s\\n\", \"" << offloading_node.input(0) << "\");" << std::endl;
        // Actually free the simulated allocation
        stream << "free(" << offloading_node.input(0) << ");" << std::endl;
    }
}

codegen::InstrumentationInfo PrintfDataOffloadingNodeDispatcher::instrumentation_info() const {
    auto& printf_node = static_cast<const PrintfDataOffloadingNode&>(node_);
    if (printf_node.is_d2h()) {
        return codegen::InstrumentationInfo(
            node_.element_id(),
            codegen::ElementType_D2HTransfer,
            TargetType_Printf,
            analysis::LoopInfo{},
            {{"bytes", language_extension_.expression(printf_node.size())}}
        );
    } else if (printf_node.is_h2d()) {
        return codegen::InstrumentationInfo(
            node_.element_id(),
            codegen::ElementType_H2DTransfer,
            TargetType_Printf,
            analysis::LoopInfo{},
            {{"bytes", language_extension_.expression(printf_node.size())}}
        );
    } else {
        return codegen::LibraryNodeDispatcher::instrumentation_info();
    }
}

// ============================================================================
// Serializer Implementation
// ============================================================================

nlohmann::json PrintfDataOffloadingNodeSerializer::serialize(const sdfg::data_flow::LibraryNode& library_node) {
    const auto& node = static_cast<const PrintfDataOffloadingNode&>(library_node);
    nlohmann::json j;

    // Library node
    j["type"] = "library_node";
    j["element_id"] = library_node.element_id();

    // Debug info
    auto& debug_info = library_node.debug_info();
    j["debug_info"]["has"] = debug_info.has();
    j["debug_info"]["filename"] = debug_info.filename();
    j["debug_info"]["start_line"] = debug_info.start_line();
    j["debug_info"]["start_column"] = debug_info.start_column();
    j["debug_info"]["end_line"] = debug_info.end_line();
    j["debug_info"]["end_column"] = debug_info.end_column();

    // Library node properties
    j["code"] = std::string(library_node.code().value());

    // Offloading node properties
    sdfg::serializer::JSONSerializer serializer;
    if (node.size().is_null()) {
        j["size"] = nlohmann::json::value_t::null;
    } else {
        j["size"] = serializer.expression(node.size());
    }
    j["transfer_direction"] = static_cast<int8_t>(node.transfer_direction());
    j["buffer_lifecycle"] = static_cast<int8_t>(node.buffer_lifecycle());

    return j;
}

data_flow::LibraryNode& PrintfDataOffloadingNodeSerializer::deserialize(
    const nlohmann::json& j, sdfg::builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
) {
    auto code = j["code"].get<std::string>();
    if (code != LibraryNodeType_Printf_Offloading.value()) {
        throw std::runtime_error("Invalid library node code for PrintfDataOffloadingNode");
    }

    sdfg::serializer::JSONSerializer serializer;
    DebugInfo debug_info = serializer.json_to_debug_info(j["debug_info"]);

    symbolic::Expression size;
    if (!j.contains("size") || j.at("size").is_null()) {
        size = SymEngine::null;
    } else {
        size = symbolic::parse(j.at("size"));
    }
    auto transfer_direction = static_cast<offloading::DataTransferDirection>(j["transfer_direction"].get<int8_t>());
    auto buffer_lifecycle = static_cast<offloading::BufferLifecycle>(j["buffer_lifecycle"].get<int8_t>());

    return builder
        .add_library_node<PrintfDataOffloadingNode>(parent, debug_info, size, transfer_direction, buffer_lifecycle);
}

} // namespace printf_target
} // namespace sdfg
