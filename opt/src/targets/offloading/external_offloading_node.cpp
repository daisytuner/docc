#include "sdfg/targets/offloading/external_offloading_node.h"
#include <cstddef>
#include <memory>
#include <nlohmann/json_fwd.hpp>
#include <string>
#include <vector>
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/codegen/code_snippet_factory.h"
#include "sdfg/codegen/dispatchers/block_dispatcher.h"
#include "sdfg/codegen/language_extension.h"
#include "sdfg/codegen/utils.h"
#include "sdfg/data_flow/data_flow_graph.h"
#include "sdfg/data_flow/data_flow_node.h"
#include "sdfg/data_flow/library_node.h"
#include "sdfg/element.h"
#include "sdfg/exceptions.h"
#include "sdfg/function.h"
#include "sdfg/graph/graph.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/targets/offloading/data_offloading_node.h"
#include "symengine/symengine_rcp.h"

namespace sdfg {
namespace offloading {

ExternalDataOffloadingNode::ExternalDataOffloadingNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const std::vector<std::string>& inputs,
    const std::string& callee_name,
    size_t transfer_index,
    DataTransferDirection transfer_direction,
    BufferLifecycle buffer_lifecycle
)
    : DataOffloadingNode(
          element_id,
          debug_info,
          vertex,
          parent,
          LibraryNodeType_External_Offloading,
          {},
          inputs,
          transfer_direction,
          buffer_lifecycle,
          SymEngine::null
      ),
      callee_name_(callee_name), transfer_index_(transfer_index) {
    if (is_H2D(transfer_direction)) {
        this->outputs_.push_back("_ret");
        this->inputs_.push_back("_ret");
    } else if (is_D2H(transfer_direction)) {
        this->outputs_.push_back(inputs.at(transfer_index));
        this->inputs_.push_back("_arg" + std::to_string(inputs.size()));
    } else if (is_ALLOC(buffer_lifecycle)) {
        this->outputs_.push_back("_ret");
    } else if (is_FREE(buffer_lifecycle)) {
        this->outputs_.push_back("_ptr");
        this->inputs_.push_back("_ptr");
    }
}

const std::string& ExternalDataOffloadingNode::callee_name() const { return this->callee_name_; }

size_t ExternalDataOffloadingNode::transfer_index() const { return this->transfer_index_; }

std::unique_ptr<data_flow::DataFlowNode> ExternalDataOffloadingNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    std::vector<std::string> inputs(this->inputs().begin(), this->inputs().end() - 1);
    if (!this->has_transfer() && this->is_alloc()) {
        inputs.push_back(this->inputs().back());
    }
    return std::make_unique<ExternalDataOffloadingNode>(
        element_id,
        this->debug_info(),
        vertex,
        parent,
        inputs,
        this->callee_name(),
        this->transfer_index(),
        this->transfer_direction(),
        this->buffer_lifecycle()
    );
}

void ExternalDataOffloadingNode::validate(const Function& function) const {
    if (this->callee_name_.empty()) {
        throw InvalidSDFGException("ExternalDataOffloadingNode: Empty callee name");
    }
    size_t inputs_size = this->inputs().size();
    if (this->has_transfer() || this->is_free()) {
        inputs_size--;
    }
    if (this->transfer_index_ >= inputs_size) {
        throw InvalidSDFGException("ExternalDataOffloadingNode: Transfer index out of range");
    }

    // Prevent copy-in and free
    if (this->is_h2d() && this->is_free()) {
        throw InvalidSDFGException("ExternalDataOffloadingNode: Combination copy-in and free is not allowed");
    }

    // Prevent copy-out and alloc
    if (this->is_d2h() && this->is_alloc()) {
        throw InvalidSDFGException("ExternalDataOffloadingNode: Combination copy-out and alloc is not allowed");
    }
}

bool ExternalDataOffloadingNode::blocking() const { return true; }

bool ExternalDataOffloadingNode::redundant_with(const offloading::DataOffloadingNode& other) const {
    if (!offloading::DataOffloadingNode::redundant_with(other)) {
        return false;
    }

    auto& other_node = static_cast<const ExternalDataOffloadingNode&>(other);
    if (this->callee_name() != other_node.callee_name()) {
        return false;
    }

    return true;
}

bool ExternalDataOffloadingNode::equal_with(const offloading::DataOffloadingNode& other) const {
    if (!offloading::DataOffloadingNode::equal_with(other)) {
        return false;
    }

    auto& other_node = static_cast<const ExternalDataOffloadingNode&>(other);
    if (this->callee_name() != other_node.callee_name()) {
        return false;
    }
    if (this->transfer_index() != other_node.transfer_index()) {
        return false;
    }

    return true;
}

ExternalDataOffloadingNodeDispatcher::ExternalDataOffloadingNodeDispatcher(
    codegen::LanguageExtension& language_extension,
    const Function& function,
    const data_flow::DataFlowGraph& data_flow_graph,
    const data_flow::LibraryNode& node
)
    : codegen::LibraryNodeDispatcher(language_extension, function, data_flow_graph, node) {}

void ExternalDataOffloadingNodeDispatcher::dispatch_code(
    codegen::PrettyPrinter& stream,
    codegen::PrettyPrinter& globals_stream,
    codegen::CodeSnippetFactory& library_snippet_factory
) {
    auto& offloading_node = static_cast<const ExternalDataOffloadingNode&>(this->node_);

    if (offloading_node.has_transfer()) {
        if (offloading_node.is_alloc()) {
            stream << "// External alloc" << std::endl;
            stream << offloading_node.output(0) << " = " << this->language_extension_.external_prefix()
                   << offloading_node.callee_name() << "_alloc_" << offloading_node.transfer_index() << "(";
            for (size_t i = 0; i < offloading_node.inputs().size() - 1; i++) {
                stream << offloading_node.input(i);
                if (i < offloading_node.inputs().size() - 2) {
                    stream << ", ";
                }
            }
            stream << ");" << std::endl;
        }

        if (offloading_node.is_h2d()) {
            stream << "// External copy-in" << std::endl;
            stream << offloading_node.output(0) << " = " << this->language_extension_.external_prefix()
                   << offloading_node.callee_name() << "_in_" << offloading_node.transfer_index() << "(";
            for (size_t i = 0; i < offloading_node.inputs().size(); i++) {
                stream << offloading_node.input(i);
                if (i < offloading_node.inputs().size() - 1) {
                    stream << ", ";
                }
            }
            stream << ");" << std::endl;
        } else if (offloading_node.is_d2h()) {
            stream << "// External copy-out" << std::endl;
            stream << this->language_extension_.external_prefix() << offloading_node.callee_name() << "_out_"
                   << offloading_node.transfer_index() << "(";
            for (size_t i = 0; i < offloading_node.inputs().size(); i++) {
                stream << offloading_node.input(i);
                if (i < offloading_node.inputs().size() - 1) {
                    stream << ", ";
                }
            }
            stream << ");" << std::endl;
        }

        if (offloading_node.is_free()) {
            stream << "// External free" << std::endl;
            stream << offloading_node.inputs().back() << " = " << this->language_extension_.external_prefix()
                   << offloading_node.callee_name() << "_free_" << offloading_node.transfer_index() << "(";
            for (size_t i = 0; i < offloading_node.inputs().size(); i++) {
                stream << offloading_node.input(i);
                if (i < offloading_node.inputs().size() - 1) {
                    stream << ", ";
                }
            }
            stream << ");" << std::endl;
        }
    } else if (offloading_node.is_alloc()) {
        if (offloading_node.is_alloc()) {
            stream << "// External alloc" << std::endl;
            stream << offloading_node.output(0) << " = " << this->language_extension_.external_prefix()
                   << offloading_node.callee_name() << "_alloc_" << offloading_node.transfer_index() << "(";
            for (size_t i = 0; i < offloading_node.inputs().size(); i++) {
                stream << offloading_node.input(i);
                if (i < offloading_node.inputs().size() - 1) {
                    stream << ", ";
                }
            }
            stream << ");" << std::endl;
        }
    } else if (offloading_node.is_free()) {
        if (offloading_node.is_free()) {
            stream << "// External free" << std::endl;
            stream << offloading_node.output(0) << " = " << this->language_extension_.external_prefix()
                   << offloading_node.callee_name() << "_free_" << offloading_node.transfer_index() << "(";
            for (size_t i = 0; i < offloading_node.inputs().size(); i++) {
                stream << offloading_node.input(i);
                if (i < offloading_node.inputs().size() - 1) {
                    stream << ", ";
                }
            }
            stream << ");" << std::endl;
        }
    }
}

nlohmann::json ExternalDataOffloadingNodeSerializer::serialize(const sdfg::data_flow::LibraryNode& library_node) {
    const auto& node = static_cast<const ExternalDataOffloadingNode&>(library_node);
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
    j["inputs"] = nlohmann::json::array();
    for (size_t i = 0; i < node.inputs().size() - 1; i++) {
        j["inputs"].push_back(node.input(i));
    }
    if (!node.has_transfer() && node.is_alloc()) {
        j["inputs"].push_back(node.inputs().back());
    }
    j["callee_name"] = node.callee_name();
    j["transfer_index"] = node.transfer_index();
    j["transfer_direction"] = static_cast<int8_t>(node.transfer_direction());
    j["buffer_lifecycle"] = static_cast<int8_t>(node.buffer_lifecycle());

    return j;
}

data_flow::LibraryNode& ExternalDataOffloadingNodeSerializer::deserialize(
    const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
) {
    auto code = j["code"].get<std::string>();
    if (code != LibraryNodeType_External_Offloading.value()) {
        throw std::runtime_error("Invalid library node code");
    }

    sdfg::serializer::JSONSerializer serializer;
    DebugInfo debug_info = serializer.json_to_debug_info(j["debug_info"]);

    std::vector<std::string> inputs = j["inputs"].get<std::vector<std::string>>();
    std::string callee_name = j["callee_name"].get<std::string>();
    size_t transfer_index = j["transfer_index"].get<size_t>();
    auto transfer_direction = static_cast<offloading::DataTransferDirection>(j["transfer_direction"].get<int8_t>());
    auto buffer_lifecycle = static_cast<offloading::BufferLifecycle>(j["buffer_lifecycle"].get<int8_t>());

    return builder.add_library_node<ExternalDataOffloadingNode>(
        parent, debug_info, inputs, callee_name, transfer_index, transfer_direction, buffer_lifecycle
    );
}

} // namespace offloading
} // namespace sdfg
