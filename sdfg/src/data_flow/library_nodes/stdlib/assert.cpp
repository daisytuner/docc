#include "sdfg/data_flow/library_nodes/stdlib/assert.h"

#include <cassert>
#include <cstddef>
#include <cstdio>
#include <memory>
#include <nlohmann/json_fwd.hpp>
#include <string>

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

namespace sdfg {
namespace stdlib {

AssertNode::AssertNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    std::string message
)
    : StdlibNode(
          element_id,
          debug_info,
          vertex,
          parent,
          LibraryNodeType_Assert,
          {},
          {"_arg"},
          true,
          data_flow::ImplementationType_NONE
      ),
      message_(message) {}

std::string& AssertNode::message() { return this->message_; }

const std::string& AssertNode::message() const { return this->message_; }

std::string AssertNode::toStr() const {
    if (this->message_.empty()) {
        return "assert(_arg)";
    } else {
        return "assert(_arg && " + this->message_ + ")";
    }
}

symbolic::SymbolSet AssertNode::symbols() const { return {}; }

std::unique_ptr<data_flow::DataFlowNode> AssertNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::make_unique<AssertNode>(element_id, this->debug_info(), vertex, parent, this->message());
}

void AssertNode::replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) {}

nlohmann::json AssertNodeSerializer::serialize(const data_flow::LibraryNode& library_node) {
    const auto& assert_node = static_cast<const AssertNode&>(library_node);

    nlohmann::json j;
    j["code"] = assert_node.code().value();

    j["message"] = assert_node.message();

    return j;
}

data_flow::LibraryNode& AssertNodeSerializer::deserialize(
    const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
) {
    assert(j.contains("code"));
    assert(j.contains("debug_info"));
    assert(j.contains("message"));

    auto code = j["code"].get<std::string>();
    if (code != LibraryNodeType_Assert.value()) {
        throw InvalidSDFGException("Invalid library node code");
    }

    sdfg::serializer::JSONSerializer serializer;
    DebugInfo debug_info = serializer.json_to_debug_info(j["debug_info"]);

    auto message = j["message"].get<std::string>();

    return builder.add_library_node<AssertNode>(parent, debug_info, message);
}

void AssertNodeDispatcher::escape_message(codegen::PrettyPrinter& stream, const std::string& message) {
    stream << "\"";
    for (unsigned char c : message) {
        switch (c) {
            case '\\':
                stream << "\\\\";
                break;
            case '\"':
                stream << "\\\"";
                break;
            case '\n':
                stream << "\\n";
                break;
            case '\r':
                stream << "\\r";
                break;
            case '\t':
                stream << "\\t";
                break;
            case '\v':
                stream << "\\v";
                break;
            case '\f':
                stream << "\\f";
                break;
            case '\b':
                stream << "\\b";
                break;
            case '\a':
                stream << "\\a";
                break;
            default:
                if (c < 0x20 || c == 0x7F) {
                    char buf[5];
                    std::snprintf(buf, sizeof(buf), "\\x%02X", (unsigned) c);
                    stream << buf;
                } else {
                    stream << static_cast<char>(c);
                }
        }
    }
    stream << "\"";
}

AssertNodeDispatcher::AssertNodeDispatcher(
    codegen::LanguageExtension& language_extension,
    const Function& function,
    const data_flow::DataFlowGraph& data_flow_graph,
    const data_flow::LibraryNode& node
)
    : codegen::LibraryNodeDispatcher(language_extension, function, data_flow_graph, node) {}

void AssertNodeDispatcher::dispatch_code(
    codegen::PrettyPrinter& stream,
    codegen::PrettyPrinter& globals_stream,
    codegen::CodeSnippetFactory& library_snippet_factory
) {
    auto& node = static_cast<const AssertNode&>(this->node_);

#ifndef __APPLE__
    std::string bool_cast = "";
#endif
    // Should be in include stream. Change when include handling is reworked!
    if (this->language_extension_.language() == "C") {
        library_snippet_factory.add_global("#include <assert.h>");
    } else {
        library_snippet_factory.add_global("#include <cassert>");
#ifndef __APPLE__
        bool_cast = "static_cast<bool>";
#endif
    }

#ifdef __APPLE__
    stream << "(__builtin_expect(!bool(" << node.input(0) << "), 0) ? __assert_rtn(__func__, __FILE__, __LINE__, \""
           << node.input(0) << "\"";
#else
    stream << "(" << bool_cast << "(" << node.input(0) << ") ? void (0) : __assert_fail(\"" << node.input(0) << "\"";
#endif
    if (!node.message().empty()) {
        stream << " \" && \" ";
        this->escape_message(stream, node.message());
    }
#ifdef __APPLE__
    stream << ") : (void) 0);" << std::endl;
#else
    stream << ", __FILE__, __LINE__, __extension__ __PRETTY_FUNCTION__));" << std::endl;
#endif
}

} // namespace stdlib
} // namespace sdfg
