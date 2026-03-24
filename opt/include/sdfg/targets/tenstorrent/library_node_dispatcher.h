#pragma once

#include "sdfg/codegen/dispatchers/block_dispatcher.h"
#include "sdfg/data_flow/library_node.h"

namespace sdfg::tenstorrent {

template<typename NodeType>
class LibraryNodeDispatcherBase : public codegen::LibraryNodeDispatcher {
public:
    LibraryNodeDispatcherBase(
        codegen::LanguageExtension& language_extension,
        const Function& function,
        const data_flow::DataFlowGraph& data_flow_graph,
        const NodeType& node
    )
        : LibraryNodeDispatcher(language_extension, function, data_flow_graph, node) {}

    virtual bool begin_node(codegen::PrettyPrinter& stream) {
        stream << "{" << std::endl;
        stream.changeIndent(+4);
        return true;
    }

    virtual void end_node(codegen::PrettyPrinter& stream, bool has_declaration) {
        if (has_declaration) {
            stream.changeIndent(-4);
            stream << "}" << std::endl;
        }
    }

protected:
    const std::string& require_param_as_var_equivalent(
        codegen::PrettyPrinter& stream, const data_flow::AccessNode* node, const std::string& name
    ) {
        if (auto const_node = dynamic_cast<const data_flow::ConstantNode*>(node)) {
            auto val = const_node->data();
            stream << language_extension_.declaration(name, const_node->type()) << " = " << val << ";" << std::endl;
            return name;
        } else {
            return node->data();
        }
    }
};

} // namespace sdfg::tenstorrent
