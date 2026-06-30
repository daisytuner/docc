#pragma once

#include "sdfg/data_flow/library_node.h"
#include "sdfg/data_flow/library_nodes/stdlib/stdlib_node.h"

#include "sdfg/codegen/dispatchers/block_dispatcher.h"
#include "sdfg/serializer/json_serializer.h"

namespace sdfg {
namespace stdlib {

inline data_flow::LibraryNodeCode LibraryNodeType_Memcpy("Memcpy");

class MemcpyNode : public StdlibNode {
private:
    symbolic::Expression count_;

public:
    MemcpyNode(
        size_t element_id,
        const DebugInfo& debug_info,
        const graph::Vertex vertex,
        data_flow::DataFlowGraph& parent,
        const symbolic::Expression count
    );

    const symbolic::Expression count() const;

    void validate(const Function& function) const override;

    symbolic::SymbolSet symbols() const override;

    void replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) override;

    void replace(const symbolic::ExpressionMapping& replacements) override;

    std::unique_ptr<DataFlowNode> clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent)
        const override;

    data_flow::PointerAccessType pointer_access_type(int input_idx) const override;

    std::string toStr() const override;
};

class MemcpyNodeSerializer : public serializer::LibraryNodeSerializer {
public:
    nlohmann::json serialize(const data_flow::LibraryNode& library_node) override;

    data_flow::LibraryNode& deserialize(
        const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
    ) override;
};

class MemcpyNodeDispatcher : public codegen::LibraryNodeDispatcher {
public:
    MemcpyNodeDispatcher(
        codegen::LanguageExtension& language_extension,
        const Function& function,
        const data_flow::DataFlowGraph& data_flow_graph,
        const MemcpyNode& node
    );

    void dispatch_code_with_edges(
        codegen::CodegenOutput& out,
        std::vector<codegen::DispatchInput>& inputs,
        std::vector<codegen::DispatchOutput>& outputs
    ) override;
};

MemcpyNode& add_memcpy_node(
    builder::StructuredSDFGBuilder& builder,
    Block& block,
    const std::string& src_ptr,
    const std::string& dst_ptr,
    const symbolic::Expression& count,
    const types::IType& ptr_type,
    DebugInfo debug_info = DebugInfo()
);

std::tuple<Block&, MemcpyNode&> add_memcpy_block(
    builder::StructuredSDFGBuilder& builder,
    Sequence& parent,
    const std::string& src_ptr,
    const std::string& dst_ptr,
    const symbolic::Expression& count,
    const types::IType& ptr_type,
    DebugInfo debug_info = DebugInfo()
);


} // namespace stdlib
} // namespace sdfg
