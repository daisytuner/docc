#pragma once

#include "sdfg/codegen/dispatchers/block_dispatcher.h"
#include "sdfg/serializer/json_serializer.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/targets/offloading/data_offloading_node.h"

namespace sdfg {
namespace tenstorrent {

inline data_flow::LibraryNodeCode LibraryNodeType_Tenstorrent_Offloading("TTOffloading");

class TTDataOffloadingNode : public offloading::DataOffloadingNode {
    friend class sdfg::builder::SDFGBuilder;
    friend class sdfg::builder::StructuredSDFGBuilder;

private:
    bool blocking_;
    symbolic::Expression page_size_;
    int cq_no_;
    std::string device_handle_;

public:
    TTDataOffloadingNode(
        size_t element_id,
        const DebugInfo& debug_info,
        const graph::Vertex vertex,
        data_flow::DataFlowGraph& parent,
        bool blocking,
        std::string device_handle,
        symbolic::Expression size,
        symbolic::Expression page_size,
        offloading::DataTransferDirection transfer_direction,
        offloading::BufferLifecycle allocation_handling,
        int cq_no = 0
    );

    void validate(const Function& function) const override;

    virtual bool blocking() const override;
    const symbolic::Expression page_size() const;
    int cq_no() const;
    std::string device_handle() const;

    virtual const symbolic::Expression alloc_size() const override;

    std::unique_ptr<data_flow::DataFlowNode>
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const override;

    symbolic::SymbolSet symbols() const override;

    void replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) override;

    virtual bool redundant_with(const offloading::DataOffloadingNode& other) const override;

    virtual bool equal_with(const offloading::DataOffloadingNode& other) const override;
};

class TTDataOffloadingNodeDispatcher : public codegen::LibraryNodeDispatcher {
public:
    TTDataOffloadingNodeDispatcher(
        codegen::LanguageExtension& language_extension,
        const Function& function,
        const data_flow::DataFlowGraph& data_flow_graph,
        const data_flow::LibraryNode& node
    );

    static void dispatch_enqueue_read_safe(
        codegen::LanguageExtension& language_extension,
        codegen::PrettyPrinter& stream,
        codegen::PrettyPrinter& globals_stream,
        codegen::CodeSnippetFactory& library_snippet_factory,
        const std::string& src_ptr,
        const std::string& dst_ptr,
        const std::string& dev_handle,
        symbolic::Expression size,
        bool blocking,
        int cq_no = 0
    );

    static void dispatch_enqueue_read_full(
        codegen::PrettyPrinter& stream,
        codegen::PrettyPrinter& globals_stream,
        codegen::CodeSnippetFactory& library_snippet_factory,
        const std::string& src_ptr,
        const std::string& dst_ptr,
        const std::string& dev_handle,
        bool blocking,
        int cq_no = 0
    );

    static void dispatch_enqueue_write_safe(
        codegen::LanguageExtension& language_extension,
        codegen::PrettyPrinter& stream,
        codegen::PrettyPrinter& globals_stream,
        codegen::CodeSnippetFactory& library_snippet_factory,
        const std::string& src_ptr,
        const std::string& dst_ptr,
        const std::string& dev_handle,
        symbolic::Expression size,
        bool blocking,
        int cq_no = 0
    );

    static void dispatch_enqueue_write_full(
        codegen::PrettyPrinter& stream,
        codegen::PrettyPrinter& globals_stream,
        codegen::CodeSnippetFactory& library_snippet_factory,
        const std::string& src_ptr,
        const std::string& dst_ptr,
        const std::string& dev_handle,
        bool blocking,
        int cq_no = 0
    );

    static void dispatch_allocate(
        codegen::PrettyPrinter& stream,
        codegen::PrettyPrinter& globals_stream,
        codegen::CodeSnippetFactory& library_snippet_factory,
        codegen::LanguageExtension& language_extension,
        const std::string& var_name,
        const std::string& device_handle,
        const symbolic::Expression size,
        const symbolic::Expression page_size
    );

    void dispatch(
        codegen::PrettyPrinter& stream,
        codegen::PrettyPrinter& globals_stream,
        codegen::CodeSnippetFactory& library_snippet_factory
    ) override;

    virtual codegen::InstrumentationInfo instrumentation_info() const override;
};

class TTDataOffloadingNodeSerializer : public serializer::LibraryNodeSerializer {
public:
    nlohmann::json serialize(const sdfg::data_flow::LibraryNode& library_node) override;

    data_flow::LibraryNode& deserialize(
        const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
    ) override;
};

} // namespace tenstorrent
} // namespace sdfg
