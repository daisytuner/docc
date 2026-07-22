#pragma once

#include <cstddef>

#include "sdfg/codegen/dispatchers/block_dispatcher.h"
#include "sdfg/data_flow/data_flow_graph.h"
#include "sdfg/data_flow/data_flow_node.h"
#include "sdfg/data_flow/library_node.h"
#include "sdfg/element.h"
#include "sdfg/function.h"
#include "sdfg/graph/graph.h"
#include "sdfg/serializer/json_serializer.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/targets/offloading/data_offloading_node.h"

namespace sdfg {
namespace rocm {

inline data_flow::LibraryNodeCode LibraryNodeType_ROCM_Offloading("ROCMOffloading");

class ROCMDataOffloadingNode : public offloading::DataOffloadingNode {
private:
    symbolic::Expression device_id_;

public:
    ROCMDataOffloadingNode(
        size_t element_id,
        const DebugInfo& debug_info,
        const graph::Vertex vertex,
        data_flow::DataFlowGraph& parent,
        offloading::DataTransferDirection transfer_direction,
        offloading::BufferLifecycle buffer_lifecycle,
        symbolic::Expression size,
        symbolic::Expression device_id
    );

    void validate(const Function& function) const override;

    const symbolic::Expression device_id() const;

    std::unique_ptr<data_flow::DataFlowNode>
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const override;

    symbolic::SymbolSet symbols() const override;

    void replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) override;

    virtual bool blocking() const override;

    virtual bool redundant_with(const offloading::DataOffloadingNode& other) const override;

    virtual bool equal_with(const offloading::DataOffloadingNode& other) const override;

    virtual bool is_same_target(const DataOffloadingNode& other) const override;
};

class ROCMDataOffloadingNodeDispatcher : public codegen::LibraryNodeDispatcher {
public:
    ROCMDataOffloadingNodeDispatcher(
        codegen::LanguageExtension& language_extension,
        const Function& function,
        const data_flow::DataFlowGraph& data_flow_graph,
        const data_flow::LibraryNode& node
    );

    void dispatch_code_with_edges(
        codegen::CodegenOutput& out,
        std::vector<codegen::DispatchInput>& inputs,
        std::vector<codegen::DispatchOutput>& outputs
    ) override;

    virtual codegen::InstrumentationInfo instrumentation_info() const override;
};

class ROCMDataOffloadingNodeSerializer : public offloading::DataOffloadingNodeSerializer {
public:
    nlohmann::json serialize(const sdfg::data_flow::LibraryNode& library_node) override;

    data_flow::LibraryNode& deserialize(
        const nlohmann::json& j,
        sdfg::builder::StructuredSDFGBuilder& builder,
        sdfg::structured_control_flow::Block& parent
    ) override;
};

// -----------------------------------------------------------------------------
// Shared data-transfer extraction helpers
//
// These build the ROCm offloading nodes (device alloc/free, H2D/D2H copies) that
// all ROCm library-node data-transfer extractions (BLAS, stdlib, FFT, ...) need.
// They are centralized here so each extraction only expresses *which* buffers to
// move, not *how* the offloading nodes are constructed.
// -----------------------------------------------------------------------------

/// Create an unmanaged device container (AMD_Generic storage) mirroring @p type.
std::string create_device_container(
    builder::StructuredSDFGBuilder& builder, const types::Pointer& type, const symbolic::Expression& size
);

/// Insert a device allocation block before @p block.
void create_allocate(
    builder::StructuredSDFGBuilder& builder,
    structured_control_flow::Sequence& sequence,
    structured_control_flow::Block& block,
    const std::string& device_container,
    const symbolic::Expression& size,
    const types::Pointer& type,
    const DebugInfo& debug_info
);

/// Insert a device deallocation block after @p block.
void create_deallocate(
    builder::StructuredSDFGBuilder& builder,
    structured_control_flow::Sequence& sequence,
    structured_control_flow::Block& block,
    const std::string& device_container,
    const types::Pointer& type,
    const DebugInfo& debug_info
);

/// Insert an H2D copy block before @p block (buffer must already be allocated).
void create_copy_to_device(
    builder::StructuredSDFGBuilder& builder,
    structured_control_flow::Sequence& sequence,
    structured_control_flow::Block& block,
    const std::string& host_container,
    const std::string& device_container,
    const symbolic::Expression& size,
    const types::Pointer& type,
    const DebugInfo& debug_info
);

/// Insert a D2H copy block after @p block (buffer stays allocated).
void create_copy_from_device(
    builder::StructuredSDFGBuilder& builder,
    structured_control_flow::Sequence& sequence,
    structured_control_flow::Block& block,
    const std::string& host_container,
    const std::string& device_container,
    const symbolic::Expression& size,
    const types::Pointer& type,
    const DebugInfo& debug_info
);

/// Insert an allocate + H2D copy block before @p block.
void create_copy_to_device_with_allocation(
    builder::StructuredSDFGBuilder& builder,
    structured_control_flow::Sequence& sequence,
    structured_control_flow::Block& block,
    const std::string& host_container,
    const std::string& device_container,
    const symbolic::Expression& size,
    const types::Pointer& type,
    const DebugInfo& debug_info
);

/// Insert a D2H copy + deallocate block after @p block.
void create_copy_from_device_with_deallocation(
    builder::StructuredSDFGBuilder& builder,
    structured_control_flow::Sequence& sequence,
    structured_control_flow::Block& block,
    const std::string& host_container,
    const std::string& device_container,
    const symbolic::Expression& size,
    const types::Pointer& type,
    const DebugInfo& debug_info
);

} // namespace rocm
} // namespace sdfg
