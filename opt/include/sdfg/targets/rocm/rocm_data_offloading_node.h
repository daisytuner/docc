#pragma once

#include <cstddef>
#include <string>

#include "sdfg/codegen/dispatchers/block_dispatcher.h"
#include "sdfg/codegen/language_extension.h"
#include "sdfg/codegen/utils.h"
#include "sdfg/data_flow/data_flow_graph.h"
#include "sdfg/data_flow/data_flow_node.h"
#include "sdfg/data_flow/library_node.h"
#include "sdfg/element.h"
#include "sdfg/function.h"
#include "sdfg/graph/graph.h"
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

void dispatch_hip_malloc(
    codegen::PrettyPrinter& stream,
    codegen::LanguageExtension& language_extension,
    const std::string& var_name,
    const symbolic::Expression& size,
    const std::string& err_var_name,
    bool async = false
);

void dispatch_hip_free(
    codegen::PrettyPrinter& stream,
    codegen::LanguageExtension& language_extension,
    const std::string& var_name,
    const std::string& err_var_name,
    bool async = false
);

void dispatch_hip_memcpy(
    codegen::PrettyPrinter& stream,
    codegen::LanguageExtension& language_extension,
    const std::string& src_var_name,
    const std::string& dst_var_name,
    const symbolic::Expression& size,
    const std::string& direction,
    const std::string& err_var_name,
    bool async = false
);

} // namespace rocm
} // namespace sdfg
