#pragma once

#include "sdfg/data_flow/library_node.h"

#include "sdfg/codegen/dispatchers/block_dispatcher.h"
#include "sdfg/serializer/json_serializer.h"

namespace sdfg {

namespace data_flow {

// Asynchronous global->shared copy plus its ordering primitives, used by the
// software-pipelining transformation. On CUDA these lower to the cp.async
// (LDGSTS) family via <cuda_pipeline.h>; on ROCm — which has no portable
// cp.async — the copy lowers to a synchronous store and commit/wait are no-ops,
// so the result stays correct (the pipeline just doesn't overlap).
inline LibraryNodeCode LibraryNodeType_CpAsyncCopy{"cp_async_copy"};
inline LibraryNodeCode LibraryNodeType_PipelineCommit{"pipeline_commit"};
inline LibraryNodeCode LibraryNodeType_PipelineWait{"pipeline_wait"};

/**
 * @brief Asynchronous copy of @p bytes from `_src` (global) to `_dst` (shared).
 * Both connectors carry addresses (reference memlets). The copy is only
 * guaranteed complete after a matching PipelineWaitNode.
 */
class CpAsyncCopyNode : public LibraryNode {
    size_t bytes_;

public:
    CpAsyncCopyNode(
        size_t element_id, const DebugInfo& debug_info, const graph::Vertex vertex, DataFlowGraph& parent, size_t bytes
    );

    size_t bytes() const { return bytes_; }

    void validate(const Function& function) const override;
    symbolic::SymbolSet symbols() const override;
    std::unique_ptr<DataFlowNode> clone(size_t element_id, const graph::Vertex vertex, DataFlowGraph& parent)
        const override;
    void replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) override;
    void replace(const symbolic::ExpressionMapping& replacements) override;
};

/** @brief Commit the outstanding async copies as one pipeline group (cp.async.commit_group). */
class PipelineCommitNode : public LibraryNode {
public:
    PipelineCommitNode(size_t element_id, const DebugInfo& debug_info, const graph::Vertex vertex, DataFlowGraph& parent);

    void validate(const Function& function) const override;
    symbolic::SymbolSet symbols() const override;
    std::unique_ptr<DataFlowNode> clone(size_t element_id, const graph::Vertex vertex, DataFlowGraph& parent)
        const override;
    void replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) override;
    void replace(const symbolic::ExpressionMapping& replacements) override;
};

/**
 * @brief Wait until at most @p keep_outstanding prior commit groups remain in flight.
 *
 * @p loads_per_group is the number of individual vector-memory loads one commit
 * group expands to on hardware without a group abstraction (CDNA: each
 * `global_load_lds` word is one vmcnt tick), so the ROCm/CDNA lowering can wait
 * on the flat vmcnt counter as `vmcnt(keep_outstanding * loads_per_group)`.
 * Ignored by the CUDA lowering, which counts commit groups directly.
 */
class PipelineWaitNode : public LibraryNode {
    size_t keep_outstanding_;
    size_t loads_per_group_;

public:
    PipelineWaitNode(
        size_t element_id,
        const DebugInfo& debug_info,
        const graph::Vertex vertex,
        DataFlowGraph& parent,
        size_t keep_outstanding,
        size_t loads_per_group = 1
    );

    size_t keep_outstanding() const { return keep_outstanding_; }
    size_t loads_per_group() const { return loads_per_group_; }
    void set_loads_per_group(size_t loads_per_group) { loads_per_group_ = loads_per_group; }

    void validate(const Function& function) const override;
    symbolic::SymbolSet symbols() const override;
    std::unique_ptr<DataFlowNode> clone(size_t element_id, const graph::Vertex vertex, DataFlowGraph& parent)
        const override;
    void replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) override;
    void replace(const symbolic::ExpressionMapping& replacements) override;
};

// ---- Serializers ---------------------------------------------------------

class CpAsyncCopyNodeSerializer : public serializer::LibraryNodeSerializer {
public:
    nlohmann::json serialize(const sdfg::data_flow::LibraryNode& library_node) override;
    data_flow::LibraryNode& deserialize(
        const nlohmann::json& j,
        sdfg::builder::StructuredSDFGBuilder& builder,
        sdfg::structured_control_flow::Block& parent
    ) override;
};

class PipelineCommitNodeSerializer : public serializer::LibraryNodeSerializer {
public:
    nlohmann::json serialize(const sdfg::data_flow::LibraryNode& library_node) override;
    data_flow::LibraryNode& deserialize(
        const nlohmann::json& j,
        sdfg::builder::StructuredSDFGBuilder& builder,
        sdfg::structured_control_flow::Block& parent
    ) override;
};

class PipelineWaitNodeSerializer : public serializer::LibraryNodeSerializer {
public:
    nlohmann::json serialize(const sdfg::data_flow::LibraryNode& library_node) override;
    data_flow::LibraryNode& deserialize(
        const nlohmann::json& j,
        sdfg::builder::StructuredSDFGBuilder& builder,
        sdfg::structured_control_flow::Block& parent
    ) override;
};

// ---- Dispatchers ---------------------------------------------------------

class CpAsyncCopyNodeDispatcher : public codegen::LibraryNodeDispatcher {
public:
    CpAsyncCopyNodeDispatcher(
        codegen::LanguageExtension& language_extension,
        const Function& function,
        const data_flow::DataFlowGraph& data_flow_graph,
        const data_flow::CpAsyncCopyNode& node
    );

    void dispatch_code_with_edges(
        codegen::CodegenOutput& out,
        std::vector<codegen::DispatchInput>& inputs,
        std::vector<codegen::DispatchOutput>& outputs
    ) override;
};

class PipelineCommitNodeDispatcher : public codegen::LibraryNodeDispatcher {
public:
    PipelineCommitNodeDispatcher(
        codegen::LanguageExtension& language_extension,
        const Function& function,
        const data_flow::DataFlowGraph& data_flow_graph,
        const data_flow::PipelineCommitNode& node
    );

    void dispatch(
        codegen::PrettyPrinter& stream,
        codegen::PrettyPrinter& globals_stream,
        codegen::CodeSnippetFactory& library_snippet_factory
    ) override;
};

class PipelineWaitNodeDispatcher : public codegen::LibraryNodeDispatcher {
public:
    PipelineWaitNodeDispatcher(
        codegen::LanguageExtension& language_extension,
        const Function& function,
        const data_flow::DataFlowGraph& data_flow_graph,
        const data_flow::PipelineWaitNode& node
    );

    void dispatch(
        codegen::PrettyPrinter& stream,
        codegen::PrettyPrinter& globals_stream,
        codegen::CodeSnippetFactory& library_snippet_factory
    ) override;
};

} // namespace data_flow
} // namespace sdfg
